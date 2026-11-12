"""Vision pick: a UR5e with a gripper and a wrist camera finds the blocks of
examples/vision-pick.json (a Robotiq 2F-85) or examples/vision-pick-suction.json
(the same cell with a Robotiq ePick) through a Viam vision service and a
viam-labs:grasp-pose tracker, and picks them into bins, so two perception
pipelines can be compared.

Usage: uv run python examples/vision_pick.py <machine address> <api key id> <api key>
           [--vision shapes|pointcloud]

Take the address and an API key from the machine's CONNECT tab.

- shapes (default): the `shape-grasps` tracker over `shape-segments`,
  detections-to-segments over the shape-finder detector `shapes`. Each object
  is labelled `<shape>-<colour>`; a block goes in the bin of its colour.
- pointcloud: the `object-grasps` tracker over `objects`, obstacles-pointcloud.
  Objects carry no label; every block goes in the clear bin.

Each round the wrist camera surveys the table from straight above and the
tracker's `grasps` command returns each object as a fitted box (centre, yaw,
size, label) and the ranked grasp poses, in the world frame. The script tries
the best grasp of the nearest object that has grasps, then that object's next
grasp, then the next object, with the other objects' boxes as obstacles for
the motion planner, drops the block in its bin, then surveys again. After each
survey it scores the objects against the blocks' true poses (the world's
get_prop_pose): position and yaw error, missed and spurious objects, and the
colour in shapes mode. At the end it reports where every block landed.
"""

import argparse
import asyncio
import math
import time

import numpy as np
from viam.components.arm import Arm
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.components.pose_tracker import PoseTracker
from viam.proto.component.arm import JointPositions
from viam.proto.common import GeometriesInFrame, Geometry, Pose, PoseInFrame, RectangularPrism, Vector3, WorldState
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

ARM, GRIPPER, CAMERA, WORLD = "pick-arm", "pick-gripper", "wrist-cam", "sim-world"
SERVICES = {"shapes": "shape-segments", "pointcloud": "objects"}
TRACKERS = {"shapes": "shape-grasps", "pointcloud": "object-grasps"}
SURVEY_MM = (500, 0, 450)  # the camera, straight over the blocks; its view covers ~690 x 520 mm
SURVEY_HEADING_DEG = 180  # the camera's x (its image's width) along world -x
# the arm's joints that put the camera at SURVEY_MM, heading SURVEY_HEADING_DEG, with the arm out of its view
# (measured on the machine; the camera rides the arm, so both grippers' cells share them)
SURVEY_JOINTS_DEG = [-206, -90, 91, -91, -90, 64]
BLOCKS = [f"{colour}_{i}" for i in range(1, 5) for colour in ("red", "blue")]
SQUARES = {"red_1", "blue_2", "red_3", "blue_4"}  # the 50 mm squares; the rest are 40 x 80 mm
BLOCK_H_MM = 40
TABLE_TOP_MM = 5  # 5 mm over the floor plane: a table flush with it z-fights in the renderer
BINS = {"red": (380, -450), "blue": (620, -450), "clear": (500, 450)}  # centres (mm)
BIN_HALF_MM = 112.5  # the bin mesh's 150 mm footprint scaled 1.5x, halved
CARRY_MM = 250  # gripper height for moving between table and bin
BIN_RIM_MM = 80  # the bin mesh's height, unscaled
RELEASE_MM = TABLE_TOP_MM + BIN_RIM_MM + BLOCK_H_MM + 30  # let go with the block just over the rim
SLOT_MM = 50  # drop in the bin's quarters in turn, from its centre
MATCH_MM = 40  # an object within this of a block's centre saw that block
MAX_PICKS, MAX_FAILURES = 16, 3


def yaw_error(found: float, truth: float, square: bool) -> float:
    period = 90 if square else 180  # a square looks the same turned a quarter
    return abs((found - truth + period / 2) % period - period / 2)


def score(found: list[dict], truths: dict[str, dict]) -> dict:
    """Objects matched to the blocks on the table by nearest centre."""
    matched, errors = set(), []
    for item in found:
        near = [
            (math.dist((item["x"], item["y"]), (t["x"], t["y"])), name)
            for name, t in truths.items() if name not in matched
        ]
        if near and min(near)[0] < MATCH_MM:
            distance, name = min(near)
            matched.add(name)
            truth = truths[name]
            colour = item.get("label", "").rsplit("-", 1)[-1] or None
            errors.append({
                "block": name, "mm": round(distance, 1),
                "yaw_deg": round(yaw_error(item["yaw"], truth["yaw"], truth["square"]), 1),
                "colour_ok": None if colour is None else colour == name.split("_")[0],
            })
    return {
        "seen": len(errors), "missed": sorted(set(truths) - matched),
        "spurious": len(found) - len(errors), "errors": errors,
    }


def in_bin(x_mm: float, y_mm: float) -> str | None:
    return next((name for name, (bx, by) in BINS.items()
                 if abs(x_mm - bx) < BIN_HALF_MM and abs(y_mm - by) < BIN_HALF_MM), None)


def pose_in_world(pose: dict, **changes: float) -> PoseInFrame:
    """A pose from the tracker (mm, orientation vector in degrees), with some fields replaced."""
    p = {**pose, **changes}
    return PoseInFrame(reference_frame="world", pose=Pose(
        x=p["x"], y=p["y"], z=p["z"], o_x=p["o_x"], o_y=p["o_y"], o_z=p["o_z"], theta=p["theta"]))


def box(obj: dict) -> Geometry:
    """An object's fitted box from the tracker, labelled by its object name: class labels repeat,
    and the planner keys geometries by label."""
    c, d = obj["center"], obj["dims_mm"]
    return Geometry(
        center=Pose(x=c["x"], y=c["y"], z=c["z"], o_x=c["o_x"], o_y=c["o_y"], o_z=c["o_z"], theta=c["theta"]),
        box=RectangularPrism(dims_mm=Vector3(x=d["x"], y=d["y"], z=d["z"])), label=obj["name"])


class Cell:
    def __init__(self, world, arm, gripper, motion, tracker, mode: str) -> None:
        self.world, self.arm, self.gripper, self.motion, self.tracker, self.mode = world, arm, gripper, motion, tracker, mode
        self.dropped = {name: 0 for name in BINS}  # blocks dropped in each bin so far

    async def truths(self) -> dict[str, dict]:
        """The blocks still on the table, from the sim."""
        found = {}
        for name in BLOCKS:
            pose = (await self.world.do_command({"command": "get_prop_pose", "prop": name}))["pose"]
            x, y, z = (v * 1000 for v in pose["position"])
            if z < TABLE_TOP_MM + BLOCK_H_MM and in_bin(x, y) is None:
                w, _, _, k = pose["orientation_wxyz"]
                yaw = math.degrees(2 * math.atan2(k, w))  # the blocks stay flat: a turn about z
                found[name] = {"x": x, "y": y, "yaw": yaw, "square": name in SQUARES}
        return found

    async def survey(self) -> list[dict]:
        """The objects in view, nearest the arm first: centre, yaw and label, their box for the
        planner, and their grasps, best first."""
        # a fixed survey pose in joints: the planner can reach the camera pose with an arm link folded into its view
        await self.arm.move_to_joint_positions(JointPositions(values=SURVEY_JOINTS_DEG))
        try:
            found = await self.tracker.do_command({"grasps": {}})
        except Exception as error:
            raise RuntimeError(f"grasp tracker {TRACKERS[self.mode]}: {error}") from error
        return [
            {"x": o["center"]["x"], "y": o["center"]["y"], "yaw": o["yaw_deg"],
             "label": o["label"] if self.mode == "shapes" else "", "box": box(o),
             "grasps": [g for g in found["grasps"] if g["object"] == o["name"]]}
            for o in found["objects"]
        ]

    async def move(self, destination: PoseInFrame, obstacles: list[Geometry]) -> None:
        state = WorldState(obstacles=[GeometriesInFrame(reference_frame="world", geometries=obstacles)])
        await self.motion.move(component_name=GRIPPER, destination=destination, world_state=state)

    async def pick(self, item: dict, others: list[Geometry]) -> bool:
        await self.gripper.open()
        error = None
        for grasp in item["grasps"]:  # best first; the next one when a plan fails
            at = grasp["pose"]
            try:
                await self.move(pose_in_world(at, z=at["z"] + 100), others)
                await self.move(pose_in_world(at), others)
                break
            except Exception as e:
                error = e
        else:
            raise error or RuntimeError("no grasps")
        await self.gripper.grab()
        await asyncio.sleep(0.5)
        held = (await self.gripper.is_holding_something()).is_holding_something
        await self.move(pose_in_world(at, z=CARRY_MM), others)
        if not held:
            return False
        colour = item["label"].rsplit("-", 1)[-1] if self.mode == "shapes" else "clear"
        name = colour if colour in BINS else "clear"
        slot = self.dropped[name] % 4
        self.dropped[name] += 1
        bx, by = BINS[name]
        bx, by = bx + (SLOT_MM if slot % 2 else -SLOT_MM), by + (SLOT_MM if slot >= 2 else -SLOT_MM)
        for z in (CARRY_MM, RELEASE_MM):
            await self.move(pose_in_world(at, x=bx, y=by, z=z), others)
        await self.gripper.open()
        await asyncio.sleep(0.5)  # let it land
        await self.move(pose_in_world(at, x=bx, y=by, z=CARRY_MM), others)
        return True

    async def run(self) -> dict:
        """Survey and pick until nothing is left. A plan that fails moves on to
        the next object in the same survey; a grasp that doesn't hold surveys
        again (the block may have moved); an object that failed twice is left
        alone; three failures in a row stop the run."""
        reports, picked, failures, tried = [], 0, 0, []  # tried: [(x, y), failures] per object

        def tries(item: dict) -> list:
            here = (item["x"], item["y"])
            entry = next((t for t in tried if math.dist(t[0], here) < MATCH_MM), None)
            if entry is None:
                entry = [here, 0]
                tried.append(entry)
            return entry

        for _ in range(MAX_PICKS):
            truths = await self.truths()
            found = await self.survey()
            report = score(found, truths)  # everything seen, before choosing what to pick
            reports.append(report)
            print(f"survey {len(reports)}: {report['seen']} seen, {len(report['missed'])} missed, "
                  f"{report['spurious']} spurious")
            if not found and len(reports) == 1:
                raise RuntimeError(f"{TRACKERS[self.mode]} found nothing on the first survey")
            # in the tracker's order; tries(f) also records each object, so a later survey finds its failures
            candidates = [f for f in found if f["grasps"] and tries(f)[1] < 2]
            if not candidates:
                break
            for target in candidates:
                others = [f["box"] for f in found if f is not target]
                try:
                    held = await self.pick(target, others)  # True held, False slipped, None no plan
                except Exception as error:  # no plan: try the next object
                    print(f"  pick at ({target['x']:.0f}, {target['y']:.0f}) failed: {error}")
                    held = None
                    if (await self.gripper.is_holding_something()).is_holding_something:
                        await self.gripper.open()  # a move after the grab failed: let the block go where it is
                    try:  # then straight up, so the next survey's joint move doesn't sweep the fingers through the blocks
                        await self.move(pose_in_world(target["grasps"][0]["pose"], z=CARRY_MM), [])
                    except Exception as lift_error:  # best effort
                        print(f"  lift failed: {lift_error}")
                if held:
                    picked, failures = picked + 1, 0
                    break
                failures += 1
                tries(target)[1] += 1
                if failures >= MAX_FAILURES:
                    print(f"stopping: {MAX_FAILURES} failures in a row")
                    return {"reports": reports, "picked": picked}
                if held is False:  # it slipped: look again before the next try
                    break
        return {"reports": reports, "picked": picked}

    async def landed(self) -> dict[str, str]:
        where = {}
        for name in BLOCKS:
            x, y, _ = (v * 1000 for v in (await self.world.do_command(
                {"command": "get_prop_pose", "prop": name}))["pose"]["position"])
            where[name] = in_bin(x, y) or "table"
        return where


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("address")
    parser.add_argument("key_id")
    parser.add_argument("key")
    parser.add_argument("--vision", choices=list(TRACKERS), default="shapes")
    args = parser.parse_args()
    opts = RobotClient.Options.with_api_key(api_key=args.key, api_key_id=args.key_id)
    async with await RobotClient.at_address(args.address, opts) as robot:
        world = Generic.from_robot(robot, WORLD)
        gripper = Gripper.from_robot(robot, GRIPPER)
        cell = Cell(world, Arm.from_robot(robot, ARM), gripper, MotionClient.from_robot(robot, "builtin"),
                    PoseTracker.from_robot(robot, TRACKERS[args.vision]), args.vision)
        await world.do_command({"command": "reset"})  # blocks back on the table
        start = time.monotonic()
        try:
            result = await cell.run()
        finally:
            await gripper.open()  # never leave a block in the fingers
        where = await cell.landed()
    errors = [e for r in result["reports"] for e in r["errors"]]
    right = [name for name, place in where.items()
             if place == (name.split("_")[0] if args.vision == "shapes" else "clear")]
    print(f"{result['picked']} picks; {len(right)}/{len(BLOCKS)} blocks in the right bin "
          f"in {time.monotonic() - start:.0f} s ({args.vision})")
    if errors:
        print(f"detections: median {np.median([e['mm'] for e in errors]):.1f} mm, "
              f"{np.median([e['yaw_deg'] for e in errors]):.1f} deg off; "
              f"{sum(len(r['missed']) for r in result['reports'])} missed, "
              f"{sum(r['spurious'] for r in result['reports'])} spurious over {len(result['reports'])} surveys")
        colours = [e["colour_ok"] for e in errors if e["colour_ok"] is not None]
        if colours:
            print(f"colour right on {sum(colours)}/{len(colours)} detections")
    for name, place in where.items():
        print(f"  {name}: {place}")


if __name__ == "__main__":
    asyncio.run(main())
