"""Vision pick: a UR5e with a Robotiq 2F-85 and a wrist camera finds the
blocks of examples/vision-pick.json through a Viam vision service and picks
them into bins, so two perception pipelines can be compared.

Usage: uv run python examples/vision_pick.py <machine address> <api key id> <api key>
           [--vision shapes|pointcloud]

Take the address and an API key from the machine's CONNECT tab.

- shapes (default): `shape-segments`, detections-to-segments over the
  shape-finder detector `shapes`. Each object is labelled `<shape>-<colour>`;
  a block goes in the bin of its colour.
- pointcloud: `objects`, obstacles-pointcloud. Objects carry no label; every
  block goes in the clear bin.

Each round the wrist camera surveys the table from straight above and the
service's GetDetections3D returns each object as a transform tree: a root
frame on the camera at the object's box, and a child holding its points.
viam-server's frame system pins each root to the world (transform_pose: the
camera rides the arm, so a root left on it would move with the next plan),
and the script keeps the points at block height and works out a grasp from
them: centre, top, yaw and width. It picks the object nearest the arm with
the fingers across its short side, the other objects' boxes passed to the
motion planner as WorldState transforms, drops it in its bin, then surveys
again. After each survey it scores the detections against
the blocks' true poses (the world's get_prop_pose): position and yaw error,
missed and spurious objects, and the colour in shapes mode. At the end it
reports where every block landed.
"""

import argparse
import asyncio
import math
import time

import numpy as np
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.proto.common import Geometry, Pose, PoseInFrame, Transform, WorldState
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient
from viam.spatialmath import EulerAngles, Quaternion

ARM, GRIPPER, CAMERA, WORLD = "pick-arm", "pick-gripper", "wrist-cam", "sim-world"
SERVICES = {"shapes": "shape-segments", "pointcloud": "objects"}
SURVEY_MM = (500, 0, 450)  # the camera, straight over the blocks; its view covers ~690 x 520 mm
SURVEY_HEADING_DEG = 180  # the camera's x (its image's width) along world -x
BLOCKS = [f"{colour}_{i}" for i in range(1, 5) for colour in ("red", "blue")]
SQUARES = {"red_1", "blue_2", "red_3", "blue_4"}  # the 50 mm squares; the rest are 40 x 80 mm
BLOCK_H_MM = 40
TABLE_TOP_MM = 5  # 5 mm over the floor plane: a table flush with it z-fights in the renderer
BINS = {"red": (380, -450), "blue": (620, -450), "clear": (500, 450)}  # centres (mm)
BIN_HALF_MM = 112.5  # the bin mesh's 150 mm footprint scaled 1.5x, halved
BAND_MM = (TABLE_TOP_MM + 5, TABLE_TOP_MM + BLOCK_H_MM + 20)  # keep points at block height
GRID_MM = 5  # top-down cell size for clustering
MIN_POINTS = 30
MAX_GRASP_MM = 80  # the 2F-85 opens 85 mm
GRIP_DEPTH_MM = 15  # fingertips this far below a block's top
CARRY_MM = 250  # gripper height for moving between table and bin
BIN_RIM_MM = 80  # the bin mesh's height, unscaled
RELEASE_MM = TABLE_TOP_MM + BIN_RIM_MM + BLOCK_H_MM + 30  # let go with the block just over the rim
SLOT_MM = 50  # drop in the bin's quarters in turn, from its centre
MATCH_MM = 40  # a detection within this of a block's centre saw that block
MAX_PICKS, MAX_FAILURES = 16, 3


def parse_pcd(data: bytes) -> np.ndarray:
    """The x, y, z (metres) of a binary PCD, as Viam's vision services return it."""
    header, rest = {}, data
    while True:
        line, rest = rest.split(b"\n", 1)
        key, _, value = line.decode().partition(" ")
        header[key] = value.split()
        if key == "DATA":
            break
    if header["DATA"] != ["binary"]:
        raise ValueError(f"expected a binary PCD, got DATA {header['DATA']}")
    kinds = {("F", "4"): "<f4", ("F", "8"): "<f8", ("U", "4"): "<u4", ("I", "4"): "<i4", ("U", "1"): "u1"}
    dtype = np.dtype([(name, kinds[(kind, size)]) for name, kind, size in zip(header["FIELDS"], header["TYPE"], header["SIZE"])])
    rows = np.frombuffer(rest, dtype=dtype, count=int(header["POINTS"][0]))
    return np.column_stack([rows["x"], rows["y"], rows["z"]]).astype(float)


def block_points(world: np.ndarray) -> np.ndarray:
    """The points at block height, and of those only the connected cluster
    (top-down, GRID_MM cells) nearest their middle: a detection's box can
    take in the table round a turned block and a neighbour's corner."""
    band = world[(world[:, 2] > BAND_MM[0]) & (world[:, 2] < BAND_MM[1])]
    if len(band) < MIN_POINTS:
        return band[:0]
    cells = np.floor(band[:, :2] / GRID_MM).astype(int)
    occupied = {tuple(c) for c in cells}
    middle = np.median(band[:, :2], axis=0) / GRID_MM
    start = min(occupied, key=lambda c: (c[0] + 0.5 - middle[0]) ** 2 + (c[1] + 0.5 - middle[1]) ** 2)
    cluster, frontier = {start}, [start]
    while frontier:
        cx, cy = frontier.pop()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (cx + dx, cy + dy)
                if cell in occupied and cell not in cluster:
                    cluster.add(cell)
                    frontier.append(cell)
    return band[np.array([tuple(c) in cluster for c in cells])]


def grasp(points: np.ndarray) -> dict:
    """A block's centre and top (mm), the yaw of its long side (degrees, in
    [-90, 90)) and its length and width: the smallest rectangle round its
    points seen from above, tried a degree at a time (a square has no main
    axis, so principal axes would turn it anywhere)."""
    xy = points[:, :2]
    best = None
    for degrees in range(90):
        a = math.radians(degrees)
        u = xy @ [math.cos(a), math.sin(a)]
        v = xy @ [-math.sin(a), math.cos(a)]
        size = (np.ptp(u), np.ptp(v))
        if best is None or size[0] * size[1] < best[0]:
            centre = ((u.max() + u.min()) / 2, (v.max() + v.min()) / 2)
            best = (size[0] * size[1], degrees, size, centre)
    _, degrees, (along_u, along_v), (cu, cv) = best
    a = math.radians(degrees)
    x, y = cu * math.cos(a) - cv * math.sin(a), cu * math.sin(a) + cv * math.cos(a)
    long_side = degrees if along_u >= along_v else degrees + 90
    return {
        "x": float(x), "y": float(y), "top": float(np.percentile(points[:, 2], 95)),
        "yaw": (long_side + 90) % 180 - 90, "length": float(max(along_u, along_v)),
        "width": float(min(along_u, along_v)),
    }


def yaw_error(found: float, truth: float, square: bool) -> float:
    period = 90 if square else 180  # a square looks the same turned a quarter
    return abs((found - truth + period / 2) % period - period / 2)


def score(found: list[dict], truths: dict[str, dict]) -> dict:
    """Detections matched to the blocks on the table by nearest centre."""
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


def down(x: float, y: float, z: float, heading_deg: float) -> PoseInFrame:
    """A world pose pointing straight down (+z along world -z) with its x
    axis turned heading_deg from world +x, counterclockwise seen from above:
    a half turn about x (roll 180), then the turn about the vertical (yaw)."""
    ov = EulerAngles(roll=math.pi, pitch=0, yaw=math.radians(heading_deg)).to_quaternion().to_orientation_vector()
    pose = Pose(x=x, y=y, z=z, o_x=ov.o_x, o_y=ov.o_y, o_z=ov.o_z, theta=math.degrees(ov.theta))
    return PoseInFrame(reference_frame="world", pose=pose)


class Cell:
    def __init__(self, robot, world, gripper, motion, vision, mode: str) -> None:
        self.robot, self.world, self.gripper, self.motion, self.vision = robot, world, gripper, motion, vision
        self.mode = mode
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
        await self.motion.move(component_name=CAMERA, destination=down(*SURVEY_MM, SURVEY_HEADING_DEG))
        try:
            detections = await self.vision.get_detections_3d(CAMERA)
        except Exception as error:
            raise RuntimeError(f"vision service {SERVICES[self.mode]}: {error}") from error
        found = []
        for detection in detections:
            # transforms[0] is the root, on the camera; its "<root>/points" child carries the
            # object's points in the root's frame (RDK's detection3d.FromSegmenter)
            root = detection.transforms[0]
            clouds = [t for t in detection.transforms[1:] if t.physical_object.HasField("pointcloud")]
            if not clouds:
                continue
            at = (await self.robot.transform_pose(root.pose_in_observer_frame, "world")).pose
            rotation = np.array(Quaternion.from_pose(at).to_rotation_matrix().elements).reshape(3, 3)
            local = parse_pcd(clouds[0].physical_object.pointcloud.point_cloud) * 1000
            points = block_points(local @ rotation.T + [at.x, at.y, at.z])
            if len(points) < MIN_POINTS:
                continue
            item = grasp(points)
            item["label"] = root.physical_object.label if self.mode == "shapes" else ""
            # the root re-parented to the world where the camera saw it, with its box: an obstacle
            # (labelled by its frame: shape labels repeat, and RDK keys geometries by label)
            box = Geometry()
            box.CopyFrom(root.physical_object)
            box.label = root.reference_frame
            item["obstacle"] = Transform(
                reference_frame=root.reference_frame,
                pose_in_observer_frame=PoseInFrame(reference_frame="world", pose=at),
                physical_object=box,
            )
            found.append(item)
        return found

    async def move(self, x: float, y: float, z: float, heading: float, obstacles: list[Transform]) -> None:
        state = WorldState(transforms=obstacles)
        await self.motion.move(component_name=GRIPPER, destination=down(x, y, z, heading), world_state=state)

    async def pick(self, item: dict, others: list[Transform]) -> bool:
        # the fingers open along the gripper's x: head it across the short side
        heading = item["yaw"] + 90
        x, y = item["x"], item["y"]
        await self.gripper.open()
        await self.move(x, y, item["top"] + 100, heading, others)
        await self.move(x, y, item["top"] - GRIP_DEPTH_MM, heading, others)
        await self.gripper.grab()
        await asyncio.sleep(0.5)
        held = (await self.gripper.is_holding_something()).is_holding_something
        # ponytail: if a move after the grab fails, the block stays in the fingers until the next pick opens them
        await self.move(x, y, CARRY_MM, heading, others)
        if not held:
            return False
        colour = item["label"].rsplit("-", 1)[-1] if self.mode == "shapes" else "clear"
        name = colour if colour in BINS else "clear"
        slot = self.dropped[name] % 4
        self.dropped[name] += 1
        bx, by = BINS[name]
        bx, by = bx + (SLOT_MM if slot % 2 else -SLOT_MM), by + (SLOT_MM if slot >= 2 else -SLOT_MM)
        await self.move(bx, by, CARRY_MM, heading, others)
        await self.move(bx, by, RELEASE_MM, heading, others)
        await self.gripper.open()
        await asyncio.sleep(0.5)  # let it land
        await self.move(bx, by, CARRY_MM, heading, others)
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
                raise RuntimeError(f"{SERVICES[self.mode]} found nothing on the first survey")
            candidates = sorted(
                (f for f in found if f["width"] <= MAX_GRASP_MM and tries(f)[1] < 2),
                key=lambda f: math.hypot(f["x"], f["y"]),  # nearest the arm first
            )
            if not candidates:
                break
            for target in candidates:
                others = [f["obstacle"] for f in found if f is not target]
                try:
                    held = await self.pick(target, others)
                except Exception as error:  # no plan: try the next object
                    print(f"  pick at ({target['x']:.0f}, {target['y']:.0f}) failed: {error}")
                    held = None
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
    parser.add_argument("--vision", choices=list(SERVICES), default="shapes")
    args = parser.parse_args()
    opts = RobotClient.Options.with_api_key(api_key=args.key, api_key_id=args.key_id)
    async with await RobotClient.at_address(args.address, opts) as robot:
        world = Generic.from_robot(robot, WORLD)
        gripper = Gripper.from_robot(robot, GRIPPER)
        cell = Cell(robot, world, gripper, MotionClient.from_robot(robot, "builtin"),
                    VisionClient.from_robot(robot, SERVICES[args.vision]), args.vision)
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
