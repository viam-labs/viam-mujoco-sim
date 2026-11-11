"""Conveyor pick: pick a parcel off the moving belt of examples/conveyor-pick.json
by suction, three ways, to compare how Viam's motion APIs keep up with a moving
target.

Usage: uv run python examples/conveyor_pick.py <machine address> <api key id> <api key>
           <teleop|stream|move> [--belt POWER] [--lookahead S] [--csv OUT]

Take the address and an API key from the machine's CONNECT tab. The belt runs
at --belt x 1.0 m/s (default 0.1); the arm aims --lookahead seconds of belt
travel ahead of the parcel (default per mode), to make up for the time a goal
takes to turn into motion. --csv writes every tick.

The modes, all fed the same goal every tick:

- teleop: the motion service's teleop DoCommands (teleop_start, teleop_move,
  teleop_stop). RDK plans each goal in the background and plays the newest
  plan without waiting for the arm to stop. RDK v1.15's teleop plans
  without the world_state obstacles it is given (the belt's rails here):
  only the frame system's own geometry is checked.
- stream: the motion service's TempStreamArmJointPositions session, which
  times a stream of joint targets for the arm. It takes joints, not poses, so
  this script solves IK itself, on the module's own UR5e kinematics; there
  are no collision checks. It needs a viam-server built with trajex, the
  `viam_rdk_cgo_have_cxx20_rt` build tag; released builds refuse the session.
- move: the motion service's Move, the usual way to move a gripper to a
  pose, once per goal. Each call plans, runs and waits for the arm to stop.
  (The module's arm leaves MoveToPosition to the motion service.)

The phases: park over the start of the arm's reach, wait for the leading
parcel, track 100 mm over it until the gripper is locked on, descend to the
parcel's top, grab, and lift 150 mm while still following the belt, as far as
the arm reaches. Prints a summary.
"""

import argparse
import asyncio
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from google.protobuf import json_format
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.components.motor import Motor
from viam.components.sensor import Sensor
from viam.gen.component.arm.v1.arm_pb2 import JointPositions, MoveOptions
from viam.gen.service.motion.v1.motion_pb2 import TempStreamArmJointPositionsRequest as StreamReq
from viam.gen.service.motion.v1.motion_pb2 import TempStreamOptions
from viam.proto.common import GeometriesInFrame, Geometry, Pose, PoseInFrame, RectangularPrism, Vector3, WorldState
from viam.proto.service.motion import MoveRequest
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

ARM, GRIPPER, MOTION = "pick-arm", "pick-gripper", "builtin"
ARM_BASE_MM = (0, 0, 400)  # pick-arm's frame in the world
TCP_MM = 130  # pick-gripper's frame along the arm's tool axis
BELT_X, BELT_TOP, BELT_W, MAX_SPEED = 600, 450, 400, 1.0  # mm, mm, mm, m/s at power 1
REACH_Y = (-450, 400)  # the parcel's y span (mm) where the gripper can be over it
HOVER_MM, CUP_GAP_MM = 100, 1  # track this far over the parcel's top; grab this far over it
DESCEND_MM_S, LIFT_MM_S, LIFT_MM = 200, 150, 150
LOCK_MM = (10, 5)  # locked on: along-belt and height errors within these for LOCK_TICKS ticks
LOCK_TICKS = 3
RATE_HZ = {"teleop": 10, "stream": 20, "move": 100}  # move is paced by its own calls
LOOKAHEAD_S = {"teleop": 0.4, "stream": 0.5, "move": 1.0}
DOWN = {"o_x": 0, "o_y": 0, "o_z": -1, "theta": 0}
RAILS = [  # the belt's frame box stops at its top: the rails, as planner obstacles
    Geometry(center=Pose(x=BELT_X + side * (BELT_W / 2 + 10), z=BELT_TOP + 20, o_z=1),
             box=RectangularPrism(dims_mm=Vector3(x=20, y=2000, z=40)), label=f"rail-{i}")
    for i, side in enumerate((-1, 1))
]
STATE = WorldState(obstacles=[GeometriesInFrame(reference_frame="world", geometries=RAILS)])


def world(x: float, y: float, z: float) -> PoseInFrame:
    return PoseInFrame(reference_frame="world", pose=Pose(x=x, y=y, z=z, **DOWN))


class Teleop:
    def __init__(self, robot: RobotClient) -> None:
        self.motion = MotionClient.from_robot(robot, MOTION)

    async def start(self, x: float, y: float, z: float) -> None:
        request = MoveRequest(name=MOTION, component_name=GRIPPER, destination=world(x, y, z), world_state=STATE)
        await self.motion.do_command({"teleop_start": json_format.MessageToJson(request)})

    async def send(self, x: float, y: float, z: float) -> None:
        await self.motion.do_command({"teleop_move": json_format.MessageToJson(world(x, y, z)),
                                      "component_name": GRIPPER})

    async def stop(self) -> None:
        await self.motion.do_command({"teleop_stop": True})


class Stream:
    """Joint targets through TempStreamArmJointPositions, by damped least
    squares IK on the UR5e kinematics, seeded with the last solution and
    keeping the parked tool orientation."""

    def __init__(self, robot: RobotClient) -> None:
        # the module's kinematics parser and forward kinematics, from this checkout
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from mujoco_module import robots

        path = Path(__file__).resolve().parents[1] / "assets/universal_robots_ur5e/ur5e_kinematics.json"
        self.kin = robots.parse_kinematics(path.read_bytes(), False, "ur5e")
        self.arm, self.motion = Arm.from_robot(robot, ARM), MotionClient.from_robot(robot, MOTION)
        self.misses = 0

    def fk(self, q: np.ndarray) -> np.ndarray:
        return self.kin.forward(q)[self.kin.end]

    def ik(self, goal: np.ndarray) -> np.ndarray:
        q = self.q.copy()
        for _ in range(60):
            pose = self.fk(q)
            dp = goal[:3, 3] - pose[:3, 3]
            r = goal[:3, :3] @ pose[:3, :3].T
            angle = math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2)))
            if np.linalg.norm(dp) < 1e-5 and angle < 1e-4:
                break
            axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
            dr = axis / (2 * math.sin(angle)) * angle if angle > 1e-9 else np.zeros(3)
            jac = np.zeros((6, 6))
            for j in range(6):  # numeric Jacobian; orientation weighted down against position
                dq = np.zeros(6)
                dq[j] = 1e-6
                p2 = self.fk(q + dq)
                r2 = p2[:3, :3] @ pose[:3, :3].T
                jac[:3, j] = (p2[:3, 3] - pose[:3, 3]) / 1e-6
                jac[3:, j] = 0.3 * np.array([r2[2, 1] - r2[1, 2], r2[0, 2] - r2[2, 0], r2[1, 0] - r2[0, 1]]) / 2e-6
            err = np.concatenate([dp, 0.3 * dr])
            q = q + jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
        return q

    async def start(self, x: float, y: float, z: float) -> None:
        self.q = np.radians((await self.arm.get_joint_positions()).values)
        self.rot = self.fk(self.q)[:3, :3]
        self.session = self.motion.client.TempStreamArmJointPositions.open()
        self.stream = await self.session.__aenter__()
        await self.stream.send_message(StreamReq(name=MOTION, init=StreamReq.Init(
            component_name=ARM,
            options=TempStreamOptions(move_options=MoveOptions(max_vel_degs_per_sec=120, max_acc_degs_per_sec2=360)),
        )))
        self.runway_ms = None

        async def acks() -> None:
            async for ack in self.stream:
                self.runway_ms = ack.motion_side_current_runway_ms

        self.reader = asyncio.create_task(acks())
        await self.send(x, y, z)

    async def send(self, x: float, y: float, z: float) -> None:
        goal = np.eye(4)
        goal[:3, :3] = self.rot
        # the arm's flange, TCP_MM over the gripper pointing down, in the arm's frame (m)
        goal[:3, 3] = (np.array([x, y, z + TCP_MM]) - ARM_BASE_MM) / 1000
        q = self.ik(goal)
        # the session plays every target it is given: never send one that missed
        if np.linalg.norm(self.fk(q)[:3, 3] - goal[:3, 3]) > 0.002:
            self.misses += 1
            return
        self.q = q
        await self.stream.send_message(StreamReq(targets=StreamReq.Targets(
            positions=[JointPositions(values=list(np.degrees(q)))])))

    async def stop(self) -> None:
        await self.stream.end()
        await self.reader
        await self.session.__aexit__(None, None, None)


class Move:
    def __init__(self, robot: RobotClient) -> None:
        self.motion = MotionClient.from_robot(robot, MOTION)

    async def start(self, x: float, y: float, z: float) -> None:
        await self.send(x, y, z)

    async def send(self, x: float, y: float, z: float) -> None:
        await self.motion.move(component_name=GRIPPER, destination=world(x, y, z), world_state=STATE)

    async def stop(self) -> None:
        pass


def top(parcel: dict) -> tuple[float, float, float]:
    """The middle of a parcel's top face (mm)."""
    x, y, z = (v * 1000 for v in parcel["pose"]["position"])
    return x, y, z + parcel["size"][2] * 500


def save(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("address")
    parser.add_argument("key_id")
    parser.add_argument("key")
    parser.add_argument("mode", choices=list(RATE_HZ))
    parser.add_argument("--belt", type=float, default=0.1, help="belt power, x 1.0 m/s")
    parser.add_argument("--lookahead", type=float, help="seconds of belt travel to aim ahead")
    parser.add_argument("--csv", help="write every tick here")
    args = parser.parse_args()
    address, key_id, key, mode, power, out = args.address, args.key_id, args.key, args.mode, args.belt, args.csv
    lookahead = LOOKAHEAD_S[mode] if args.lookahead is None else args.lookahead
    speed = power * MAX_SPEED * 1000  # mm/s
    opts = RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id)
    async with await RobotClient.at_address(address, opts) as robot:
        belt, feed = Motor.from_robot(robot, "belt"), Sensor.from_robot(robot, "feed")
        gripper, motion = Gripper.from_robot(robot, GRIPPER), MotionClient.from_robot(robot, MOTION)
        driver = {"teleop": Teleop, "stream": Stream, "move": Move}[mode](robot)

        async def parcels() -> list[dict]:
            return (await feed.do_command({"command": "get_parcels"}))["parcels"]

        await motion.do_command({"teleop_stop": True})  # a teleop pipeline left by a stopped run
        await gripper.open()  # let go of a parcel the last run lifted
        # park over the start of the reach, so the arm tracks instead of chasing
        await motion.move(component_name=GRIPPER, world_state=STATE,
                          destination=world(BELT_X, REACH_Y[0], BELT_TOP + 150 + HOVER_MM))
        await feed.do_command({"command": "spawn", "force": True})
        await belt.set_power(power)

        def coming(p: dict) -> bool:  # on the belt, not yet at the arm
            x, y, z = (v * 1000 for v in p["pose"]["position"])
            return abs(x - BELT_X) < BELT_W and z > BELT_TOP and y < REACH_Y[0] + 100

        deadline = time.monotonic() + 60
        while not (on := [p for p in await parcels() if coming(p)]) or max(
                p["pose"]["position"][1] for p in on) * 1000 < REACH_Y[0]:
            if time.monotonic() > deadline:
                raise RuntimeError("no parcel reached the arm in 60 s")
            await asyncio.sleep(0.02)
        name = max(on, key=lambda p: p["pose"]["position"][1])["name"]

        rows, phase, above, locked = [], "track", HOVER_MM, 0
        held = grabbed_at = grab = None
        t0 = last = time.monotonic()
        started = False
        try:
            while True:
                tick = time.monotonic()
                dt, last = tick - last, tick
                live = next(p for p in await parcels() if p["name"] == name)
                x, y, z = top(live)
                if (grab is None and y > REACH_Y[1]) or (phase == "lift" and tick - grabbed_at > 2.0):
                    break
                if grab is not None:  # gripped: where the belt would have taken it, within reach
                    x, y, z = grab
                    y += min(speed * (tick - grabbed_at), REACH_Y[1] - y)
                if phase == "descend":
                    above = max(CUP_GAP_MM, above - DESCEND_MM_S * dt)
                elif phase in ("grab", "lift") and tick - grabbed_at > 0.5:  # the cup has had time to seal
                    if held is None:
                        held = (await gripper.is_holding_something()).is_holding_something
                        phase = "lift"
                    above = min(CUP_GAP_MM + LIFT_MM, above + LIFT_MM_S * dt)
                # lead the height ramps by the lookahead too, as the belt is led
                lead = {"descend": -DESCEND_MM_S, "lift": LIFT_MM_S}.get(phase, 0) * lookahead
                goal_above = min(max(CUP_GAP_MM, above + lead), CUP_GAP_MM + LIFT_MM)
                goal = (x, min(y + speed * lookahead, REACH_Y[1]), z + goal_above)  # never past the reach
                await (driver.send if started else driver.start)(*goal)
                started = True
                g = (await motion.get_pose(component_name=GRIPPER, destination_frame="world")).pose
                # where the gripper should be now: the belt has moved on while the goal was sent
                now_y = y + speed * (time.monotonic() - tick)
                if grab is not None:
                    now_y = min(now_y, REACH_Y[1])
                err = (g.x - x, g.y - now_y, g.z - (z + above))
                if phase == "track":
                    locked = locked + 1 if abs(err[1]) < LOCK_MM[0] and abs(err[2]) < LOCK_MM[1] else 0
                    if locked >= LOCK_TICKS:
                        phase = "descend"
                elif phase == "descend" and above == CUP_GAP_MM and abs(err[2]) < 3:
                    await gripper.grab()
                    phase, grabbed_at, grab = "grab", time.monotonic(), (x, y, z)
                rows.append({"t": round(tick - t0, 3), "phase": phase, "above": round(above, 1),
                             "parcel_y": round(top(live)[1], 1), "parcel_z": round(top(live)[2], 1),
                             **{f"err_{a}": round(e, 1) for a, e in zip("xyz", err)},
                             "runway_ms": getattr(driver, "runway_ms", None)})
                await asyncio.sleep(max(0.0, 1 / RATE_HZ[mode] - (time.monotonic() - tick)))
        finally:
            if started:
                await driver.stop()
            await belt.stop()
        final = next(p for p in await parcels() if p["name"] == name)
        first = {}
        for r in rows:
            first.setdefault(r["phase"], r)
        gripped = first.get("grab")
        summary = {
            "mode": mode, "belt_mm_s": speed, "lookahead_s": lookahead, "ticks": len(rows),
            "tick_ms_median": round(1000 * float(np.median(np.diff([r["t"] for r in rows]))), 1) if len(rows) > 1 else None,
            "locked_at_s": first["descend"]["t"] if "descend" in first else None,
            "grabbed_at_s": gripped["t"] if gripped else None,
            "err_at_grab_mm": [gripped[f"err_{a}"] for a in "xyz"] if gripped else None,
            "held": bool(held), "holding_now": (await gripper.is_holding_something()).is_holding_something,
            "lifted_mm": round(top(final)[2] - gripped["parcel_z"], 1) if gripped else None,
        }
        if mode == "stream":
            summary["ik_misses"] = driver.misses
        await gripper.open()
    if out and rows:
        save(out, rows)
    print(json.dumps(summary, indent=1))
    if not summary["holding_now"] or not summary["lifted_mm"] or summary["lifted_mm"] < 100:
        sys.exit("missed the parcel")


if __name__ == "__main__":
    asyncio.run(main())
