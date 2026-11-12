"""Machine tending on a rail: a UR5e on a 2.6 m floor rail moves parts from a
parts table through machine A, then machine B, to an outfeed, keeping both
machines busy. Every move is one motion.move of the gripper; RDK plans the
rail and the arm together (README, "Machine tending on a rail").

Usage: uv run python examples/rail_tending.py <machine address> <api key id> <api key> [--parts N]

Load examples/rail-tending.json on the machine.
"""

import argparse
import asyncio
import math
import time

from viam.components.gantry import Gantry
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

WORLD, RAIL, GRIPPER = "sim-world", "rail", "gripper"
STATIONS = {"parts": -1100.0, "A": -350.0, "B": 400.0, "out": 1150.0}  # world x, mm
# where a part sits (world x, y mm): the parts table's row, the machines' centres, the outfeed's row
SLOTS = {
    "parts1": (-1280, 560), "parts2": (-1160, 560), "parts3": (-1040, 560), "parts4": (-920, 560),
    "A": (-350, 650), "B": (400, 650),
    "out1": (960, 560), "out2": (1080, 560), "out3": (1200, 560), "out4": (1320, 560),
}
# Transit: above a station's front half, 120 mm over the machines' walls (tops at 630 mm). RDK
# executes a combined plan one component at a time, in either order, so long rail-and-arm moves
# go transit to transit, through open air; only short moves happen near the walls.
TRANSIT_Y, TRANSIT_Z = 550.0, 750.0
GRIP_Z = 478.0  # the gripper's tool point for a 40 mm part on a 450 mm surface (centre 470)
PLACE_Z = GRIP_Z + 5  # let go just above the surface
PRE_DZ = 150.0  # pre-grasp, lift and up: this far above
CYCLE_S = 2.0  # a machine "runs" this long after it's loaded (a wait: the arm idles meanwhile)
MOVE_TIMEOUT_S = 120.0  # a plan and its execution
SLOT_TOLERANCE_MM = 30.0
MAX_PARTS = 4


def station_of(slot: str) -> str:
    return next(s for s in STATIONS if slot.startswith(s))


def pipeline(parts: int) -> list[tuple[int, str, str]]:
    """(part, from slot, to slot) in order: load A, move A's part on to B, then load
    the next part into A before emptying B, so both machines stay busy."""
    steps = [(1, "parts1", "A"), (1, "A", "B")]
    for k in range(2, parts + 1):
        steps += [(k, f"parts{k}", "A"), (k - 1, "B", f"out{k - 1}"), (k, "A", "B")]
    return steps + [(parts, "B", f"out{parts}")]


def transfer_steps(src: str, dst: str) -> list[tuple]:
    """A transfer as ("move", label, (x, y, z)) gripper poses and ("grab",) / ("open",)."""
    (sx, sy), (dx, dy) = SLOTS[src], SLOTS[dst]
    a, b = station_of(src), station_of(dst)
    return [
        ("move", f"transit {a}", (STATIONS[a], TRANSIT_Y, TRANSIT_Z)),
        ("move", f"pre {src}", (sx, sy, GRIP_Z + PRE_DZ)),
        ("move", f"grip {src}", (sx, sy, GRIP_Z)),
        ("grab",),
        ("move", f"lift {src}", (sx, sy, GRIP_Z + PRE_DZ)),
        ("move", f"transit {a}", (STATIONS[a], TRANSIT_Y, TRANSIT_Z)),
        ("move", f"transit {b}", (STATIONS[b], TRANSIT_Y, TRANSIT_Z)),
        ("move", f"pre {dst}", (dx, dy, GRIP_Z + PRE_DZ)),
        ("move", f"place {dst}", (dx, dy, PLACE_Z)),
        ("open",),
        ("move", f"up {dst}", (dx, dy, GRIP_Z + PRE_DZ)),
        ("move", f"transit {b}", (STATIONS[b], TRANSIT_Y, TRANSIT_Z)),
    ]


def on_slot(position_m, slot: str) -> bool:
    """Whether a part's world position (m) is on a slot (within SLOT_TOLERANCE_MM)."""
    x, y = SLOTS[slot]
    return math.hypot(position_m[0] * 1000 - x, position_m[1] * 1000 - y) <= SLOT_TOLERANCE_MM


class Cell:
    def __init__(self, robot: RobotClient) -> None:
        self.motion = MotionClient.from_robot(robot, "builtin")
        self.rail = Gantry.from_robot(robot, RAIL)
        self.gripper = Gripper.from_robot(robot, GRIPPER)
        self.world = Generic.from_robot(robot, WORLD)
        self.rail_travel_mm = 0.0

    async def move(self, x: float, y: float, z: float) -> None:
        """The gripper to a world pose pointing down; RDK moves the rail and the arm."""
        before = (await self.rail.get_position())[0]
        if not await self.motion.move(component_name=GRIPPER, destination=PoseInFrame(
                reference_frame="world", pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=-1, theta=0)),
                timeout=MOVE_TIMEOUT_S):
            raise RuntimeError(f"motion.move to ({x:.0f}, {y:.0f}, {z:.0f}) failed")
        self.rail_travel_mm += abs((await self.rail.get_position())[0] - before)

    async def part(self, k: int) -> list[float]:
        return (await self.world.do_command({"command": "get_prop_pose", "prop": f"part_{k}"}))["pose"]["position"]

    async def transfer(self, k: int, src: str, dst: str) -> str:
        started, travel = time.monotonic(), self.rail_travel_mm
        for step in transfer_steps(src, dst):
            if step[0] == "move":
                await self.move(*step[2])
            elif step[0] == "grab":
                await self.gripper.grab()
                await asyncio.sleep(0.5)  # let the fingers close on it
            else:
                await self.gripper.open()
                await asyncio.sleep(0.8)  # let the part settle
        landed = await self.part(k)
        x, y = SLOTS[dst]
        off = math.hypot(landed[0] * 1000 - x, landed[1] * 1000 - y)
        where = "on" if on_slot(landed, dst) else "OFF"
        line = (f"part {k}: {src} -> {dst} in {time.monotonic() - started:.0f} s, {where} its slot ({off:.0f} mm), "
                f"rail {(self.rail_travel_mm - travel) / 1000:.1f} m")
        if dst in ("A", "B"):
            await asyncio.sleep(CYCLE_S)  # the machine runs (not counted in the transfer's time)
        return line


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("address")
    parser.add_argument("key_id")
    parser.add_argument("key")
    parser.add_argument("--parts", type=int, default=MAX_PARTS, choices=range(1, MAX_PARTS + 1))
    args = parser.parse_args()
    opts = RobotClient.Options.with_api_key(api_key=args.key, api_key_id=args.key_id)
    async with await RobotClient.at_address(args.address, opts) as robot:
        cell = Cell(robot)
        await cell.world.do_command({"command": "reset"})
        await cell.gripper.open()
        # first to the parts table's transit, before the rail ever moves: the reset pose holds the
        # gripper over the stations (the first transfer then starts with a near no-op move there)
        await cell.move(STATIONS["parts"], TRANSIT_Y, TRANSIT_Z)
        started, failed = time.monotonic(), False
        for k, src, dst in pipeline(args.parts):
            try:
                print(await cell.transfer(k, src, dst), flush=True)
            except Exception as e:  # report the failed move and stop: the cell's state is unknown
                print(f"part {k}: {src} -> {dst} FAILED: {e}", flush=True)
                failed = True
                await robot.stop_all()  # a timed-out move may still be running on the machine
                await cell.gripper.open()
                break
        # a list, not a generator: an await inside a generator expression makes it async, which sum() can't take
        delivered = sum([on_slot(await cell.part(k), f"out{k}") for k in range(1, args.parts + 1)])
        print(f"{delivered}/{args.parts} parts on the outfeed in {time.monotonic() - started:.0f} s; "
              f"rail travel {cell.rail_travel_mm / 1000:.1f} m")
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
