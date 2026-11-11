"""Gantry sort: move the blocks of examples/gantry-sort.json from the table
into the bin of their colour, with the XYZ gantry and its ePick.

Usage: uv run python examples/gantry_sort.py <machine address> <api key id> <api key> [--direct]

Take the address and an API key from the machine's CONNECT tab. For each
block (red_1..red_4, blue_1..blue_4): read its pose with the world's
get_prop_pose, go over it at carry height, down to its top, grab and check
the cup holds it, up, over a free quarter of its bin, down, let go, up. Then
read every block's pose and report any that isn't in its bin.

By default every move goes through the motion service (motion.move on the
gripper), with the other blocks as obstacles (the bins are obstacle
components, which the planner already sees) and a held block as a
transform on the gripper; the planner moves the gantry's three axes.
With --direct the same moves are gantry.move_to_position calls, the world
point turned into axis positions here.
"""

import argparse
import asyncio
import time

from viam.components.gantry import Gantry
from viam.components.gripper import Gripper
from viam.components.generic import Generic
from viam.proto.common import (
    GeometriesInFrame,
    Geometry,
    Pose,
    PoseInFrame,
    RectangularPrism,
    Transform,
    Vector3,
    WorldState,
)
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

GANTRY, GRIPPER, WORLD = "gantry", "cup", "sim-world"
GANTRY_ORIGIN_MM = (-500, -400, 1050)  # the gantry's frame: the tool point with every axis at 0
TCP_MM = 130  # the cup's frame below the gantry's tool point
BLOCK_MM = 60
BLOCKS = [f"{colour}_{i}" for i in range(1, 5) for colour in ("red", "blue")]
BINS = {"red": (-350, 0), "blue": (350, 0)}  # bin centres (mm)
BIN_HALF_MM = 150  # half the bin's 300 mm footprint (the 150 mm mesh scaled 2x)
BIN_RIM_MM = 400 + 80 * 1.5  # on the table (400 mm), 80 mm tall scaled 1.5x
BIN_FLOOR_MM = 400 + 5 * 1.5
SLOT_MM = 70  # quarter centres, from the bin's centre
CUP_GAP_MM = 1  # stop the cup this far above a block
DROP_MM = 10  # let go with the block this far above the bin floor
CARRY_MM = BIN_RIM_MM + BLOCK_MM + 20  # cup height for travel: a held block clears the rims
RELEASE_MM = BIN_FLOOR_MM + DROP_MM + BLOCK_MM + CUP_GAP_MM  # cup height to let go over a bin
DOWN = {"o_x": 0, "o_y": 0, "o_z": -1, "theta": 0}  # the gantry's tool can't turn: goals must point down


def box(name: str, x: float, y: float, z: float, size: tuple[float, float, float]) -> Geometry:
    return Geometry(center=Pose(x=x, y=y, z=z, o_z=1), box=RectangularPrism(dims_mm=Vector3(x=size[0], y=size[1], z=size[2])), label=name)


HELD = Transform(  # a held block, just past the cup, in the cup's frame (+z along the tool)
    reference_frame="held_block",
    pose_in_observer_frame=PoseInFrame(reference_frame=GRIPPER, pose=Pose(z=BLOCK_MM / 2 + CUP_GAP_MM, o_z=1)),
    physical_object=Geometry(center=Pose(o_z=1), box=RectangularPrism(dims_mm=Vector3(x=BLOCK_MM, y=BLOCK_MM, z=BLOCK_MM - 4)), label="held_block"),
)


def axes(x: float, y: float, z: float) -> list[float]:
    """The gantry positions (mm) that put the cup's frame at world (x, y, z)."""
    ox, oy, oz = GANTRY_ORIGIN_MM
    return [x - ox, y - oy, oz - (z + TCP_MM)]


def in_bin(position_m: list[float], colour: str) -> bool:
    x, y, z = (v * 1000 for v in position_m)
    bx, by = BINS[colour]
    return abs(x - bx) < BIN_HALF_MM and abs(y - by) < BIN_HALF_MM and z < BIN_RIM_MM


class Cell:
    def __init__(self, world, gantry, gripper, motion, direct: bool) -> None:
        self.world, self.gantry, self.gripper, self.motion, self.direct = world, gantry, gripper, motion, direct

    async def where(self, name: str) -> list[float]:
        return (await self.world.do_command({"command": "get_prop_pose", "prop": name}))["pose"]["position"]

    async def move(self, x: float, y: float, z: float, obstacles: list[Geometry], holding: bool = False) -> None:
        if self.direct:
            await self.gantry.move_to_position(axes(x, y, z), [])
            return
        state = WorldState(
            obstacles=[GeometriesInFrame(reference_frame="world", geometries=obstacles)],
            transforms=[HELD] if holding else [],
        )
        destination = PoseInFrame(reference_frame="world", pose=Pose(x=x, y=y, z=z, **DOWN))
        await self.motion.move(component_name=GRIPPER, destination=destination, world_state=state)

    async def sort(self) -> None:
        await self.gripper.open()  # let go of anything a stopped run left on the cup
        filled = {colour: 0 for colour in BINS}
        for name in BLOCKS:
            colour = name.split("_")[0]
            others = [
                box(other, *(v * 1000 for v in await self.where(other)), (BLOCK_MM,) * 3)
                for other in BLOCKS if other != name
            ]
            x, y, z = (v * 1000 for v in await self.where(name))
            top = z + BLOCK_MM / 2
            await self.move(x, y, CARRY_MM, others)
            await self.move(x, y, top + CUP_GAP_MM, others)
            await self.gripper.grab()
            await asyncio.sleep(0.5)  # Grab returns at once; give the cup a moment to seal
            if not (await self.gripper.is_holding_something()).is_holding_something:
                raise RuntimeError(f"{name}: the cup is not holding it")
            await self.move(x, y, CARRY_MM, others, holding=True)
            slot = filled[colour]
            filled[colour] += 1
            bx, by = BINS[colour]
            dx, dy = (SLOT_MM if slot % 2 else -SLOT_MM), (SLOT_MM if slot >= 2 else -SLOT_MM)
            await self.move(bx + dx, by + dy, CARRY_MM, others, holding=True)
            await self.move(bx + dx, by + dy, RELEASE_MM, others, holding=True)
            await self.gripper.open()
            await asyncio.sleep(0.5)  # let it land
            await self.move(bx + dx, by + dy, CARRY_MM, others)
            print(f"{name} -> {colour} bin")

    async def misplaced(self) -> list[str]:
        return [name for name in BLOCKS if not in_bin(await self.where(name), name.split("_")[0])]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("address")
    parser.add_argument("key_id")
    parser.add_argument("key")
    parser.add_argument("--direct", action="store_true", help="gantry.move_to_position instead of the motion service")
    args = parser.parse_args()
    opts = RobotClient.Options.with_api_key(api_key=args.key, api_key_id=args.key_id)
    async with await RobotClient.at_address(args.address, opts) as robot:
        world = Generic.from_robot(robot, WORLD)
        gripper = Gripper.from_robot(robot, GRIPPER)
        cell = Cell(world, Gantry.from_robot(robot, GANTRY), gripper, MotionClient.from_robot(robot, "builtin"), args.direct)
        await world.do_command({"command": "reset"})  # blocks back on the table
        start = time.monotonic()
        try:
            await cell.sort()
        finally:
            await gripper.open()  # never leave a block stuck to the cup
        wrong = await cell.misplaced()
    print(f"{len(BLOCKS)} blocks in {time.monotonic() - start:.1f} s ({'direct' if args.direct else 'motion service'})")
    if wrong:
        raise SystemExit(f"not in their bin: {wrong}")


if __name__ == "__main__":
    asyncio.run(main())
