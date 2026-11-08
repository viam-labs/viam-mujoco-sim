"""Parcel sorting: weigh the parcels examples/parcel-sort.json feeds onto the
belt and drop each in the light or heavy bin, with the UR5e and its ePick.

Usage: uv run python examples/parcel_sort.py <machine address> <api key id> <api key> [parcels]

Take the address and an API key from the machine's CONNECT tab. Sorts 10
parcels unless told otherwise.

For each parcel: run the belt until the photo-eye counts one, stop it, find
the leading parcel with the spawner's get_parcels, pick it by suction, set
it on the scale, read its weight once stable, pick it again and drop it in
its bin. Every move plans around the belt's rails and the other parcels on
the belt, given to the motion service as obstacles, and carries a held
parcel as a transform on the gripper. At the end, get_parcels tells where
each parcel landed.
"""

import asyncio
import math
import sys

from viam.components.gripper import Gripper
from viam.components.motor import Motor
from viam.components.sensor import Sensor
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
from viam.proto.service.motion import Constraints, OrientationConstraint
from viam.services.motion import MotionClient
from viam.spatialmath import Quaternion

GRIPPER = "pick-gripper"
BELT = (600, 450, 400)  # the belt's centre x, top z and width (mm)
BELT_LENGTH_MM = 2000
RAIL_MM = (20, 40)  # the belt's side rails: thickness, and height over the belt (the conveyor's defaults)
SCALE = (0, 550, 450)  # the scale platform's centre x, y and top z (mm)
BINS = {"light": (0, -600), "heavy": (-600, 0)}  # bin centres (mm)
BIN_HALF_MM = 300
BIN_RIM_MM = 200
LIGHT_KG = 1.0  # lighter goes in the light bin
BELT_POWER = 0.6
CARRY_Z_MM = 800  # TCP height for carrying: a held parcel clears the belt, the scale and the bins
ABOVE_MM = 100  # approach and retreat height over a pick or a drop
CUP_GAP_MM = 1  # stop the cup this far above a parcel, within its 2 mm contact margin
DROP_MM = 10  # let go with the parcel this far above what it lands on
PARCEL_TIMEOUT_S = 60  # wait this long for a parcel at the photo-eye
CARRY_TILT_DEG = 5  # keep a held parcel within this of level for the whole move
DOWN = {"o_x": 0, "o_y": 0, "o_z": -1, "theta": 0}


# The belt's frame box stops at the belt top, so it won't enclose the parcels
# on it: the planner sees the side rails only as these. Without them it can
# lower the upper arm onto a rail.
RAILS = [
    Geometry(
        center=Pose(x=BELT[0] + side * (BELT[2] + RAIL_MM[0]) / 2, z=BELT[1] + RAIL_MM[1] / 2, o_z=1),
        box=RectangularPrism(dims_mm=Vector3(x=RAIL_MM[0], y=BELT_LENGTH_MM, z=RAIL_MM[1])),
        label=f"belt-rail-{i}",
    )
    for i, side in enumerate((-1, 1))
]


def size_mm(parcel: dict) -> tuple[float, float, float]:
    x, y, z = (v * 1000 for v in parcel["size"])
    return x, y, z


def box(parcel: dict) -> Geometry:
    """A parcel, from get_parcels, as a planner obstacle."""
    x, y, z = (v * 1000 for v in parcel["pose"]["position"])
    ov = Quaternion(*parcel["pose"]["orientation_wxyz"]).to_orientation_vector()
    center = Pose(x=x, y=y, z=z, o_x=ov.o_x, o_y=ov.o_y, o_z=ov.o_z, theta=math.degrees(ov.theta))
    sx, sy, sz = size_mm(parcel)
    return Geometry(center=center, box=RectangularPrism(dims_mm=Vector3(x=sx, y=sy, z=sz)), label=parcel["name"])


def held(size: tuple[float, float, float]) -> Transform:
    """A held parcel in the gripper's frame (+z along the tool; with the tool
    pointing down its x stays along world x), just past the gripper's
    planner box and 4 mm shorter than the parcel, so it never touches what
    the parcel sat on."""
    return Transform(
        reference_frame="held_parcel",
        pose_in_observer_frame=PoseInFrame(reference_frame=GRIPPER, pose=Pose(z=size[2] / 2 + CUP_GAP_MM, o_z=1)),
        physical_object=Geometry(
            center=Pose(o_z=1),  # RDK rejects a geometry without a centre pose
            box=RectangularPrism(dims_mm=Vector3(x=size[0], y=size[1], z=size[2] - 4)),
            label="held_parcel",
        ),
    )


def on_belt(parcel: dict) -> bool:
    x, y, z = (v * 1000 for v in parcel["pose"]["position"])
    return abs(x - BELT[0]) < BELT[2] / 2 and abs(y) < 1000 and z > BELT[1]


def in_bin(parcel: dict, name: str) -> bool:
    x, y, _ = (v * 1000 for v in parcel["pose"]["position"])
    bx, by = BINS[name]
    return abs(x - bx) < BIN_HALF_MM and abs(y - by) < BIN_HALF_MM


class Cell:
    def __init__(self, belt, eye, feed, scale, gripper, motion) -> None:
        self.belt, self.eye, self.feed, self.scale = belt, eye, feed, scale
        self.gripper, self.motion = gripper, motion

    async def parcels(self) -> list[dict]:
        return (await self.feed.do_command({"command": "get_parcels"}))["parcels"]

    async def move(self, x: float, y: float, z: float, obstacles: list[Geometry], size=None) -> None:
        state = WorldState(
            obstacles=[GeometriesInFrame(reference_frame="world", geometries=[*RAILS, *obstacles])],
            transforms=[held(size)] if size else [],
        )
        destination = PoseInFrame(reference_frame="world", pose=Pose(x=x, y=y, z=z, **DOWN))
        # carrying: keep the cup pointing down along the whole path, not just at the ends
        constraints = (
            Constraints(orientation_constraint=[OrientationConstraint(orientation_tolerance_degs=CARRY_TILT_DEG)])
            if size
            else None
        )
        await self.motion.move(
            component_name=GRIPPER, destination=destination, world_state=state, constraints=constraints
        )

    async def pick(self, parcel: dict, others: list[Geometry]) -> None:
        x, y, z = (v * 1000 for v in parcel["pose"]["position"])
        size = size_mm(parcel)
        top = z + size[2] / 2
        # over it at carry height, then down; the parcel stays an obstacle
        # until the last, straight-down approach
        await self.move(x, y, CARRY_Z_MM, [*others, box(parcel)])
        await self.move(x, y, top + ABOVE_MM, [*others, box(parcel)])
        await self.move(x, y, top + CUP_GAP_MM, others)
        await self.gripper.grab()
        await asyncio.sleep(0.5)  # Grab returns at once; give the cup a moment to seal
        if not (await self.gripper.is_holding_something()).is_holding_something:
            raise RuntimeError(f"{parcel['name']}: the cup is not holding it")
        await self.move(x, y, CARRY_Z_MM, others, size)

    async def place(self, x: float, y: float, under_mm: float, size, others: list[Geometry]) -> None:
        """Set the held parcel down DROP_MM above under_mm and let go."""
        release = under_mm + DROP_MM + size[2] + CUP_GAP_MM
        await self.move(x, y, CARRY_Z_MM, others, size)
        await self.move(x, y, release + ABOVE_MM, others, size)
        await self.move(x, y, release, others, size)
        await self.gripper.open()
        await asyncio.sleep(1.0)  # let it land
        await self.move(x, y, CARRY_Z_MM, others)

    async def bin_drop(self, bin_name: str, name: str, size) -> tuple[float, float, float, list[Geometry]]:
        """Where to let go over a bin: the quarter with the lowest pile, and
        high enough to clear what is already in it; plus the binned parcels,
        as obstacles."""
        bx, by = BINS[bin_name]
        binned = [p for p in await self.parcels() if p["name"] != name and in_bin(p, bin_name)]
        # a quarter's centre, pulled in so the whole parcel lands inside the walls
        off = max(0.0, min(BIN_HALF_MM / 2, BIN_HALF_MM - 15 - max(size[:2]) / 2))

        def pile(cx: float, cy: float) -> float:
            tops = [BIN_RIM_MM]
            for p in binned:
                px, py, pz = (v * 1000 for v in p["pose"]["position"])
                reach = max(size_mm(p)) / 2  # a binned parcel may lie at any angle
                if abs(px - cx) < reach + max(size[:2]) / 2 and abs(py - cy) < reach + max(size[:2]) / 2:
                    tops.append(pz + reach)
            return max(tops)

        spots = [(bx + sx * off, by + sy * off) for sx in (-1, 1) for sy in (-1, 1)]
        x, y = min(spots, key=lambda s: pile(*s))
        return x, y, pile(x, y), [box(p) for p in binned]

    async def weigh(self) -> float:
        await asyncio.sleep(0.5)  # landing
        for _ in range(100):
            reading = await self.scale.get_readings()
            if reading["stable"]:
                return reading["weight_kg"]
            await asyncio.sleep(0.1)
        raise RuntimeError("the scale never settled")

    async def next_parcel(self) -> dict:
        """Run the belt until the photo-eye counts a parcel; the leading one."""
        seen = (await self.eye.get_readings())["count"]
        await self.belt.set_power(BELT_POWER)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + PARCEL_TIMEOUT_S
        while (await self.eye.get_readings())["count"] == seen:
            if loop.time() > deadline:
                await self.belt.stop()
                why = " (the spawner's pool is exhausted)" if (await self.feed.get_readings())["pool_exhausted"] else ""
                raise RuntimeError(f"no parcel reached the photo-eye in {PARCEL_TIMEOUT_S} s{why}")
            await asyncio.sleep(0.05)
        await self.belt.stop()
        await asyncio.sleep(0.5)  # let it stop
        return max((p for p in await self.parcels() if on_belt(p)), key=lambda p: p["pose"]["position"][1])

    async def sort(self, count: int) -> list[tuple[str, str]]:
        """Sort `count` parcels; (parcel, bin) for each."""
        await self.gripper.open()  # let go of anything a stopped run left on the cup
        sorted_ = []
        for index in range(count):
            parcel = await self.next_parcel()
            name, size = parcel["name"], size_mm(parcel)
            others = [box(p) for p in await self.parcels() if on_belt(p) and p["name"] != name]
            await self.pick(parcel, others)
            await self.place(SCALE[0], SCALE[1], SCALE[2], size, others)
            weight = await self.weigh()
            (parcel,) = [p for p in await self.parcels() if p["name"] == name]
            await self.pick(parcel, others)
            bin_name = "light" if weight < LIGHT_KG else "heavy"
            x, y, under, binned = await self.bin_drop(bin_name, name, size)
            await self.place(x, y, under, size, [*others, *binned])
            print(f"{index + 1}: {name} ({parcel['class']}) weighs {weight:.2f} kg -> {bin_name} bin")
            sorted_.append((name, bin_name))
        return sorted_

    async def check(self, sorted_: list[tuple[str, str]]) -> list[str]:
        """The parcels that are not in their right bin."""
        where = {p["name"]: p for p in await self.parcels()}
        wrong = []
        for name, _ in sorted_:
            parcel = where.get(name)
            right = "light" if parcel and parcel["mass_kg"] < LIGHT_KG else "heavy"
            if parcel is None or not in_bin(parcel, right):
                wrong.append(name)
        return wrong


async def main() -> None:
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    address, api_key_id, api_key = sys.argv[1:4]
    count = int(sys.argv[4]) if len(sys.argv) == 5 else 10
    robot = await RobotClient.at_address(
        address, RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id)
    )
    gripper = Gripper.from_robot(robot, GRIPPER)
    belt = Motor.from_robot(robot, "belt")
    cell = Cell(
        belt, Sensor.from_robot(robot, "eye"), Sensor.from_robot(robot, "feed"),
        Sensor.from_robot(robot, "scale"), gripper, MotionClient.from_robot(robot, "builtin"),
    )
    try:
        await cell.feed.do_command({"command": "clear"})  # binned parcels never despawn: start with a full pool
        wrong = await cell.check(await cell.sort(count))
    finally:
        await belt.stop()
        await gripper.open()  # never leave a parcel stuck to the cup
        await robot.close()
    if wrong:
        sys.exit(f"not in the right bin: {wrong}")
    print(f"all {count} parcels sorted into the right bin")


if __name__ == "__main__":
    asyncio.run(main())
