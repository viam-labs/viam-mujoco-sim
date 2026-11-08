"""De-palletize: carry the 12 cartons of examples/depalletize.json from the
pallet to the conveyor with the UR5e and its ePick, in one client session.

Usage: uv run python examples/depalletize.py <machine address> <api key id> <api key>

Take the address and an API key from the machine's CONNECT tab.

Every move plans around the cartons still on the pallet, given to the motion
service as obstacles, and, while a carton is held, carries it as the
held_carton transform on the gripper so the planner keeps it clear too.
"""

import asyncio
import math
import sys

from viam.components.generic import Generic
from viam.components.gripper import Gripper
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
from viam.spatialmath import Quaternion

CARTONS = [f"carton_{i}" for i in range(1, 13)]  # carton_1..6 are the top layer
SIZE_MM = (190, 150, 150)
DROP_MM = (-550, 0)  # above the conveyor
CONVEYOR_TOP_MM = 460
CONVEYOR_MM = (300, 1000)  # its footprint, centred on DROP_MM
CARRY_Z_MM = 620  # TCP height for carrying: the held carton clears the top layer
ABOVE_MM = 100  # approach and retreat height over a pick or drop
CUP_GAP_MM = 1  # stop the cup this far above a carton, within its 2 mm contact margin
BREAKAWAY_MM = 25  # the first part of a lift, which ignores the cartons below
DOWN = {"o_x": 0, "o_y": 0, "o_z": -1, "theta": 0}

# The held carton in the gripper's frame (+z along the tool; with the tool
# pointing down, x is world -x, so the carton's 190 mm side stays along x).
# Its centre is 76 mm out, just past the gripper's planner box, and it is
# 4 mm shorter than the carton so it never touches what the carton sat on.
HELD = Transform(
    reference_frame="held_carton",
    pose_in_observer_frame=PoseInFrame(reference_frame="pick-gripper", pose=Pose(z=76, o_z=1)),
    physical_object=Geometry(
        center=Pose(o_z=1),  # RDK rejects a geometry without a centre pose
        box=RectangularPrism(dims_mm=Vector3(x=SIZE_MM[0], y=SIZE_MM[1], z=SIZE_MM[2] - 4)),
        label="held_carton",
    ),
)


def box(name: str, pose: dict) -> Geometry:
    """A carton, from its get_prop_pose pose, as a planner obstacle."""
    x, y, z = (v * 1000 for v in pose["position"])
    ov = Quaternion(*pose["orientation_wxyz"]).to_orientation_vector()
    center = Pose(x=x, y=y, z=z, o_x=ov.o_x, o_y=ov.o_y, o_z=ov.o_z, theta=math.degrees(ov.theta))
    size = Vector3(x=SIZE_MM[0], y=SIZE_MM[1], z=SIZE_MM[2])
    return Geometry(center=center, box=RectangularPrism(dims_mm=size), label=name)


def parked(index: int) -> list[float]:
    """Off the conveyor's far end, two by two on the floor (m)."""
    return [-0.45 - 0.2 * (index % 2), 0.65 + 0.17 * (index // 2), 0.076]


async def depalletize(world, gripper, motion) -> list[str]:
    """Move every carton to the conveyor; return the ones that missed it."""

    async def pose(name: str) -> dict:
        return (await world.do_command({"command": "get_prop_pose", "prop": name}))["pose"]

    async def move(x: float, y: float, z: float, obstacles: list[Geometry], held: bool) -> None:
        state = WorldState(
            obstacles=[GeometriesInFrame(reference_frame="world", geometries=obstacles)],
            transforms=[HELD] if held else [],
        )
        destination = PoseInFrame(reference_frame="world", pose=Pose(x=x, y=y, z=z, **DOWN))
        await motion.move(component_name="pick-gripper", destination=destination, world_state=state)

    # let go of anything a stopped run left on the cup, then restack: every
    # carton back to its spawn pose on the pallet
    await gripper.open()
    await world.do_command({"command": "park_props", "props": CARTONS})
    on_pallet = {name: await pose(name) for name in CARTONS}
    missed = []
    for index, name in enumerate(CARTONS):
        here = on_pallet.pop(name)
        x, y, z = (v * 1000 for v in here["position"])
        top = z + SIZE_MM[2] / 2
        print(f"{name} ({'top' if index < 6 else 'bottom'} layer) at ({x:.0f}, {y:.0f}, {z:.0f}) mm")
        others = [box(n, p) for n, p in on_pallet.items()]
        # the cartons the held one sits on, ignored while it breaks away from them
        beside = [
            box(n, p) for n, p in on_pallet.items()
            if p["position"][2] * 1000 > z - 1
            or abs(p["position"][0] * 1000 - x) > SIZE_MM[0] / 2
            or abs(p["position"][1] * 1000 - y) > SIZE_MM[1] / 2
        ]
        # the target stays an obstacle until the last, straight-down approach,
        # or the planner may route the arm through it
        await move(x, y, top + ABOVE_MM, [*others, box(name, here)], held=False)
        await move(x, y, top + CUP_GAP_MM, others, held=False)
        await gripper.grab()
        await asyncio.sleep(0.5)  # Grab returns at once; give the cup a moment to seal
        if not (await gripper.is_holding_something()).is_holding_something:
            raise RuntimeError(f"{name}: the cup is not holding it")
        await move(x, y, top + CUP_GAP_MM + BREAKAWAY_MM, beside, held=True)
        await move(x, y, CARRY_Z_MM, others, held=True)
        drop_z = CONVEYOR_TOP_MM + SIZE_MM[2] + 10  # let go 10 mm above the belt
        await move(*DROP_MM, drop_z + ABOVE_MM, others, held=True)
        await move(*DROP_MM, drop_z, others, held=True)
        await gripper.open()
        await asyncio.sleep(1.0)  # let the carton drop onto the belt before reading it back
        await move(*DROP_MM, drop_z + ABOVE_MM, others, held=False)
        px, py, pz = (v * 1000 for v in (await pose(name))["position"])
        landed = (  # on the 300 x 1000 mm belt, resting on its top
            abs(px - DROP_MM[0]) < CONVEYOR_MM[0] / 2 and abs(py) < CONVEYOR_MM[1] / 2
            and abs(pz - (CONVEYOR_TOP_MM + SIZE_MM[2] / 2)) < 10
        )
        print(f"  {'on the conveyor' if landed else 'MISSED the conveyor'}: ({px:.0f}, {py:.0f}, {pz:.0f}) mm")
        if not landed:
            missed.append(name)
        # the stand-in for the belt moving: off its far end, clearing the drop spot
        await world.do_command({
            "command": "set_prop_pose", "prop": name,
            "position": parked(index), "orientation_wxyz": [1, 0, 0, 0],
        })
    return missed


async def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    address, api_key_id, api_key = sys.argv[1:]
    robot = await RobotClient.at_address(
        address, RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id)
    )
    gripper = Gripper.from_robot(robot, "pick-gripper")
    try:
        missed = await depalletize(
            Generic.from_robot(robot, "sim-world"), gripper, MotionClient.from_robot(robot, "builtin")
        )
    finally:
        await gripper.open()  # never leave a carton stuck to the cup
        await robot.close()
    if missed:
        sys.exit(f"missed the conveyor: {missed}")
    print(f"all {len(CARTONS)} cartons reached the conveyor")


if __name__ == "__main__":
    asyncio.run(main())
