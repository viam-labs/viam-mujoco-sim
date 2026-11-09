"""Mobile pick: the Husky of examples/husky-ur5e.json drives its UR5e to
table A, picks up the block, drives to table B and puts it down, in one
client session.

Usage: uv run python examples/mobile_pick.py <machine address> <api key id> <api key>

Take the address and an API key from the machine's CONNECT tab.

RDK keeps the Husky at its configured pose in the frame system however far
the sim base drives, so every motion.move plans in the "husky" frame: the
block's and the tables' world poses are moved into it with the Husky's
odometry pose (exact in this config, and zero at the Husky's spawn, the
world origin). The tables' own obstacle components stay where RDK puts them,
more than 2 m from the Husky's configured pose, out of the arm's reach.
"""

import asyncio
import math
import sys

from viam.components.base import Base
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.components.movement_sensor import MovementSensor
from viam.proto.common import (
    GeometriesInFrame,
    Geometry,
    Pose,
    PoseInFrame,
    RectangularPrism,
    Vector3,
    WorldState,
)
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

BLOCK = "block"
BLOCK_MM = 31.3
# name: (centre x, y, z), (size x, y, z), mm in the world, as in husky-ur5e.json
TABLES = {
    "table-a": ((0, 2780, 225), (800, 500, 450)),
    "table-b": ((-2780, 1600, 225), (500, 800, 450)),
}
TABLE_TOP_MM = 450
PLACE_MM = (-2780, 1600)  # on table B, 250 mm in from its near edge
CHASSIS_FRONT_MM = 493.7  # the Husky's bumper, ahead of its centre
SPEED_MM_S = 300  # gentle: the ideal drive starts and stops at once
TO_TABLE_A_MM = 2000  # the bumper stops 36 mm short of table A
BACK_UP_MM = 400  # then the arm clears table A while the Husky turns
TO_TABLE_B_MM = 2000  # the bumper stops 36 mm short of table B
ABOVE_MM = 100  # approach and retreat height
DOWN = {"o_x": 0, "o_y": 0, "o_z": -1, "theta": 0}
# the carry pose, in the husky frame: close over the Husky, pointing down, with the
# gripper's box (158 mm below the tool point) clear of the 450 mm tables
CARRY_MM = (0, 400, 750)


def to_husky(x: float, y: float, husky: tuple[float, float, float]) -> tuple[float, float]:
    """A world point (mm) in the husky frame; husky is its world pose (x, y mm, yaw rad)."""
    hx, hy, yaw = husky
    dx, dy = x - hx, y - hy
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def box(
    label: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    husky: tuple[float, float, float],
) -> Geometry:
    """A world-aligned box (mm) as a planner obstacle in the husky frame,
    labelled apart from the obstacle components' own geometry."""
    x, y = to_husky(center[0], center[1], husky)
    pose = Pose(x=x, y=y, z=center[2], o_z=1, theta=-math.degrees(husky[2]))
    return Geometry(center=pose, box=RectangularPrism(dims_mm=Vector3(x=size[0], y=size[1], z=size[2])), label=f"{label}-near")


def on_table_b(x: float, y: float, z: float) -> bool:
    (cx, cy, _), (sx, sy, _) = TABLES["table-b"]
    return (
        abs(x - cx) < (sx - BLOCK_MM) / 2
        and abs(y - cy) < (sy - BLOCK_MM) / 2
        and abs(z - (TABLE_TOP_MM + BLOCK_MM / 2)) < 5
    )


async def husky_pose(odometry: MovementSensor) -> tuple[float, float, float]:
    """The Husky's world pose (x, y mm, yaw rad): odometry is zero at its spawn, the world origin."""
    position, _ = await odometry.get_position(extra={"return_relative_pos_m": True})
    orientation = await odometry.get_orientation()
    return position.longitude * 1000, position.latitude * 1000, math.radians(orientation.theta)


async def mobile_pick(world, base, odometry, gripper, motion) -> tuple[float, float, float]:
    """Carry the block from table A to table B; return where it ends up (mm, world)."""

    async def block_mm() -> tuple[float, float, float]:
        pose = (await world.do_command({"command": "get_prop_pose", "prop": BLOCK}))["pose"]
        x, y, z = (v * 1000 for v in pose["position"])
        return x, y, z

    # No held-block transform: the block sits between the fingers, inside the
    # gripper's planner box but for its last ~4 mm (less than the place clearance).
    tool = CARRY_MM  # where the tool point last went, in the husky frame

    async def move_husky(hx: float, hy: float, z: float, husky, step: str, *extra: Geometry) -> None:
        """The gripper's tool point to (hx, hy, z) mm in the husky frame, pointing down."""
        nonlocal tool
        tables = [box(name, *table, husky) for name, table in TABLES.items()]
        state = WorldState(obstacles=[GeometriesInFrame(reference_frame="husky", geometries=[*tables, *extra])])
        destination = PoseInFrame(reference_frame="husky", pose=Pose(x=hx, y=hy, z=z, **DOWN))
        if not await motion.move(component_name="gripper", destination=destination, world_state=state):
            raise RuntimeError(f"motion.move failed: {step}")
        tool = (hx, hy, z)

    async def move(x: float, y: float, z: float, husky, step: str, *extra: Geometry) -> None:
        """The same, to world (x, y, z) mm."""
        hx, hy = to_husky(x, y, husky)
        await move_husky(hx, hy, z, husky, step, *extra)

    async def carry(husky, step: str) -> None:
        """Lift straight up, then pull in close over the Husky, before a drive."""
        await move_husky(tool[0], tool[1], CARRY_MM[2], husky, f"{step}: lift")
        await move_husky(*CARRY_MM, husky, f"{step}: pull in")

    # start over: the Husky, arm and block at their spawn, odometry at zero
    await world.do_command({"command": "reset"})
    await gripper.open()
    await carry((0.0, 0.0, 0.0), "before driving to table A")

    await base.move_straight(TO_TABLE_A_MM, SPEED_MM_S)
    husky = await husky_pose(odometry)
    x, y, z = await block_mm()
    print(f"at table A: Husky at ({husky[0]:.0f}, {husky[1]:.0f}) mm, block at ({x:.0f}, {y:.0f}, {z:.0f}) mm")
    # the block stays an obstacle until the last, straight-down approach
    await move(x, y, z + ABOVE_MM, husky, "above the block", box(BLOCK, (x, y, z), (BLOCK_MM,) * 3, husky))
    await move(x, y, z, husky, "down to the block")
    await gripper.grab()
    await asyncio.sleep(1.0)  # Grab returns at once; let the jaws close
    if not (await gripper.is_holding_something()).is_holding_something:
        raise RuntimeError("the gripper is not holding the block")
    await move(x, y, z + ABOVE_MM, husky, "lift the block")
    await carry(husky, "before backing up")

    await base.move_straight(-BACK_UP_MM, SPEED_MM_S)
    await base.spin(90, 45)  # left: now facing world -x
    await base.move_straight(TO_TABLE_B_MM, SPEED_MM_S)
    husky = await husky_pose(odometry)
    x, y = PLACE_MM
    z = TABLE_TOP_MM + BLOCK_MM / 2
    print(f"at table B: Husky at ({husky[0]:.0f}, {husky[1]:.0f}) mm")
    await move(x, y, z + ABOVE_MM, husky, "above table B")
    await move(x, y, z + 2, husky, "down onto table B")  # let go 2 mm above the table
    await gripper.open()
    await asyncio.sleep(1.0)  # let the block settle before reading it back
    await move(x, y, z + ABOVE_MM, husky, "retreat from table B")
    return await block_mm()


async def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    address, api_key_id, api_key = sys.argv[1:]
    robot = await RobotClient.at_address(
        address, RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id)
    )
    try:
        x, y, z = await mobile_pick(
            Generic.from_robot(robot, "sim-world"),
            Base.from_robot(robot, "husky"),
            MovementSensor.from_robot(robot, "odometry"),
            Gripper.from_robot(robot, "gripper"),
            MotionClient.from_robot(robot, "builtin"),
        )
    finally:
        await robot.close()
    print(f"block at ({x:.0f}, {y:.0f}, {z:.0f}) mm")
    if not on_table_b(x, y, z):
        sys.exit("the block is not on table B")
    print("the block is on table B")


if __name__ == "__main__":
    asyncio.run(main())
