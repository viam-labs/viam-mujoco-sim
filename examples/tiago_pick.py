"""TIAGo pick: the TIAGo++ of examples/tiago-pick.json drives to the low
table, looks at it with its head camera, picks the block up with its left
gripper, drives to the shelf and puts the block on it, in one client session.

Usage: uv run python examples/tiago_pick.py <machine address> <api key id> <api key>

Take the address and an API key from the machine's CONNECT tab.

As in mobile_pick.py, RDK keeps the base at its configured pose in the frame
system however far the sim base drives, so every motion.move plans in the
"tiago-base" frame: the block's and the tables' world poses are moved into it
with the base's odometry pose. The left arm is framed on the torso arm, so the
torso's lift is part of the left gripper's chain: RDK lowers it to reach the
low table and raises it for the shelf (the script prints the torso before and
after each move). From the lift on, the planner is given the held block as a
transform on the gripper, except where it would collide by construction (the
lift starts with it on the table, the put-down ends with it below the shelf top).
"""

import asyncio
import math
import sys
from pathlib import Path

from viam.components.arm import Arm, JointPositions
from viam.components.base import Base
from viam.components.camera import Camera
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.components.movement_sensor import MovementSensor
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

BLOCK = "block"
BLOCK_MM = 40
# name: (centre x, y, z), (size x, y, z), mm in the world, as in tiago-pick.json
TABLES = {
    "table": ((0, 2650, 225), (800, 500, 450)),
    "shelf": ((-2650, 1300, 400), (500, 800, 800)),
}
TABLE_TOP_MM = 450
SHELF_TOP_MM = 800
BASE_FRONT_MM = 361.9  # the front of pal_tiago_base's box, ahead of its centre
SPEED_MM_S = 300
TO_TABLE_MM = 2000  # the base stops 38 mm short of the table
BACK_UP_MM = 400
# the base stops 38 mm short of the shelf, which sits 300 mm to the left of its
# path: the right arm stays at home, its gripper ~400 mm ahead at the shelf's height
TO_SHELF_MM = 2000
# where the left gripper picks and places, in the tiago-base frame: ahead and to the left
REACH_MM = (-200, 500)
PLACE_MM = (-2500, 1400)  # on the shelf, 100 mm in from its near edge
ABOVE_MM = 80  # approach and retreat height: the held block slips 25-35 mm down in the grip
DOWN = {"o_x": 0, "o_y": 0, "o_z": -1, "theta": 0}
# the carry pose, in the tiago-base frame: pulled in behind the shelf's face
CARRY_MM = (-250, 300, 750)
LOOK_DOWN_DEG = -40  # the head's tilt to see the table
IMAGE = "tiago-table"
# the held block in the gripper's frame (+z along the tool), slipped 25-35 mm
# down the tool since the pick; 10 mm taller than the block to cover the slip
HELD = Transform(
    reference_frame="held_block",
    pose_in_observer_frame=PoseInFrame(reference_frame="left-gripper", pose=Pose(z=30, o_z=1)),
    physical_object=Geometry(
        center=Pose(o_z=1),  # RDK rejects a geometry without a centre pose
        box=RectangularPrism(dims_mm=Vector3(x=BLOCK_MM, y=BLOCK_MM, z=BLOCK_MM + 10)),
        label="held_block",
    ),
)


def to_base(x: float, y: float, base: tuple[float, float, float]) -> tuple[float, float]:
    """A world point (mm) in the tiago-base frame; base is its world pose (x, y mm, yaw rad)."""
    bx, by, yaw = base
    dx, dy = x - bx, y - by
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def box(
    label: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    base: tuple[float, float, float],
) -> Geometry:
    """A world-aligned box (mm) as a planner obstacle in the tiago-base frame,
    labelled apart from the obstacle components' own geometry."""
    x, y = to_base(center[0], center[1], base)
    pose = Pose(x=x, y=y, z=center[2], o_z=1, theta=-math.degrees(base[2]))
    return Geometry(center=pose, box=RectangularPrism(dims_mm=Vector3(x=size[0], y=size[1], z=size[2])), label=f"{label}-near")


def on_top(name: str, x: float, y: float, z: float, top: float) -> bool:
    """The block (centre x, y, z mm) rests on the table or shelf `name`, whose top is at `top`."""
    (cx, cy, _), (sx, sy, _) = TABLES[name]
    return (
        abs(x - cx) < (sx - BLOCK_MM) / 2
        and abs(y - cy) < (sy - BLOCK_MM) / 2
        and abs(z - (top + BLOCK_MM / 2)) < 5
    )


async def base_pose(odometry: MovementSensor) -> tuple[float, float, float]:
    """The base's world pose (x, y mm, yaw rad): odometry is zero at its spawn, the world origin."""
    position, _ = await odometry.get_position(extra={"return_relative_pos_m": True})
    orientation = await odometry.get_orientation()
    return position.longitude * 1000, position.latitude * 1000, math.radians(orientation.theta)


async def tiago_pick(world, base, odometry, torso, head, head_cam, gripper, motion) -> tuple[float, float, float]:
    """Carry the block from the table to the shelf; return where it ends up (mm, world)."""

    async def block_mm() -> tuple[float, float, float]:
        pose = (await world.do_command({"command": "get_prop_pose", "prop": BLOCK}))["pose"]
        x, y, z = (v * 1000 for v in pose["position"])
        return x, y, z

    async def move_base(bx: float, by: float, z: float, pose, step: str, *extra: Geometry, held: bool = False) -> None:
        """The left gripper's tool point to (bx, by, z) mm in the tiago-base frame, pointing down.
        held: the planner sees the block in the gripper."""
        lift = (await torso.get_joint_positions()).values[0]
        tables = [box(name, *table, pose) for name, table in TABLES.items()]
        state = WorldState(
            obstacles=[GeometriesInFrame(reference_frame="tiago-base", geometries=[*tables, *extra])],
            transforms=[HELD] if held else [],
        )
        destination = PoseInFrame(reference_frame="tiago-base", pose=Pose(x=bx, y=by, z=z, **DOWN))
        if not await motion.move(component_name="left-gripper", destination=destination, world_state=state):
            raise RuntimeError(f"motion.move failed: {step}")
        print(f"{step}: torso {lift:.0f} -> {(await torso.get_joint_positions()).values[0]:.0f} mm")

    async def move(x: float, y: float, z: float, pose, step: str, *extra: Geometry, held: bool = False) -> None:
        """The same, to world (x, y, z) mm."""
        await move_base(*to_base(x, y, pose), z, pose, step, *extra, held=held)

    # start over: the base, the arms and the block at their spawn, odometry at zero
    await world.do_command({"command": "reset"})
    await gripper.open()

    await base.move_straight(TO_TABLE_MM, SPEED_MM_S)
    pose = await base_pose(odometry)
    x, y, z = await block_mm()
    print(f"at the table: base at ({pose[0]:.0f}, {pose[1]:.0f}) mm, block at ({x:.0f}, {y:.0f}, {z:.0f}) mm")
    await head.move_to_joint_positions(JointPositions(values=[0, LOOK_DOWN_DEG]))
    images, _ = await head_cam.get_images()
    path = f"{IMAGE}.{images[0].mime_type.split('/')[-1]}"
    await asyncio.to_thread(Path(path).write_bytes, images[0].data)
    print(f"saved the head camera's view of the table to {path}")
    await head.move_to_joint_positions(JointPositions(values=[0, 0]))

    # the block stays an obstacle until the last, straight-down approach
    await move(x, y, z + ABOVE_MM, pose, "above the block", box(BLOCK, (x, y, z), (BLOCK_MM,) * 3, pose))
    await move(x, y, z, pose, "down to the block")
    await gripper.grab()
    await asyncio.sleep(1.0)  # Grab returns at once; let the jaws close
    if not (await gripper.is_holding_something()).is_holding_something:
        raise RuntimeError("the gripper is not holding the block")
    await move(x, y, z + ABOVE_MM, pose, "lift the block")
    await move_base(*CARRY_MM, pose, "pull in", held=True)

    await base.move_straight(-BACK_UP_MM, SPEED_MM_S)
    await base.spin(90, 45)  # left: now facing world -x
    await base.move_straight(TO_SHELF_MM, SPEED_MM_S)
    pose = await base_pose(odometry)
    x, y = PLACE_MM
    z = SHELF_TOP_MM + BLOCK_MM / 2
    print(f"at the shelf: base at ({pose[0]:.0f}, {pose[1]:.0f}) mm")
    await move(x, y, z + ABOVE_MM, pose, "above the shelf", held=True)
    await move(x, y, z + 2, pose, "down onto the shelf")  # let go 2 mm above it
    await gripper.open()
    await asyncio.sleep(1.0)  # let the block settle before reading it back
    await move(x, y, z + ABOVE_MM, pose, "retreat from the shelf")
    return await block_mm()


async def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    address, api_key_id, api_key = sys.argv[1:]
    robot = await RobotClient.at_address(
        address, RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id)
    )
    try:
        x, y, z = await tiago_pick(
            Generic.from_robot(robot, "sim-world"),
            Base.from_robot(robot, "tiago-base"),
            MovementSensor.from_robot(robot, "odometry"),
            Arm.from_robot(robot, "torso"),
            Arm.from_robot(robot, "head"),
            Camera.from_robot(robot, "head-cam"),
            Gripper.from_robot(robot, "left-gripper"),
            MotionClient.from_robot(robot, "builtin"),
        )
    finally:
        await robot.close()
    print(f"block at ({x:.0f}, {y:.0f}, {z:.0f}) mm")
    if not on_top("shelf", x, y, z, SHELF_TOP_MM):
        sys.exit("the block is not on the shelf")
    print("the block is on the shelf")


if __name__ == "__main__":
    asyncio.run(main())
