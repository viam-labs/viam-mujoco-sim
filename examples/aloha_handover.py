"""ALOHA handover: in examples/aloha-handover.json, the left arm picks up the
block and sets it down in the middle of the table; the right arm picks it up
from there and sets it down on its own side. Both arms are limbs of one robot
file (viam-labs:mujoco-sim:robot) and plan with the builtin motion service.

Usage: uv run python examples/aloha_handover.py <machine address> <api key id> <api key>

Take the address and an API key from the machine's CONNECT tab. Poses are in
the world frame, mm; the grippers point straight down (each arm's end effector
+Z is out of its fingers).
"""

import asyncio
import sys

from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

BLOCK = "block"
BLOCK_MM = 30
MIDDLE_MM = (0, 50)  # where the left arm sets the block down
RIGHT_MM = (200, 50)  # where the right arm sets it down
ABOVE_MM = 100  # approach and retreat height above the grasp
GRASP_MM = 25  # tool point height to grasp at: the fingertips reach 19 mm past it, ~6 mm above the table
DOWN = {"o_x": 0, "o_y": 0, "o_z": -1, "theta": 0}


async def handover(world, grippers, motion) -> tuple[float, float, float]:
    """Pass the block left to middle, then middle to right; return where it ends (mm, world)."""

    async def block_mm() -> tuple[float, float, float]:
        pose = (await world.do_command({"command": "get_prop_pose", "prop": BLOCK}))["pose"]
        x, y, z = (v * 1000 for v in pose["position"])
        return x, y, z

    async def move(arm: str, x: float, y: float, z: float, step: str) -> None:
        destination = PoseInFrame(reference_frame="world", pose=Pose(x=x, y=y, z=z, **DOWN))
        if not await motion.move(component_name=f"{arm}-gripper", destination=destination):
            raise RuntimeError(f"motion.move failed: {arm} {step}")

    async def pick(arm: str, x: float, y: float) -> None:
        await move(arm, x, y, GRASP_MM + ABOVE_MM, "above the block")
        await move(arm, x, y, GRASP_MM, "down to the block")
        await grippers[arm].grab()
        await asyncio.sleep(1.0)  # Grab returns at once; let the jaws close
        if not (await grippers[arm].is_holding_something()).is_holding_something:
            raise RuntimeError(f"the {arm} gripper is not holding the block")
        await move(arm, x, y, GRASP_MM + ABOVE_MM, "lift the block")

    async def place(arm: str, x: float, y: float) -> None:
        await move(arm, x, y, GRASP_MM + ABOVE_MM, "above the spot")
        await move(arm, x, y, GRASP_MM + 2, "down onto the table")  # let go 2 mm above it
        await grippers[arm].open()
        await asyncio.sleep(1.0)  # let the block settle
        await move(arm, x, y, GRASP_MM + ABOVE_MM, "retreat")

    await world.do_command({"command": "reset"})
    for gripper in grippers.values():
        await gripper.open()
    x, y, _ = await block_mm()
    print(f"block at ({x:.0f}, {y:.0f}) mm")
    await pick("left", x, y)
    await place("left", *MIDDLE_MM)
    # out of the right arm's way: back over its own side
    await move("left", x, y, GRASP_MM + ABOVE_MM, "back to its side")
    x, y, z = await block_mm()
    print(f"left arm set it down at ({x:.0f}, {y:.0f}, {z:.0f}) mm")
    await pick("right", x, y)
    await place("right", *RIGHT_MM)
    return await block_mm()


async def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    address, api_key_id, api_key = sys.argv[1:]
    robot = await RobotClient.at_address(
        address, RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id)
    )
    try:
        x, y, z = await handover(
            Generic.from_robot(robot, "sim-world"),
            {side: Gripper.from_robot(robot, f"{side}-gripper") for side in ("left", "right")},
            MotionClient.from_robot(robot, "builtin"),
        )
    finally:
        await robot.close()
    print(f"block at ({x:.0f}, {y:.0f}, {z:.0f}) mm")
    if abs(x - RIGHT_MM[0]) > 10 or abs(y - RIGHT_MM[1]) > 10 or abs(z - BLOCK_MM / 2) > 5:
        sys.exit("the block is not where the right arm put it")
    print("the block made it across")


if __name__ == "__main__":
    asyncio.run(main())
