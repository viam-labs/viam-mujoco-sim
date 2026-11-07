"""Contract tests: the shapes examples/scripted_pick.py (viam-isaac-sim)
depends on."""

import asyncio

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

import mujoco_module.models  # noqa: F401
from mujoco_module.models.arm import MujocoArm
from mujoco_module.models.camera import MujocoCamera
from mujoco_module.models.gripper import MujocoGripper
from mujoco_module.models.world import MujocoWorld
from conftest import BLOCK


def config(name, attrs):
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def cell(running_sim):
    world = MujocoWorld.new(config("sim-world", {"props": [BLOCK], "real_time": False}), {})
    arm = MujocoArm.new(config("pick-arm", {"world": "sim-world", "asset": "lite6"}), {})
    gripper = MujocoGripper.new(
        config("pick-gripper", {"world": "sim-world", "arm": "pick-arm", "sensorless": True}),
        {},
    )
    return world, arm, gripper


def test_validate_returns_world_and_arm_dependencies():
    deps, _ = MujocoGripper.validate_config(
        config("g", {"world": "sim-world", "arm": "pick-arm"})
    )
    assert deps == ["sim-world", "pick-arm"]
    with pytest.raises(ValueError, match="asset must be one of"):
        MujocoArm.validate_config(config("a", {"world": "sim-world", "asset": "ur5e"}))
    with pytest.raises(ValueError, match="grip_force_n"):
        MujocoGripper.validate_config(
            config("g", {"world": "w", "arm": "a", "grip_force_n": 20})
        )


def test_world_commands(cell):
    world, _, _ = cell
    status = run(world.do_command({"command": "status"}))
    assert {"playing", "sim_time", "real_time_factor", "slips", "warnings"} <= set(status)
    assert run(world.do_command({"command": "pause"})) == {"ok": True}  # hold the block still
    placed = run(
        world.do_command(
            {
                "command": "set_prop_pose",
                "prop": "pick_block",
                "position": [0.3, 0.05, 0.02],
                "orientation_wxyz": [1, 0, 0, 0],
            }
        )
    )
    # scripted_pick reads these at the top level
    assert placed["position"] == pytest.approx([0.3, 0.05, 0.02])
    assert placed["orientation_wxyz"] == pytest.approx([1, 0, 0, 0])
    pose = run(world.do_command({"command": "get_prop_pose", "prop": "pick_block"}))
    assert pose["pose"]["position"] == pytest.approx([0.3, 0.05, 0.02])
    assert run(world.do_command({"command": "play"})) == {"ok": True}


def test_gripper_holding_reports_firm_grasp(cell):
    _, _, gripper = cell
    assert run(gripper.grab()) is True
    status = run(gripper.is_holding_something())
    assert isinstance(status.meta["firm_grasp"], bool)
    assert status.meta["firm_grasp"] is False
    assert status.is_holding_something is True  # sensorless: the latched grip
    run(gripper.open())
    assert run(gripper.get_current_inputs()) == [0.0]


def test_arm_dynamics_and_kinematics_url(cell):
    _, arm, _ = cell
    meta = run(arm.do_command({"command": "dynamics_metadata"}))
    assert isinstance(meta["max_velocity_rad_s"], list)
    assert arm._kinematics_url().endswith("lite6_kinematics.json")
    pose = run(arm.get_end_position())  # flange, mm, Viam orientation vector
    assert pose.z > 100
    assert (pose.o_x, pose.o_y, pose.o_z) == pytest.approx((0, 0, -1), abs=0.05)


def test_streamed_trajectory_yields_an_update(cell):
    from datetime import timedelta

    from viam.components.arm import Arm

    _, arm, _ = cell
    home = [0.0, 0.0, 89.954, 0.0, 89.954, 0.0]

    async def batches():
        yield [
            Arm.TrajectoryPoint(time=timedelta(0), positions=home),
            Arm.TrajectoryPoint(time=timedelta(seconds=0.2), positions=home),
        ]

    async def consume():
        return [update async for update in arm.move_through_joint_positions_streamed(batches())]

    assert len(run(consume())) == 1


def test_camera_properties(cell):
    camera = MujocoCamera.new(
        config("scene-cam", {"world": "sim-world", "target": [0.3, 0, 0.15]}), {}
    )
    properties = run(camera.get_properties())
    assert properties.intrinsic_parameters.width_px == 640
    images, _ = run(camera.get_images())
    assert images[0].data[:2] == b"\xff\xd8"  # JPEG


def test_arm_rejects_non_finite_targets(cell):
    from viam.components.arm import JointPositions

    _, arm, _ = cell
    with pytest.raises(ValueError, match="finite"):
        run(arm.move_to_joint_positions(JointPositions(values=[float("nan")] + [0.0] * 5)))
