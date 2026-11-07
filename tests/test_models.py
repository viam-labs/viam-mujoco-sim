"""Contract tests: the shapes examples/scripted_pick.py (viam-isaac-sim)
depends on."""

import asyncio
import math
import time

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Vector3
from viam.utils import dict_to_struct

import mujoco_module.models  # noqa: F401
from mujoco_module.models.arm import MujocoArm
from mujoco_module.models.base import MujocoBase
from mujoco_module.models.camera import MujocoCamera
from mujoco_module.models.gripper import MujocoGripper
from mujoco_module.models.lidar import MujocoLidar
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
    assert [image.name for image in images] == ["scene-cam"]
    assert images[0].data[:2] == b"\xff\xd8"  # JPEG
    assert properties.supports_pcd


def test_wrist_camera_returns_depth_and_point_cloud(cell):
    cfg = config("wrist-cam", {"world": "sim-world", "depth": True, "width": 64, "height": 48})
    cfg.frame.parent = "pick-arm"
    cfg.frame.translation.z = 50  # mm along the tool axis
    camera = MujocoCamera.new(cfg, {})
    images, _ = run(camera.get_images())
    assert [image.name for image in images] == ["wrist-cam", "depth"]
    depth = images[1].bytes_to_depth_array()
    assert (len(depth), len(depth[0])) == (24, 32)
    only_depth, _ = run(camera.get_images(filter_source_names=["depth"]))
    assert [image.name for image in only_depth] == ["depth"]
    data, mime = run(camera.get_point_cloud())
    assert mime == "pointcloud/pcd"
    assert data.startswith(b"VERSION .7\nFIELDS x y z rgb")


def test_arm_rejects_non_finite_targets(cell):
    from viam.components.arm import JointPositions

    _, arm, _ = cell
    with pytest.raises(ValueError, match="finite"):
        run(arm.move_to_joint_positions(JointPositions(values=[float("nan")] + [0.0] * 5)))


@pytest.mark.parametrize(
    "model, attrs",
    [
        (MujocoArm, {"asset": "lite6", "move_timeout_sec": -1}),
        (MujocoArm, {"asset": "lite6", "robot_control_freq_hz": float("nan")}),
        (MujocoArm, {"asset": "lite6", "path_max_step_deg": 45}),
        (MujocoArm, {"asset": "lite6", "path_tolerance_delta_deg": True}),
        (MujocoCamera, {"width": 640.5}),
        (MujocoCamera, {"fov_deg": 180}),
    ],
)
def test_validate_config_rejects_bad_values(model, attrs):
    with pytest.raises(ValueError):
        model.validate_config(config("x", {"world": "sim-world", **attrs}))


def test_arm_moves_through_model(cell):
    from viam.components.arm import JointPositions

    _, arm, _ = cell
    target = [10.0, -10.0, 80.0, 0.0, 80.0, 10.0]
    run(arm.move_to_joint_positions(JointPositions(values=target)))
    assert run(arm.get_joint_positions()).values == pytest.approx(target, abs=0.6)
    run(arm.move_through_joint_positions([JointPositions(values=[0, 0, 90, 0, 90, 0])]))
    assert run(arm.get_joint_positions()).values == pytest.approx([0, 0, 90, 0, 90, 0], abs=0.6)


def test_densify_ends_on_waypoint_within_step():
    from mujoco_module.models.arm import _densify

    step = 0.05
    out = _densify([0.0, 0.0], [[0.3, -0.1], [0.3, 0.2]], step)
    assert out[-1] == [0.3, 0.2]
    prev = [0.0, 0.0]
    for point in out:
        assert max(abs(a - b) for a, b in zip(point, prev)) <= step + 1e-12
        prev = point


def test_kinematics_is_vendored(cell):
    import json

    from viam.proto.common import KinematicsFileFormat

    _, arm, _ = cell
    fmt, data = run(arm.get_kinematics())
    assert fmt == KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA
    assert json.loads(data)["name"] == "lite6"


def test_arm_frame_parent_must_be_world():
    cfg = config("x", {"world": "sim-world", "asset": "lite6"})
    cfg.frame.parent = "world"
    MujocoArm.validate_config(cfg)
    cfg.frame.parent = "pick-arm"
    with pytest.raises(ValueError, match="frame.parent must be"):
        MujocoArm.validate_config(cfg)


def test_camera_frame_parent_may_be_an_arm_but_not_aimed():
    cfg = config("wrist", {"world": "sim-world"})
    cfg.frame.parent = "pick-arm"
    MujocoCamera.validate_config(cfg)
    cfg = config("wrist", {"world": "sim-world", "target": [0, 0, 0]})
    cfg.frame.parent = "pick-arm"
    with pytest.raises(ValueError, match="target can't aim"):
        MujocoCamera.validate_config(cfg)


def test_gripper_frame_parent_may_be_its_arm():
    cfg = config("g", {"world": "sim-world", "arm": "pick-arm"})
    cfg.frame.parent = "pick-arm"
    MujocoGripper.validate_config(cfg)


def test_park_props_rejects_a_bare_string(cell):
    world, _, _ = cell
    with pytest.raises(ValueError):
        run(world.do_command({"command": "park_props", "props": "cube"}))


@pytest.fixture
def rover_cell(running_sim):
    world = MujocoWorld.new(config("sim-world", {"real_time": False}), {})
    base = MujocoBase.new(config("rover", {"world": "sim-world", "asset": "viam_rover"}), {})
    return world, base


def test_camera_on_the_rover_looks_forward(rover_cell):
    _, base = rover_cell
    cfg = config("rover-cam", {"world": "sim-world"})
    cfg.frame.parent = "rover"
    cfg.frame.translation.y = 120
    cfg.frame.translation.z = 150
    cfg.frame.orientation.quaternion.w = 0.7071068
    cfg.frame.orientation.quaternion.x = -0.7071068
    cfg.frame.orientation.quaternion.y = 0
    cfg.frame.orientation.quaternion.z = 0
    MujocoCamera.new(cfg, {})
    sim = base._h()._sim

    def forward():
        return -sim.data.cam_xmat[sim.model.camera("rover-cam").id].reshape(3, 3)[:, 2]

    assert sim.run(forward) == pytest.approx([0, 1, 0], abs=1e-3)


@pytest.mark.parametrize(
    "attrs, message",
    [
        ({}, "asset must be one of"),
        ({"asset": "jetbot"}, "asset must be one of"),
        ({"asset": "viam_rover", "max_linear_mps": 0}, "max_linear_mps"),
        ({"asset": "viam_rover", "max_angular_dps": -5}, "max_angular_dps"),
        ({"asset": "viam_rover", "move_timeout_margin_sec": "soon"}, "move_timeout_margin_sec"),
    ],
)
def test_base_validate_rejects_bad_values(attrs, message):
    with pytest.raises(ValueError, match=message):
        MujocoBase.validate_config(config("rover", {"world": "sim-world", **attrs}))


def test_base_frame_parent_must_be_world():
    cfg = config("rover", {"world": "sim-world", "asset": "viam_rover"})
    cfg.frame.parent = "pick-arm"
    with pytest.raises(ValueError, match="frame.parent must be"):
        MujocoBase.validate_config(cfg)
    assert MujocoBase.validate_config(config("rover", {"world": "sim-world", "asset": "viam_rover"}))[0] == ["sim-world"]


def test_base_drives_through_the_model(rover_cell):
    _, base = rover_cell
    handle = base._h()
    sim = handle._sim

    def joint(name):
        return sim.run(lambda: float(sim.data.qpos[sim.model.joint(f"rover/{name}").qposadr[0]]))

    run(base.move_straight(300, 300))
    assert joint("y") == pytest.approx(0.3, abs=0.01)
    run(base.spin(90, 180))
    assert joint("yaw") == pytest.approx(math.radians(90), abs=math.radians(2))
    assert not run(base.is_moving())
    run(base.set_velocity(Vector3(y=200), Vector3(z=0)))
    assert run(base.is_moving())
    assert handle._v == pytest.approx(0.2)
    run(base.set_power(Vector3(y=-0.5), Vector3(z=0.25)))
    assert handle._v == pytest.approx(-0.5 * handle.max_linear_mps)
    assert handle._w == pytest.approx(0.25 * handle.max_angular_rad_s)
    run(base.stop())
    properties = run(base.get_properties())
    assert properties.width_meters == pytest.approx(0.356)
    assert properties.turning_radius_meters == 0
    assert properties.wheel_circumference_meters == pytest.approx(0.381)
    (geometry,) = run(base.get_geometries())
    assert geometry.box.dims_mm.y == pytest.approx(300)
    assert geometry.center.z == pytest.approx(80)


def test_base_set_power_rejects_nan(rover_cell):
    _, base = rover_cell
    with pytest.raises(ValueError, match="finite"):
        run(base.set_power(Vector3(y=float("nan")), Vector3()))


def test_base_rejects_zero_velocity(rover_cell):
    _, base = rover_cell
    with pytest.raises(ValueError, match="non-zero"):
        run(base.move_straight(100, 0))


def _cancel_mid_move(base, cancel_after_s):
    async def flow():
        task = asyncio.ensure_future(base.move_straight(100000, 50))
        await asyncio.sleep(cancel_after_s)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.3)  # let a delayed start land and the callback cancel it
        for _ in range(100):
            if not await base.is_moving():
                break
            await asyncio.sleep(0.01)
        assert not await base.is_moving()

    run(flow())
    assert base._h()._goal is None


def test_base_cancel_during_wait_stops_the_rover(rover_cell):
    _, base = rover_cell
    _cancel_mid_move(base, 0.05)


def test_base_cancel_during_start_stops_the_rover(rover_cell, monkeypatch):
    _, base = rover_cell
    handle = base._h()
    real = handle.move_straight

    def slow_start(*args):
        time.sleep(0.2)
        return real(*args)

    monkeypatch.setattr(handle, "move_straight", slow_start)
    _cancel_mid_move(base, 0.05)


@pytest.mark.parametrize(
    "attrs, message",
    [
        ({"range_min_m": -1}, "range_min_m"),
        ({"range_max_m": 0}, "range_max_m"),
        ({"range_min_m": 5, "range_max_m": 2}, "range_max_m must be greater"),
        ({"angular_resolution_deg": 0}, "angular_resolution_deg"),
        ({"angular_resolution_deg": 0.05}, "angular_resolution_deg"),
        ({"angular_resolution_deg": 400}, "angular_resolution_deg"),
    ],
)
def test_lidar_validate_rejects_bad_values(attrs, message):
    with pytest.raises(ValueError, match=message):
        MujocoLidar.validate_config(config("lidar", {"world": "sim-world", **attrs}))


def test_lidar_frame_parent_may_be_a_base():
    cfg = config("lidar", {"world": "sim-world"})
    cfg.frame.parent = "rover"
    assert MujocoLidar.validate_config(cfg)[0] == ["sim-world"]


def test_lidar_serves_point_clouds_only(rover_cell):
    cfg = config("lidar", {"world": "sim-world"})
    cfg.frame.parent = "rover"
    cfg.frame.translation.z = 160
    lidar = MujocoLidar.new(cfg, {})
    data, mime = run(lidar.get_point_cloud())
    assert mime == "pointcloud/pcd"
    assert data.startswith(b"VERSION .7\nFIELDS x y z\n")
    assert run(lidar.get_properties()).supports_pcd
    with pytest.raises(NotImplementedError):
        run(lidar.get_images())
