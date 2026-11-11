"""Contract tests: the shapes examples/scripted_pick.py (viam-isaac-sim)
depends on."""

import asyncio
import math
import shutil
import time
from pathlib import Path

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import KinematicsFileFormat, Vector3
from viam.utils import dict_to_struct

import mujoco_module.models  # noqa: F401
from mujoco_module.models.arm import MujocoArm
from mujoco_module.models.base import MujocoBase
from mujoco_module.models.camera import MujocoCamera
from mujoco_module.models.gripper import MujocoGripper
from mujoco_module.models.lidar import MujocoLidar
from mujoco_module.models.movement_sensor import MujocoMovementSensor
from mujoco_module.models.world import MujocoWorld
from conftest import BLOCK

KINEMATICS_FILE = Path(__file__).parents[1] / "assets" / "ufactory_lite6" / "lite6_kinematics.json"


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
        MujocoArm.validate_config(config("a", {"world": "sim-world", "asset": "puma560"}))
    with pytest.raises(ValueError, match="grip_force_n"):
        MujocoGripper.validate_config(
            config("g", {"world": "w", "arm": "a", "grip_force_n": -1})
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
    assert {"jaw", "open", "closed"} <= set(status.meta)
    # the Lite6's jaw is a slide, so its meters keep their old names
    assert status.meta["jaw_m"] == status.meta["jaw"]
    assert (status.meta["open_m"], status.meta["closed_m"]) == pytest.approx((-0.0081, -1e-5))
    run(gripper.open())
    assert run(gripper.get_current_inputs()) == [0.0]


def test_arm_dynamics_and_kinematics_url(cell):
    _, arm, _ = cell
    meta = run(arm.do_command({"command": "dynamics_metadata"}))
    assert isinstance(meta["max_velocity_rad_s"], list)
    assert arm._h().kinematics_url.endswith("lite6_kinematics.json")
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


def test_move_through_streams_no_faster_than_the_joint_limit(running_sim, monkeypatch):
    """Targets pushed faster than max_joint_deg_s make the rate-limited servo
    chase the newest target and cut corners off the planned path."""
    from viam.components.arm import JointPositions

    MujocoWorld.new(config("sim-world", {"props": [BLOCK], "real_time": False}), {})
    arm = MujocoArm.new(
        config("pick-arm", {"world": "sim-world", "asset": "lite6", "max_joint_deg_s": 30}), {}
    )
    handle = arm._h()
    pushed = []
    original = handle.set_joint_targets
    monkeypatch.setattr(handle, "set_joint_targets", lambda t: (pushed.append(list(t)), original(t)))
    start = run(arm.get_joint_positions()).values
    run(arm.move_through_joint_positions([JointPositions(values=[20, 0, 90, 0, 90, 0])]))
    prev = [math.radians(v) for v in start]
    for targets in pushed:
        # 30 deg/s at the default 50 Hz is 0.6 deg per push
        assert max(abs(a - b) for a, b in zip(targets, prev)) <= math.radians(0.6) + 1e-9
        prev = targets


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


def test_arm_frame_parent_is_a_dependency():
    cfg = config("x", {"world": "sim-world", "asset": "ur5e"})
    cfg.frame.parent = "world"
    assert MujocoArm.validate_config(cfg) == (["sim-world"], [])
    cfg.frame.parent = "husky"
    assert MujocoArm.validate_config(cfg) == (["sim-world", "husky"], [])


def test_an_arm_framed_on_a_base_rides_it(running_sim):
    MujocoWorld.new(config("sim-world", {"real_time": False}), {})
    husky = config("husky", {"world": "sim-world", "asset": "clearpath_husky"})
    husky.frame.parent = "world"
    husky.frame.translation.x = 1000  # off the origin, so a world-mounted arm can't pass
    base = MujocoBase.new(husky, {})
    cfg = config("arm", {"world": "sim-world", "asset": "ur5e"})
    cfg.frame.parent = "husky"
    cfg.frame.translation.y, cfg.frame.translation.z = 250, 385
    arm = MujocoArm.new(cfg, {})
    try:
        frame = running_sim.run(lambda: running_sim.data.site("arm/arm_frame").xpos.copy())
        assert frame == pytest.approx([1.0, 0.25, 0.385], abs=1e-6)
        pose = run(arm.get_end_position())
        # the UR5e's home end pose in its own frame, not the world's (+385 mm up)
        assert (pose.x, pose.y, pose.z) == pytest.approx((-134, 492, 488), abs=5)
    finally:
        run(arm.close())
        run(base.close())


def test_camera_frame_parent_may_be_an_arm_but_not_aimed():
    cfg = config("wrist", {"world": "sim-world"})
    cfg.frame.parent = "pick-arm"
    MujocoCamera.validate_config(cfg)
    cfg = config("wrist", {"world": "sim-world", "target": [0, 0, 0]})
    cfg.frame.parent = "pick-arm"
    with pytest.raises(ValueError, match="target can't aim"):
        MujocoCamera.validate_config(cfg)


def test_bundled_ur5e_and_2f85_validate():
    assert MujocoArm.validate_config(config("a", {"world": "sim-world", "asset": "ur5e"}))[0] == ["sim-world"]
    gripper = {"world": "sim-world", "arm": "a", "asset": "robotiq_2f85"}
    assert MujocoGripper.validate_config(config("g", gripper))[0] == ["sim-world", "a"]
    epick = {**gripper, "asset": "robotiq_epick"}
    assert MujocoGripper.validate_config(config("g", epick))[0] == ["sim-world", "a"]
    with pytest.raises(ValueError, match="asset must be one of"):
        MujocoGripper.validate_config(config("g", {**gripper, "asset": "suction"}))


def test_gripper_validate_mjcf():
    base = {"world": "sim-world", "arm": "pick-arm"}
    MujocoGripper.validate_config(config("g", {**base, "mjcf": "ufactory_lite6/lite6.xml"}))
    with pytest.raises(ValueError, match="asset or mjcf, not both"):
        MujocoGripper.validate_config(
            config("g", {**base, "asset": "lite6_gripper", "mjcf": "ufactory_lite6/lite6.xml"})
        )
    with pytest.raises(ValueError, match="mjcf: file .*nope.xml not found"):
        MujocoGripper.validate_config(config("g", {**base, "mjcf": "nope.xml"}))
    with pytest.raises(ValueError, match="open"):
        MujocoGripper.validate_config(config("g", {**base, "open": "wide"}))
    with pytest.raises(ValueError, match="mesh_dir must be a folder path"):
        MujocoGripper.validate_config(config("g", {**base, "mesh_dir": ""}))


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
        ({}, r"asset must be one of \['clearpath_husky', 'pal_tiago_base', 'viam_rover'\], got None; or set mjcf"),
        ({"asset": "jetbot"}, "asset must be one of"),
        ({"asset": "viam_rover", "mjcf": "viam_rover/rover.xml"}, "set asset or mjcf, not both"),
        ({"mjcf": ""}, "mjcf must be a file path"),
        ({"mjcf": "nope.xml"}, "mjcf: file .*nope.xml not found"),
        ({"mjcf": "clearpath_husky/LICENSE"}, r"mjcf must be an .xml \(MJCF\) or .urdf file"),
        ({"asset": "viam_rover", "drive": "tank"}, 'drive must be "ideal" or "wheels"'),
        ({"asset": "viam_rover", "chassis": ""}, "chassis must be a body name"),
        ({"mjcf": "viam_rover/rover.xml", "mesh_dir": 3}, "mesh_dir must be a folder path"),
        ({"asset": "viam_rover", "forward_axis": "x"}, r"forward_axis must be one of \['\+x', '-x', '\+y', '-y'\]"),
        ({"asset": "viam_rover", "wheel_radius_m": 0}, "wheel_radius_m"),
        ({"asset": "viam_rover", "wheels": ["a", "b"]}, r'wheels must be \{"left": \[joint, ...\], "right": \[joint, ...\]\}'),
        ({"asset": "viam_rover", "wheels": {"left": ["a"]}}, "wheels must be"),
        ({"asset": "viam_rover", "wheels": {"left": ["a", "b"], "right": ["b"]}}, "wheels joint 'b' is on both sides"),
        ({"asset": ["x"]}, "asset must be one of"),
        ({"asset": "viam_rover", "wheels": {"left": ["a"], "right": []}}, "wheels must be"),
        ({"asset": "viam_rover", "max_linear_mps": 0}, "max_linear_mps"),
        ({"asset": "viam_rover", "max_angular_dps": -5}, "max_angular_dps"),
        ({"mjcf": "viam_rover/rover.xml", "max_angular_dps": "fast"}, "max_angular_dps"),
        ({"asset": "viam_rover", "move_timeout_margin_sec": "soon"}, "move_timeout_margin_sec"),
    ],
)
def test_base_validate_rejects_bad_values(attrs, message):
    with pytest.raises(ValueError, match=message):
        MujocoBase.validate_config(config("rover", {"world": "sim-world", **attrs}))


def test_base_validate_accepts_a_file_and_overrides():
    attrs = {
        "world": "sim-world", "mjcf": "clearpath_husky/husky.xml", "drive": "wheels",
        "chassis": "base_link", "forward_axis": "+x", "wheel_radius_m": 0.17,
        "wheels": {"left": ["front_left_wheel"], "right": ["front_right_wheel"]},
    }
    assert MujocoBase.validate_config(config("husky", attrs)) == (["sim-world"], [])


def test_a_base_from_a_file_reports_its_inferred_shape(running_sim, tmp_path):
    from test_base import ROS_BASE

    MujocoWorld.new(config("sim-world", {"real_time": False}), {})
    path = tmp_path / "skid.xml"
    path.write_text(ROS_BASE)
    attrs = {"world": "sim-world", "mjcf": str(path), "drive": "wheels"}
    with pytest.raises(ValueError, match="set max_linear_mps and max_angular_dps"):
        MujocoBase.new(config("skid", attrs), {})
    base = MujocoBase.new(config("skid", {**attrs, "max_linear_mps": 1, "max_angular_dps": 90}), {})
    properties = run(base.get_properties())
    assert properties.width_meters == pytest.approx(0.5)
    assert properties.wheel_circumference_meters == pytest.approx(2 * math.pi * 0.1)
    (geometry,) = run(base.get_geometries())
    # the 0.6 x 0.4 box along the file's x is 0.6 along the base's +Y, its centre 0.2 above the floor
    assert (geometry.box.dims_mm.x, geometry.box.dims_mm.y, geometry.box.dims_mm.z) == pytest.approx((400, 600, 100))
    assert (geometry.center.x, geometry.center.y, geometry.center.z) == pytest.approx((0, 0, 200))
    assert geometry.label == "skid-chassis"
    run(base.close())


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

def odometry_config(**attrs):
    return config("odometry", {"world": "sim-world", "base": "rover", **attrs})


def test_movement_sensor_validate():
    deps, _ = MujocoMovementSensor.validate_config(odometry_config())
    assert deps == ["sim-world", "rover"]
    for attrs, message in [
        ({"base": ""}, "base"),
        ({"base": 3}, "base"),
        ({"linear_scale_error": -1}, "linear_scale_error"),
        ({"linear_scale_error": float("nan")}, "linear_scale_error"),
        ({"linear_scale_error": True}, "linear_scale_error"),
        ({"heading_drift_dps": float("inf")}, "heading_drift_dps"),
        ({"heading_drift_dps": "fast"}, "heading_drift_dps"),
    ]:
        with pytest.raises(ValueError, match=message):
            MujocoMovementSensor.validate_config(odometry_config(**attrs))
    with pytest.raises(ValueError, match="world"):
        MujocoMovementSensor.validate_config(config("odometry", {"base": "rover"}))


def test_movement_sensor_serves_odometry(rover_cell):
    _, base = rover_cell
    sensor = MujocoMovementSensor.new(odometry_config(), {})
    try:
        props = run(sensor.get_properties())
        assert props.linear_velocity_supported and props.angular_velocity_supported
        assert props.orientation_supported and props.position_supported
        assert props.compass_heading_supported and not props.linear_acceleration_supported

        run(base.move_straight(300, 300))
        run(base.spin(90, 90))
        run(base.set_velocity(Vector3(y=200), Vector3(z=0)))
        time.sleep(0.3)
        velocity = run(sensor.get_linear_velocity())
        assert (velocity.x, velocity.y) == pytest.approx((0.0, 0.2), abs=0.01)
        assert run(sensor.get_angular_velocity()).z == pytest.approx(0.0, abs=0.5)
        run(base.stop())
        time.sleep(0.2)

        orientation = run(sensor.get_orientation())
        assert (orientation.o_x, orientation.o_y, orientation.o_z) == (0, 0, 1)
        assert orientation.theta == pytest.approx(90, abs=2)
        assert run(sensor.get_compass_heading()) == pytest.approx(270, abs=2)

        # rover_cell free-runs the sim (real_time false), so the sleep above drove an
        # unknown distance; compare with the base's joints, not a fixed number.
        # The base spawned at identity, so odometry (x, y) = joint (x, y).
        sim = base._h()._sim

        def joint(name):
            return sim.run(lambda: float(sim.data.qpos[sim.model.joint(f"rover/{name}").qposadr[0]]))

        relative, altitude = run(sensor.get_position(extra={"return_relative_pos_m": True}))
        assert altitude == 0
        assert relative.latitude == pytest.approx(joint("y"), abs=0.01)  # y, m
        assert relative.longitude == pytest.approx(joint("x"), abs=0.01)
        assert relative.longitude < -0.03  # drove toward -X after the left turn
        geo, _ = run(sensor.get_position())
        assert geo.latitude > 0 and geo.longitude < 0  # north-west of (0, 0)
        assert abs(geo.latitude) < 1e-4  # metres, not degrees

        readings = run(sensor.get_readings())
        assert set(readings) == {
            "position", "altitude", "linear_velocity", "angular_velocity",
            "compass", "orientation",
        }
        assert readings["compass"] == pytest.approx(run(sensor.get_compass_heading()), abs=1e-6)
        with pytest.raises(NotImplementedError):
            run(sensor.get_linear_acceleration())
        assert run(sensor.get_geometries()) == []

        assert run(sensor.do_command({"command": "reset_odometry"})) == {}
        relative, _ = run(sensor.get_position(extra={"return_relative_pos_m": True}))
        assert (relative.latitude, relative.longitude) == pytest.approx((0, 0), abs=1e-3)
        with pytest.raises(ValueError, match="unknown command"):
            run(sensor.do_command({"command": "fly"}))
    finally:
        run(sensor.close())


def test_rover_nav_example_validates():
    import json
    import pathlib

    example = json.loads(
        (pathlib.Path(__file__).parent.parent / "examples" / "rover-nav.json").read_text()
    )
    (sensor,) = [c for c in example["components"] if c["name"] == "odometry"]
    deps, _ = MujocoMovementSensor.validate_config(config("odometry", sensor["attributes"]))
    assert deps == ["sim-world", "rover"]
    names = {c["name"] for c in example["components"]}
    slam = next(s for s in example["services"] if s["name"] == "slam")["attributes"]
    assert {slam["base"], slam["movement_sensor"], slam["lidar"]} <= names


@pytest.mark.parametrize(
    "example",
    [
        "rover-lidar.json", "rover-nav.json", "husky-lidar.json", "lite6-pick.json", "ur5e-pick.json",
        "depalletize.json", "husky-ur5e.json",
    ],
)
def test_example_obstacles_validate(example):
    import json
    import pathlib

    from google.protobuf.json_format import ParseDict
    from viam.proto.app.robot import Frame

    from mujoco_module.models.obstacle import MujocoObstacle

    cfg = json.loads((pathlib.Path(__file__).parent.parent / "examples" / example).read_text())
    obstacles = [c for c in cfg["components"] if c["model"] == "viam-labs:mujoco-sim:obstacle"]
    assert obstacles
    world = next(c for c in cfg["components"] if c["model"] == "viam-labs:mujoco-sim:world")
    for item in obstacles:
        component = config(item["name"], item["attributes"])
        frame = item["frame"]
        parsed = {"parent": frame["parent"], "translation": frame["translation"]}
        if "geometry" in frame:  # mesh obstacles have none
            g = frame["geometry"]
            shapes = {
                "box": {"box": {"dims_mm": {"x": g.get("x"), "y": g.get("y"), "z": g.get("z")}}},
                "sphere": {"sphere": {"radius_mm": g.get("r")}},
                "capsule": {"capsule": {"radius_mm": g.get("r"), "length_mm": g.get("l")}},
            }
            parsed["geometry"] = dict(shapes[g["type"]])
            if "translation" in g:
                parsed["geometry"]["center"] = {**g["translation"], "o_z": 1}
        component.frame.CopyFrom(ParseDict(parsed, Frame()))
        assert MujocoObstacle.validate_config(component) == ([world["name"]], [])


def test_husky_lidar_example_validates():
    import json
    import pathlib

    cfg = json.loads((pathlib.Path(__file__).parent.parent / "examples" / "husky-lidar.json").read_text())
    models = {
        "viam-labs:mujoco-sim:base": MujocoBase,
        "viam-labs:mujoco-sim:lidar": MujocoLidar,
        "viam-labs:mujoco-sim:movement_sensor": MujocoMovementSensor,
    }
    checked = []
    for item in cfg["components"]:
        if item["model"] in models:
            component = config(item["name"], item["attributes"])
            if "frame" in item:
                component.frame.parent = item["frame"]["parent"]
            models[item["model"]].validate_config(component)
            checked.append(item["name"])
    assert checked == ["husky", "lidar", "odometry"]


def test_arm_kinematics_url_may_be_a_path(running_sim, tmp_path):
    MujocoWorld.new(config("sim-world", {"real_time": False}), {})
    source = tmp_path / "mine.json"
    shutil.copy(KINEMATICS_FILE, source)
    attrs = {"world": "sim-world", "asset": "lite6", "kinematics_url": str(source)}
    MujocoArm.validate_config(config("a", attrs))
    arm = MujocoArm.new(config("a", attrs), {})
    fmt, data = run(arm.get_kinematics())
    assert fmt == KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA
    assert data == source.read_bytes()
    with pytest.raises(ValueError, match="kinematics_url: file .*nope.json not found"):
        MujocoArm.validate_config(config("a", {**attrs, "kinematics_url": "nope.json"}))
    with pytest.raises(ValueError, match="a: kinematics_url scheme must be file, http or https"):
        MujocoArm.validate_config(config("a", {**attrs, "kinematics_url": "ftp://x/arm.json"}))


@pytest.mark.parametrize(
    "attrs, message",
    [
        ({"asset": "lite6", "mjcf": "x.xml", "kinematics_url": "k.json"}, "asset or mjcf, not both"),
        ({"mjcf": "ufactory_lite6/lite6.xml"}, "mjcf needs kinematics_url"),
        ({"mjcf": "nope.xml", "kinematics_url": "ufactory_lite6/lite6_kinematics.json"}, "mjcf: file .*nope.xml not found"),
        ({"asset": "lite6", "joints": "joint1"}, "joints must be a list"),
        ({"asset": "lite6", "flange_site": 3}, "flange_site must be a name"),
        ({"asset": "lite6", "mesh_dir": 3}, "mesh_dir must be a folder path"),
        ({"asset": "lite6", "collision": "hull"}, 'collision must be "kinematics" or "mesh"'),
        ({"asset": "lite6", "flange_offset": {"orientation": {"theta": "half"}}}, "flange_offset"),
    ],
)
def test_arm_validate_rejects_bad_robot_attributes(attrs, message):
    with pytest.raises(ValueError, match=message):
        MujocoArm.validate_config(config("a", {"world": "sim-world", **attrs}))


def test_arm_validate_accepts_mjcf_and_kinematics_url():
    attrs = {
        "world": "sim-world",
        "mjcf": "ufactory_lite6/lite6_gripper_wide.xml",
        "kinematics_url": "ufactory_lite6/lite6_kinematics.json",
        "flange_offset": {"orientation": {"o_z": 1, "theta": 180}},
    }
    assert MujocoArm.validate_config(config("a", attrs)) == (["sim-world"], [])



@pytest.mark.parametrize("example", ["lite6-pick.json", "ur5e-pick.json", "depalletize.json"])
def test_pick_example_arm_and_gripper_validate(example):
    import json
    import pathlib

    cfg = json.loads((pathlib.Path(__file__).parent.parent / "examples" / example).read_text())
    models = {"viam-labs:mujoco-sim:arm": MujocoArm, "viam-labs:mujoco-sim:gripper": MujocoGripper}
    checked = 0
    for item in cfg["components"]:
        if item["model"] in models:
            component = config(item["name"], item["attributes"])
            component.frame.parent = item["frame"]["parent"]
            models[item["model"]].validate_config(component)
            checked += 1
    assert checked == 2



def test_depalletize_cartons_are_two_spaced_layers_on_the_pallet():
    import itertools
    import json
    import pathlib

    from mujoco_module.sim import WorldConfig

    cfg = json.loads((pathlib.Path(__file__).parent.parent / "examples" / "depalletize.json").read_text())
    components = {c["name"]: c for c in cfg["components"]}
    props = WorldConfig.from_attrs(components["sim-world"]["attributes"]).props
    assert [p.name for p in props] == [f"carton_{i}" for i in range(1, 13)]
    assert all(p.geom_size == pytest.approx((0.095, 0.075, 0.075)) and p.mass == 1.0 for p in props)
    pallet = components["pallet"]["frame"]
    top = (pallet["translation"]["z"] + pallet["geometry"]["z"] / 2) / 1000
    # carton_1..6 on carton_7..12 on the pallet
    assert {round(p.position[2] - 0.075, 4) for p in props[:6]} == {round(top + 0.15, 4)}
    assert {round(p.position[2] - 0.075, 4) for p in props[6:]} == {round(top, 4)}
    for p in props:
        assert abs(p.position[0] - pallet["translation"]["x"] / 1000) + 0.095 <= pallet["geometry"]["x"] / 2000
        assert abs(p.position[1] - pallet["translation"]["y"] / 1000) + 0.075 <= pallet["geometry"]["y"] / 2000
    for a, b in itertools.combinations(props, 2):
        if a.position[2] == b.position[2]:  # planner boxes in a layer don't touch
            dx, dy = (abs(u - v) for u, v in zip(a.position[:2], b.position[:2]))
            assert max(dx - 0.19, dy - 0.15) >= 0.005, (a.name, b.name)


def test_depalletize_script_imports_and_builds_its_world_state():
    import importlib.util
    import json
    import pathlib

    examples = pathlib.Path(__file__).parent.parent / "examples"
    spec = importlib.util.spec_from_file_location("depalletize", examples / "depalletize.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    cfg = json.loads((examples / "depalletize.json").read_text())
    world = next(c for c in cfg["components"] if c["name"] == "sim-world")
    assert script.CARTONS == [p["name"] for p in world["attributes"]["props"]]
    turned = [0.7071068, 0, 0, 0.7071068]  # a quarter turn about z
    carton = script.box("carton_1", {"position": [0.401, -0.158, 0.369], "orientation_wxyz": turned})
    center = carton.center
    assert (center.x, center.y, center.z, center.o_z, center.theta) == pytest.approx((401, -158, 369, 1, 90))
    assert (carton.box.dims_mm.x, carton.box.dims_mm.y, carton.box.dims_mm.z) == (190, 150, 150)
    held = script.HELD
    assert held.pose_in_observer_frame.reference_frame == "pick-gripper"
    assert held.physical_object.HasField("center")  # RDK rejects a geometry without one
    # clear of the gripper's planner box, which ends at the tool point
    assert held.pose_in_observer_frame.pose.z - held.physical_object.box.dims_mm.z / 2 > 0

def test_husky_ur5e_example_validates_and_matches_its_script():
    import importlib.util
    import json
    import pathlib

    examples = pathlib.Path(__file__).parent.parent / "examples"
    cfg = json.loads((examples / "husky-ur5e.json").read_text())
    components = {c["name"]: c for c in cfg["components"]}
    models = {
        "viam-labs:mujoco-sim:base": MujocoBase,
        "viam-labs:mujoco-sim:arm": MujocoArm,
        "viam-labs:mujoco-sim:gripper": MujocoGripper,
        "viam-labs:mujoco-sim:movement_sensor": MujocoMovementSensor,
    }
    deps = {}
    for item in cfg["components"]:
        if item["model"] in models:
            component = config(item["name"], item["attributes"])
            if "frame" in item:
                component.frame.parent = item["frame"]["parent"]
            deps[item["name"]] = models[item["model"]].validate_config(component)[0]
    assert deps == {
        "husky": ["sim-world"],
        "arm": ["sim-world", "husky"],  # built after the base it rides
        "gripper": ["sim-world", "arm"],
        "odometry": ["sim-world", "husky"],
    }
    assert components["arm"]["frame"]["translation"] == {"x": 0, "y": 250, "z": 385}  # on the top plate

    spec = importlib.util.spec_from_file_location("mobile_pick", examples / "mobile_pick.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    for name, (center, size) in script.TABLES.items():
        frame = components[name]["frame"]
        t, g = frame["translation"], frame["geometry"]
        assert ((t["x"], t["y"], t["z"]), (g["x"], g["y"], g["z"])) == (center, size)
        assert center[2] * 2 == size[2] == script.TABLE_TOP_MM  # standing on the floor
        # more than 2 m from the Husky's start, where RDK keeps it: out of the arm's reach there
        nearest = [max(abs(c) - s / 2, 0) for c, s in zip(center[:2], size[:2])]
        assert math.hypot(*nearest) > 2000
    (block,) = components["sim-world"]["attributes"]["props"]
    assert block["name"] == script.BLOCK and block["size"] * 1000 == pytest.approx(script.BLOCK_MM)
    bx, by, bz = (v * 1000 for v in block["position"])
    (ax, ay, _), (_, a_depth, _) = script.TABLES["table-a"]
    a_edge = ay - a_depth / 2
    assert (bx, by - a_edge) == (ax, 250)  # 250 mm in from the near edge
    assert bz == pytest.approx(script.TABLE_TOP_MM + script.BLOCK_MM / 2, abs=0.5)
    # the drives stop the bumper a few cm short of each table
    assert 0 < a_edge - (script.TO_TABLE_A_MM + script.CHASSIS_FRONT_MM) < 60
    stop_b = (-script.TO_TABLE_B_MM, script.TO_TABLE_A_MM - script.BACK_UP_MM, math.pi / 2)
    (b_x, b_y, _), (b_depth, _, _) = script.TABLES["table-b"]
    b_edge = b_x + b_depth / 2
    assert 0 < (stop_b[0] - script.CHASSIS_FRONT_MM) - b_edge < 60
    assert (b_y, script.PLACE_MM) == (stop_b[1], (b_edge - 250, b_y))
    # seen from the Husky there, the place point is straight ahead, 530 mm past the arm's base
    assert script.to_husky(*script.PLACE_MM, stop_b) == pytest.approx((0, 780))
    table = script.box("table-b", *script.TABLES["table-b"], stop_b)
    assert (table.center.x, table.center.y, table.center.theta) == pytest.approx((0, 780, -90))
    assert table.label == "table-b-near"  # not the obstacle component's own geometry name
    assert script.on_table_b(*script.PLACE_MM, script.TABLE_TOP_MM + script.BLOCK_MM / 2)
    assert not script.on_table_b(bx, by, bz)


def test_meta_models_are_registered():
    import json
    import pathlib

    from viam.resource.registry import Registry
    from viam.resource.types import API, Model


    meta = json.loads((pathlib.Path(__file__).parent.parent / "meta.json").read_text())
    for entry in meta["models"]:
        Registry.lookup_resource_creator(API.from_string(entry["api"]), Model.from_string(entry["model"]))


@pytest.mark.parametrize("example,model", [
    ("husky-ur5e.json", "viam-labs:mujoco-sim:imu"),
    ("ur5e-pick.json", "viam-labs:mujoco-sim:force_torque"),
])
def test_example_sensors_validate(example, model):
    import json
    import pathlib

    from google.protobuf.json_format import ParseDict
    from viam.proto.app.robot import ComponentConfig
    from viam.utils import dict_to_struct

    from mujoco_module.models.force_torque import MujocoForceTorque
    from mujoco_module.models.imu import MujocoImu

    cfg = json.loads((pathlib.Path(__file__).parent.parent / "examples" / example).read_text())
    (item,) = [c for c in cfg["components"] if c["model"] == model]
    component = ComponentConfig(name=item["name"], attributes=dict_to_struct(item["attributes"]))
    ParseDict(item["frame"], component.frame)
    cls = MujocoImu if model.endswith(":imu") else MujocoForceTorque
    deps, _ = cls.validate_config(component)
    assert deps[0] == "sim-world"
    assert item["frame"]["parent"] in {c["name"] for c in cfg["components"]}
