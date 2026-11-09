"""An arm riding a base: the bundled UR5e and 2F-85 on the Husky."""

import json
import math

import mujoco
import numpy as np
import pytest

from mujoco_module import robots
from mujoco_module.handles import ArmHandle, BaseHandle, CameraHandle, GripperHandle, JointTrajectoryPoint
from mujoco_module.sim import WorldConfig
from test_arm import THREE_JOINT_SVA, three_joint_files  # noqa: F401
from test_base import finish, frame_pose, settle

HOME = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
ON_TOP = [0.0, 0.25, 0.385]  # the example's mount: 250 mm forward, on the top plate


def husky(sim):
    return BaseHandle(sim, "husky", {"asset": "clearpath_husky", "drive": "ideal"})


def ur5e(sim, name="arm", **attrs):
    return ArmHandle(sim, name, {"asset": "ur5e", "parent": "husky", "position": ON_TOP, **attrs})


@pytest.fixture
def rig(sim):
    sim.configure_world(WorldConfig.from_attrs({"props": []}))
    base = husky(sim)
    arm = ur5e(sim)
    gripper = GripperHandle(sim, "gripper", "arm", {"asset": "robotiq_2f85"})
    return base, arm, gripper


def exists(sim, kind, name):
    return mujoco.mj_name2id(sim.model, kind, name) >= 0


def test_the_arm_is_attached_under_the_chassis_at_its_offset(sim, rig):
    _, arm, _ = rig
    root = sim.model.body(f"arm/{arm.root_body}")
    assert root.parentid[0] == sim.model.body("husky/base_link").id
    assert root.weldid[0] == sim.model.body("husky/base_link").weldid[0]  # rides the chassis
    site = sim.model.site("husky/viam_frame").id
    frame = sim.model.site("arm/arm_frame").id
    expected = sim.data.site_xpos[site] + sim.data.site_xmat[site].reshape(3, 3) @ ON_TOP
    assert sim.data.site_xpos[frame] == pytest.approx(expected, abs=1e-9)
    assert sim.data.xpos[root.id] == pytest.approx(expected, abs=1e-9)
    assert arm.get_joint_positions() == pytest.approx(HOME)


def test_end_position_ignores_where_the_base_drives(sim, rig):
    base, arm, _ = rig
    world = ArmHandle(sim, "world-arm", {"asset": "ur5e", "position": [3.0, 0.0, 0.0]})
    sim.advance(0.5)
    start, start_quat = arm.get_end_pose()
    assert start == pytest.approx(world.get_end_pose()[0], abs=1e-3)
    mount = sim.data.site("arm/arm_frame").xpos.copy()
    finish(sim, base.move_straight(1.0, 0.5)).result()
    finish(sim, base.spin(math.radians(90), math.radians(60))).result()
    settle(sim)
    assert np.linalg.norm(sim.data.site("arm/arm_frame").xpos - mount) > 0.7  # went with the base (0.79 m)
    position, quat = arm.get_end_pose()
    assert position == pytest.approx(start, abs=1e-3)
    assert abs(np.dot(quat, start_quat)) == pytest.approx(1.0, abs=1e-5)


def test_no_arm_chassis_contacts_at_rest(sim, rig):
    sim.advance(1.0)
    names = [
        {sim.model.body(sim.model.geom_bodyid[g]).name.split("/")[0] for g in (c.geom1, c.geom2)}
        for c in sim.data.contact[: sim.data.ncon]
    ]
    assert not [pair for pair in names if "husky" in pair and pair & {"arm", "gripper"}]


def test_an_arm_waits_for_its_missing_base(sim):
    sim.configure_world(WorldConfig.from_attrs({"props": []}))
    arm = ur5e(sim)  # constructing it succeeds
    assert not exists(sim, mujoco.mjtObj.mjOBJ_BODY, f"arm/{arm.root_body}")
    sim.advance(0.05)
    assert "arm:arm" in sim.step_callbacks  # the step skipped instead of failing
    with pytest.raises(RuntimeError, match="arm 'arm': parent 'husky' has no viam_frame in the scene"):
        arm.get_joint_positions()
    with pytest.raises(RuntimeError, match="no viam_frame in the scene"):
        arm.set_joint_targets(HOME)
    with pytest.raises(RuntimeError, match="no viam_frame in the scene"):
        arm.get_end_pose()
    arm.stop()  # viam-server's Stop (session end, e-stop) still succeeds
    base = husky(sim)
    assert arm.get_joint_positions() == pytest.approx(HOME)  # init ran when it appeared
    base.close()
    sim.advance(0.05)
    assert "arm:arm" in sim.step_callbacks
    husky(sim)
    assert arm.get_joint_positions() == pytest.approx(HOME)
    sim.advance(0.5)
    assert arm.get_joint_positions() == pytest.approx(HOME, abs=1e-3)


def test_a_chain_on_the_base_rebuilds_in_any_order(sim, rig):
    base, arm, _ = rig
    CameraHandle(sim, "wrist", {"parent": "arm", "position": [0, 0, 0.05]})
    present = (
        (mujoco.mjtObj.mjOBJ_BODY, f"arm/{arm.root_body}"),
        (mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper/fingers_actuator"),
        (mujoco.mjtObj.mjOBJ_CAMERA, "wrist"),
    )
    base.close()  # the arm, gripper and camera wait
    assert not any(exists(sim, *item) for item in present)
    husky(sim)  # now inserted after all of them
    assert all(exists(sim, *item) for item in present)
    arm.close()
    ur5e(sim)  # the arm last: the gripper and camera still come back
    assert all(exists(sim, *item) for item in present)


def test_a_running_move_fails_promptly_when_the_base_goes(sim, rig):
    base, arm, _ = rig
    mid = [value + 0.3 for value in HOME]
    future = arm.start_trajectory([JointTrajectoryPoint(0.0, tuple(HOME)), JointTrajectoryPoint(5.0, tuple(mid))])
    sim.advance(0.1)
    assert not future.done()
    base.close()
    sim.advance(0.05)
    assert future.done()
    with pytest.raises(RuntimeError, match="arm 'arm': parent 'husky' has no viam_frame in the scene"):
        future.result()


def test_a_rotated_frame_and_base_offset_compose_on_a_base(sim, three_joint_files, tmp_path):  # noqa: F811
    # test_arm's arm: kinematics base 400 mm up and turned 90 deg about z, so base_offset is non-identity
    sva = json.loads(json.dumps(THREE_JOINT_SVA))
    sva["links"][0]["translation"]["z"] = 400
    sva["links"][0]["orientation"] = {"type": "ov_degrees", "value": {"x": 0, "y": 0, "z": 1, "th": 90}}
    kinematics_file = tmp_path / "turned.json"
    kinematics_file.write_text(json.dumps(sva))
    sim.configure_world(WorldConfig.from_attrs({"props": []}))
    husky(sim)
    turn = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]  # the frame is turned 90 deg about x
    arm = ArmHandle(sim, "mine", {
        **three_joint_files, "kinematics_url": str(kinematics_file), "parent": "husky",
        "base_offset": {"translation": {"x": 0, "y": 0, "z": 100}, "orientation": {"o_z": 1, "theta": 90}},
        "position": ON_TOP, "orientation_wxyz": turn,
    })
    kinematics = robots.parse_kinematics(json.dumps(sva).encode(), urdf=False, label="turned")
    expected = kinematics.forward([0.1, 0.2, 0.3])["tool"]
    position, quat = arm.get_end_pose()
    assert position == pytest.approx(expected[:3, 3], abs=1e-6)
    assert robots.transform(quat=quat)[:3, :3] == pytest.approx(expected[:3, :3], abs=1e-6)
    site = sim.model.site("husky/viam_frame").id
    rotation = sim.data.site_xmat[site].reshape(3, 3)
    spot = sim.data.site_xpos[site] + rotation @ ON_TOP
    assert sim.data.site("mine/arm_frame").xpos == pytest.approx(spot, abs=1e-9)  # at the frame
    frame_rotation = robots.transform(quat=turn)[:3, :3]
    mount = spot + rotation @ frame_rotation @ [0, 0, 0.4]  # the MJCF world (base_offset 0.1 up, and the base body 0.3 up in it)
    assert sim.data.body("mine/base").xpos == pytest.approx(mount, abs=1e-9)


def test_a_fast_arm_move_barely_moves_an_idle_base(sim, rig):
    """The arm's reaction swings the idle base (about 0.6 deg of yaw mid-move);
    what a grasp planned from an earlier odometry read needs is that it ends
    up where it was."""
    _, arm, _ = rig
    sim.advance(0.5)
    before = frame_pose(sim, "husky")
    swung = [HOME[0] + math.pi / 2, -1.0, 1.0, *HOME[3:]]  # pan a quarter turn, reach out
    for target in (swung, HOME):
        arm.set_joint_targets(target)  # at the default 180 deg/s
        finish(sim, arm.wait_for_joint_targets(target, math.radians(0.5)))
    sim.advance(0.5)
    x, y, yaw = frame_pose(sim, "husky")
    assert math.hypot(x - before[0], y - before[1]) < 0.001
    assert abs(math.degrees(yaw - before[2])) < 0.1
