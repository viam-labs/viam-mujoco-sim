import math
from pathlib import Path

import numpy as np
import pytest

from mujoco_module.handles import ArmHandle, JointTrajectoryPoint
from mujoco_module.sim import WorldConfig
from viam.spatialmath import AxisAngle
from sva import _quat_matrix, forward_kinematics
from conftest import BLOCK

KINEMATICS = Path(__file__).parents[1] / "assets" / "ufactory_lite6" / "lite6_kinematics.json"
HOME = [0.0, 0.0, 1.57, 0.0, 1.57, 0.0]


def make_arm(sim, **attrs):
    return ArmHandle(sim, "pick-arm", {"asset": "lite6", **attrs})


def test_arm_starts_at_home(sim):
    arm = make_arm(sim)
    assert arm.get_joint_positions() == pytest.approx(HOME)


@pytest.mark.parametrize("seed", range(4))
def test_flange_matches_viam_kinematics(sim, seed):
    """If this fails, the motion service plans against the wrong arm."""
    q = AxisAngle(0, 0, 1, math.radians(30)).to_quaternion()
    yaw = (q.w, q.i, q.j, q.k)
    arm = make_arm(sim, position=[0.1, 0.2, 0.05], orientation_wxyz=list(yaw))
    rng = np.random.default_rng(seed)
    joints = list(np.asarray(HOME) + rng.uniform(-0.8, 0.8, 6))
    ids = arm._now()[1]
    sim.data.qpos[ids.qpos] = joints
    import mujoco

    mujoco.mj_kinematics(sim.model, sim.data)
    (x, y, z), quat = arm.get_end_pose()
    expected = forward_kinematics(KINEMATICS, joints, "gripper_mount")
    assert np.allclose([x, y, z], expected[:3, 3], atol=1.5e-3)
    assert np.allclose(_quat_matrix(quat), expected[:3, :3], atol=1e-2)
    # and the base really sits at the configured frame in the world
    base = sim.data.body("pick-arm/link_base")
    assert np.allclose(base.xpos, [0.1, 0.2, 0.05])
    assert np.allclose(base.xquat, yaw)


def test_move_settles_within_velocity_limit(sim):
    arm = make_arm(sim, max_joint_deg_s=60)
    target = [0.5, -0.3, 1.2, 0.2, 1.0, -0.4]
    arm.set_joint_targets(target)
    done = arm.wait_for_joint_targets(target, math.radians(0.5))
    previous = np.asarray(arm.get_joint_positions())
    peak = 0.0
    while not done.done() and sim.data.time < 10.0:
        sim.advance(0.01)
        now = np.asarray(arm.get_joint_positions())
        peak = max(peak, float(np.max(np.abs(now - previous)) / 0.01))
        previous = now
    assert done.done()
    assert arm.get_joint_positions() == pytest.approx(target, abs=math.radians(0.5))
    assert peak <= math.radians(60) * 1.1


def test_trajectory_hits_waypoints_on_time(sim):
    arm = make_arm(sim)
    mid = [0.3, 0.0, 1.57, 0.0, 1.57, 0.0]
    end = [0.6, 0.0, 1.57, 0.0, 1.57, 0.0]
    points = [
        JointTrajectoryPoint(0.0, tuple(HOME)),
        JointTrajectoryPoint(1.0, tuple(mid)),
        JointTrajectoryPoint(2.0, tuple(end)),
    ]
    done = arm.start_trajectory(points)
    sim.advance(1.0)
    # the kp=2000/kv=200 servo trails a moving target by about 0.1 s
    assert arm.get_joint_positions()[0] == pytest.approx(0.3, abs=0.04)
    sim.advance(1.05)
    assert done.done()
    sim.advance(0.5)
    assert arm.get_joint_positions() == pytest.approx(end, abs=0.01)


def test_stop_trajectory_holds_measured(sim):
    arm = make_arm(sim)
    done = arm.start_trajectory(
        [JointTrajectoryPoint(0.0, tuple(HOME)), JointTrajectoryPoint(2.0, (1.0, 0, 1.57, 0, 1.57, 0))]
    )
    sim.advance(0.5)
    arm.stop_trajectory()
    assert done.cancelled()
    held = arm.get_joint_positions()
    sim.advance(0.5)
    assert arm.get_joint_positions() == pytest.approx(held, abs=0.01)


def test_dynamics_metadata(sim):
    meta = make_arm(sim, max_joint_deg_s=30, max_joint_deg_s2=60).dynamics_metadata()
    assert meta["max_velocity_rad_s"] == pytest.approx([math.radians(30)] * 6)
    assert meta["max_acceleration_rad_s2"] == pytest.approx([math.radians(60)] * 6)


def test_rebuild_keeps_arm_state_and_time(sim):
    arm = make_arm(sim)
    target = [0.4, -0.2, 1.3, 0.1, 1.2, 0.2]
    arm.set_joint_targets(target)
    sim.advance(2.0)
    before_q = arm.get_joint_positions()
    before_t = sim.data.time
    distractor = {"name": "distractor", "size": 0.03, "position": [0.2, 0.2, 0.015]}
    sim.configure_world(WorldConfig.from_attrs({"props": [BLOCK, distractor]}))
    assert sim.data.time == before_t
    assert arm.get_joint_positions() == pytest.approx(before_q)
    sim.advance(0.5)
    assert arm.get_joint_positions() == pytest.approx(target, abs=0.01)


def test_bad_trajectory_is_rejected_up_front(sim):
    arm = make_arm(sim)
    with pytest.raises(ValueError):
        arm.start_trajectory([JointTrajectoryPoint(0.0, (0.0,) * 5)])
    with pytest.raises(ValueError):
        arm.start_trajectory([JointTrajectoryPoint(1.0, tuple(HOME)), JointTrajectoryPoint(0.5, tuple(HOME))])
    assert "arm:pick-arm" in sim.step_callbacks


def test_batch_starting_after_zero_interpolates_from_command(sim):
    arm = make_arm(sim)
    sim.advance(0.5)
    done = arm.start_trajectory([JointTrajectoryPoint(0.5, (0.5, 0, 1.57, 0, 1.57, 0))])
    sim.step_once()
    ids = arm._now()[1]
    assert sim.data.ctrl[ids.ctrl][0] == pytest.approx(0.0, abs=0.01)  # no backwards jump
    sim.advance(0.6)
    assert done.done()


def test_close_fails_inflight_moves(sim):
    arm = make_arm(sim)
    done = arm.start_trajectory([JointTrajectoryPoint(0.0, tuple(HOME)), JointTrajectoryPoint(5.0, tuple(HOME))])
    wait = arm.wait_for_joint_targets([1.0] * 6, 1e-6)
    arm.close()
    for future in (done, wait):
        assert future.done() and not future.cancelled()
        assert isinstance(future.exception(), RuntimeError)
        assert "closed" in str(future.exception())
    assert "arm:pick-arm" not in sim.step_callbacks
