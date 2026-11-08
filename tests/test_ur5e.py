"""The bundled UR5e and Robotiq 2F-85, as compiled scenes."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_module import robots
from mujoco_module.handles import ArmHandle, GripperHandle

KINEMATICS = Path(__file__).parents[1] / "assets" / "universal_robots_ur5e" / "ur5e_kinematics.json"
HOME = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]


@pytest.fixture
def ur5e(sim):
    arm = ArmHandle(sim, "ur", {"asset": "ur5e"})
    gripper = GripperHandle(sim, "grip", "ur", {"asset": "robotiq_2f85"})
    return arm, gripper


def kinematics():
    return robots.parse_kinematics(KINEMATICS.read_bytes(), urdf=False, label="ur5e")


def test_ur5e_starts_at_home_and_its_end_pose_matches_viam(sim):
    arm = ArmHandle(sim, "ur", {"asset": "ur5e"})
    assert arm.joint_names() == (
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    )
    assert arm.get_joint_positions() == pytest.approx(HOME)
    expected = kinematics().forward(HOME)["ee_link"]
    position, quat = arm.get_end_pose()
    assert np.linalg.norm(np.asarray(position) - expected[:3, 3]) < 0.005
    assert robots.transform(quat=quat)[:3, :3] == pytest.approx(expected[:3, :3], abs=0.02)


def test_ur5e_holds_a_loaded_pose_without_sagging(sim, ur5e):
    arm, _ = ur5e
    # shoulder and elbow loaded by the arm's and gripper's weight
    target = [-3.108, -1.374, 1.982, -2.178, -1.571, 1.604]
    arm.set_joint_targets(target)
    sim.advance(5.0)
    error = np.degrees(np.abs(np.asarray(arm.get_joint_positions()) - target))
    assert error.max() < 0.05


def test_ur5e_collides_with_the_planner_shapes(sim):
    ArmHandle(sim, "ur", {"asset": "ur5e"})
    kin = kinematics()
    viam = kin.forward(HOME)
    for frame in kin.frames.values():
        if frame.geometry:
            expected = viam[frame.parent] @ robots.sva_pose(frame.geometry, frame.name)
            geom = sim.data.geom(f"ur/viam_{frame.name}")
            assert np.linalg.norm(geom.xpos - expected[:3, 3]) < 0.005, frame.name
    # every link got a real shape, so Menagerie's own capsules are off
    model = sim.model
    colliding = [
        model.geom(i).name for i in range(model.ngeom)
        if model.body(model.geom_bodyid[i]).name.startswith("ur/")
        and (model.geom_contype[i] or model.geom_conaffinity[i])
    ]
    assert colliding and all(name.startswith("ur/viam_") for name in colliding)


def test_2f85_attaches_with_elliptic_cones_and_moves(sim, ur5e):
    _, gripper = ur5e
    assert sim.model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert sim.model.opt.impratio == 10
    assert (gripper.open_q, gripper.closed_q) == (0.0, pytest.approx(0.8))
    assert not gripper.slide  # a hinge: jaw is in radians
    gripper.close_jaws()
    sim.advance(1.0)
    assert gripper.jaw() > 0.7
    gripper.open_jaws()
    sim.advance(1.0)
    assert gripper.jaw() < 0.05


def test_2f85_grasps_and_carries_a_block(sim, ur5e):
    arm, gripper = ur5e
    gripper.open_jaws()
    sim.advance(1.0)
    pinch = sim.data.site("grip/pinch").xpos.copy()
    sim.set_prop_pose("pick_block", list(pinch), [1, 0, 0, 0])
    gravity = sim.model.opt.gravity.copy()
    sim.model.opt.gravity[:] = 0  # weightless while the jaws close
    gripper.close_jaws()
    sim.advance(1.0)
    sim.model.opt.gravity[:] = gravity
    sim.advance(0.5)
    assert gripper.firm_grasp()
    target = list(HOME)
    target[0] += 0.5  # swing the arm about its base
    arm.set_joint_targets(target)
    sim.advance(3.0)
    assert gripper.firm_grasp()
    moved = sim.data.site("grip/pinch").xpos
    assert np.linalg.norm(moved - pinch) > 0.1
    block = np.asarray(sim.get_prop_pose("pick_block")["position"])
    assert np.linalg.norm(block - moved) < 0.02


def test_one_2f85_pad_touching_a_block_is_not_holding(sim, ur5e):
    _, gripper = ur5e
    gripper.open_jaws()
    sim.advance(1.0)
    right = sim.data.geom("grip/right_pad1").xpos.copy()
    left = sim.data.geom("grip/left_pad1").xpos.copy()
    outward = (right - left) / np.linalg.norm(right - left)
    sim.set_prop_pose("pick_block", list(right + outward * 0.015), [1, 0, 0, 0])
    mujoco.mj_forward(sim.model, sim.data)
    block = sim.model.body("pick_block").id
    assert any(
        block in (sim.model.geom_bodyid[c.geom1], sim.model.geom_bodyid[c.geom2])
        for c in sim.data.contact[: sim.data.ncon]
    )
    assert not gripper.firm_grasp()


def test_2f85_on_the_lite6_is_an_error(sim):
    ArmHandle(sim, "pick-arm", {"asset": "lite6"})
    with pytest.raises(ValueError, match="already has a gripper in its MJCF"):
        GripperHandle(sim, "grip", "pick-arm", {"asset": "robotiq_2f85"})
