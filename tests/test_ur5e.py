"""The bundled UR5e and Robotiq 2F-85, as compiled scenes."""

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_module import robots
from mujoco_module.handles import ArmHandle, GripperHandle
from mujoco_module.models.arm import _densify
from mujoco_module.sim import WorldConfig

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


def test_epick_attaches_with_its_cup_130mm_out_along_the_tool_axis(sim):
    ArmHandle(sim, "ur", {"asset": "ur5e"})
    gripper = GripperHandle(sim, "grip", "ur", {"asset": "robotiq_epick"})
    assert gripper.suction
    assert sim.model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    end = sim.data.site("ur/viam_frame")
    axes = end.xmat.reshape(3, 3)
    assert axes.T @ (sim.data.site("grip/pinch").xpos - end.xpos) == pytest.approx([0, 0, 0.130], abs=1e-4)
    # the housing mesh runs from the flange to just short of the cup
    geom = sim.model.geom("grip/housing")
    mesh = sim.model.mesh(geom.dataid[0])
    local = sim.model.mesh_vert[mesh.vertadr[0] : mesh.vertadr[0] + mesh.vertnum[0]]
    world = sim.data.geom_xpos[geom.id] + local @ sim.data.geom_xmat[geom.id].reshape(3, 3).T
    along = (world - end.xpos) @ axes[:, 2]
    assert -0.005 < along.min() < 0 and 0.099 < along.max() < 0.105
    assert np.abs((world - end.xpos) @ axes[:, :2]).max() < 0.045


# Tool straight down above (-0.133, 0.492), the UR5e's home TCP position: at
# z 0.15 (the top of a carton on the floor), 0.40 and 0.25.
PICK = [-1.5708, -1.4357, 1.9828, -2.1179, -1.5708, 0.0]
LIFT = [-1.5708, -1.5655, 1.4579, -1.4632, -1.5708, 0.0]
LOWER = [-1.5708, -1.5354, 1.8134, -1.8488, -1.5708, 0.0]
CARTON = {"name": "carton", "type": "cube", "size": 1, "scale": [0.19, 0.15, 0.15], "mass": 1.0,
          "position": [1.0, 1.0, 0.075]}


def test_epick_carries_a_carton_through_ramped_moves_and_drops_it(sim):
    """Sets the ePick's adhesion gain (100 N). Moves stream targets as
    MoveThroughJointPositions does, at 30 deg/s with abrupt starts and stops."""
    sim.configure_world(WorldConfig.from_attrs({"props": [CARTON]}))
    arm = ArmHandle(sim, "ur", {"asset": "ur5e", "max_joint_deg_s": 30})
    gripper = GripperHandle(sim, "grip", "ur", {"asset": "robotiq_epick"})
    worst = 0.0

    def gap() -> float:
        """Distance from the cup's face to the carton's top centre."""
        carton = sim.data.body("carton")
        top = carton.xpos + carton.xmat.reshape(3, 3) @ [0, 0, 0.075]
        return float(np.linalg.norm(sim.data.site("grip/pinch").xpos - top))

    def move(target):
        nonlocal worst
        stream = _densify(arm.get_joint_positions(), [target], math.radians(30) * 0.02)
        for step in stream + [target] * 50:  # then a second to settle
            arm.set_joint_targets(step)
            sim.advance(0.02)
            worst = max(worst, gap())

    move(PICK)
    pinch = sim.data.site("grip/pinch").xpos.copy()
    sim.set_prop_pose("carton", list(pinch - [0, 0, 0.075]), [1, 0, 0, 0])
    sim.advance(0.5)
    gripper.close_jaws()
    sim.advance(0.2)
    assert gripper.firm_grasp()
    worst = 0.0
    move(LIFT)
    move([0.0, *LIFT[1:]])  # a quarter turn about the base
    move([0.0, *LOWER[1:]])
    assert gripper.firm_grasp()
    assert worst < 0.005  # within the cup's 2 mm margin, plus a little
    carried = sim.get_prop_pose("carton")["position"]
    assert carried[2] > 0.16  # centre lifted 9 cm clear of where it sat
    gripper.open_jaws()
    sim.advance(1.0)
    assert not gripper.firm_grasp()
    assert sim.get_prop_pose("carton")["position"][2] == pytest.approx(0.075, abs=0.002)
