import pytest

from mujoco_module.handles import ArmHandle, GripperHandle

HOME = [0.0, 0.0, 1.57, 0.0, 1.57, 0.0]


@pytest.fixture
def cell(sim):
    arm = ArmHandle(sim, "pick-arm", {"asset": "lite6"})
    gripper = GripperHandle(sim, "pick-gripper", "pick-arm", grip_force_n=5.0)
    return arm, gripper


def test_open_and_close_move_the_jaw(sim, cell):
    _, gripper = cell
    gripper.open_jaws()
    sim.advance(0.5)
    assert gripper.jaw_m() == pytest.approx(GripperHandle.OPEN_M, abs=0.001)
    assert not gripper.is_moving()
    gripper.close_jaws()
    sim.advance(0.5)
    assert gripper.jaw_m() == pytest.approx(GripperHandle.CLOSED_M, abs=0.001)
    assert not gripper.firm_grasp()  # closed on nothing


def test_grasp_holds_and_lifts_block(sim, cell):
    arm, gripper = cell
    gripper.open_jaws()
    sim.advance(0.5)
    tip = sim.data.site("pick-arm/end_effector").xpos.copy()
    # Place the block between the open jaws, weightless while they close.
    sim.set_prop_pose("pick_block", list(tip + [0, 0, 0.006]), [1, 0, 0, 0])
    gravity = sim.model.opt.gravity.copy()
    sim.model.opt.gravity[:] = 0
    gripper.close_jaws()
    sim.advance(0.5)
    sim.model.opt.gravity[:] = gravity
    sim.advance(0.5)
    assert gripper.firm_grasp()
    start_z = sim.get_prop_pose("pick_block")["position"][2]
    arm.set_joint_targets([0.0, -0.4, 1.57, 0.0, 1.57, 0.0])
    sim.advance(3.0)
    assert gripper.firm_grasp()
    assert sim.get_prop_pose("pick_block")["position"][2] - start_z > 0.05


def test_gripper_requires_attached_arm(sim):
    with pytest.raises(RuntimeError, match="configure the arm first"):
        GripperHandle(sim, "pick-gripper", "missing-arm", grip_force_n=5.0)
