import pytest

from mujoco_module.handles import ArmHandle, GripperHandle


@pytest.fixture
def cell(sim):
    arm = ArmHandle(sim, "pick-arm", {"asset": "lite6"})
    gripper = GripperHandle(sim, "pick-gripper", "pick-arm", {"grip_force_n": 5.0})
    return arm, gripper


def test_open_and_close_move_the_jaw(sim, cell):
    _, gripper = cell
    gripper.open_jaws()
    sim.advance(0.5)
    assert gripper.jaw() == pytest.approx(gripper.open_q, abs=0.001)
    assert not gripper.is_moving()
    gripper.close_jaws()
    sim.advance(0.01)
    assert gripper.is_moving()
    sim.advance(0.5)
    assert gripper.jaw() == pytest.approx(gripper.closed_q, abs=0.001)
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
        GripperHandle(sim, "pick-gripper", "missing-arm", {})


def test_gripper_survives_arm_removal_and_releases_on_close(sim, cell):
    arm, gripper = cell
    arm.close()
    sim.advance(0.05)  # arm absent: the callback must not crash
    with pytest.raises(RuntimeError, match="not in the scene"):
        gripper.jaw()
    ArmHandle(sim, "pick-arm", {"asset": "lite6"})
    gripper.open_jaws()
    sim.advance(0.5)
    assert gripper.jaw() == pytest.approx(gripper.open_q, abs=0.001)
    gripper.close()
    actuator = sim.model.actuator("pick-arm/gripper").id
    assert sim.data.ctrl[actuator] == 0.0


def test_one_finger_touching_a_block_is_not_holding(sim, cell):
    import mujoco
    import numpy as np

    _, gripper = cell
    gripper.open_jaws()
    sim.advance(0.5)
    left = sim.data.geom("pick-arm/gripper_left_finger").xpos.copy()
    right = sim.data.geom("pick-arm/gripper_right_finger").xpos.copy()
    outward = (left - right) / np.linalg.norm(left - right)
    # the block overlaps the left fingertip from outside the jaws
    sim.set_prop_pose("pick_block", list(left + outward * 0.015), [1, 0, 0, 0])
    mujoco.mj_forward(sim.model, sim.data)
    block = sim.model.body("pick_block").id
    fingers = {"pick-arm/gripper_left_finger", "pick-arm/gripper_right_finger"}
    touching = [
        c for c in sim.data.contact[: sim.data.ncon]
        if block in (sim.model.geom_bodyid[c.geom1], sim.model.geom_bodyid[c.geom2])
        and {sim.model.body(sim.model.geom_bodyid[g]).name for g in (c.geom1, c.geom2)} & fingers
    ]
    assert touching
    assert not gripper.firm_grasp()


def test_force_stop_releases_and_grip_force_is_bounded_by_ctrlrange(sim, cell):
    _, gripper = cell
    gripper.close_jaws()
    sim.advance(0.1)
    gripper.stop()
    sim.step_once()
    assert sim.data.ctrl[sim.model.actuator("pick-arm/gripper").id] == 0.0
    with pytest.raises(ValueError, match=r"grip_force_n 20.0 is outside the actuator's ctrlrange \[-10.0, 10.0\]"):
        GripperHandle(sim, "strong", "pick-arm", {"grip_force_n": 20})


POSITION_GRIPPER_MJCF = """<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.3">
      <geom size="0.03" pos="0 0 -0.06"/>
      <body name="l1"><joint name="a" axis="0 0 1"/><geom size="0.02"/>
        <site name="attachment_site"/>
        <body name="hand" pos="0 0 0.05"><geom size="0.02" contype="0" conaffinity="0"/>
          <body name="finger"><joint name="f" type="slide" axis="0 1 0" range="0 0.04"/>
            <geom type="box" size="0.005 0.005 0.005" pos="0 0 0.03"/></body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="act_a" joint="a" kp="50" kv="5"/>
    <position name="grip" joint="f" kp="100" kv="5" ctrlrange="0 0.04"/>
  </actuator>
</mujoco>"""

ONE_JOINT_SVA = {
    "name": "one",
    "links": [
        {"id": "base", "parent": "world", "translation": {"x": 0, "y": 0, "z": 300}},
        {"id": "tool", "parent": "a", "translation": {"x": 0, "y": 0, "z": 0}},
    ],
    "joints": [{"id": "a", "type": "revolute", "parent": "base", "axis": {"x": 0, "y": 0, "z": 1}}],
}


@pytest.fixture
def position_arm(sim, tmp_path):
    import json

    def make(mjcf=POSITION_GRIPPER_MJCF):
        (tmp_path / "arm.xml").write_text(mjcf)
        (tmp_path / "arm.json").write_text(json.dumps(ONE_JOINT_SVA))
        return ArmHandle(
            sim, "arm", {"mjcf": str(tmp_path / "arm.xml"), "kinematics_url": str(tmp_path / "arm.json")}
        )

    return make


def test_position_gripper_opens_closes_and_stops_where_it_is(sim, position_arm):
    position_arm()
    # grip_force_n means nothing to a position gripper
    gripper = GripperHandle(sim, "g", "arm", {"grip_force_n": 1000})
    assert (gripper.open_q, gripper.closed_q) == (0.0, pytest.approx(0.04))
    gripper.close_jaws()
    sim.advance(0.5)
    assert gripper.jaw() == pytest.approx(0.04, abs=1e-3)
    gripper.open_jaws()
    sim.advance(0.03)
    gripper.stop()
    stopped = gripper.jaw()
    assert 0.002 < stopped < 0.038
    sim.advance(0.5)
    assert gripper.jaw() == pytest.approx(stopped, abs=2e-3)
    assert not gripper.is_moving()


def test_gripper_actuator_inference_errors(sim, position_arm):
    position_arm(POSITION_GRIPPER_MJCF.replace('<position name="grip" joint="f" kp="100" kv="5" ctrlrange="0 0.04"/>', ""))
    with pytest.raises(ValueError, match="has no gripper actuator"):
        GripperHandle(sim, "g", "arm", {})
    with pytest.raises(ValueError, match="actuator 'nope' is not in the MJCF"):
        GripperHandle(sim, "g", "arm", {"actuator": "nope"})


BARE_ARM_MJCF = POSITION_GRIPPER_MJCF.replace(
    '<position name="grip" joint="f" kp="100" kv="5" ctrlrange="0 0.04"/>', ""
)

GRIPPER_MJCF = """<mujoco>
  <option cone="elliptic" impratio="10"/>
  <worldbody>
    <body name="palm"><geom type="box" size="0.02 0.02 0.01"/>
      <body name="jaw" pos="0 0 0.03">
        <joint name="slide" type="slide" axis="0 1 0" range="0 0.03"/>
        <geom type="box" size="0.005 0.005 0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator><position name="drive" joint="slide" kp="100" kv="5" ctrlrange="0 0.03"/></actuator>
</mujoco>"""


@pytest.fixture
def gripper_file(tmp_path):
    path = tmp_path / "gripper.xml"
    path.write_text(GRIPPER_MJCF)
    return str(path)


def test_gripper_mjcf_attaches_at_the_flange_with_its_options(sim, position_arm, gripper_file):
    import mujoco

    position_arm(BARE_ARM_MJCF)
    gripper = GripperHandle(sim, "g", "arm", {"mjcf": gripper_file})
    assert sim.data.body("g/palm").xpos == pytest.approx(sim.data.site("arm/attachment_site").xpos)
    assert sim.model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert sim.model.opt.impratio == 10
    gripper.close_jaws()
    sim.advance(0.5)
    assert gripper.jaw() == pytest.approx(0.03, abs=1e-3)
    gripper.open_jaws()
    sim.advance(0.5)
    assert gripper.jaw() == pytest.approx(0.0, abs=1e-3)
    gripper.close()
    assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "g/palm") < 0
    defaults = mujoco.MjSpec().option
    assert sim.model.opt.cone == defaults.cone
    assert sim.model.opt.impratio == defaults.impratio


def test_attached_gripper_follows_its_arm_out_and_back(sim, position_arm, gripper_file):
    arm = position_arm(BARE_ARM_MJCF)
    gripper = GripperHandle(sim, "g", "arm", {"mjcf": gripper_file})
    arm.close()
    sim.advance(0.05)
    with pytest.raises(RuntimeError, match="not in the scene"):
        gripper.jaw()
    position_arm(BARE_ARM_MJCF)
    gripper.close_jaws()
    sim.advance(0.5)
    assert gripper.jaw() == pytest.approx(0.03, abs=1e-3)


def test_gripper_mjcf_on_an_arm_with_its_own_gripper_is_an_error(sim, cell, gripper_file):
    with pytest.raises(ValueError, match="arm 'pick-arm' already has a gripper in its MJCF"):
        GripperHandle(sim, "g", "pick-arm", {"mjcf": gripper_file})


def test_gripper_mjcf_with_two_actuators_is_an_error(sim, position_arm, tmp_path):
    position_arm(BARE_ARM_MJCF)
    path = tmp_path / "two.xml"
    path.write_text(GRIPPER_MJCF.replace("</actuator>", '<motor name="extra" joint="slide"/></actuator>'))
    with pytest.raises(ValueError, match=r"more than one gripper actuator \['drive', 'extra'\]; set actuator"):
        GripperHandle(sim, "g", "arm", {"mjcf": str(path)})
    # naming one picks it
    GripperHandle(sim, "g", "arm", {"mjcf": str(path), "actuator": "drive"})


def test_unnamed_finger_bodies_under_a_named_root_still_grip(sim, position_arm):
    position_arm(POSITION_GRIPPER_MJCF.replace('<body name="finger">', "<body>"))
    gripper = GripperHandle(sim, "g", "arm", {})
    gripper.close_jaws()
    sim.advance(0.5)
    assert gripper.jaw() == pytest.approx(0.04, abs=1e-3)
    assert not gripper.firm_grasp()  # reads without a KeyError


def test_unnamed_root_body_is_an_error(sim, position_arm, tmp_path):
    position_arm(BARE_ARM_MJCF)
    unnamed_root = tmp_path / "root.xml"
    unnamed_root.write_text(GRIPPER_MJCF.replace('<body name="palm">', "<body>"))
    with pytest.raises(ValueError, match="root body has no name"):
        GripperHandle(sim, "g", "arm", {"mjcf": str(unnamed_root)})
