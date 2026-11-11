import json
import math

import mujoco
import numpy as np
import pytest

from mujoco_module import robots
from mujoco_module.handles import ArmHandle, BaseHandle, CameraHandle
from mujoco_module.robot_file import RobotHandle
from mujoco_module.sim import WorldConfig
from test_base import finish, settle

# A fixed torso with two 2-joint chains (shoulder about z, elbow about y), and
# a slide gripper on the left one. Both chains hang from the torso, their
# chain base. Keyframe "spread" differs from the arms' "home".
TORSO = """
<mujoco>
  <compiler angle="radian"/>
  <default><joint damping="5" armature="0.01"/><position kp="500"/></default>
  <worldbody>
    <body name="torso" pos="0 0 0.5">
      <geom type="box" size="0.1 0.1 0.1" mass="5"/>
      <body name="left/upper" pos="-0.2 0 0.2">
        <joint name="left/shoulder" axis="0 0 1" range="-3 3"/>
        <geom type="capsule" fromto="0 0 0 -0.2 0 0" size="0.02" mass="0.5"/>
        <body name="left/fore" pos="-0.2 0 0">
          <joint name="left/elbow" axis="0 1 0" range="-2 2"/>
          <geom type="capsule" fromto="0 0 0 -0.2 0 0" size="0.02" mass="0.5"/>
          <site name="left/tip" pos="-0.2 0 0"/>
          <body name="left/finger" pos="-0.2 0 0">
            <joint name="left/finger_slide" type="slide" axis="1 0 0" range="0 0.04"/>
            <geom type="box" size="0.01 0.01 0.01" mass="0.1"/>
          </body>
        </body>
      </body>
      <body name="right/upper" pos="0.2 0 0.2">
        <joint name="right/shoulder" axis="0 0 1" range="-3 3"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.5"/>
        <body name="right/fore" pos="0.2 0 0">
          <joint name="right/elbow" axis="0 1 0" range="-2 2"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.5"/>
          <site name="right/tip" pos="0.2 0 0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="left/shoulder" joint="left/shoulder" ctrlrange="-3 3"/>
    <position name="left/elbow" joint="left/elbow" ctrlrange="-2 2"/>
    <position name="left/grip" joint="left/finger_slide" ctrlrange="0 0.04"/>
    <position name="right/shoulder" joint="right/shoulder" ctrlrange="-3 3"/>
    <position name="right/elbow" joint="right/elbow" ctrlrange="-2 2"/>
  </actuator>
  <keyframe>
    <key name="spread" qpos="0.5 0.5 0.02 0.5 0.5" ctrl="0.5 0.5 0.02 0.5 0.5"/>
  </keyframe>
</mujoco>
"""
TORSO_POSE = [0, 0, 0.5]  # each arm's frame: the torso, relative to the robot


def kinematics(sign):
    """SVA for one chain: shoulder 200 mm out along `sign` x and 200 mm up
    from the torso, elbow 200 mm further, tip 200 mm beyond."""
    links = [
        {"id": "base", "parent": "world", "translation": {"x": sign * 200, "y": 0, "z": 200}},
        {"id": "upper", "parent": "shoulder", "translation": {"x": sign * 200, "y": 0, "z": 0}},
        {"id": "tip", "parent": "elbow", "translation": {"x": sign * 200, "y": 0, "z": 0}},
    ]
    joints = [
        {"id": "shoulder", "type": "revolute", "parent": "base", "axis": {"x": 0, "y": 0, "z": 1}, "min": -171, "max": 171},
        {"id": "elbow", "type": "revolute", "parent": "upper", "axis": {"x": 0, "y": 1, "z": 0}, "min": -114, "max": 114},
    ]
    return {"name": "chain", "kinematic_param_type": "SVA", "links": links, "joints": joints}


@pytest.fixture
def torso(tmp_path):
    (tmp_path / "torso.xml").write_text(TORSO)
    (tmp_path / "left.json").write_text(json.dumps(kinematics(-1)))
    (tmp_path / "right.json").write_text(json.dumps(kinematics(1)))
    return tmp_path


def limb(path, side, **extra):
    return {
        "robot": "bot", "parent": "bot", "position": TORSO_POSE,
        "joints": [f"{side}/shoulder", f"{side}/elbow"], "flange_site": f"{side}/tip",
        "kinematics_url": str(path / f"{side}.json"), **extra,
    }


def robot(sim, path, **extra):
    return RobotHandle(sim, "bot", {"mjcf": str(path / "torso.xml"), **extra})


# ---- binding ----------------------------------------------------------------


def test_two_arms_bind_to_one_copy_and_move_independently(sim, torso):
    robot(sim, torso)
    left = ArmHandle(sim, "left-arm", limb(torso, "left"))
    right = ArmHandle(sim, "right-arm", limb(torso, "right"))
    names = [sim.model.body(b).name for b in range(sim.model.nbody)]
    assert names.count("bot/torso") == 1 and names.count("bot/left/upper") == 1
    assert left.joint_names() == ("left/shoulder", "left/elbow")
    left.set_joint_targets([0.4, -0.3])
    sim.advance(2.0)
    assert left.get_joint_positions() == pytest.approx([0.4, -0.3], abs=1e-3)
    assert right.get_joint_positions() == pytest.approx([0.0, 0.0], abs=1e-3)


def test_viam_sites_carry_the_arm_name(sim, torso):
    robot(sim, torso)
    left = ArmHandle(sim, "left-arm", limb(torso, "left"))
    tip = sim.data.site("bot/left/tip")
    end = sim.data.site("left-arm/viam_frame")
    assert end.xpos == pytest.approx(tip.xpos)
    assert sim.data.site("left-arm/arm_frame").xpos == pytest.approx(sim.data.body("bot/torso").xpos)
    assert left.flange_site_name == "bot/left/tip"
    CameraHandle(sim, "wrist-cam", {"parent": "left-arm"})  # mounts on the arm's viam_frame
    assert sim.data.cam("wrist-cam").xpos == pytest.approx(tip.xpos, abs=1e-6)


def test_the_frame_must_be_the_chain_base(sim, torso):
    robot(sim, torso)
    with pytest.raises(ValueError, match=r'set frame.translation to \{"x": 0.0, "y": 0.0, "z": 500.0\}'):
        ArmHandle(sim, "left-arm", limb(torso, "left", position=[0, 0, 0.4]))
    with pytest.raises(ValueError, match='needs frame.parent "bot"'):
        ArmHandle(sim, "left-arm", limb(torso, "left", parent=None))


def test_a_jointed_chain_base_is_an_error(sim, torso):
    (torso / "torso.xml").write_text(TORSO.replace('<body name="torso" pos="0 0 0.5">', '<body name="torso" pos="0 0 0.5"><joint name="spin" axis="0 0 1"/>', 1).replace('qpos="0.5', 'qpos="0 0.5'))
    robot(sim, torso)
    with pytest.raises(ValueError, match="torso moves relative to robot bot"):
        ArmHandle(sim, "left-arm", limb(torso, "left"))


# ---- claims, homes and lifecycle --------------------------------------------


def test_a_joint_can_be_claimed_once(sim, torso):
    robot(sim, torso)
    left = ArmHandle(sim, "left-arm", limb(torso, "left"))
    with pytest.raises(ValueError, match="joint left/shoulder is already left-arm's"):
        ArmHandle(sim, "other", limb(torso, "left"))
    left.close()
    ArmHandle(sim, "other", limb(torso, "left"))  # released on close


def test_a_failed_arm_holds_no_claims(sim, torso):
    robot(sim, torso)
    with pytest.raises(ValueError, match="set frame.translation"):
        ArmHandle(sim, "left-arm", limb(torso, "left", position=[0, 0, 0]))
    assert "joint:left/shoulder" not in sim.robot_claims.get("bot", {})
    ArmHandle(sim, "left-arm", limb(torso, "left"))


def test_the_robots_home_leaves_claimed_joints_alone_in_any_order(sim, torso):
    robot(sim, torso)  # home_keyframe: the file's first, "spread"
    left = ArmHandle(sim, "left-arm", limb(torso, "left"))  # home: zeros (no "home" keyframe)
    sim.reset()
    sim.run(lambda: sim._contributions["bot"].init(sim.model, sim.data))  # the robot's init last
    assert left.get_joint_positions() == pytest.approx([0.0, 0.0])
    right = sim.data.joint("bot/right/shoulder").qpos[0]
    assert right == pytest.approx(0.5)  # unclaimed: the robot's keyframe


def test_the_robot_goes_and_comes_back(sim, torso):
    bot = robot(sim, torso)
    left = ArmHandle(sim, "left-arm", limb(torso, "left"))
    left.set_joint_targets([0.4, 0.0])
    sim.advance(1.0)
    bot.close()
    with pytest.raises(RuntimeError, match="robot 'bot' is not in the scene"):
        left.get_joint_positions()
    robot(sim, torso)  # a new robot handle; the arm and its claims stay
    assert left.get_joint_positions() == pytest.approx([0.0, 0.0])  # back at home
    assert sim.robot_claims["bot"]["joint:left/shoulder"] == "left-arm"


def test_one_arm_closing_keeps_the_others_state(sim, torso):
    robot(sim, torso)
    left = ArmHandle(sim, "left-arm", limb(torso, "left"))
    right = ArmHandle(sim, "right-arm", limb(torso, "right"))
    right.set_joint_targets([0.3, 0.2])
    sim.advance(2.0)
    left.close()
    assert right.get_joint_positions() == pytest.approx([0.3, 0.2], abs=1e-3)


def test_a_failed_contribution_leaves_nothing_behind(sim, torso, monkeypatch):
    robot(sim, torso)

    def fail(name, contribution):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(sim, "set_contribution", fail)
    with pytest.raises(RuntimeError, match="compile failed"):
        ArmHandle(sim, "left-arm", limb(torso, "left"))
    assert "left-arm" not in sim.arms
    assert "arm:left-arm" not in sim.step_callbacks
    assert "joint:left/shoulder" not in sim.robot_claims.get("bot", {})


@pytest.mark.parametrize("insert,message", [
    ('<body name="loose"><freejoint/><geom size="0.01"/></body>', "can't have a free joint yet"),
    ('<body name="ghost" mocap="true"><geom size="0.01" contype="0" conaffinity="0"/></body>', "can't have mocap bodies"),
])
def test_robot_file_rejects(sim, torso, insert, message):
    (torso / "torso.xml").write_text(TORSO.replace("  <worldbody>\n", f"  <worldbody>\n    {insert}\n", 1))
    with pytest.raises(ValueError, match=message):
        robot(sim, torso)


def test_two_arms_racing_for_one_chain_leave_one(running_sim, torso):
    import threading

    sim = running_sim
    sim.configure_world(WorldConfig.from_attrs({}))
    robot(sim, torso)
    results: dict[str, object] = {}

    def build(name):
        try:
            results[name] = ArmHandle(sim, name, limb(torso, "left"))
        except ValueError as error:
            results[name] = error

    threads = [threading.Thread(target=build, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    winners = [name for name, result in results.items() if isinstance(result, ArmHandle)]
    assert len(winners) == 1
    loser = ({"a", "b"} - set(winners)).pop()
    assert "is already" in str(results[loser])
    assert sim.run(lambda: (sorted(sim.arms), sim.contributes(loser))) == ([winners[0]], False)



# ---- a robot on a base, an arm on an arm --------------------------------------

# The fixture with a slide "torso" between the root and the limbs: the lift
# carries both chains, so their chain base is the lift; its flange site is
# 100 mm above it. Its box sits in the torso's and collides with nothing.
LIFTED = (
    TORSO.replace(
        '<geom type="box" size="0.1 0.1 0.1" mass="5"/>',
        '<geom type="box" size="0.1 0.1 0.1" mass="5"/>\n      <body name="lift">'
        '<joint name="lift" type="slide" axis="0 0 1" range="0 0.3" damping="200"/>'
        '<geom type="box" size="0.05 0.05 0.05" mass="1" contype="0" conaffinity="0"/><site name="lift/top" pos="0 0 0.1"/>',
        1,
    )
    .replace("    </body>\n  </worldbody>", "    </body></body>\n  </worldbody>")
    .replace("  <actuator>\n", '  <actuator>\n    <position name="lift" joint="lift" ctrlrange="0 0.3" kp="5000"/>\n')
    .replace('qpos="0.5', 'qpos="0 0.5').replace('ctrl="0.5', 'ctrl="0 0.5')
)
LIFT_SVA = {
    "name": "lift", "kinematic_param_type": "SVA",
    "links": [{"id": "base", "parent": "world"}, {"id": "top", "parent": "lift", "translation": {"x": 0, "y": 0, "z": 100}}],
    "joints": [{"id": "lift", "type": "prismatic", "parent": "base", "axis": {"x": 0, "y": 0, "z": 1}, "min": 0, "max": 300}],
}
# a box on two wheels for the robot to ride (forward +x, as a ROS base)
BOX_BASE = """
<mujoco>
  <worldbody>
    <body name="chassis">
      <geom type="box" size="0.2 0.15 0.05" pos="0 0 0.1" mass="10"/>
      <body name="left" pos="0 0.17 0.05"><joint name="left" axis="0 1 0"/>
        <geom type="cylinder" size="0.05 0.02" euler="1.5708 0 0" mass="0.5"/></body>
      <body name="right" pos="0 -0.17 0.05"><joint name="right" axis="0 1 0"/>
        <geom type="cylinder" size="0.05 0.02" euler="1.5708 0 0" mass="0.5"/></body>
    </body>
  </worldbody>
</mujoco>
"""
ON_BASE = [0, 0, 0.1]  # the robot's frame on the base: its torso box clears the chassis
ON_LIFT = [0, 0, -0.1]  # a limb's frame on the lift arm: the lift body, 100 mm below its flange


@pytest.fixture
def lifted(torso):
    (torso / "lifted.xml").write_text(LIFTED)
    (torso / "lift.json").write_text(json.dumps(LIFT_SVA))
    (torso / "base.xml").write_text(BOX_BASE)
    return torso


def lifted_robot(sim, path, **extra):
    return RobotHandle(sim, "bot", {"mjcf": str(path / "lifted.xml"), **extra})


def lift(path, **extra):
    return {
        "robot": "bot", "parent": "bot", "position": TORSO_POSE, "joints": ["lift"],
        "flange_site": "lift/top", "kinematics_url": str(path / "lift.json"), **extra,
    }


def box_base(sim, path):
    return BaseHandle(sim, "base", {"mjcf": str(path / "base.xml"), "max_linear_mps": 0.5, "max_angular_dps": 90})


def contacts(sim):
    """Body-name pairs in contact."""
    model = sim.model
    return [{model.body(model.geom_bodyid[g]).name for g in (c.geom1, c.geom2)} for c in sim.data.contact[: sim.data.ncon]]


def test_a_robot_rides_its_base(sim, lifted):
    sim.configure_world(WorldConfig.from_attrs({"props": []}))
    base = box_base(sim, lifted)
    robot(sim, lifted, parent="base", position=ON_BASE)  # the fixed-torso file
    left = ArmHandle(sim, "left-arm", limb(lifted, "left"))
    sim.advance(0.5)
    assert not [pair for pair in contacts(sim) if any(n.startswith("base/") for n in pair) and any(n.startswith("bot/") for n in pair)]
    start, start_quat = left.get_end_pose()

    def relative(name):
        frame = sim.data.site(name)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, frame.xmat)
        return robots.transform(frame.xpos, quat)

    before = np.linalg.inv(relative("base/viam_frame")) @ relative("bot/viam_frame")
    assert before[:3, 3] == pytest.approx(ON_BASE, abs=1e-9)
    mount = sim.data.site("bot/viam_frame").xpos.copy()
    finish(sim, base.move_straight(0.5, 0.3)).result()
    finish(sim, base.spin(math.radians(90), math.radians(60))).result()
    settle(sim)
    assert np.linalg.norm(sim.data.site("bot/viam_frame").xpos - mount) > 0.4  # it went with the base
    after = np.linalg.inv(relative("base/viam_frame")) @ relative("bot/viam_frame")
    assert after == pytest.approx(before, abs=1e-6)
    position, quat = left.get_end_pose()  # in the arm's own frame: unchanged
    assert position == pytest.approx(start, abs=1e-3)
    assert abs(np.dot(quat, start_quat)) == pytest.approx(1.0, abs=1e-5)


def test_the_robot_and_its_arms_drop_out_with_the_base(sim, lifted):
    sim.configure_world(WorldConfig.from_attrs({"props": []}))
    base = box_base(sim, lifted)
    lifted_robot(sim, lifted, parent="base", position=ON_BASE)
    torso = ArmHandle(sim, "torso", lift(lifted))
    left = ArmHandle(sim, "left-arm", limb(lifted, "left", parent="torso", position=ON_LIFT))
    base.close()
    assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "bot/torso") < 0
    for arm in (torso, left):
        with pytest.raises(RuntimeError, match="robot 'bot' is not in the scene"):
            arm.get_joint_positions()
    box_base(sim, lifted)
    assert left.get_joint_positions() == pytest.approx([0.0, 0.0])  # back, at home


def test_an_arm_on_an_arm(sim, lifted):
    lifted_robot(sim, lifted)
    torso = ArmHandle(sim, "torso", lift(lifted))
    with pytest.raises(ValueError, match=(
        r'set frame.translation to \{"x": 0.0, "y": 0.0, "z": -100.0\}.*relative to arm torso\'s end effector'
    )):
        ArmHandle(sim, "left-arm", limb(lifted, "left", parent="torso", position=[0, 0, 0]))
    left = ArmHandle(sim, "left-arm", limb(lifted, "left", parent="torso", position=ON_LIFT))
    right = ArmHandle(sim, "right-arm", limb(lifted, "right", parent="torso", position=ON_LIFT))
    claims = sim.robot_claims["bot"]
    assert (claims["joint:lift"], claims["joint:left/shoulder"], claims["joint:right/shoulder"]) == (
        "torso", "left-arm", "right-arm")
    # the torso moves the limb's frame in the world; its end position, in that frame, stays
    sim.advance(0.5)
    start, _ = left.get_end_pose()
    end = sim.data.site("left-arm/viam_frame").xpos.copy()
    torso.set_joint_targets([0.1])
    sim.advance(2.0)
    assert torso.get_joint_positions() == pytest.approx([0.1], abs=1e-3)
    assert sim.data.site("left-arm/viam_frame").xpos - end == pytest.approx([0, 0, 0.1], abs=1e-3)
    assert left.get_end_pose()[0] == pytest.approx(start, abs=1e-4)
    assert right.get_joint_positions() == pytest.approx([0.0, 0.0], abs=1e-3)


def test_a_limb_must_hang_from_its_parent_arms_last_link(sim, torso):
    robot(sim, torso)
    ArmHandle(sim, "right-arm", limb(torso, "right"))
    with pytest.raises(ValueError, match=r"torso moves relative to right/fore \(where arm right-arm's end effector is\)"):
        ArmHandle(sim, "left-arm", limb(torso, "left", parent="right-arm"))


def test_a_parent_arm_must_be_bound_to_the_same_robot(sim, torso):
    robot(sim, torso)
    RobotHandle(sim, "bot2", {"mjcf": str(torso / "torso.xml"), "position": [2, 0, 0]})
    ArmHandle(sim, "other", {**limb(torso, "right"), "robot": "bot2", "parent": "bot2"})
    with pytest.raises(ValueError, match='an arm in robot bot needs frame.parent "bot" or another of its arms'):
        ArmHandle(sim, "left-arm", limb(torso, "left", parent="other"))


def test_a_parent_arm_must_be_in_the_scene(sim, torso):
    robot(sim, torso)
    with pytest.raises(ValueError, match="arm 'right-arm' is not in the scene; configure it first"):
        ArmHandle(sim, "left-arm", limb(torso, "left", parent="right-arm"))


# ---- slide joints -------------------------------------------------------------


def test_a_slide_joint_moves_at_up_to_100_mm_s(sim, lifted):
    lifted_robot(sim, lifted)
    torso = ArmHandle(sim, "torso", lift(lifted))
    assert torso.dynamics_metadata()["max_velocity_rad_s"] == [0.1]  # m/s
    torso.set_joint_targets([0.12])
    sim.advance(0.5)
    assert torso.get_joint_positions()[0] == pytest.approx(0.05, abs=0.005)
    sim.advance(1.5)
    assert torso.get_joint_positions() == pytest.approx([0.12], abs=1e-3)
    hinged = ArmHandle(sim, "left-arm", limb(lifted, "left", parent="torso", position=ON_LIFT))
    assert hinged.dynamics_metadata()["max_velocity_rad_s"] == pytest.approx([math.pi] * 2)  # unchanged


def test_a_slide_joint_reads_mm(running_sim, lifted):
    import asyncio

    from viam.components.arm import JointPositions
    from viam.proto.app.robot import ComponentConfig
    from viam.utils import dict_to_struct

    from mujoco_module.models.arm import MujocoArm
    from mujoco_module.models.robot import MujocoRobot
    from mujoco_module.models.world import MujocoWorld
    from test_robot_models import config

    MujocoWorld.new(ComponentConfig(name="sim-world", attributes=dict_to_struct({})), {})
    bot = MujocoRobot.new(config("bot", {"mjcf": str(lifted / "lifted.xml")}), {})
    attrs = {k: v for k, v in lift(lifted).items() if k not in ("parent", "position")}
    torso = MujocoArm.new(config("torso", attrs, parent="bot", translation={"x": 0, "y": 0, "z": 500}), {})
    left = MujocoArm.new(config("left-arm", {k: v for k, v in limb(lifted, "left").items() if k not in ("parent", "position")},
                                parent="torso", translation={"x": 0, "y": 0, "z": -100}), {})

    async def run():
        assert torso._tolerance_rad.tolist() == [0.001]  # 1 mm
        assert left._tolerance_rad == pytest.approx([math.radians(0.5)] * 2)  # a hinge arm's, unchanged
        await torso.move_to_joint_positions(JointPositions(values=[120]))
        assert (await torso.get_joint_positions()).values == pytest.approx([120], abs=1)
        assert (await torso.do_command({"command": "get_joint_positions_radians"}))["values"] == pytest.approx([0.12], abs=1e-3)
        await left.move_to_joint_positions(JointPositions(values=[20, -10]))
        assert (await left.get_joint_positions()).values == pytest.approx([20, -10], abs=0.5)
        for component in (left, torso, bot):
            await component.close()

    asyncio.run(run())


def test_a_slide_joint_moves_through_mm_waypoints(running_sim, lifted):
    import asyncio

    from viam.components.arm import JointPositions
    from viam.proto.app.robot import ComponentConfig
    from viam.utils import dict_to_struct

    from mujoco_module.models.arm import MujocoArm
    from mujoco_module.models.robot import MujocoRobot
    from mujoco_module.models.world import MujocoWorld
    from test_robot_models import config

    MujocoWorld.new(ComponentConfig(name="sim-world", attributes=dict_to_struct({})), {})
    bot = MujocoRobot.new(config("bot", {"mjcf": str(lifted / "lifted.xml")}), {})
    attrs = {k: v for k, v in lift(lifted).items() if k not in ("parent", "position")}
    torso = MujocoArm.new(config("torso", attrs, parent="bot", translation={"x": 0, "y": 0, "z": 500}), {})

    async def run():
        await torso.move_through_joint_positions([JointPositions(values=[60]), JointPositions(values=[120])])
        assert (await torso.get_joint_positions()).values == pytest.approx([120], abs=1)
        for component in (torso, bot):
            await component.close()

    asyncio.run(run())
