import json

import pytest

from mujoco_module.handles import ArmHandle, CameraHandle
from mujoco_module.robot_file import RobotHandle
from mujoco_module.sim import WorldConfig

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

