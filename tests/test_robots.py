import json
import math
from pathlib import Path

import numpy as np
import pytest

from mujoco_module import robots

ASSETS = Path(__file__).parents[1] / "assets"
LITE6_KINEMATICS = ASSETS / "ufactory_lite6" / "lite6_kinematics.json"

# Two revolute joints: base -> j1 (z) -> upper (0.1 m up) -> j2 (y) -> tool (0.2 m along x)
SVA = {
    "name": "two",
    "links": [
        {"id": "base", "parent": "world", "translation": {"x": 0, "y": 0, "z": 0}},
        {"id": "upper", "parent": "j1", "translation": {"x": 0, "y": 0, "z": 100},
         "geometry": {"r": 20, "l": 100, "translation": {"x": 0, "y": 0, "z": 50}}},
        {"id": "tool", "parent": "j2", "translation": {"x": 200, "y": 0, "z": 0},
         "orientation": {"type": "ov_degrees", "value": {"x": 0, "y": 0, "z": 1, "th": 90}}},
    ],
    "joints": [
        {"id": "j1", "type": "revolute", "parent": "base", "axis": {"x": 0, "y": 0, "z": 1}},
        {"id": "j2", "type": "revolute", "parent": "upper", "axis": {"x": 0, "y": 1, "z": 0}},
    ],
}

URDF = """<robot name="two">
  <link name="base"/><link name="upper"/><link name="tool"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="upper"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/></joint>
  <joint name="j2" type="revolute"><parent link="upper"/><child link="tool_mount"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/><axis xyz="0 1 0"/></joint>
  <link name="tool_mount"/>
  <joint name="tip" type="fixed"><parent link="tool_mount"/><child link="tool"/>
    <origin xyz="0.2 0 0" rpy="0 0 1.5707963"/></joint>
</robot>"""


def expected_tool(q1, q2):
    """Hand-computed tool position for both test files."""
    c1, s1, c2, s2 = math.cos(q1), math.sin(q1), math.cos(q2), math.sin(q2)
    # in the plane of joint 1, joint 2 swings the 0.2 m arm about y
    x, z = 0.2 * c2, 0.1 - 0.2 * s2
    return np.array([c1 * x, s1 * x, z])


def test_sva_chain_and_forward_kinematics():
    kin = robots.parse_kinematics(json.dumps(SVA).encode(), urdf=False, label="two.json")
    assert kin.joints == ("j1", "j2")
    assert kin.end == "tool"
    poses = kin.forward([0.3, 0.4])
    assert poses["tool"][:3, 3] == pytest.approx(expected_tool(0.3, 0.4))
    # tool is turned 90 degrees about z relative to j2's frame
    at_zero = kin.forward([0.0, 0.0])["tool"]
    assert at_zero[:3, 0] == pytest.approx([0, 1, 0], abs=1e-9)
    assert kin.frames["upper"].geometry["l"] == 100


def test_urdf_chain_matches_the_sva_one():
    kin = robots.parse_kinematics(URDF.encode(), urdf=True, label="two.urdf")
    assert kin.joints == ("j1", "j2")
    assert kin.end == "tool"
    sva = robots.parse_kinematics(json.dumps(SVA).encode(), urdf=False, label="two.json")
    for q in ([0.0, 0.0], [0.3, 0.4], [-1.0, 1.2]):
        assert kin.forward(q)["tool"] == pytest.approx(sva.forward(q)["tool"], abs=1e-6)


def test_lite6_file_parses():
    kin = robots.parse_kinematics(LITE6_KINEMATICS.read_bytes(), urdf=False, label="lite6")
    assert kin.joints == ("waist", "shoulder", "elbow", "forearm_rot", "wrist", "gripper_rot")
    assert kin.end == "gripper_mount"
    # home: the flange points straight down
    flange = kin.forward([0, 0, 1.57, 0, 1.57, 0])["gripper_mount"]
    assert flange[:3, 2] == pytest.approx([0, 0, -1], abs=1e-2)


def test_prismatic_joint_moves_along_its_axis():
    doc = {
        "links": [{"id": "carriage", "parent": "rail", "translation": {"x": 0, "y": 0, "z": 0}}],
        "joints": [{"id": "rail", "type": "prismatic", "parent": "world", "axis": {"x": 1, "y": 0, "z": 0}}],
    }
    kin = robots.parse_kinematics(json.dumps(doc).encode(), urdf=False, label="rail")
    assert kin.forward([0.25])["carriage"][:3, 3] == pytest.approx([0.25, 0, 0])


def test_urdf_prismatic_axis_is_normalized():
    urdf = """<robot name="r"><link name="base"/><link name="slide"/>
      <joint name="rail" type="prismatic"><parent link="base"/><child link="slide"/>
        <axis xyz="0 0 2"/></joint></robot>"""
    kin = robots.parse_kinematics(urdf.encode(), urdf=True, label="rail.urdf")
    assert kin.forward([0.5])["slide"][:3, 3] == pytest.approx([0, 0, 0.5])


def test_duplicate_link_and_joint_name_is_rejected():
    doc = {
        "links": [{"id": "a", "parent": "a"}],
        "joints": [{"id": "a", "type": "revolute", "parent": "world", "axis": {"x": 0, "y": 0, "z": 1}}],
    }
    with pytest.raises(ValueError, match="duplicate frame name"):
        robots.parse_kinematics(json.dumps(doc).encode(), urdf=False, label="dup.json")


@pytest.mark.parametrize(
    "doc, message",
    [
        ({"links": [{"id": "a", "parent": "world"}, {"id": "b", "parent": "world"}]}, "one chain"),
        ({"links": [{"id": "a", "parent": "nowhere"}]}, "unknown parent"),
        ({"links": [{"id": "a", "parent": "world",
                     "orientation": {"type": "euler", "value": {}}}]}, "not supported"),
        ({"links": [{"parent": "world"}]}, "not a valid SVA JSON"),
    ],
)
def test_bad_sva_files(doc, message):
    with pytest.raises(ValueError, match=message):
        robots.parse_kinematics(json.dumps(doc).encode(), urdf=False, label="bad.json")


def test_bad_urdf_is_labelled():
    with pytest.raises(ValueError, match="bad.urdf: not a valid URDF"):
        robots.parse_kinematics(b"<robot", urdf=True, label="bad.urdf")


def test_transform_round_trips_a_quaternion():
    q = np.array([math.cos(0.3), 0, math.sin(0.3), 0])
    assert robots.quat(robots.transform((1, 2, 3), q)) == pytest.approx(q)
    a, b = robots.transform(), robots.transform((0.003, 0, 0), (math.cos(0.01), 0, 0, math.sin(0.01)))
    metres, degrees = robots.pose_error(a, b)
    assert metres == pytest.approx(0.003)
    assert degrees == pytest.approx(math.degrees(0.02))



ARM = """<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <geom size="0.05"/>
      <body name="l1"><joint name="a" axis="0 0 1" range="-3 3"/><geom size="0.05"/>
        <body name="l2" pos="0 0 0.1"><joint name="b" axis="0 1 0" range="-2 2"/><geom size="0.05"/>
          <body name="l3" pos="0.1 0 0"><joint name="c" axis="0 1 0" range="-2 2"/><geom size="0.05"/>
            <site name="attachment_site"/>
            <body name="hand"><geom size="0.02"/>
              <body name="finger"><joint name="f" type="slide" axis="0 1 0" range="0 0.04"/>
                <geom size="0.01"/></body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="act_c" joint="c" kp="100"/>
    <position name="act_a" joint="a" kp="100"/>
    <general name="act_b" joint="b" biastype="affine" gainprm="100" biasprm="0 -100 -10"/>
    <motor name="grip" joint="f" ctrlrange="-5 5"/>
  </actuator>
  <keyframe><key name="home" qpos="0.1 0.2 0.3 0" ctrl="0.3 0.1 0.2 0"/></keyframe>
</mujoco>"""


def compile_xml(xml):
    import mujoco

    return mujoco.MjSpec.from_string(xml).compile()


def names(model, kind, ids):
    import mujoco

    return [mujoco.mj_id2name(model, kind, i) for i in ids]


def test_arm_joints_and_actuators_are_inferred_by_target():
    import mujoco

    model = compile_xml(ARM)
    joints = robots.arm_joints(model, 3, "arm.xml")
    assert names(model, mujoco.mjtObj.mjOBJ_JOINT, joints) == ["a", "b", "c"]
    actuators = robots.arm_actuators(model, joints, "arm.xml")
    assert names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuators) == ["act_a", "act_b", "act_c"]
    assert robots.home(model, "home", actuators) == pytest.approx([0.1, 0.2, 0.3])
    assert robots.home(model, "missing", actuators) == pytest.approx([0, 0, 0])


def test_joints_override():
    import mujoco

    model = compile_xml(ARM)
    joints = robots.arm_joints(model, 2, "arm.xml", ["b", "c"])
    assert names(model, mujoco.mjtObj.mjOBJ_JOINT, joints) == ["b", "c"]
    with pytest.raises(ValueError, match=r"joints \['nope'\] are not in the MJCF"):
        robots.arm_joints(model, 1, "arm.xml", ["nope"])
    with pytest.raises(ValueError, match="lists 2 joints but the kinematics file has 3"):
        robots.arm_joints(model, 3, "arm.xml", ["b", "c"])


def test_too_few_joints_lists_what_was_found():
    with pytest.raises(ValueError, match=r"arm.xml: found 4 actuated joints \['a', 'b', 'c', 'f'\].* has 7; set joints"):
        robots.arm_joints(compile_xml(ARM), 7, "arm.xml")


def test_a_branch_before_n_joints_is_an_error():
    xml = """<mujoco><worldbody>
      <body name="left"><joint name="l1"/><geom size="0.1"/>
        <body><joint name="l2"/><geom size="0.1"/></body></body>
      <body name="right"><joint name="r1"/><geom size="0.1"/></body>
    </worldbody><actuator>
      <position joint="l1"/><position joint="l2"/><position joint="r1"/>
    </actuator></mujoco>"""
    with pytest.raises(ValueError, match="not one chain"):
        robots.arm_joints(compile_xml(xml), 3, "two-arms.xml")


def test_arm_actuators_must_be_position():
    model = compile_xml(ARM.replace('<position name="act_a" joint="a" kp="100"/>', '<motor name="act_a" joint="a"/>'))
    joints = robots.arm_joints(model, 3, "arm.xml")
    with pytest.raises(ValueError, match="'act_a' must be a position actuator"):
        robots.arm_actuators(model, joints, "arm.xml")


@pytest.mark.parametrize(
    "actuator",
    [
        '<position name="act_a" joint="a" kp="100" gear="2"/>',
        '<general name="act_a" joint="a" biastype="affine" gainprm="50" biasprm="0 -100 0"/>',
    ],
)
def test_arm_actuators_ctrl_must_be_the_joint_position(actuator):
    model = compile_xml(ARM.replace('<position name="act_a" joint="a" kp="100"/>', actuator))
    joints = robots.arm_joints(model, 3, "arm.xml")
    with pytest.raises(ValueError, match="gear 1 and gain == kp"):
        robots.arm_actuators(model, joints, "arm.xml")


def test_home_falls_back_to_keyframe_qpos():
    model = compile_xml(ARM.replace(' ctrl="0.3 0.1 0.2 0"', ""))
    actuators = robots.arm_actuators(model, robots.arm_joints(model, 3, "arm.xml"), "arm.xml")
    assert robots.home(model, "home", actuators) == pytest.approx([0.1, 0.2, 0.3])


@pytest.mark.parametrize(
    "actuator, kind",
    [
        ('<motor name="g" joint="f"/>', "force"),
        ('<general name="g" joint="f"/>', "force"),
        ('<position name="g" joint="f" kp="10"/>', "position"),
        ('<position name="g" joint="f" kp="10" timeconst="0.1"/>', "position"),
        ('<general name="g" joint="f" biastype="affine" gainprm="10" biasprm="0 -10 0"/>', "position"),
        ('<adhesion name="g" body="hand" ctrlrange="0 1" gain="1"/>', "suction"),
    ],
)
def test_control_type(actuator, kind):
    model = compile_xml(ARM.replace('<motor name="grip" joint="f" ctrlrange="-5 5"/>', actuator).replace(' ctrl="0.3 0.1 0.2 0"', ""))
    assert robots.control_type(model, model.actuator("g").id, "arm.xml") == kind


@pytest.mark.parametrize(
    "actuator, what",
    [
        ('<velocity name="g" joint="f" kv="10"/>', "a velocity"),
        ('<intvelocity name="g" joint="f" kp="10" actrange="0 1"/>', "an unsupported"),
    ],
)
def test_unsupported_control_types_are_named(actuator, what):
    model = compile_xml(ARM.replace('<motor name="grip" joint="f" ctrlrange="-5 5"/>', actuator).replace(' ctrl="0.3 0.1 0.2 0"', ""))
    with pytest.raises(ValueError, match=f"'g' is {what} actuator"):
        robots.control_type(model, model.actuator("g").id, "arm.xml")


@pytest.mark.parametrize(
    "actuator",
    [
        '<general name="g" body="hand" ctrlrange="-1 1" gainprm="5"/>',
        '<general name="g" body="hand" ctrllimited="false" gainprm="5"/>',
        '<general name="g" body="hand" ctrlrange="0 1" gainprm="5" biastype="affine" biasprm="1"/>',
        '<adhesion name="g" body="hand" ctrlrange="0.2 1" gain="1"/>',
        '<adhesion name="g" body="hand" ctrlrange="0 1" gain="0"/>',
    ],
)
def test_body_actuators_other_than_adhesion_are_errors(actuator):
    model = compile_xml(ARM.replace('<motor name="grip" joint="f" ctrlrange="-5 5"/>', actuator).replace(' ctrl="0.3 0.1 0.2 0"', ""))
    with pytest.raises(ValueError, match="'g' pulls on a body but is not a suction actuator"):
        robots.control_type(model, model.actuator("g").id, "arm.xml")


def test_integrated_gripper_actuator_jaw_and_bodies():
    import mujoco

    model = compile_xml(ARM)
    joints = robots.arm_joints(model, 3, "arm.xml")
    arm = robots.arm_actuators(model, joints, "arm.xml")
    grip = robots.gripper_actuator(model, "arm.xml", exclude=arm)
    assert model.actuator(grip).name == "grip"
    assert robots.control_type(model, grip, "arm.xml") == "force"
    jaw = robots.jaw(model, grip, -5.0, 5.0, "arm.xml")
    assert (model.joint(jaw.joint).name, jaw.open, jaw.closed) == ("f", 0.0, 0.04)
    # the open command reversed: open drives toward the other end
    assert robots.jaw(model, grip, 5.0, -5.0, "arm.xml").open == 0.04
    bodies = robots.integrated_gripper_bodies(model, joints[-1], int(model.jnt_bodyid[jaw.joint]))
    assert names(model, mujoco.mjtObj.mjOBJ_BODY, bodies) == ["hand", "finger"]
    # a suction cup: the body itself, not a jaw joint
    assert robots.integrated_gripper_bodies(model, joints[-1], model.body("hand").id) == bodies
    assert robots.gripper_actuator(model, "arm.xml", name="act_a") == model.actuator("act_a").id
    with pytest.raises(ValueError, match="'nope' is not in the MJCF"):
        robots.gripper_actuator(model, "arm.xml", name="nope")


def test_gripper_actuator_count_errors():
    model = compile_xml(ARM)
    with pytest.raises(ValueError, match="more than one gripper actuator.*set actuator"):
        robots.gripper_actuator(model, "arm.xml")
    with pytest.raises(ValueError, match="arm.xml: has no gripper actuator"):
        robots.gripper_actuator(model, "arm.xml", exclude=range(model.nu))


def test_tendon_jaw_uses_the_first_joint_and_its_coefficient():
    xml = """<mujoco><compiler angle="radian"/><worldbody><body name="palm">
      <body name="l"><joint name="left" axis="1 0 0" range="0 0.8"/><geom size="0.01"/></body>
      <body name="r"><joint name="right" axis="1 0 0" range="0 0.8"/><geom size="0.01"/></body>
    </body></worldbody>
    <tendon><fixed name="split"><joint joint="right" coef="0.5"/><joint joint="left" coef="0.5"/></fixed></tendon>
    <actuator><general name="fingers" tendon="split" ctrlrange="0 255" biastype="affine"
      gainprm="0.3137255" biasprm="0 -100 -10"/></actuator></mujoco>"""
    model = compile_xml(xml)
    assert robots.control_type(model, 0, "g.xml") == "position"
    jaw = robots.jaw(model, 0, 0.0, 255.0, "g.xml")
    assert (model.joint(jaw.joint).name, jaw.open, jaw.closed) == ("right", 0.0, pytest.approx(0.8))


def test_load_mjcf_names_unnamed_actuators_after_their_target():
    spec = robots.load_mjcf(ASSETS / "ufactory_lite6" / "lite6_gripper_wide.xml")
    assert [a.name for a in spec.actuators] == [f"joint{i}" for i in range(1, 7)] + ["gripper"]


def test_load_file_takes_mesh_dir_only_for_a_urdf():
    with pytest.raises(ValueError, match=r"lite6: mesh_dir is for a .urdf mjcf; an MJCF names its own meshdir"):
        robots.load_file(ASSETS / "ufactory_lite6" / "lite6.xml", "lite6", mesh_dir="meshes")
    with pytest.raises(ValueError, match=r"mjcf must be an .xml \(MJCF\) or .urdf file, got lite6.sdf"):
        robots.load_file(ASSETS / "lite6.sdf", "lite6")


def test_load_kinematics_reads_paths_and_file_urls(tmp_path, monkeypatch):
    data = robots.load_kinematics("ufactory_lite6/lite6_kinematics.json", "arm")
    assert json.loads(data)["name"] == "lite6"
    assert robots.load_kinematics(LITE6_KINEMATICS.as_uri(), "arm") == data
    assert robots.load_kinematics(str(LITE6_KINEMATICS), "arm") == data
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "k.json").write_bytes(data)
    assert robots.load_kinematics("~/k.json", "arm") == data
    with pytest.raises(ValueError, match="arm: file .*missing.json not found"):
        robots.load_kinematics("missing.json", "arm")


def test_is_urdf():
    assert robots.is_urdf("https://example.com/arm.URDF")
    assert robots.is_urdf("arm.xacro")
    assert not robots.is_urdf("arm.json")
    assert robots.is_urdf("https://host/arm.urdf?sig=abc")
    assert not robots.is_urdf("https://host/arm.json?f=x.urdf")


def test_load_kinematics_names_the_arm_on_read_failure(tmp_path):
    with pytest.raises(ValueError, match="arm a: cannot read"):
        robots.load_kinematics((tmp_path / "missing.json").as_uri(), "arm a")


def test_parse_pose():
    assert robots.parse_pose(None, "x") == pytest.approx(np.eye(4))
    assert robots.parse_pose({}, "x") == pytest.approx(np.eye(4))
    pose = robots.parse_pose(
        {"translation": {"x": 10, "z": -20}, "orientation": {"o_z": 1, "theta": 180}}, "x"
    )
    assert pose[:3, 3] == pytest.approx([0.01, 0, -0.02])
    assert pose[:3, :3] == pytest.approx(np.diag([-1.0, -1.0, 1.0]), abs=1e-9)
    # no axis given: Viam's default, +Z
    turned = robots.parse_pose({"orientation": {"theta": 90}}, "x")
    assert turned[:3, :3] == pytest.approx(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), abs=1e-9)


@pytest.mark.parametrize(
    "raw, message",
    [
        ([0, 0, 0], "must be {translation"),
        ({"offset": {}}, "must be {translation"),
        ({"translation": {"x": "far"}}, "must be numbers"),
        ({"orientation": {"o_x": 0, "o_y": 0, "o_z": 0, "theta": 5}}, "not all zero"),
        ({"translation": {"x": float("nan")}}, "finite"),
    ],
)
def test_parse_pose_rejects(raw, message):
    with pytest.raises(ValueError, match=f"arm: flange_offset.*{message}"):
        robots.parse_pose(raw, "arm: flange_offset")


def test_format_pose_round_trips_through_parse_pose():
    pose = robots.parse_pose(
        {"translation": {"x": 12.5, "y": -3, "z": 0}, "orientation": {"o_x": 0, "o_y": 1, "o_z": 0, "theta": 30}},
        "x",
    )
    again = robots.parse_pose(json.loads(robots.format_pose(pose)), "x")
    assert again == pytest.approx(pose, abs=1e-3)


@pytest.mark.parametrize(
    "geometry, kind, size",
    [
        ({"x": 100, "y": 40, "z": 20}, "box", (0.05, 0.02, 0.01)),
        ({"r": 30}, "sphere", (0.03,)),
        ({"r": 20, "l": 100}, "capsule", (0.02, 0.03)),  # Viam's l includes the caps
        ({"r": 20, "l": 40}, "sphere", (0.02,)),  # no straight part left
    ],
)
def test_viam_geometries_become_mujoco_shapes(geometry, kind, size):
    got_kind, got_size, _ = robots._mj_shape(geometry)
    assert (got_kind, got_size) == (kind, pytest.approx(size))


# A world link, an offset joint from it, and two end links (tool and a side sensor)
WORLD_URDF = """<robot name="two">
  <link name="world"/><link name="base"/><link name="upper"/><link name="tool"/><link name="sensor"/>
  <joint name="mount" type="fixed"><parent link="world"/><child link="base"/>
    <origin xyz="0 0 0.05"/></joint>
  <joint name="j1" type="revolute"><parent link="base"/><child link="upper"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/></joint>
  <joint name="j2" type="revolute"><parent link="upper"/><child link="tool"/>
    <origin xyz="0.2 0 0"/><axis xyz="0 1 0"/></joint>
  <joint name="side" type="fixed"><parent link="base"/><child link="sensor"/>
    <origin xyz="0.1 0 0"/></joint>
</robot>"""


def test_a_world_link_is_the_world_frame():
    pruned = robots.prune_urdf(WORLD_URDF.encode(), "tool")
    kin = robots.parse_kinematics(pruned, urdf=True, label="two")  # hung before the world link was skipped
    assert (kin.joints, kin.end) == (("j1", "j2"), "tool")
    assert kin.frames["mount"].parent == "world"
    assert kin.forward([0.0, 0.0])["tool"][:3, 3] == pytest.approx([0.2, 0, 0.15])


def test_prune_urdf_keeps_the_path_from_the_root_to_the_link():
    from xml.etree import ElementTree

    root = ElementTree.fromstring(robots.prune_urdf(WORLD_URDF.encode(), "tool"))
    assert [link.get("name") for link in root.findall("link")] == ["base", "upper", "tool"]
    assert [joint.get("name") for joint in root.findall("joint")] == ["mount", "j1", "j2"]
    with pytest.raises(ValueError, match=r"expected one chain with one end frame, found ends \['tool', 'sensor'\]"):
        robots.parse_kinematics(WORLD_URDF.encode(), urdf=True, label="two")
