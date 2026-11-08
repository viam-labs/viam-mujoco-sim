import json
import math
import re
import shutil
from pathlib import Path
from xml.etree import ElementTree

import mujoco
import numpy as np
import pytest

from mujoco_module import robots
from mujoco_module.handles import ArmHandle, GripperHandle

DATA = Path(__file__).parent / "data"
ARM = DATA / "arm" / "arm.urdf"  # three joints, meshes/ beside it, end links tool0 and sensor_link
GRIPPER = DATA / "gripper.urdf"  # left_joint driven, right_joint its <mimic>
JOINTS = ("joint1", "joint2", "joint3")
EFFORTS = (50.0, 50.0, 10.0)


def urdf_arm(sim, **attrs):
    return ArmHandle(sim, "arm", {"mjcf": str(ARM), "flange_link": "tool0", **attrs})


def edited_arm(tmp_path, old, new):
    """A copy of the fixture arm (with its meshes) with `old` replaced by `new`."""
    folder = shutil.copytree(ARM.parent, tmp_path / "arm")
    text = ARM.read_text()
    assert text.count(old) == 1
    (folder / "arm.urdf").write_text(text.replace(old, new))
    return str(folder / "arm.urdf")


def test_several_end_links_need_flange_link(sim):
    with pytest.raises(ValueError, match=r"has end links \['tool0', 'sensor_link'\]; set flange_link"):
        ArmHandle(sim, "arm", {"mjcf": str(ARM)})
    with pytest.raises(ValueError, match=r"flange_link 'tool' is not a link; use one of \['base_link', "):
        urdf_arm(sim, flange_link="tool")
    with pytest.raises(ValueError, match="flange_link 'world' is not a link"):
        urdf_arm(sim, flange_link="world")


def test_flange_attributes_follow_the_file_type(sim):
    with pytest.raises(ValueError, match="flange_site is for an MJCF arm; set flange_link for a .urdf"):
        urdf_arm(sim, flange_site="attachment_site")
    with pytest.raises(ValueError, match="flange_link is for a .urdf arm; set flange_site for an MJCF"):
        ArmHandle(sim, "pick-arm", {"asset": "lite6", "flange_link": "link6"})


def test_default_kinematics_are_the_urdf_pruned_to_the_flange_link(sim):
    arm = urdf_arm(sim)
    assert arm.kinematics_url == str(ARM)
    root = ElementTree.fromstring(arm.kinematics_data)
    assert [link.get("name") for link in root.findall("link")] == ["base_link", "link1", "link2", "link3", "tool0"]
    assert root.find("joint").get("name") == "world_joint"  # kept: RDK reads its parent as the world frame
    assert (arm.kinematics.joints, arm.kinematics.end) == (JOINTS, "tool0")
    assert arm.joint_names() == JOINTS


def test_servos_are_sized_from_effort_and_inertia(sim):
    arm = urdf_arm(sim)
    plain = robots.load_mjcf(ARM).compile()
    data = mujoco.MjData(plain)
    mujoco.mj_forward(plain, data)
    inertia = np.zeros((plain.nv, plain.nv))
    mujoco.mj_fullM(plain, data, inertia)
    model = arm.model
    for joint, actuator, effort in zip(arm.joints, arm.actuators, EFFORTS):
        kp = effort / robots.EFFORT_ERROR_RAD
        dof = model.jnt_dofadr[joint]
        assert model.actuator(actuator).name == model.joint(joint).name
        assert model.actuator_gainprm[actuator, 0] == pytest.approx(kp)
        assert model.actuator_biasprm[actuator, 2] == pytest.approx(-2 * math.sqrt(kp * inertia[dof, dof]))
        assert model.actuator_ctrlrange[actuator] == pytest.approx(model.jnt_range[joint])
        assert model.jnt_actfrcrange[joint] == pytest.approx([-effort, effort])  # the force limit
    assert model.body_gravcomp == pytest.approx(np.ones(model.nbody))


def test_a_joint_without_an_effort_limit_is_an_error(sim, tmp_path):
    mjcf = edited_arm(tmp_path, 'effort="10" ', "")
    with pytest.raises(ValueError, match="joint 'joint3' has no effort limit; add <limit effort> to the URDF"):
        ArmHandle(sim, "arm", {"mjcf": mjcf, "flange_link": "tool0"})


def test_the_flange_is_an_attachment_site_at_the_flange_link(sim):
    urdf_arm(sim, position=[0.5, 0, 0])
    assert sim.data.site("arm/attachment_site").xpos == pytest.approx(sim.data.body("arm/tool0").xpos)
    # the world joint's offset stays: base_link is 5 cm above the arm's frame
    assert sim.data.body("arm/base_link").xpos == pytest.approx([0.5, 0, 0.05])


def test_home_contacts_are_excluded_and_the_arm_reaches_and_holds_targets(sim):
    arm = urdf_arm(sim)
    base, link1 = (sim.model.body(f"arm/{name}").id for name in ("base_link", "link1"))
    assert (base << 16) + link1 in sim.model.exclude_signature  # their meshes overlap 0.5 mm at home
    target = [1.0, -0.6, 0.8]
    arm.set_joint_targets(target)
    sim.advance(3.0)
    assert arm.get_joint_positions() == pytest.approx(target, abs=math.radians(0.5))
    arm.set_joint_targets([0.0, 0.0, 0.0])  # the forearm straight out, against gravity
    sim.advance(3.0)
    assert arm.get_joint_positions() == pytest.approx([0.0, 0.0, 0.0], abs=math.radians(0.5))


def test_explicit_kinematics_url(sim, tmp_path):
    pruned = tmp_path / "pruned.urdf"
    pruned.write_bytes(robots.prune_urdf(ARM.read_bytes(), "tool0"))
    with pytest.raises(ValueError, match=r"kinematics_url .*pruned.urdf has mesh collisions"):
        urdf_arm(sim, kinematics_url=str(pruned))
    boxes = tmp_path / "boxes.urdf"
    boxes.write_bytes(re.sub(rb"<mesh [^>]*/>", b'<box size="0.05 0.05 0.05"/>', pruned.read_bytes()))
    arm = urdf_arm(sim, kinematics_url=str(boxes))
    assert arm.kinematics.joints == JOINTS and arm.kinematics_meshes == {}
    sva = tmp_path / "arm.json"
    sva.write_text(json.dumps({
        "name": "three",
        "links": [{"id": "base", "parent": "world"}],
        "joints": [{"id": "a", "type": "revolute", "parent": "base", "axis": {"x": 0, "y": 0, "z": 1}}],
    }))
    with pytest.raises(ValueError, match="a .urdf arm with an SVA kinematics_url needs joints"):
        urdf_arm(sim, kinematics_url=str(sva))


def test_a_urdf_that_is_not_named_urdf_is_refused(tmp_path):
    xml = shutil.copytree(ARM.parent, tmp_path / "arm") / "arm.xml"
    (tmp_path / "arm" / "arm.urdf").rename(xml)
    with pytest.raises(ValueError, match=r"arm: .*arm.xml is a URDF; rename it to .urdf"):
        robots.load_file(xml, "arm")


def test_flange_attributes_are_checked_at_validate_time():
    from viam.proto.app.robot import ComponentConfig
    from viam.utils import dict_to_struct

    from mujoco_module.models.arm import MujocoArm

    def check(attrs):
        MujocoArm.validate_config(
            ComponentConfig(name="a", attributes=dict_to_struct({"world": "w", **attrs}))
        )

    with pytest.raises(ValueError, match="flange_site is for an MJCF arm"):
        check({"mjcf": str(ARM), "flange_site": "s"})
    with pytest.raises(ValueError, match="flange_link is for a .urdf arm"):
        check({"mjcf": "ufactory_lite6/lite6.xml", "kinematics_url": str(ARM), "flange_link": "l"})


def test_a_urdf_arm_validates_without_kinematics_url():
    from viam.proto.app.robot import ComponentConfig
    from viam.utils import dict_to_struct

    from mujoco_module.models.arm import MujocoArm

    def config(attrs):
        return ComponentConfig(name="a", attributes=dict_to_struct({"world": "w", **attrs}))

    assert MujocoArm.validate_config(config({"mjcf": str(ARM), "flange_link": "tool0"})) == (["w"], [])
    with pytest.raises(ValueError, match="flange_link must be a name"):
        MujocoArm.validate_config(config({"mjcf": str(ARM), "flange_link": ""}))


def test_get_kinematics_serves_the_pruned_urdf_and_its_collision_meshes(running_sim):
    import asyncio

    from viam.proto.app.robot import ComponentConfig
    from viam.proto.common import KinematicsFileFormat
    from viam.utils import dict_to_struct

    from mujoco_module.models.arm import MujocoArm
    from mujoco_module.models.world import MujocoWorld

    def config(name, attrs):
        return ComponentConfig(name=name, attributes=dict_to_struct(attrs))

    MujocoWorld.new(config("w", {"real_time": False}), {})
    arm = MujocoArm.new(config("a", {"world": "w", "mjcf": str(ARM), "flange_link": "tool0"}), {})
    fmt, data, meshes = asyncio.run(arm.get_kinematics())
    assert fmt == KinematicsFileFormat.KINEMATICS_FILE_FORMAT_URDF
    assert data == robots.prune_urdf(ARM.read_bytes(), "tool0")
    # keyed as RDK normalizes package://three_description/meshes/base.stl
    assert sorted(meshes) == ["meshes/base.stl", "meshes/link1.stl"]
    assert meshes["meshes/base.stl"].content_type == "stl"
    assert meshes["meshes/base.stl"].mesh == (ARM.parent / "meshes" / "base.stl").read_bytes()


def test_kinematics_meshes_must_be_stl_or_ply(tmp_path):
    urdf = b"""<robot name="r"><link name="a"><collision><geometry>
      <mesh filename="package://r/meshes/a.obj"/></geometry></collision></link></robot>"""
    with pytest.raises(ValueError, match="r.urdf: collision mesh package://r/meshes/a.obj is not STL or PLY"):
        robots.kinematics_meshes(urdf, tmp_path, "r.urdf")


def test_urdf_gripper_on_the_urdf_arm(sim):
    urdf_arm(sim)
    gripper = GripperHandle(sim, "g", "arm", {"mjcf": str(GRIPPER)})
    # attached at the arm's stored flange site, which the module added at tool0
    assert sim.data.body("g/palm").xpos == pytest.approx(sim.data.site("arm/attachment_site").xpos)
    assert gripper._joint == "g/left_joint" and gripper._actuator == "g/left_joint"
    assert (gripper.open_q, gripper.closed_q) == (0.0, pytest.approx(0.034))  # the ctrlrange ends
    actuator = sim.model.actuator("g/left_joint").id
    assert sim.model.actuator_gainprm[actuator, 0] == pytest.approx(20 / robots.EFFORT_ERROR_RAD)
    gripper.close_jaws()
    sim.advance(0.5)
    assert gripper.jaw() == pytest.approx(0.034, abs=1e-3)
    assert sim.data.joint("g/right_joint").qpos[0] == pytest.approx(0.034, abs=1e-3)  # the mimic follows
    gripper.open_jaws()
    sim.advance(0.5)
    assert gripper.jaw() == pytest.approx(0.0, abs=1e-3)
    assert not gripper.is_moving()


def test_urdf_gripper_grabs_a_block(sim):
    urdf_arm(sim)
    gripper = GripperHandle(sim, "g", "arm", {"mjcf": str(GRIPPER)})
    sim.advance(0.1)
    # between the fingertips, weightless while they close
    left, right = (sim.data.body(f"g/{side}_finger").xpos for side in ("left", "right"))
    tool_z = sim.data.site("arm/attachment_site").xmat.reshape(3, 3)[:, 2]
    sim.set_prop_pose("pick_block", list((left + right) / 2 + 0.035 * tool_z), [1, 0, 0, 0])
    gravity = sim.model.opt.gravity.copy()
    sim.model.opt.gravity[:] = 0
    gripper.close_jaws()
    sim.advance(0.5)
    sim.model.opt.gravity[:] = gravity
    sim.advance(0.5)
    assert gripper.firm_grasp()
    assert gripper.jaw() < 0.025  # stopped on the 31 mm block


@pytest.mark.parametrize(
    "old, new, found",
    [
        ('<mimic joint="left_joint" multiplier="1" offset="0"/>', "", "['left_joint', 'right_joint']"),
        ('type="prismatic"', 'type="fixed"', "[]"),
    ],
)
def test_urdf_gripper_needs_one_driven_joint(sim, tmp_path, old, new, found):
    urdf_arm(sim)
    path = tmp_path / "gripper.urdf"
    path.write_text(GRIPPER.read_text().replace(old, new))
    with pytest.raises(ValueError, match=rf"needs one movable joint that isn't a <mimic>, found {re.escape(found)}; use an MJCF gripper"):
        GripperHandle(sim, "g", "arm", {"mjcf": str(path)})
