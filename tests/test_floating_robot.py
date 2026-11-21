"""A free-floating robot file (a legged robot's trunk on a free joint) in viam-labs:mujoco-sim:robot,
its BAM actuators, read-only limb arms and the io DoCommand. A synthetic two-legged robot stands in
for the MicroDuck, whose meshes are CC BY-NC-SA and stay out of this repo."""

import mujoco
import numpy as np
import pytest

from mujoco_module.bam import BamConfig, rewrite_spec
from mujoco_module.handles import ArmHandle
from mujoco_module.robot_file import RobotHandle

# A 0.3 kg trunk on a free joint, and two 2-hinge legs about y with wide flat box feet, so it
# stands statically. Actuator order interleaves the legs the way the duck's file does (left, then
# right), so io's joint order is the file's actuator order. Geometry at STAND (hip 0.15,
# knee -0.3):
# - each 0.1 m segment drops 0.1*cos(0.15) = 0.0989 m;
# - the shin's world pitch is -0.15, so the foot box is pitched +0.15 on the shin to lie flat;
# - trunk centre to sole is 0.03 + 2*0.0989 + 0.005 = 0.2328, so the trunk starts at 0.234,
#   about 1 mm above the floor.
# STAND must differ from qpos0 (all zeros): several tests use that to catch a qpos wipe.
FLOATER = """
<mujoco>
  <compiler angle="radian"/>
  <default><joint damping="0.05" frictionloss="0.01" armature="0.001"/><position kp="5"/></default>
  <worldbody>
    <body name="trunk" pos="0 0 0.234">
      <freejoint name="root"/>
      <geom type="box" size="0.05 0.04 0.03" mass="0.3"/>
      <body name="l_thigh" pos="0 0.03 -0.03">
        <joint name="l_hip" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" mass="0.05"/>
        <body name="l_shin" pos="0 0 -0.1">
          <joint name="l_knee" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" mass="0.05"/>
          <geom type="box" size="0.04 0.02 0.005" pos="0 0 -0.105" euler="0 0.15 0" mass="0.02"/>
          <site name="l_foot" pos="0 0 -0.1"/>
        </body>
      </body>
      <body name="r_thigh" pos="0 -0.03 -0.03">
        <joint name="r_hip" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" mass="0.05"/>
        <body name="r_shin" pos="0 0 -0.1">
          <joint name="r_knee" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" mass="0.05"/>
          <geom type="box" size="0.04 0.02 0.005" pos="0 0 -0.105" euler="0 0.15 0" mass="0.02"/>
          <site name="r_foot" pos="0 0 -0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="l_hip" joint="l_hip"/>
    <position name="l_knee" joint="l_knee"/>
    <position name="r_hip" joint="r_hip"/>
    <position name="r_knee" joint="r_knee"/>
  </actuator>
  <keyframe>
    <key name="stand" qpos="0 0 0.234 1 0 0 0  0.15 -0.3 0.15 -0.3" ctrl="0.15 -0.3 0.15 -0.3"/>
  </keyframe>
</mujoco>
"""
STAND = [0.15, -0.3, 0.15, -0.3]
TRUNK_Z = 0.234


@pytest.fixture
def floater(tmp_path):
    path = tmp_path / "floater.xml"
    path.write_text(FLOATER)
    return path


def robot(sim, path, **extra):
    return RobotHandle(sim, "duck", {"mjcf": str(path), **extra})


def free_qpos(sim, name="duck/root"):
    adr = sim.model.joint(name).qposadr[0]
    return sim.data.qpos[adr : adr + 7].copy()


def test_a_free_root_loads_and_homes_from_its_keyframe(sim, floater):
    robot(sim, floater)
    assert sim.model.joint("duck/root").type == 0  # mjJNT_FREE
    assert free_qpos(sim) == pytest.approx([0, 0, TRUNK_Z, 1, 0, 0, 0])


def test_a_floating_robot_has_weight(sim, floater):
    # load_mjcf gravity-compensates every body (for arms); a floating robot must not hover
    robot(sim, floater)
    trunk = sim.model.body("duck/trunk").id
    assert sim.model.body_gravcomp[trunk] == 0.0
    assert sim.model.body_gravcomp[sim.model.body("duck/l_shin").id] == 0.0


def test_a_free_root_homes_relative_to_the_robots_frame(sim, floater):
    # the keyframe's trunk pose is in the file's world; the robot's frame moves it
    robot(sim, floater, position=[1.0, 0.5, 0.0], orientation_wxyz=[0.7071068, 0, 0, 0.7071068])
    q = free_qpos(sim)
    assert q[:3] == pytest.approx([1.0, 0.5, TRUNK_Z], abs=1e-6)
    assert q[3:] == pytest.approx([0.7071068, 0, 0, 0.7071068], abs=1e-6)


@pytest.mark.parametrize(
    "edit, message",
    [
        # a free joint below the root
        (('<body name="l_thigh" pos="0 0.03 -0.03">', '<body name="l_thigh" pos="0 0.03 -0.03"><freejoint name="loose"/>'),
         "free joint on 'l_thigh', which isn't a root body"),
        # an unnamed free joint, and an unnamed trunk
        (('<freejoint name="root"/>', "<freejoint/>"), "free joint and its body need names"),
        (('<body name="trunk" pos', '<body pos'), "free joint and its body need names"),
        # a second floating root
        (("</worldbody>", '<body name="other"><freejoint/><geom size="0.01"/></body></worldbody>'),
         "more than one floating root"),
    ],
)
def test_free_joint_errors(sim, floater, edit, message):
    floater.write_text(FLOATER.replace(*edit, 1))
    with pytest.raises(ValueError, match=message):
        robot(sim, floater)


def test_a_floating_robot_cannot_ride_a_base(sim, floater):
    with pytest.raises(ValueError, match="floating robot .* must be framed on the world"):
        robot(sim, floater, parent="rover")


def leg(side, **extra):
    # chain base is the trunk itself, so the arm's frame relative to the robot is the identity
    return {"robot": "duck", "parent": "duck", "joints": [f"{side}_hip", f"{side}_knee"],
            "flange_site": f"{side}_foot", "home_joints": STAND[:2], **extra}


def test_limbs_bind_to_a_floating_trunk(sim, floater):
    robot(sim, floater)
    left = ArmHandle(sim, "left-leg", leg("l"))
    ArmHandle(sim, "right-leg", leg("r"))
    assert left.joint_names() == ("l_hip", "l_knee")
    assert left.get_joint_positions() == pytest.approx(STAND[:2], abs=1e-9)
    # the arm frame site sits on the trunk and moves with it
    assert sim.data.site("left-leg/arm_frame").xpos == pytest.approx(sim.data.body("duck/trunk").xpos)


def test_a_limb_frame_off_the_trunk_says_what_to_paste(sim, floater):
    robot(sim, floater)
    # -?: inv(parent_end) @ base can leave -1e-17, printed as -0.0
    with pytest.raises(ValueError, match=r'set frame.translation to \{"x": -?0.0, "y": -?0.0, "z": -?0.0\}'):
        ArmHandle(sim, "left-leg", leg("l", position=[0, 0, 0.1]))


POLLEN_SOLREF = (-5.0e4, -2.0e2)
POLLEN_SOLIMP = (0.99, 0.9999, 0.001, 0.5, 2.0)


def pollen_rewrite(spec, bam_model):
    """microduck_rl@0daab16 scripts/infer_policy.py load_mujoco_with_bam, the spec part."""
    force_limit = bam_model.actuator.vin * bam_model.kt.value / bam_model.R.value
    for act in spec.actuators:
        if act.target.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        joint = spec.joint(act.target)
        joint.damping = np.zeros(np.shape(joint.damping))
        joint.frictionloss = 0.0
        joint.solref_friction = POLLEN_SOLREF
        joint.solimp_friction = POLLEN_SOLIMP


def test_bam_config_defaults_and_errors():
    cfg = BamConfig.from_attrs({"model": "bam"}, "robot duck")
    assert (cfg.motor, cfg.bam_model, cfg.kp_fw, cfg.vin, cfg.vin_drop_gain, cfg.vin_min) == (
        "xl330", "m6", 200.0, 7.4, 0.1, 6.0)
    with pytest.raises(ValueError, match='actuators.model must be "bam"'):
        BamConfig.from_attrs({"model": "pd"}, "robot duck")
    with pytest.raises(ValueError, match="unknown actuators keys"):
        BamConfig.from_attrs({"model": "bam", "kp": 3}, "robot duck")
    with pytest.raises(ValueError, match="BAM has no motor"):
        BamConfig.from_attrs({"model": "bam", "motor": "nope"}, "robot duck")


def test_rewrite_matches_pollens_load_mujoco_with_bam():
    cfg = BamConfig.from_attrs({"model": "bam"}, "t")
    ours, theirs = mujoco.MjSpec.from_string(FLOATER), mujoco.MjSpec.from_string(FLOATER)
    names = rewrite_spec(ours, cfg, cfg.load())
    pollen_rewrite(theirs, cfg.load())
    assert names == ["l_hip", "l_knee", "r_hip", "r_knee"]
    a, b = ours.compile(), theirs.compile()
    for field in ("actuator_gainprm", "actuator_biasprm", "actuator_gear", "actuator_forcerange",
                  "actuator_forcelimited", "actuator_ctrllimited", "actuator_dyntype",
                  "dof_damping", "dof_frictionloss", "dof_solref", "dof_solimp"):
        assert getattr(a, field) == pytest.approx(getattr(b, field)), field
