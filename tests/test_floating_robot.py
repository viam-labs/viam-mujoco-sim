"""A free-floating robot file (a legged robot's trunk on a free joint) in viam-labs:mujoco-sim:robot,
its BAM actuators, read-only limb arms and the io DoCommand. A synthetic two-legged robot stands in
for the MicroDuck, whose meshes are CC BY-NC-SA and stay out of this repo."""

import numpy as np
import pytest

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
