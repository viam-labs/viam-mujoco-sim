import math
import shutil
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_module import bases, robots

URDF = Path(__file__).parent / "data" / "skid.urdf"
MESHES = Path(__file__).parents[1] / "assets" / "meshes"

# A Viam-axes differential drive: axles along x (forward +Y), chassis origin
# at axle height, and a rear caster (a swivel with a wheel under it).
DIFF = """<mujoco>
  <worldbody>
    <body name="chassis" pos="3 4 5" euler="0 0 30">
      <freejoint/>
      <geom name="shell" type="box" size="0.1 0.15 0.04" pos="0 0 0.03" mass="2"/>
      <body name="left_wheel" pos="-0.15 0 0">
        <joint name="left" axis="1 0 0"/>
        <geom type="cylinder" size="0.05 0.01" euler="0 90 0" mass="0.1"/>
      </body>
      <body name="right_wheel" pos="0.15 0 0">
        <joint name="right" axis="-1 0 0"/>
        <geom type="cylinder" size="0.05 0.01" euler="0 90 0" mass="0.1"/>
      </body>
      <body name="swivel" pos="0 -0.12 -0.01">
        <joint name="swivel" axis="0 0 1"/>
        <geom type="sphere" size="0.005" mass="0.01"/>
        <body name="caster" pos="0 -0.01 -0.02">
          <joint name="caster" axis="1 0 0"/>
          <geom type="sphere" size="0.02" mass="0.01"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"""

# A ROS-axes skid-steer: axles along y (forward +X), four wheels, two of
# them on a jointless axle body.
SKID = """<mujoco>
  <worldbody>
    <body name="base_link">
      <geom type="box" size="0.3 0.2 0.05" pos="0 0 0.1" mass="10"/>
      <body name="axle">
        <body name="fl" pos="0.2 0.25 0"><joint name="fl" axis="0 1 0"/>
          <geom type="cylinder" size="0.1 0.03" euler="90 0 0" mass="1"/></body>
        <body name="fr" pos="0.2 -0.25 0"><joint name="fr" axis="0 1 0"/>
          <geom type="cylinder" size="0.1 0.03" euler="90 0 0" mass="1"/></body>
      </body>
      <body name="rl" pos="-0.2 0.25 0"><joint name="rl" axis="0 1 0"/>
        <geom type="cylinder" size="0.1 0.03" euler="90 0 0" mass="1"/></body>
      <body name="rr" pos="-0.2 -0.25 0"><joint name="rr" axis="0 1 0"/>
        <geom type="cylinder" size="0.1 0.03" euler="90 0 0" mass="1"/></body>
    </body>
  </worldbody>
</mujoco>"""


def base_model(tmp_path, xml, chassis=None, name="base.xml"):
    path = tmp_path / name
    path.write_text(xml)
    spec = robots.load_file(path, "base.xml")
    bases.take_chassis(spec, chassis, "base.xml")
    return spec.compile()


def infer(tmp_path, xml, **overrides):
    return bases.infer(base_model(tmp_path, xml), "base.xml", **overrides)


def test_differential_drive_with_a_caster(tmp_path):
    base = infer(tmp_path, DIFF)
    assert base.chassis == "chassis"
    # the caster hangs off the swivel and the swivel's axis is vertical: neither is driven
    assert [(w.joint, w.left, w.sign) for w in base.wheels] == [("left", True, -1.0), ("right", False, 1.0)]
    assert base.radius == pytest.approx(0.05)
    assert base.track == pytest.approx(0.3)
    assert base.forward == "+y" and base.yaw == 0.0
    # the wheels and the caster both reach 5 cm below the chassis origin
    assert base.lift == pytest.approx(0.05)
    assert base.mass == pytest.approx(2.22)
    center, size = base.box
    assert center == pytest.approx([0, 0, 0.08]) and size == pytest.approx([0.2, 0.3, 0.08])


def ply_points(ply: bytes) -> np.ndarray:
    head, body = ply.decode().split("end_header\n", 1)
    count = int(next(line for line in head.splitlines() if line.startswith("element vertex")).split()[-1])
    return np.array([[float(v) for v in row.split()] for row in body.splitlines()[:count]])


@pytest.mark.parametrize("xml", ["DIFF", "SKID"])
def test_the_chassis_mesh_spans_the_chassis_box_in_the_base_frame(tmp_path, xml):
    model = base_model(tmp_path, globals()[xml])
    base = bases.infer(model, "base.xml")
    points = ply_points(bases.chassis_mesh(model, base))  # metres, the viam frame (turned and lifted)
    center, size = base.box
    assert points.min(axis=0) == pytest.approx(center - size / 2, abs=1e-6)
    assert points.max(axis=0) == pytest.approx(center + size / 2, abs=1e-6)


def test_take_chassis_drops_the_free_joint_and_the_file_pose(tmp_path):
    model = base_model(tmp_path, DIFF)
    assert model.njnt == 4 and model.body_pos[1] == pytest.approx([0, 0, 0])
    assert model.body_quat[1] == pytest.approx([1, 0, 0, 0])


def test_skid_steer_with_ros_axes(tmp_path):
    base = infer(tmp_path, SKID)
    left = sorted(w.joint for w in base.wheels if w.left)
    right = sorted(w.joint for w in base.wheels if not w.left)
    assert (left, right) == (["fl", "rl"], ["fr", "rr"])  # fl, fr on the jointless axle count
    assert all(w.sign == 1.0 for w in base.wheels)
    assert base.forward == "+x" and base.yaw == pytest.approx(math.pi / 2)
    assert (base.radius, base.track, base.lift) == pytest.approx((0.1, 0.5, 0.1))
    center, size = base.box  # 0.6 long along the file's x, which becomes the viam frame's y
    assert center == pytest.approx([0, 0, 0.2]) and size == pytest.approx([0.4, 0.6, 0.1])
    # 10 kg box (0.6 x 0.4) plus four 1 kg wheels 0.32 m out, each with its own izz
    wheel_izz = 1 * (3 * 0.1**2 + 0.06**2) / 12
    assert base.izz == pytest.approx(10 * (0.6**2 + 0.4**2) / 12 + 4 * (0.2**2 + 0.25**2 + wheel_izz))


def test_forward_axis_flips_a_base_that_drives_the_other_way(tmp_path):
    base = infer(tmp_path, SKID, forward_axis="-x")
    assert base.yaw == pytest.approx(-math.pi / 2)
    assert sorted(w.joint for w in base.wheels if w.left) == ["fr", "rr"]
    assert all(w.sign == -1.0 for w in base.wheels)


def test_only_actuated_hinges_count_when_the_file_has_actuators(tmp_path):
    xml = SKID.replace("</worldbody>", "</worldbody><actuator>"
                       '<velocity joint="fl" kv="1"/><velocity joint="fr" kv="1"/></actuator>')
    assert sorted(w.joint for w in infer(tmp_path, xml).wheels) == ["fl", "fr"]


def test_wheels_override_names_the_driven_joints_and_sides(tmp_path):
    base = infer(tmp_path, SKID, wheels={"left": ["rl"], "right": ["rr"]})
    assert [(w.joint, w.left) for w in base.wheels] == [("rl", True), ("rr", False)]


def test_mesh_wheel_radius_is_its_furthest_point_from_the_axle(tmp_path):
    square = " ".join(f"{x} {y} {z}" for x in (-0.01, 0.01) for y in (-0.06, 0.06) for z in (-0.06, 0.06))
    xml = DIFF.replace("<worldbody>", f'<asset><mesh name="w" vertex="{square}"/></asset><worldbody>').replace(
        '<geom type="cylinder" size="0.05 0.01" euler="0 90 0" mass="0.1"/>', '<geom type="mesh" mesh="w"/>'
    )
    assert infer(tmp_path, xml).radius == pytest.approx(0.06 * math.sqrt(2))


def test_wheel_radius_override(tmp_path):
    assert infer(tmp_path, DIFF, wheel_radius=0.07).radius == 0.07


@pytest.mark.parametrize(
    "xml, overrides, message",
    [
        (DIFF.replace('axis="-1 0 0"', 'axis="0 0 1"'), {}, r"needs driven wheels on both sides, found left \['left'\] and right \[\] among hinge joints \['left', 'right', 'swivel', 'caster'\]; set wheels"),
        (DIFF.replace('axis="1 0 0"', 'axis="0 0 1"').replace('axis="-1 0 0"', 'axis="0 0 1"'), {}, r"found left \[\] and right \[\]"),
        (SKID.replace('<joint name="rr" axis="0 1 0"/>', '<joint name="rr" axis="1 0.3 0"/>'), {}, r"wheel axles are not parallel: .*'rr': \[0.958, 0.287, 0.0\]\}; set wheels"),
        (DIFF.replace('axis="1 0 0"', 'axis="1 1 0"').replace('axis="-1 0 0"', 'axis="-1 -1 0"'), {}, r"cannot tell forward from wheel axle \[0.707, 0.707, 0.0\]; set forward_axis"),
        (SKID, {"forward_axis": "+y"}, r"forward_axis \+y is not across the wheel axles \[0.0, 1.0, 0.0\]; set forward_axis"),
        (DIFF.replace('<geom type="cylinder" size="0.05 0.01" euler="0 90 0" mass="0.1"/>', '<geom type="box" size="0.05 0.05 0.05"/>', 1), {}, r"radius of wheels \['left'\] .*; set wheel_radius_m"),
        (SKID.replace('size="0.1 0.03"', 'size="0.2 0.03"', 1), {}, r"radii differ by more than 5%: \{'fl': 0.2, 'fr': 0.1, 'rl': 0.1, 'rr': 0.1\}; set wheel_radius_m"),
        (SKID, {"wheels": {"left": ["fl", "axle"], "right": ["fr"]}}, r"wheels \['axle'\] are not hinge joints; use names from \['fl', 'fr', 'rl', 'rr'\]"),
    ],
)
def test_inference_errors_name_the_attribute_that_fixes_them(tmp_path, xml, overrides, message):
    with pytest.raises(ValueError, match=message):
        infer(tmp_path, xml, **overrides)


def test_chassis_must_be_a_single_top_level_body(tmp_path):
    two = SKID.replace("</worldbody>", '<body name="crate"/></worldbody>')
    with pytest.raises(ValueError, match=r"has 2 top-level bodies \['base_link', 'crate'\]; set chassis"):
        base_model(tmp_path, two)
    with pytest.raises(ValueError, match=r"has 0 top-level bodies \[\]; set chassis"):
        base_model(tmp_path, "<mujoco><worldbody/></mujoco>")
    with pytest.raises(ValueError, match=r"chassis 'fl' is not a top-level body; use one of \['base_link', 'crate'\]"):
        base_model(tmp_path, two, chassis="fl")
    model = base_model(tmp_path, two, chassis="base_link")
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "crate") < 0


def test_file_extension_must_be_xml_or_urdf(tmp_path):
    with pytest.raises(ValueError, match=r"mjcf must be an .xml \(MJCF\) or .urdf file, got base.sdf"):
        base_model(tmp_path, SKID, name="base.sdf")


def test_wheel_actuators_and_speed_limits(tmp_path):
    xml = SKID.replace("</worldbody>", "</worldbody><actuator>"
                       '<velocity joint="fl" kv="5" ctrlrange="-12 12"/><velocity joint="fr" kv="5" ctrlrange="-10 10"/>'
                       '<velocity joint="rl" kv="5"/><velocity joint="rr" kv="5"/></actuator>')
    model = base_model(tmp_path, xml)
    base = bases.infer(model, "base.xml")
    assert bases.wheel_actuators(model, base, "base.xml") == {"fl": "fl", "fr": "fr", "rl": "rl", "rr": "rr"}
    linear, angular = bases.speed_limits(model, base)
    assert linear == pytest.approx(1.0)  # 10 rad/s x 0.1 m
    assert angular == pytest.approx(math.degrees(2 * 1.0 / 0.5))
    assert bases.speed_limits(base_model(tmp_path, SKID), infer(tmp_path, SKID)) is None


def test_a_non_velocity_wheel_actuator_is_an_error_for_drive_wheels(tmp_path):
    xml = SKID.replace("</worldbody>", '</worldbody><actuator><motor name="push" joint="fl"/>'
                       '<velocity joint="fr"/><velocity joint="rl"/><velocity joint="rr"/></actuator>')
    model = base_model(tmp_path, xml)
    with pytest.raises(ValueError, match=r"actuator 'push' on wheel 'fl' is not a velocity actuator"):
        bases.wheel_actuators(model, bases.infer(model, "base.xml"), "base.xml")


def test_traction_is_the_higher_priority_friction(tmp_path):
    model = base_model(tmp_path, SKID.replace('mass="1"/>', 'mass="1" friction="0.6" priority="1"/>', 1)
                       .replace('mass="1"/>', 'mass="1" friction="0.6"/>', 1))
    assert bases.traction(model, "fl") == pytest.approx(0.6)  # priority over the floor
    assert bases.traction(model, "fr") == pytest.approx(1.0)  # the floor's 1 wins the max


def test_a_urdf_loads_with_mesh_dir():
    spec = robots.load_file(URDF, "skid.urdf", mesh_dir="../../assets/meshes")
    assert bases.take_chassis(spec, None, "skid.urdf") == "base_link"  # the root link
    assert robots.density_links(spec) == ["base_link"]  # it has a geom and no <inertial>
    base = bases.infer(spec.compile(), "skid.urdf")
    # the front wheels hang off front_axle_link, a fixed link: they count
    assert sorted(w.joint for w in base.wheels if w.left) == ["front_left_wheel", "rear_left_wheel"]
    assert sorted(w.joint for w in base.wheels if not w.left) == ["front_right_wheel", "rear_right_wheel"]
    assert base.forward == "+x" and base.radius == pytest.approx(0.1)
    assert base.mass == pytest.approx(0.6 * 0.4 * 0.1 * 1000 + 0.5 + 4 + 0.3)  # base_link by density


def test_urdf_meshes_default_to_a_meshes_folder_beside_the_file(tmp_path):
    shutil.copy(URDF, tmp_path / "skid.urdf")
    with pytest.raises(
        ValueError, match=r"mesh package://skid_description/meshes/bin.stl not found as .*/bin.stl; set mesh_dir"
    ):
        robots.load_file(tmp_path / "skid.urdf", "skid.urdf")
    (tmp_path / "meshes").mkdir()
    shutil.copy(MESHES / "bin.stl", tmp_path / "meshes")
    spec = robots.load_file(tmp_path / "skid.urdf", "skid.urdf")
    assert spec.meshes[0].file == str(tmp_path / "meshes" / "bin.stl")
    spec.compile()


def test_unreadable_files_name_the_path(tmp_path):
    truncated = tmp_path / "truncated.xml"
    truncated.write_text(SKID[:80])
    with pytest.raises(ValueError, match=r"cannot load mjcf .*truncated.xml"):
        robots.load_file(truncated, "base")


def test_zero_track_is_an_error(tmp_path):
    with pytest.raises(ValueError, match=r"same lateral offset .*; set wheels"):
        infer(tmp_path, SKID, wheels={"left": ["fl"], "right": ["rl"]})


# ---- swerve -------------------------------------------------------------------

# A ROS-axes swerve base (forward +X): four modules, each a steer hinge (vertical) on a housing
# fixed to the chassis with a drive wheel under it (axle along the housing's y). The front-right
# housing is turned a quarter turn, as Mabel's are, so its wheel rolls along y at steer 0.
SWERVE = """<mujoco>
  <worldbody>
    <body name="chassis">
      <geom type="box" size="0.25 0.25 0.05" pos="0 0 0.1" mass="10"/>
      <body name="fl" pos="0.2 0.2 0"><body name="fl_steer"><joint name="fl_steer" axis="0 0 1"/>
        <geom type="box" size="0.01 0.01 0.01" mass="0.1"/>
        <body name="fl_wheel" pos="0 0 -0.02"><joint name="fl_drive" axis="0 1 0"/>
          <geom type="cylinder" size="0.05 0.02" euler="90 0 0" mass="0.5"/></body></body></body>
      <body name="fr" pos="0.2 -0.2 0" euler="0 0 90"><body name="fr_steer"><joint name="fr_steer" axis="0 0 1"/>
        <geom type="box" size="0.01 0.01 0.01" mass="0.1"/>
        <body name="fr_wheel" pos="0 0 -0.02"><joint name="fr_drive" axis="0 1 0"/>
          <geom type="cylinder" size="0.05 0.02" euler="90 0 0" mass="0.5"/></body></body></body>
      <body name="rl" pos="-0.2 0.2 0"><body name="rl_steer"><joint name="rl_steer" axis="0 0 1"/>
        <geom type="box" size="0.01 0.01 0.01" mass="0.1"/>
        <body name="rl_wheel" pos="0 0 -0.02"><joint name="rl_drive" axis="0 1 0"/>
          <geom type="cylinder" size="0.05 0.02" euler="90 0 0" mass="0.5"/></body></body></body>
      <body name="rr" pos="-0.2 -0.2 0"><body name="rr_steer"><joint name="rr_steer" axis="0 0 1"/>
        <geom type="box" size="0.01 0.01 0.01" mass="0.1"/>
        <body name="rr_wheel" pos="0 0 -0.02"><joint name="rr_drive" axis="0 1 0"/>
          <geom type="cylinder" size="0.05 0.02" euler="90 0 0" mass="0.5"/></body></body></body>
    </body>
  </worldbody>
  <actuator>
    <position joint="fl_steer" kp="50" kv="1"/><velocity joint="fl_drive" kv="2" ctrlrange="-20 20"/>
    <position joint="fr_steer" kp="50" kv="1"/><velocity joint="fr_drive" kv="2" ctrlrange="-20 20"/>
    <position joint="rl_steer" kp="50" kv="1"/><velocity joint="rl_drive" kv="2" ctrlrange="-20 20"/>
    <position joint="rr_steer" kp="50" kv="1"/><velocity joint="rr_drive" kv="2" ctrlrange="-20 20"/>
  </actuator>
</mujoco>"""


def test_a_swerve_base_is_found_from_its_actuated_steer_hinges(tmp_path):
    base = infer(tmp_path, SWERVE)
    assert base.kinematics == "swerve" and base.wheels == ()
    assert [(m.steer, m.drive) for m in base.modules] == [
        ("fl_steer", "fl_drive"), ("fr_steer", "fr_drive"), ("rl_steer", "rl_drive"), ("rr_steer", "rr_drive")]
    # forward defaults to +X (ROS): the file's +X is the viam frame's +Y, its +Y (left) is -X
    assert base.forward == "+x" and base.yaw == pytest.approx(math.pi / 2)
    positions = {m.steer: m.position for m in base.modules}
    assert positions["fl_steer"] == pytest.approx((-0.2, 0.2))
    assert positions["rr_steer"] == pytest.approx((0.2, -0.2))
    # at steer 0 a wheel rolls along its axle x up: the file's +X (viam +Y, 90 deg) for the plain
    # housings, and the turned one's along the file's +Y (viam -X, 180 deg)
    roll = {m.steer: math.degrees(m.roll0) for m in base.modules}
    assert roll["fl_steer"] == pytest.approx(90) and roll["fr_steer"] == pytest.approx(180)
    assert base.radius == pytest.approx(0.05)
    assert base.lift == pytest.approx(0.07)  # the wheels reach 7 cm below the chassis origin


def test_a_caster_doesnt_make_a_base_swerve(tmp_path):
    base = infer(tmp_path, DIFF)  # its swivel is a vertical hinge with a wheel, but unactuated
    assert base.kinematics == "differential" and base.modules == ()


def test_kinematics_overrides_the_inference(tmp_path):
    with pytest.raises(ValueError, match='kinematics "swerve" needs steer hinges'):
        infer(tmp_path, SKID, kinematics="swerve")
    plain = SWERVE.split("<actuator>")[0] + "</mujoco>"  # no actuators: not inferred, but stated
    assert len(infer(tmp_path, plain, kinematics="swerve").modules) == 4


def test_swerve_targets_forward_sideways_and_spinning(tmp_path):
    modules = {m.steer: m for m in infer(tmp_path, SWERVE).modules}
    fl, fr = modules["fl_steer"], modules["fr_steer"]
    zero = [0.0, 0.0]
    # forward (viam +Y): fl already rolls that way at 0; fr (rolling 180 deg at 0) turns -90 deg
    angles, speeds = bases.swerve_targets([fl, fr], 0.0, 1.0, 0.0, zero)
    assert np.degrees(angles) == pytest.approx([0, -90]) and speeds == pytest.approx([1, 1])
    # sideways right (viam +X): fl turns -90 deg; fr already lies along x, so it rolls backward
    angles, speeds = bases.swerve_targets([fl, fr], 1.0, 0.0, 0.0, zero)
    assert np.degrees(angles) == pytest.approx([-90, 0]) and speeds == pytest.approx([1, -1])
    # spinning left in place: each wheel at w x r, square to its position
    angles, speeds = bases.swerve_targets([fl], 0.0, 0.0, 1.0, [0.0])
    direction = fl.roll0 + fl.steer_sign * angles[0]
    # the wheel's ground velocity (either wheel direction): w x position = (-0.2, -0.2)
    assert speeds[0] * np.array([math.cos(direction), math.sin(direction)]) == pytest.approx([-0.2, -0.2])


def test_swerve_targets_flip_for_the_shorter_turn_and_hold_when_still(tmp_path):
    fl = next(m for m in infer(tmp_path, SWERVE).modules if m.steer == "fl_steer")
    # backward from a wheel at 0: reverse the wheel rather than turn it 180 deg
    angles, speeds = bases.swerve_targets([fl], 0.0, -1.0, 0.0, [0.0])
    assert angles[0] == pytest.approx(0.0) and speeds[0] == pytest.approx(-1.0)
    # from 350 deg (unwrapped), forward is 360, not 0
    angles, _ = bases.swerve_targets([fl], 0.0, 1.0, 0.0, [math.radians(350)])
    assert math.degrees(angles[0]) == pytest.approx(360)
    # stopped: keep the angle, no speed
    angles, speeds = bases.swerve_targets([fl], 0.0, 0.0, 0.0, [0.7])
    assert angles[0] == pytest.approx(0.7) and speeds[0] == 0.0
