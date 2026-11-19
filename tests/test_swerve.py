"""A swerve base: each wheel on its own steering hinge, so it drives sideways and
diagonally as well as forward and turning (bases.py infers the modules)."""

import math

import pytest
from test_base import finish, pose, settle
from test_bases import SWERVE

from mujoco_module.handles import BaseHandle


@pytest.fixture
def swerve(sim, tmp_path):
    path = tmp_path / "swerve.xml"
    path.write_text(SWERVE)
    return BaseHandle(sim, "swerve", {"mjcf": str(path)})


def joint(sim, name):
    return float(sim.data.qpos[sim.model.joint(f"swerve/{name}").qposadr[0]])


def test_speed_limits_come_from_the_drive_actuators(swerve):
    # 20 rad/s on 50 mm wheels; turning, the corners (0.28 m out) are the fastest wheels
    assert swerve.max_linear_mps == pytest.approx(1.0)
    assert swerve.max_angular_rad_s == pytest.approx(1.0 / math.hypot(0.2, 0.2))


def test_ideal_swerve_drives_sideways_steering_its_modules(sim, swerve):
    swerve.set_velocity(0.0, 0.0, lateral_mps=0.3)
    sim.advance(1.0)
    swerve.stop()
    settle(sim)
    x, y, yaw = pose(sim, "swerve")
    assert x == pytest.approx(0.3, abs=0.02) and y == pytest.approx(0.0, abs=1e-3) and yaw == pytest.approx(0.0, abs=1e-3)
    # every module steered so its wheel moves the base along +X (one of the two wheel
    # directions, whichever was the shorter turn), and rolled the distance travelled
    for module in swerve.base.modules:
        angle, rolled = joint(sim, module.steer), joint(sim, module.drive)
        direction = module.roll0 + module.steer_sign * angle
        assert math.copysign(1, rolled) * math.cos(direction) == pytest.approx(1, abs=1e-3), module.steer
        assert abs(rolled) == pytest.approx(x / 0.05, rel=0.02)
    assert not swerve.is_moving()


def test_ideal_swerve_drives_diagonally_and_spins(sim, swerve):
    swerve.set_velocity(0.2, 0.0, lateral_mps=-0.2)  # forward and left
    sim.advance(1.0)
    swerve.stop()
    settle(sim)
    x, y, _ = pose(sim, "swerve")
    assert (x, y) == pytest.approx((-0.2, 0.2), abs=0.02)
    finish(sim, swerve.spin(math.radians(90), math.radians(60))).result()
    settle(sim)
    assert pose(sim, "swerve")[2] == pytest.approx(math.radians(90), abs=math.radians(2))


def test_a_differential_base_rejects_sideways(sim):
    rover = BaseHandle(sim, "rover", {"asset": "viam_rover"})
    with pytest.raises(ValueError, match="sideways motion needs a swerve base"):
        rover.set_velocity(0.0, 0.0, lateral_mps=0.1)


def test_a_still_swerve_base_keeps_its_steer_angles(sim, swerve):
    sim.advance(1.0)  # settling at spawn: tiny motions must not swing the modules
    assert [joint(sim, m.steer) for m in swerve.base.modules] == pytest.approx([0.0] * 4, abs=1e-9)
    swerve.set_velocity(0.0, 0.0, lateral_mps=0.3)
    sim.advance(0.5)
    swerve.stop()
    angles = [joint(sim, m.steer) for m in swerve.base.modules]
    sim.advance(1.0)  # stopping keeps them where they were
    assert [joint(sim, m.steer) for m in swerve.base.modules] == pytest.approx(angles)


def test_a_blocked_swerve_base_wheels_roll_only_as_far_as_it_moves(sim, tmp_path):
    from test_base import wall_world

    wall_world(sim, y=0.5)  # across the path ahead
    path = tmp_path / "swerve.xml"
    path.write_text(SWERVE)
    swerve = BaseHandle(sim, "swerve", {"mjcf": str(path)})
    swerve.set_velocity(0.5, 0.0)
    sim.advance(3.0)
    travelled = pose(sim, "swerve")[1]
    assert travelled < 0.4  # stopped at the wall
    for module in swerve.base.modules:
        assert abs(joint(sim, module.drive)) == pytest.approx(travelled / 0.05, rel=0.05)


# ---- drive "wheels": the steer servos and drive motors move it through friction ----------


@pytest.fixture
def driven(sim, tmp_path):
    path = tmp_path / "swerve.xml"
    path.write_text(SWERVE)
    return BaseHandle(sim, "swerve", {"mjcf": str(path), "drive": "wheels"})


def chassis_xy_yaw(sim):
    site = sim.data.site("swerve/viam_frame")
    forward = site.xmat.reshape(3, 3)[:, 1]
    return float(site.xpos[0]), float(site.xpos[1]), math.atan2(-forward[0], forward[1])


def test_wheels_swerve_drives_forward_and_sideways_through_friction(sim, driven):
    # once under way (the wheels slip for a few tenths of a second from a standstill),
    # it moves at the commanded speed, in the commanded direction
    for forward, lateral in ((0.3, 0.0), (0.0, 0.3), (0.2, -0.2)):
        driven.set_velocity(forward, 0.0, lateral_mps=lateral)
        sim.advance(1.0)
        x0, y0, _ = chassis_xy_yaw(sim)
        sim.advance(1.0)
        x1, y1, yaw = chassis_xy_yaw(sim)
        # in the base's own frame (+X right, +Y forward)
        c, s_ = math.cos(yaw), math.sin(yaw)
        right, ahead = (x1 - x0) * c + (y1 - y0) * s_, -(x1 - x0) * s_ + (y1 - y0) * c
        assert (right, ahead) == pytest.approx((lateral, forward), abs=0.03)
        assert abs(yaw) < math.radians(5)
        driven.stop()
        sim.advance(0.5)
    assert not driven.is_moving()


def test_wheels_swerve_finishes_moves(sim, driven):
    finish(sim, driven.move_straight(0.4, 0.3)).result()
    settle(sim)
    x, y, _ = chassis_xy_yaw(sim)
    assert y == pytest.approx(0.4, abs=0.03) and abs(x) < 0.03
    finish(sim, driven.spin(math.radians(90), math.radians(60))).result()
    settle(sim)
    assert chassis_xy_yaw(sim)[2] == pytest.approx(math.radians(90), abs=math.radians(5))


def test_wheels_swerve_adds_missing_actuators(sim, tmp_path):
    path = tmp_path / "plain.xml"
    path.write_text(SWERVE.split("<actuator>")[0] + "</mujoco>")
    base = BaseHandle(sim, "swerve", {"mjcf": str(path), "drive": "wheels", "kinematics": "swerve",
                                      "max_linear_mps": 0.5, "max_angular_dps": 90})
    names = {sim.model.actuator(a).name for a in range(sim.model.nu)}
    assert {"swerve/fl_steer_servo", "swerve/fl_drive_drive"} <= names
    base.set_velocity(0.0, 0.0, lateral_mps=-0.3)
    sim.advance(2.0)
    assert chassis_xy_yaw(sim)[0] == pytest.approx(-0.6, abs=0.12)
