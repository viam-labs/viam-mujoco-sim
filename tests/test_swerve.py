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
