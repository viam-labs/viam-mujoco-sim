import math

import pytest

from mujoco_module.handles import BaseHandle
from mujoco_module.odometry import OdometryHandle, odometry_geo_point
from test_base import finish, pose


@pytest.fixture
def rover(sim):
    return BaseHandle(sim, "rover", {"asset": "viam_rover"})


def odom(sim, **attrs):
    return OdometryHandle(sim, "odometry", {"base": "rover", **attrs})


def test_exact_odometry_tracks_the_rover(sim, rover):
    sensor = odom(sim)
    finish(sim, rover.move_straight(0.5, 0.2)).result()
    finish(sim, rover.spin(math.radians(90), math.radians(90))).result()
    rover.set_velocity(0.2, math.radians(30))
    sim.advance(1.5)
    rover.stop()
    sim.advance(0.3)
    reading = sensor.read()
    x, y, yaw = pose(sim)
    assert (reading.x, reading.y) == pytest.approx((x, y), abs=0.005)
    assert reading.yaw == pytest.approx(yaw, abs=math.radians(0.5))


def test_velocities_are_in_the_body_frame(sim, rover):
    sensor = odom(sim)
    finish(sim, rover.spin(math.radians(90), math.radians(90))).result()
    rover.set_velocity(0.3, 0.0)
    sim.advance(0.5)
    reading = sensor.read()
    # facing world -X now, but the body frame is still +Y forward
    assert (reading.lateral, reading.forward) == pytest.approx((0.0, 0.3), abs=1e-3)
    assert reading.yaw_rate == pytest.approx(0.0, abs=1e-3)
    rover.set_velocity(0.0, math.radians(60))
    sim.advance(0.5)
    assert sensor.read().yaw_rate == pytest.approx(math.radians(60), abs=1e-3)


def test_scale_error_stretches_distance(sim, rover):
    sensor = odom(sim, linear_scale_error=0.02)
    finish(sim, rover.move_straight(1.0, 0.25)).result()
    sim.advance(0.3)
    reading = sensor.read()
    assert pose(sim)[1] == pytest.approx(1.0, abs=0.01)
    assert reading.y == pytest.approx(1.02 * pose(sim)[1], abs=0.002)


def test_heading_drift_accumulates_while_parked(sim, rover):
    sensor = odom(sim, heading_drift_dps=1.0)
    sim.advance(2.0)
    reading = sensor.read()
    assert reading.yaw == pytest.approx(math.radians(2.0), abs=math.radians(0.01))
    assert reading.yaw_rate == pytest.approx(math.radians(1.0))
    assert (reading.x, reading.y) == pytest.approx((0.0, 0.0), abs=1e-6)


def test_odometry_starts_at_zero_facing_the_rovers_forward(sim):
    rotated = BaseHandle(
        sim, "rover", {"asset": "viam_rover", "position": [1.0, -2.0, 0.0],
                       "orientation_wxyz": [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]},
    )
    sensor = odom(sim)
    finish(sim, rotated.move_straight(0.4, 0.2)).result()
    sim.advance(0.3)
    reading = sensor.read()
    assert (reading.x, reading.y, reading.yaw) == pytest.approx((0.0, 0.4, 0.0), abs=0.005)


def test_integration_does_not_depend_on_read_rate(sim, rover):
    polled, unpolled = odom(sim), OdometryHandle(sim, "other", {"base": "rover"})
    rover.set_velocity(0.3, math.radians(40))
    for _ in range(100):
        sim.advance(0.01)
        polled.read()
    a, b = polled.read(), unpolled.read()
    assert (a.x, a.y, a.yaw) == (b.x, b.y, b.yaw)


def test_reset_and_reset_odometry_zero_the_pose(sim, rover):
    sensor = odom(sim)
    rover.set_velocity(0.3, 0.5)
    sim.advance(1.0)
    assert sensor.read().y != 0.0
    sensor.reset_odometry()
    reading = sensor.read()
    assert (reading.x, reading.y, reading.yaw) == (0.0, 0.0, 0.0)
    sim.advance(0.5)
    sim.reset()
    reading = sensor.read()
    assert (reading.x, reading.y, reading.yaw) == (0.0, 0.0, 0.0)


def test_missing_base_errors_until_it_is_added(sim):
    sensor = odom(sim)
    with pytest.raises(RuntimeError, match="base 'rover' is not in the scene"):
        sensor.read()
    sim.advance(0.1)  # the callback must survive steps without a base
    assert "odom:odometry" in sim.step_callbacks
    rover = BaseHandle(sim, "rover", {"asset": "viam_rover"})
    rover.set_velocity(0.2, 0.0)
    sim.advance(0.5)
    assert sensor.read().forward == pytest.approx(0.2, abs=1e-3)


def test_close_detaches(sim, rover):
    sensor = odom(sim)
    sensor.close()
    assert "odom:odometry" not in sim.step_callbacks
    assert "odometry" not in sim._contributions


def test_geo_point_matches_rdk_wheeled_odometry():
    assert odometry_geo_point(0.0, 0.0) == (0.0, 0.0)
    # 1 km along +Y (bearing 0, north): latitude moves 1/6371 rad
    lat, lng = odometry_geo_point(0.0, 1000.0)
    assert lat == pytest.approx(math.degrees(1 / 6371), rel=1e-9)
    assert lng == pytest.approx(0.0, abs=1e-12)
    # 1 km along +X (bearing 90°, east): longitude moves, latitude doesn't
    lat, lng = odometry_geo_point(1000.0, 0.0)
    assert lat == pytest.approx(0.0, abs=1e-12)
    assert lng == pytest.approx(math.degrees(1 / 6371), rel=1e-9)
