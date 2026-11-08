"""The bundled Clearpath Husky, in both drives."""

import math
from pathlib import Path
from xml.etree import ElementTree

import pytest

from mujoco_module import bases, robots
from mujoco_module.handles import BaseHandle
from mujoco_module.odometry import OdometryHandle
from mujoco_module.sim import WorldConfig
from test_base import finish, frame_pose, settle, turned_by

HUSKY = Path(__file__).parents[1] / "assets" / "clearpath_husky" / "husky.xml"


def test_husky_inference():
    spec = robots.load_file(HUSKY, "husky.xml")
    bases.take_chassis(spec, None, "husky.xml")
    base = bases.infer(spec.compile(), "husky.xml")
    assert sorted(w.joint for w in base.wheels if w.left) == ["front_left_wheel", "rear_left_wheel"]
    assert sorted(w.joint for w in base.wheels if not w.left) == ["front_right_wheel", "rear_right_wheel"]
    assert base.forward == "+x"
    assert (base.radius, base.track) == pytest.approx((0.1651, 0.555))
    assert base.lift == pytest.approx(0.1651 - 0.03282, abs=1e-6)  # base_link above the floor
    # the mass is the file's inertials (geoms carry none)
    inertials = ElementTree.parse(HUSKY).getroot().iter("inertial")
    assert base.mass == pytest.approx(sum(float(i.get("mass")) for i in inertials))


@pytest.fixture(params=["ideal", "wheels"])
def husky(sim, request):
    sim.configure_world(WorldConfig.from_attrs({"props": []}))
    return BaseHandle(sim, "husky", {"asset": "clearpath_husky", "drive": request.param})


def test_husky_spawns_on_its_wheels(sim, husky):
    sim.advance(1.0)
    site = sim.model.site("husky/viam_frame").id
    assert sim.data.site_xpos[site] == pytest.approx([0, 0, 0], abs=0.001)
    for wheel in ("front_left", "rear_right"):
        body = sim.model.body(f"husky/{wheel}_wheel_link").id
        assert sim.data.xpos[body][2] == pytest.approx(0.1651, abs=0.001)
    assert not husky.is_moving()


def test_husky_moves_straight_and_spins(sim, husky):
    finish(sim, husky.move_straight(1.0, 0.5)).result()
    settle(sim)
    x, y, yaw = frame_pose(sim, "husky")
    assert (x, y) == pytest.approx((0.0, 1.0), abs=0.02)
    assert yaw == pytest.approx(0.0, abs=math.radians(2))
    for target in (math.radians(90), math.radians(-90), math.radians(270)):
        before = frame_pose(sim, "husky")[2]
        finish(sim, husky.spin(target, math.radians(90))).result()
        settle(sim)
        assert turned_by(before, frame_pose(sim, "husky")[2], target) == pytest.approx(target, abs=math.radians(2))


def test_a_skid_steer_husky_under_turns_open_loop(sim):
    sim.configure_world(WorldConfig.from_attrs({"props": []}))
    husky = BaseHandle(sim, "husky", {"asset": "clearpath_husky", "drive": "wheels"})
    husky.set_velocity(0.0, 1.0)
    sim.advance(2.0)
    turned = frame_pose(sim, "husky")[2]
    assert 0.05 < turned < 1.0  # well short of the 2 rad commanded: the wheels scrub sideways


def test_husky_odometry_tracks_the_wheel_driven_chassis(sim):
    sim.configure_world(WorldConfig.from_attrs({"props": []}))
    husky = BaseHandle(sim, "husky", {"asset": "clearpath_husky", "drive": "wheels"})
    sensor = OdometryHandle(sim, "odometry", {"base": "husky"})
    finish(sim, husky.move_straight(0.8, 0.5)).result()
    finish(sim, husky.spin(math.radians(90), math.radians(60))).result()
    finish(sim, husky.move_straight(0.5, 0.5)).result()
    settle(sim)
    reading = sensor.read()
    x, y, yaw = frame_pose(sim, "husky")
    assert (reading.x, reading.y) == pytest.approx((x, y), abs=0.01)
    assert reading.yaw == pytest.approx(yaw, abs=math.radians(1))


def test_a_wall_stalls_the_wheel_driven_husky(sim):
    sim.configure_world(WorldConfig.from_attrs({"props": [
        {"name": "wall", "fixed": True, "size": 1.0, "scale": [2, 0.04, 0.5], "position": [0, 1.5, 0.25]},
    ]}))
    husky = BaseHandle(sim, "husky", {"asset": "clearpath_husky", "drive": "wheels", "move_timeout_margin_sec": 0.5})
    future = finish(sim, husky.move_straight(3.0, 0.5))
    with pytest.raises(RuntimeError, match="stalled"):
        future.result()
    # the chassis box reaches 0.4937 m ahead of base_link: it stops at the wall's face (1.48)
    assert frame_pose(sim, "husky")[1] == pytest.approx(1.48 - 0.4937, abs=0.02)
