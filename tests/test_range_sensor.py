import asyncio
import math

import numpy as np
import pytest
from viam.proto.app.robot import ComponentConfig, Frame, Translation
from viam.utils import dict_to_struct

from mujoco_module.handles import BaseHandle, RangeSensorHandle, cone_rays
from mujoco_module.models.range_sensor import MujocoRangeSensor
from mujoco_module.sim import WorldConfig
from test_lidar import wall_ahead_x

FORWARD = [0.70711, 0.0, 0.0, 0.70711]  # +X onto the rover's forward, +Y
# the example's right bumper: 85 deg about Z (the rover's forward, turned 5 deg to its right,
# +X), then -12 deg about the sensor's own Y, so the 10 deg cone's lowest ray is 2 deg above level
BUMPER_UP = [0.73324, 0.07062, -0.07707, 0.67189]
README_MOUNT = [0.70323, 0.07391, -0.07391, 0.70323]
BUMPER_DOWN = [0.73324, -0.07062, 0.07707, 0.67189]  # the same, pitched +12 deg: wrong sign


def props(sim, *items):
    sim.configure_world(WorldConfig.from_attrs({"props": list(items)}))


def pole(name, x, y, radius=0.015):
    return {"name": name, "type": "cylinder", "fixed": True, "size": 1.0,
            "scale": [2 * radius, 2 * radius, 0.6], "position": [x, y, 0.3]}


def test_cone_rays_layout():
    rays = cone_rays(20.0, 2, 8)
    assert rays.shape == (17, 3)
    assert np.linalg.norm(rays, axis=1) == pytest.approx(np.ones(17))
    angles = np.degrees(np.arccos(rays[:, 0]))
    assert angles[0] == pytest.approx(0)
    assert angles[1:9] == pytest.approx(np.full(8, 5.0))
    assert angles[9:] == pytest.approx(np.full(8, 10.0))
    assert rays[1] == pytest.approx([math.cos(math.radians(5)), math.sin(math.radians(5)), 0])
    assert cone_rays(0.0, 2, 8).tolist() == [[1.0, 0.0, 0.0]]


def test_a_single_ray_reads_the_distance_to_a_wall(sim):
    wall_ahead_x(sim)
    sensor = RangeSensorHandle(sim, "range", {"position": [0, 0, 0.2], "cone_deg": 0})
    assert sensor.readings() == {"distance": pytest.approx(1.0, abs=1e-3)}


def test_the_cone_catches_a_thin_pole_off_axis(sim):
    props(sim, pole("pole", 1.0, 0.17))
    single = RangeSensorHandle(sim, "single", {"position": [0, 0, 0.2], "cone_deg": 0})
    cone = RangeSensorHandle(sim, "cone", {"position": [0, 0, 0.2], "cone_deg": 20})
    assert single.readings() == {"distance": 4.0}
    assert cone.readings()["distance"] == pytest.approx(1.0, abs=0.03)


def test_the_nearest_hit_wins_and_a_too_close_hit_reads_range_min(sim):
    props(sim, pole("near", 0.5, 0.0), {"name": "wall", "fixed": True, "size": 1.0,
                                        "scale": [0.04, 2, 0.6], "position": [1.02, 0, 0.3]})
    sensor = RangeSensorHandle(sim, "range", {"position": [0, 0, 0.2]})
    assert sensor.readings()["distance"] == pytest.approx(0.485, abs=0.01)
    # the pole, nearer than range_min_m, hides the wall: too close to measure, never an open path
    far = RangeSensorHandle(sim, "far", {"position": [0, 0, 0.2], "range_min_m": 0.6})
    assert far.readings() == {"distance": 0.6}


def test_nothing_in_range_reads_range_max(sim):
    # from 0.2 m up the cone's lowest ray (10 deg down) meets the floor at 1.15 m; within 1 m
    # it passes over the fixture's 3 cm block
    sensor = RangeSensorHandle(sim, "range", {"position": [0, 0, 0.2], "range_max_m": 1.0})
    assert sensor.readings() == {"distance": 1.0}


def test_on_the_rover_it_ignores_the_chassis_and_moves_with_it(sim):
    props(sim, {"name": "wall", "fixed": True, "size": 1.0, "scale": [2, 0.04, 0.6],
                "position": [0, 1.02, 0.3]})
    BaseHandle(sim, "rover", {"asset": "viam_rover"})
    # inside the shell: only hiding the rover's geoms keeps it out. A single ray: 8 cm up, a
    # cone would read the floor first
    sensor = RangeSensorHandle(sim, "range", {"parent": "rover", "position": [0, 0, 0.08],
                                              "orientation_wxyz": FORWARD, "cone_deg": 0})
    assert sensor.readings()["distance"] == pytest.approx(1.0, abs=0.01)


def test_the_example_bumper_mount_does_not_see_the_floor(sim):
    BaseHandle(sim, "rover", {"asset": "viam_rover"})
    up = RangeSensorHandle(sim, "up", {"parent": "rover", "position": [0.06, 0.155, 0.08],
                                       "orientation_wxyz": BUMPER_UP, "range_max_m": 1.5})
    down = RangeSensorHandle(sim, "down", {"parent": "rover", "position": [0.06, 0.155, 0.08],
                                           "orientation_wxyz": BUMPER_DOWN, "range_max_m": 1.5})
    assert up.readings() == {"distance": 1.5}
    assert down.readings()["distance"] < 0.5  # the floor: the test can tell the signs apart


def test_the_readme_mount_does_not_see_the_floor(sim):
    BaseHandle(sim, "rover", {"asset": "viam_rover"})
    # 90 deg about Z, then -12 deg about the sensor's own Y: README.md's range_sensor example
    sensor = RangeSensorHandle(sim, "range", {"parent": "rover", "position": [0, 0.155, 0.08],
                                              "orientation_wxyz": README_MOUNT, "range_max_m": 1.5})
    assert sensor.readings() == {"distance": 1.5}


def test_it_waits_for_its_parent(sim):
    sensor = RangeSensorHandle(sim, "range", {"parent": "rover", "position": [0, 0, 0.08],
                                              "cone_deg": 0, "range_max_m": 1.0})
    with pytest.raises(RuntimeError, match="not in the scene"):
        sensor.readings()
    BaseHandle(sim, "rover", {"asset": "viam_rover"})
    assert sensor.readings() == {"distance": 1.0}


def config(attrs, parent="world", y=0, z=0):
    cfg = ComponentConfig(name="range", attributes=dict_to_struct({"world": "sim-world", **attrs}))
    cfg.frame.CopyFrom(Frame(parent=parent, translation=Translation(y=y, z=z)))
    return cfg


@pytest.mark.parametrize("attrs, field", [
    ({"cone_deg": 180}, "cone_deg"),
    ({"cone_deg": -1}, "cone_deg"),
    ({"rings": 0}, "rings"),
    ({"cone_deg": 0, "rings": 0}, "rings"),
    ({"rays_per_ring": 2.5}, "rays_per_ring"),
    ({"range_min_m": -0.1}, "range_min_m"),
    ({"range_max_m": 0}, "range_max_m"),
    ({"range_min_m": 2, "range_max_m": 1}, "range_max_m"),
])
def test_validation_names_the_field(attrs, field):
    with pytest.raises(ValueError, match=field):
        MujocoRangeSensor.validate_config(config(attrs))


def test_validation_accepts_defaults_and_a_single_ray():
    assert MujocoRangeSensor.validate_config(config({}, parent="rover")) == (["sim-world"], [])
    assert MujocoRangeSensor.validate_config(config({"cone_deg": 0})) == (["sim-world"], [])


def test_the_model_serves_readings(running_sim):
    from mujoco_module.models.world import MujocoWorld

    MujocoWorld.new(ComponentConfig(name="sim-world", attributes=dict_to_struct({"real_time": False})), {})
    # the cone meets the floor at 1.15 m from 0.2 m up; read within 1 m
    sensor = MujocoRangeSensor.new(config({"range_max_m": 1.0}, z=200), {})

    async def run():
        assert await sensor.get_readings() == {"distance": 1.0}
        assert await sensor.get_geometries() == []
        assert await sensor.do_command({"anything": 1}) == {}
        await sensor.close()

    asyncio.run(run())
