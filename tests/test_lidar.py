import math

import mujoco
import numpy as np
import pytest

from mujoco_module.handles import BaseHandle, LidarHandle
from mujoco_module.sim import Contribution, WorldConfig


def wall_ahead_x(sim, face=1.0):
    """A tall fixed wall whose near face is at x = face."""
    sim.configure_world(WorldConfig.from_attrs({"props": [
        {"name": "wall", "fixed": True, "size": 1.0, "scale": [0.04, 2, 0.6],
         "position": [face + 0.02, 0, 0.3]},
    ]}))


def test_world_lidar_sees_a_wall(sim):
    wall_ahead_x(sim)
    lidar = LidarHandle(sim, "lidar", {"position": [0, 0, 0.2]})
    points = lidar.scan()
    assert points.dtype == np.float32 and points.shape[1] == 3
    assert np.all(points[:, 2] == 0)
    ahead = points[np.argmin(np.abs(points[:, 1]))]
    assert ahead == pytest.approx([1.0, 0.0, 0.0], abs=0.01)
    # the wall spans y in [-1, 1] at x = 1: about 45° either side at 1° steps
    assert 85 <= len(points) <= 95
    angles = np.sort(np.degrees(np.arctan2(points[:, 1], points[:, 0])))
    assert np.diff(angles) == pytest.approx(1.0, abs=1e-3)


def test_nothing_in_range_gives_an_empty_cloud(sim):
    # the fixture's block is 3 cm tall; horizontal rays at 20 cm miss it and the floor
    assert len(LidarHandle(sim, "lidar", {"position": [0, 0, 0.2]}).scan()) == 0


def test_range_limits_and_resolution(sim):
    wall_ahead_x(sim)
    assert len(LidarHandle(sim, "near", {"position": [0, 0, 0.2], "range_max_m": 0.9}).scan()) == 0
    assert len(LidarHandle(sim, "far", {"position": [0, 0, 0.2], "range_min_m": 1.5}).scan()) == 0
    coarse = LidarHandle(sim, "coarse", {"position": [0, 0, 0.2], "angular_resolution_deg": 5})
    assert 17 <= len(coarse.scan()) <= 19


def test_lidar_on_the_rover_moves_with_it_and_ignores_the_chassis(sim):
    sim.configure_world(WorldConfig.from_attrs({"props": [
        {"name": "wall", "fixed": True, "size": 1.0, "scale": [2, 0.04, 0.6],
         "position": [0, 1.02, 0.3]},
    ]}))
    rover = BaseHandle(sim, "rover", {"asset": "viam_rover"})
    # mounted inside the chassis box: only the body exclusion keeps it out of the scan
    lidar = LidarHandle(sim, "lidar", {"parent": "rover", "position": [0, 0, 0.08]})
    points = lidar.scan()
    assert np.min(np.linalg.norm(points, axis=1)) > 0.5
    forward = points[np.argmin(np.abs(points[:, 0]))]  # sensor +Y is rover-forward
    assert forward[1] == pytest.approx(1.0, abs=0.01)
    rover.set_velocity(0.25, 0.0)
    sim.advance(2.0)
    rover.stop()
    sim.advance(0.1)
    y = float(sim.data.qpos[sim.model.joint("rover/y").qposadr[0]])
    forward = lidar.scan()
    forward = forward[np.argmin(np.abs(forward[:, 0]))]
    assert forward[1] == pytest.approx(1.0 - y, abs=0.01)


def test_lidar_waits_for_its_parent(sim):
    lidar = LidarHandle(sim, "lidar", {"parent": "rover", "position": [0, 0, 0.16]})
    with pytest.raises(RuntimeError, match="not in the scene"):
        lidar.scan()
    BaseHandle(sim, "rover", {"asset": "viam_rover"})
    assert lidar.scan().shape[1] == 3


def test_ray_zero_points_along_sensor_x(sim):
    wall_ahead_x(sim)
    # a quarter turn about Z points sensor +X at world +Y, away from the wall
    turned = LidarHandle(sim, "turned", {
        "position": [0, 0, 0.2],
        "orientation_wxyz": [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)],
    })
    points = turned.scan()
    ahead = points[np.argmin(np.abs(points[:, 0]))]  # the wall is now along sensor -Y
    assert ahead == pytest.approx([0.0, -1.0, 0.0], abs=0.01)


def test_a_mounted_lidar_sees_past_its_parents_child_bodies(sim):
    wall_ahead_x(sim)

    def build(spec):
        # a parent with a frame site and, along +X, a jointed child with a child of its own
        body = spec.worldbody.add_body(name="probe", pos=[0, 0, 0.1])
        body.add_site(name="probe/viam_frame")
        fin = body.add_body(name="probe/fin")
        fin.add_joint(name="probe/fin", axis=[0, 0, 1])
        fin.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.05, 0.05], pos=[0.3, 0, 0])
        tip = fin.add_body(name="probe/tip", pos=[0.6, 0, 0])
        tip.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.05, 0.05])

    sim.set_contribution("probe", Contribution(build=build))
    points = LidarHandle(sim, "lidar", {"parent": "probe"}).scan()
    ahead = points[np.argmin(np.abs(points[:, 1]))]
    assert ahead == pytest.approx([1.0, 0.0, 0.0], abs=0.01)
    assert np.min(np.linalg.norm(points, axis=1)) > 0.9
