import asyncio

import mujoco
import pytest
from viam.proto.app.robot import ComponentConfig, Frame, Orientation, Translation
from viam.proto.common import Capsule, Geometry, Mesh, Pose, RectangularPrism, Sphere, Vector3
from viam.utils import dict_to_struct

from mujoco_module.handles import BaseHandle
from mujoco_module.models.obstacle import MujocoObstacle, obstacle_contribution
from test_base import finish, pose


def config(geometry=None, parent="world", attrs=None, translation=(0, 0, 0), yaw_deg=None, frame=True):
    cfg = ComponentConfig(name="table", attributes=dict_to_struct({"world": "sim-world", **(attrs or {})}))
    if frame:
        f = Frame(parent=parent, translation=Translation(x=translation[0], y=translation[1], z=translation[2]))
        if yaw_deg is not None:
            f.orientation.CopyFrom(Orientation(vector_degrees=Orientation.OrientationVectorDegrees(z=1, theta=yaw_deg)))
        if geometry is not None:
            f.geometry.CopyFrom(geometry)
        cfg.frame.CopyFrom(f)
    return cfg


BOX = Geometry(box=RectangularPrism(dims_mm=Vector3(x=600, y=800, z=20)))


def test_box_obstacle_builds_at_frame_and_center_pose(sim):
    geometry = Geometry(
        center=Pose(z=-10, o_z=1), box=RectangularPrism(dims_mm=Vector3(x=600, y=800, z=20))
    )
    sim.set_contribution("table", obstacle_contribution(config(geometry, translation=(400, 0, 300), yaw_deg=90)))
    geom = sim.model.geom("table/obstacle")
    assert geom.type[0] == mujoco.mjtGeom.mjGEOM_BOX
    assert tuple(geom.size) == pytest.approx((0.3, 0.4, 0.01))
    assert sim.data.geom("table/obstacle").xpos == pytest.approx([0.4, 0, 0.29])
    # yawed 90 degrees: the geom's local X points along world Y
    assert sim.data.geom("table/obstacle").xmat.reshape(3, 3)[:, 0] == pytest.approx([0, 1, 0], abs=1e-9)
    assert sim.model.body("table/obstacle").jntnum[0] == 0  # fixed


def test_center_offset_and_rotation_compose_with_the_frame(sim):
    geometry = Geometry(
        center=Pose(x=100, z=-10, o_x=1, theta=0), box=RectangularPrism(dims_mm=Vector3(x=100, y=200, z=20))
    )
    sim.set_contribution("table", obstacle_contribution(config(geometry, translation=(400, 0, 300), yaw_deg=90)))
    geom = sim.data.geom("table/obstacle")
    # frame +X is world +Y after the yaw, so the 100 mm offset along frame X lands at y = 0.1
    assert geom.xpos == pytest.approx([0.4, 0.1, 0.29])
    # the geom's Z axis points along the frame's X, which is world +Y
    assert geom.xmat.reshape(3, 3)[:, 2] == pytest.approx([0, 1, 0], abs=1e-9)


def test_sphere_and_capsule_obstacles(sim):
    ball = config(Geometry(sphere=Sphere(radius_mm=50)))
    ball.name = "ball"
    sim.set_contribution("ball", obstacle_contribution(ball))
    assert sim.model.geom("ball/obstacle").type[0] == mujoco.mjtGeom.mjGEOM_SPHERE
    capsule = Geometry(capsule=Capsule(radius_mm=50, length_mm=300))
    cfg = config(capsule)
    cfg.name = "post"
    sim.set_contribution("post", obstacle_contribution(cfg))
    assert tuple(sim.model.geom("post/obstacle").size[:2]) == pytest.approx((0.05, 0.1))


def test_rover_stalls_against_an_obstacle_and_reset_leaves_it(sim):
    wall = Geometry(box=RectangularPrism(dims_mm=Vector3(x=2000, y=40, z=300)))
    wall_cfg = config(wall, translation=(0, 1000, 150))
    wall_cfg.name = "wall"
    sim.set_contribution("wall", obstacle_contribution(wall_cfg))
    rover = BaseHandle(sim, "rover", {"asset": "viam_rover"})
    future = rover.move_straight(1.5, 0.3)
    with pytest.raises(RuntimeError, match="stalled"):
        finish(sim, future).result()
    assert pose(sim)[1] < 1.0 - 0.02 - 0.15 + 0.01  # chassis front stays outside the wall
    sim.reset()
    assert sim.data.geom("wall/obstacle").xpos == pytest.approx([0, 1.0, 0.15])


@pytest.mark.parametrize("cfg,message", [
    (config(BOX, frame=False), "frame"),
    (config(BOX, parent="pick-arm"), "frame.parent must be"),
    (config(None), "set frame.geometry to a box, sphere or capsule"),
    (config(Geometry(mesh=Mesh(content_type="stl", mesh=b"x"))), "mesh obstacles aren't supported yet"),
    (config(Geometry(box=RectangularPrism(dims_mm=Vector3(x=0, y=1, z=1)))), "must be positive"),
    (config(BOX, attrs={"color": [1, 0]}), "color"),
    (config(BOX, attrs={"static_friction": -1}), "static_friction"),
])
def test_validate_rejects(cfg, message):
    with pytest.raises(ValueError, match=message):
        MujocoObstacle.validate_config(cfg)


def test_validate_and_geometries(running_sim):
    from mujoco_module.models.world import MujocoWorld

    assert MujocoObstacle.validate_config(config(BOX)) == (["sim-world"], [])
    MujocoWorld.new(ComponentConfig(name="sim-world", attributes=dict_to_struct({"real_time": False})), {})
    obstacle = MujocoObstacle.new(config(BOX, translation=(400, 0, 10)), {})
    try:
        (geometry,) = asyncio.run(obstacle.get_geometries())
        assert geometry.label == "table"
        assert geometry.box.dims_mm.x == pytest.approx(600)
        assert running_sim.run(lambda: running_sim.model.geom("table/obstacle").id) >= 0
    finally:
        asyncio.run(obstacle.close())
    assert running_sim.run(lambda: mujoco.mj_name2id(
        running_sim.model, mujoco.mjtObj.mjOBJ_GEOM, "table/obstacle")) == -1
