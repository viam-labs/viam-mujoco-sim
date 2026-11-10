import asyncio

import mujoco
import pytest
from viam.proto.app.robot import ComponentConfig, Frame, Orientation, Translation
from viam.proto.common import Capsule, Geometry, Mesh, Pose, RectangularPrism, Sphere, Vector3
from viam.utils import dict_to_struct

from mujoco_module.handles import BaseHandle
from mujoco_module.models.obstacle import MujocoObstacle, obstacle_contribution
from test_base import finish, pose
from test_meshes import box_piece
from test_textures import blue, grey, halves, look_down, red, untinted, write_image


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
    (config(None), "set frame.geometry to a box, sphere or capsule, or set mesh"),
    (config(Geometry(mesh=Mesh(content_type="stl", mesh=b"x"))), "set the mesh attribute instead"),
    (config(Geometry(box=RectangularPrism(dims_mm=Vector3(x=0, y=1, z=1)))), "must be positive"),
    (config(BOX, attrs={"color": [1, 0]}), "color"),
    (config(BOX, attrs={"static_friction": -1}), "static_friction"),
    (config(BOX, attrs={"mesh": "/a.stl"}), "not both"),
    (config(BOX, attrs={"scale": 2}), "only apply with mesh"),
    (config(None, attrs={"mesh": "/a.step"}), r"\.stl, \.obj or \.ply"),
    (config(None, attrs={"mesh": "/a.stl", "threshold": 0}), "threshold"),
    (config(None, attrs={"mesh": "/a.stl", "scale": -1}), "scale"),
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


def test_mesh_obstacle_builds_one_geom_per_piece_at_the_frame(sim):
    pieces = [box_piece((0.1, 0.1, 0.1)), box_piece((0.1, 0.1, 0.1), (0.2, 0, 0))]
    cfg = config(None, attrs={"mesh": "/unused.stl"}, translation=(400, 0, 300), yaw_deg=90)
    sim.set_contribution("table", obstacle_contribution(cfg, pieces))
    assert sim.model.geom("table/obstacle/0").type[0] == mujoco.mjtGeom.mjGEOM_MESH
    assert sim.model.body("table/obstacle").jntnum[0] == 0
    # piece 1 sits 0.2 m along the frame's X, which is world +Y after the yaw
    assert sim.data.geom("table/obstacle/1").xpos == pytest.approx([0.4, 0.2, 0.3], abs=1e-6)


def test_a_box_dropped_into_a_u_obstacle_rests_inside_it(sim, u_pieces):
    cfg = config(None, attrs={"mesh": "/unused.stl"}, translation=(0, 0, 0))
    sim.set_contribution("table", obstacle_contribution(cfg, u_pieces))
    sim.set_prop_pose("pick_block", [0, 0, 0.2], [1, 0, 0, 0])
    sim.advance(1.5)
    # inside: on the 5 mm floor (block half height ~16 mm), well below the 80 mm rim;
    # a single hull would leave it resting on top at ~96 mm
    assert sim.data.body("pick_block").xpos[2] < 0.05


def test_mesh_obstacle_geometries_are_its_pieces(running_sim, u_pieces, monkeypatch):
    from mujoco_module.models import obstacle as obstacle_module
    from mujoco_module.models.world import MujocoWorld

    monkeypatch.setattr(obstacle_module, "load_pieces", lambda label, mesh: u_pieces)
    MujocoWorld.new(ComponentConfig(name="sim-world", attributes=dict_to_struct({"real_time": False})), {})
    cfg = config(None, attrs={"mesh": "/unused.stl"}, translation=(400, 0, 0))
    assert MujocoObstacle.validate_config(cfg) == (["sim-world"], [])
    obstacle = MujocoObstacle.new(cfg, {})
    try:
        geometries = asyncio.run(obstacle.get_geometries())
        assert len(geometries) >= 2
        assert [g.label for g in geometries] == [f"piece{i}" for i in range(len(geometries))]
        assert all(g.mesh.content_type == "ply" for g in geometries)
        assert geometries[0].mesh.mesh.startswith(b"ply\nformat ascii 1.0\n")
        count = running_sim.run(lambda: sum(
            running_sim.model.geom(i).name.startswith("table/obstacle/")
            for i in range(running_sim.model.ngeom)))
        assert count == len(geometries)
    finally:
        asyncio.run(obstacle.close())


@pytest.mark.parametrize("mesh", [False, True])
def test_textured_obstacle_shows_its_image_not_its_color(sim, mesh):
    attrs = {"texture": "/unused.png", "color": [0.2, 0.8, 0.2]}
    if mesh:
        cfg = config(None, attrs={**attrs, "mesh": "/unused.stl"}, translation=(0, 0, 50))
        contribution = obstacle_contribution(cfg, [box_piece((0.1, 0.1, 0.1))], halves())
    else:
        cube = Geometry(box=RectangularPrism(dims_mm=Vector3(x=100, y=100, z=100)))
        contribution = obstacle_contribution(config(cube, attrs=attrs, translation=(0, 0, 50)), texture=halves())
    sim.set_contribution("table", contribution)
    look_down(sim, 0, 0, 0.5, 10)
    rgb, _ = sim.render("cam", 64, 64)
    assert any(red(p) for p in rgb[32]) and any(blue(p) for p in rgb[32])
    assert rgb[..., 1].max() < 100


@pytest.mark.parametrize("mesh", [False, True])
def test_color_does_not_tint_a_textured_obstacle(sim, mesh):
    attrs = {"texture": "/unused.png", "color": [0.2, 0.8, 0.2]}
    if mesh:
        cfg = config(None, attrs={**attrs, "mesh": "/unused.stl"}, translation=(0, 0, 50))
        contribution = obstacle_contribution(cfg, [box_piece((0.1, 0.1, 0.1))], grey())
    else:
        cube = Geometry(box=RectangularPrism(dims_mm=Vector3(x=100, y=100, z=100)))
        contribution = obstacle_contribution(config(cube, attrs=attrs, translation=(0, 0, 50)), texture=grey())
    sim.set_contribution("table", contribution)
    look_down(sim, 0, 0, 0.5, 10)
    rgb, _ = sim.render("cam", 64, 64)
    assert untinted(rgb[32, 32])


def test_textured_obstacle_needs_its_texture_loaded():
    cube = Geometry(box=RectangularPrism(dims_mm=Vector3(x=100, y=100, z=100)))
    with pytest.raises(ValueError, match="table: a textured obstacle needs its texture loaded"):
        obstacle_contribution(config(cube, attrs={"texture": "/unused.png"}))


def test_validate_rejects_a_bad_texture():
    with pytest.raises(ValueError, match=r"table: texture must be a \.png"):
        MujocoObstacle.validate_config(config(BOX, attrs={"texture": "/a.gif"}))


def test_obstacle_component_loads_its_texture(running_sim, tmp_path):
    from mujoco_module.models.world import MujocoWorld

    MujocoWorld.new(ComponentConfig(name="sim-world", attributes=dict_to_struct({"real_time": False})), {})
    image = str(write_image(tmp_path / "t.png"))
    obstacle = MujocoObstacle.new(config(BOX, attrs={"texture": image}), {})
    try:
        assert running_sim.run(
            lambda: running_sim.model.geom("table/obstacle").matid[0]
            == running_sim.model.material("table/obstacle/texture").id
        )
    finally:
        asyncio.run(obstacle.close())
