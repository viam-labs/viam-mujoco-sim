import asyncio
import math
import warnings

import mujoco
import pytest
from viam.proto.app.robot import ComponentConfig, Frame, Orientation, Translation
from viam.proto.common import Geometry, RectangularPrism, Vector3
from viam.utils import dict_to_struct

from mujoco_module.models.scene import MujocoScene, scene_contribution
from mujoco_module.scenes import load_scene
from mujoco_module.sim import FREE_CONTYPE, PROP_CONTACT_PRIORITY, PROP_SOLREF

# a table (box, visual sphere, cylinder leg, wedge mesh, ellipsoid knob), a
# free cube on it, a hinged door, a floor plane; its <option> and <size>
# differ from the world's
CELL = """
<mujoco model="cell">
  <option timestep="0.01" gravity="0 0 -5"/>
  <size memory="10M"/>
  <asset>
    <mesh name="wedge" vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"/>
    <mesh name="cube" vertex="-.05 -.05 -.05  .05 -.05 -.05  -.05 .05 -.05  .05 .05 -.05
                               -.05 -.05 .05  .05 -.05 .05  -.05 .05 .05  .05 .05 .05  0 0 0"/>
  </asset>
  <default><default class="visual"><geom contype="0" conaffinity="0" group="2"/></default></default>
  <worldbody>
    <light name="lamp" pos="0 0 2"/>
    <geom name="ground" type="plane" size="0 0 0.05"/>
    <body name="table" pos="0.5 0 0.2">
      <geom name="top" type="box" size="0.3 0.2 0.02"/>
      <geom name="deco" type="sphere" size="0.05" class="visual" pos="0 0 0.1"/>
      <geom name="leg" type="cylinder" size="0.02 0.1" pos="0.25 0.15 -0.1"/>
      <geom name="wedge" type="mesh" mesh="wedge" pos="-0.2 0 0.02"/>
      <geom name="ball" type="sphere" size="0.02" pos="0.2 0 0.04"/>
      <geom name="rod" type="capsule" size="0.01 0.05" pos="-0.1 0 0.04"/>
      <geom name="knob" type="ellipsoid" size="0.01 0.02 0.03" pos="0 -0.19 0"/>
      <geom name="block" type="mesh" mesh="cube" pos="-0.25 -0.15 0.07"/>
    </body>
    <body name="cube" pos="0.5 0 0.3">
      <freejoint name="cube_joint"/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.05"/>
    </body>
    <body name="door" pos="0 0.5 0.3">
      <joint name="hinge" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.2 0.01 0.3" euler="0 0 90"/>
    </body>
  </worldbody>
  <keyframe><key name="home" qpos="0.5 0 0.3 1 0 0 0 0"/></keyframe>
</mujoco>
"""

YAW_90 = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))


def write(tmp_path, xml=CELL, name="cell.xml"):
    path = tmp_path / name
    path.write_text(xml)
    return str(path)


def by_label(scene):
    return {g.label: g for g in scene.geometries}


def test_planner_geometries_in_the_scene_frame(tmp_path):
    scene = load_scene("cell", write(tmp_path))
    geometries = by_label(scene)
    # visual-only geoms, the plane and the free cube are left out
    assert sorted(geometries) == ["ball", "block", "door:0", "knob", "leg", "rod", "top", "wedge"]
    top = geometries["top"]
    assert top.box.dims_mm.x == pytest.approx(600) and top.box.dims_mm.z == pytest.approx(40)
    assert (top.center.x, top.center.y, top.center.z) == pytest.approx((500, 0, 200))
    assert geometries["leg"].mesh.content_type == "ply"  # cylinders go as meshes
    assert geometries["knob"].box.dims_mm.z == pytest.approx(60)  # ellipsoid: box of its semi-axes
    assert geometries["door:0"].box.dims_mm.y == pytest.approx(20)
    door = geometries["door:0"].center  # turned 90 degrees about z
    assert (door.o_z, door.theta) == pytest.approx((1, 90))
    assert geometries["ball"].sphere.radius_mm == pytest.approx(20)
    assert geometries["rod"].capsule.length_mm == pytest.approx(120)  # tip to tip


def test_mesh_geometry_is_its_hull_in_the_scene_frame(tmp_path):
    block = by_label(load_scene("cell", write(tmp_path)))["block"]
    header, body = block.mesh.mesh.decode().split("end_header\n")
    vertices = [tuple(map(float, line.split())) for line in body.splitlines() if not line.startswith("3 ")]
    assert "element vertex 8" in header  # the interior vertex isn't on the hull
    xs, zs = [v[0] for v in vertices], [v[2] for v in vertices]
    # the block sits at (0.5 - 0.25, -0.15, 0.27) with a 0.1 m edge, in meters (RDK scales PLY)
    assert (min(xs), max(xs)) == pytest.approx((0.2, 0.3), abs=1e-6)
    assert (min(zs), max(zs)) == pytest.approx((0.22, 0.32), abs=1e-6)


def test_props_spawn_where_the_frame_puts_them(tmp_path):
    import numpy as np

    xml = CELL.replace('<body name="cube" pos="0.5 0 0.3">', '<body name="cube" pos="0.5 0 0.3" euler="90 0 0">')
    scene = load_scene("cell", write(tmp_path, xml), (1.0, 0.0, 0.0), YAW_90)
    (prop,) = scene.props
    assert prop.name == "cell/cube" and not prop.fixed
    assert prop.position == pytest.approx((1.0, 0.5, 0.3))  # (0.5, 0, 0.3) turned 90 degrees, moved 1 m
    expected = np.zeros(4)
    mujoco.mju_mulQuat(expected, np.array(YAW_90), np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]))
    assert prop.orientation_wxyz == pytest.approx(tuple(expected))
    assert scene.has_plane


def test_a_plane_on_a_body_is_not_a_floor(tmp_path):
    xml = CELL.replace('<geom name="ground" type="plane" size="0 0 0.05"/>', "").replace(
        '<geom name="top"', '<geom name="wall" type="plane" size="1 1 0.05"/><geom name="top"'
    )
    assert not load_scene("cell", write(tmp_path, xml)).has_plane
    visual = CELL.replace(
        '<geom name="ground" type="plane"', '<geom name="ground" contype="0" conaffinity="0" type="plane"'
    )
    assert not load_scene("cell", write(tmp_path, visual)).has_plane


def test_unnamed_geoms_and_bodies_get_labels(tmp_path):
    xml = CELL.replace('<body name="door"', '<body')
    assert "body3:0" in by_label(load_scene("cell", write(tmp_path, xml)))


def test_free_joints_take_their_body_name(tmp_path):
    scene = load_scene("cell", write(tmp_path))
    assert [j.name for j in scene.spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE] == ["cube"]
    assert not list(scene.spec.keys)  # keyframes are dropped


def test_movables_get_prop_contacts(tmp_path):
    geom = load_scene("cell", write(tmp_path)).spec.body("cube").first_geom()
    assert geom.contype == FREE_CONTYPE and geom.priority == PROP_CONTACT_PRIORITY
    assert tuple(geom.solref) == pytest.approx(PROP_SOLREF)


@pytest.mark.parametrize("insert,message", [
    ('<actuator><motor joint="hinge"/></actuator>', "can't have actuators"),
    ('<tendon><fixed name="t"><joint joint="hinge" coef="1"/></fixed></tendon>', "can't have tendons"),
    ('<equality><joint joint1="hinge"/></equality>', "can't have equalities"),
])
def test_rejected_elements(tmp_path, insert, message):
    with pytest.raises(ValueError, match=f"cell: a scene file {message}"):
        load_scene("cell", write(tmp_path, CELL.replace("</mujoco>", insert + "</mujoco>")))


def test_passive_slide_and_ball_joints_are_allowed(tmp_path):
    xml = CELL.replace("  </worldbody>", """    <body name="drawer" pos="0 -0.5 0.1">
      <joint type="slide" axis="1 0 0"/><geom type="box" size="0.1 0.1 0.05"/>
    </body>
    <body name="swivel" pos="0 -0.8 0.3">
      <joint type="ball"/><geom type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>""")
    labels = by_label(load_scene("cell", write(tmp_path, xml)))
    assert "drawer:0" in labels and "swivel:0" in labels


def test_rejected_mocap_and_unnamed_free_bodies(tmp_path):
    mocap = CELL.replace('<body name="door" pos="0 0.5 0.3">', '<body name="door" pos="0 0.5 0.3" mocap="true">')
    mocap = mocap.replace('<joint name="hinge" type="hinge" axis="0 0 1"/>', "")
    with pytest.raises(ValueError, match="can't have mocap bodies"):
        load_scene("cell", write(tmp_path, mocap))
    with pytest.raises(ValueError, match="free body #2 needs a name"):
        load_scene("cell", write(tmp_path, CELL.replace('<body name="cube"', "<body")))


def test_load_errors(tmp_path):
    with pytest.raises(ValueError, match="cell: file .*nope.xml not found"):
        load_scene("cell", str(tmp_path / "nope.xml"))
    with pytest.raises(ValueError, match=r"cell: a scene must be MJCF \(\.xml\)"):
        load_scene("cell", "robot.urdf")
    with pytest.raises(ValueError, match="cell: cannot load mjcf"):
        load_scene("cell", write(tmp_path, "<mujoco><worldbody><geom type='box'/>"))


def test_free_joint_name_clash_names_the_scene(tmp_path):
    xml = CELL.replace('<joint name="hinge"', '<joint name="cube"')
    with pytest.raises(ValueError, match="free body 'cube' names its free joint"):
        load_scene("cell", write(tmp_path, xml))


def test_shared_helpers_on_a_free_body_with_a_child(tmp_path):
    from mujoco_module.scenes import geom_geometry, subtree_geoms

    xml = CELL.replace(
        '<geom type="box" size="0.02 0.02 0.02" mass="0.05"/>',
        '<geom type="box" size="0.02 0.02 0.02" mass="0.05"/>'
        '<body pos="0 0 0.1"><geom type="box" size="0.01 0.01 0.01"/></body>',
    )
    scene = load_scene("cell", write(tmp_path, xml))
    model = scene.spec.copy().compile()
    cube = model.body("cube").id
    geoms = subtree_geoms(model, cube)
    assert len(geoms) == 2
    assert all(g.contype == FREE_CONTYPE for g in scene.spec.body("cube").find_all(mujoco.mjtObj.mjOBJ_GEOM))
    assert not {"cube:0", "cube:1"} & set(by_label(scene))
    assert len(scene.spec.body("cube").find_all(mujoco.mjtObj.mjOBJ_GEOM)) == 2
    box = geom_geometry(model, None, geoms[0], "x")
    assert (box.center.x, box.center.y, box.center.z, box.center.o_z) == (0, 0, 0, 1)
    assert box.box.dims_mm.x == pytest.approx(40)


# ---- in the sim ----------------------------------------------------------


def attach(sim, tmp_path, position=(1.0, 0.0, 0.0), quat=YAW_90, xml=CELL):
    scene = load_scene("cell", write(tmp_path, xml), position, quat)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # MuJoCo's attach-conflict warning fails the test
        sim.set_contribution("cell", scene_contribution("cell", scene, position, quat))
    return scene


def test_scene_attaches_at_its_frame_and_the_world_keeps_its_options(sim, tmp_path):
    attach(sim, tmp_path)
    assert sim.data.geom("cell/top").xpos == pytest.approx([1.0, 0.5, 0.2])
    assert sim.model.opt.timestep == pytest.approx(0.002)
    assert sim.model.opt.gravity[2] == pytest.approx(-9.81)


def test_scene_plane_replaces_the_floor_until_removed(sim, tmp_path):
    attach(sim, tmp_path)
    assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "floor") == -1
    sim.remove_contribution("cell")
    assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "floor") >= 0


def test_scene_props_move_rest_and_reset(sim, tmp_path):
    attach(sim, tmp_path)
    sim.set_prop_pose("cell/cube", [1.0, 0.5, 0.5], [1, 0, 0, 0])
    sim.advance(1.0)
    # rests on the table top (0.22 m) by its half height (0.02 m)
    assert sim.get_prop_pose("cell/cube")["position"] == pytest.approx([1.0, 0.5, 0.24], abs=0.003)
    sim.reset()
    assert sim.get_prop_pose("cell/cube")["position"] == pytest.approx([1.0, 0.5, 0.3], abs=1e-6)
    sim.set_prop_pose("cell/cube", [1.0, 0.5, 0.5], [1, 0, 0, 0])
    parked = sim.park_props(["cell/cube"])["poses"]["cell/cube"]
    assert parked["position"] == pytest.approx([1.0, 0.5, 0.3], abs=1e-6)
    with pytest.raises(ValueError, match="unknown prop 'cell/nope'"):
        sim.get_prop_pose("cell/nope")


def test_scene_survives_a_rebuild(sim, tmp_path):
    from mujoco_module.models.obstacle import obstacle_contribution
    from test_obstacle import BOX, config as obstacle_config

    attach(sim, tmp_path)
    sim.set_prop_pose("cell/cube", [0.5, 0.5, 0.6], [1, 0, 0, 0])
    sim.set_contribution("table", obstacle_contribution(obstacle_config(BOX, translation=(0, -1000, 10))))
    assert sim.get_prop_pose("cell/cube")["position"] == pytest.approx([0.5, 0.5, 0.6], abs=1e-6)
    assert [p.name for p in sim.run(sim.scene_props)] == ["cell/cube"]


def test_floor_texture_warning_is_logged_once(sim, tmp_path, monkeypatch):
    from mujoco_module import sim as sim_module
    from mujoco_module.models.obstacle import obstacle_contribution
    from mujoco_module.sim import WorldConfig
    from test_obstacle import BOX, config as obstacle_config
    from test_textures import halves

    warned = []
    monkeypatch.setattr(sim_module.LOGGER, "warning", lambda *args: warned.append(args))
    sim.configure_world(WorldConfig.from_attrs({"floor_texture": "/unused.png"}), textures={"floor": halves()})
    attach(sim, tmp_path)
    sim.set_contribution("table", obstacle_contribution(obstacle_config(BOX, translation=(0, -1000, 10))))
    assert [a[0] for a in warned].count("floor_texture is ignored: scene %s brings its own floor") == 1


def test_floor_texture_added_after_a_scene_still_warns(sim, tmp_path, monkeypatch):
    from mujoco_module import sim as sim_module
    from mujoco_module.sim import WorldConfig
    from test_textures import halves

    warned = []
    monkeypatch.setattr(sim_module.LOGGER, "warning", lambda *args: warned.append(args))
    attach(sim, tmp_path)
    sim.configure_world(WorldConfig.from_attrs({"floor_texture": "/unused.png"}), textures={"floor": halves()})
    assert [a[0] for a in warned].count("floor_texture is ignored: scene %s brings its own floor") == 1


# ---- the component -------------------------------------------------------


def config(source, frame=True, parent="world", geometry=None):
    cfg = ComponentConfig(name="cell", attributes=dict_to_struct({"world": "sim-world", "mjcf": source}))
    if frame:
        f = Frame(parent=parent, translation=Translation(x=1000, y=0, z=0))
        f.orientation.CopyFrom(Orientation(vector_degrees=Orientation.OrientationVectorDegrees(z=1, theta=90)))
        if geometry is not None:
            f.geometry.CopyFrom(geometry)
        cfg.frame.CopyFrom(f)
    return cfg


@pytest.mark.parametrize("cfg,message", [
    (config("/a/cell.xml", frame=False), 'set a frame with parent "world"'),
    (config("/a/cell.xml", parent="arm"), 'frame.parent must be "world"'),
    (config("/a/cell.xml", geometry=Geometry(box=RectangularPrism(dims_mm=Vector3(x=1, y=1, z=1)))),
     "remove frame.geometry"),
    (config("/a/cell.urdf"), "mjcf must be a .xml MJCF file path"),
])
def test_validate_rejects(cfg, message):
    with pytest.raises(ValueError, match=message):
        MujocoScene.validate_config(cfg)


def test_component_serves_geometries_and_closes(running_sim, tmp_path):
    from mujoco_module.models.world import MujocoWorld

    MujocoWorld.new(ComponentConfig(name="sim-world", attributes=dict_to_struct({"real_time": False})), {})
    cfg = config(write(tmp_path))
    assert MujocoScene.validate_config(cfg) == (["sim-world"], [])
    scene = MujocoScene.new(cfg, {})
    try:
        labels = sorted(g.label for g in asyncio.run(scene.get_geometries()))
        assert labels == ["ball", "block", "door:0", "knob", "leg", "rod", "top", "wedge"]
        position = running_sim.run(lambda: running_sim.data.geom("cell/top").xpos.tolist())
        assert position == pytest.approx([1.0, 0.5, 0.2])
    finally:
        asyncio.run(scene.close())
    assert running_sim.run(
        lambda: mujoco.mj_name2id(running_sim.model, mujoco.mjtObj.mjOBJ_GEOM, "cell/top")
    ) == -1


def test_world_state_publishes_scene_props(running_sim, tmp_path):
    from mujoco_module.models.world import MujocoWorld
    from mujoco_module.models.world_state import MujocoWorldState

    MujocoWorld.new(ComponentConfig(name="sim-world", attributes=dict_to_struct({"real_time": False})), {})
    scene = MujocoScene.new(config(write(tmp_path)), {})
    store = MujocoWorldState("world-state")
    try:
        running_sim.pause()  # the cube stays where it's put
        running_sim.set_prop_pose("cell/cube", [1.1, 0.4, 0.6], [0.7071068, 0, 0, 0.7071068])
        transforms = [store._transform(e) for e in store._snapshot()]
        (cube,) = [t for t in transforms if t.reference_frame == "cell/cube"]
        assert cube.physical_object.box.dims_mm.x == pytest.approx(40)
        pose = cube.pose_in_observer_frame.pose
        assert (pose.x, pose.y, pose.z) == pytest.approx((1100, 400, 600), abs=0.01)
        assert pose.o_z == pytest.approx(1, abs=0.01)
        assert pose.theta == pytest.approx(90, abs=0.01)
    finally:
        asyncio.run(scene.close())


def test_bundled_workbench_loads():
    scene = load_scene("workbench", "scenes/workbench.xml")
    assert [p.name for p in scene.props] == ["workbench/block_a", "workbench/block_b"]
    assert scene.has_plane and scene.geometries
    free = [j.name for j in scene.spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]
    assert free == ["block_a", "block_b"]  # unnamed <freejoint/>s take their body's name


def test_urdf_named_xml_rejected(tmp_path):
    path = tmp_path / "robot.xml"
    path.write_text('<robot name="r"><link name="a"/></robot>')
    with pytest.raises(ValueError, match="got a URDF"):
        load_scene("s", str(path))
