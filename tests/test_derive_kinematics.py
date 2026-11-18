"""SVA kinematics derived from an MJCF arm (derive_kinematics.py)."""

import itertools
import json
import math

import mujoco
import numpy as np
import pytest

from mujoco_module import derive_kinematics as derive
from mujoco_module import robots
from mujoco_module.sim import KNOWN_ASSETS


def bundled(asset):
    """The bundled arm's compiled model, its flange site id and its offsets, as configured."""
    attrs = KNOWN_ASSETS[asset]
    spec = robots.load_mjcf(robots.resolve_path(attrs["mjcf"], asset), attrs.get("mesh_dir"))
    model = spec.compile()
    site = model.site(attrs.get("flange_site", "attachment_site")).id
    base = robots.parse_pose(attrs.get("base_offset"), asset)
    flange = robots.parse_pose(attrs.get("flange_offset"), asset)
    return model, site, base, flange


def compiled(xml):
    return mujoco.MjSpec.from_string(xml).compile()


@pytest.mark.parametrize("asset", ["lite6", "ur5e"])
def test_chain_is_the_hinges_from_the_base_to_the_flange(asset):
    model, site, _, _ = bundled(asset)
    joints = derive.chain_joints(model, site, asset)
    assert len(joints) == 6
    # base first: each joint's body is below the previous one's
    for before, after in itertools.pairwise(joints):
        assert robots._is_ancestor(model, int(model.jnt_bodyid[before]), int(model.jnt_bodyid[after]))


ARM = """<mujoco><worldbody><body name="b0">{root}
  <body name="b1" pos="0 0 0.1"><joint name="j1" {j1}/><geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.1"/>
    <body name="b2" pos="0 0 0.1"><joint name="j2" axis="0 1 0" range="-1 1"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.1 0 0"/><site name="attachment_site" pos="0.1 0 0"/>
    </body></body></body></worldbody></mujoco>"""


def arm(root="", j1='axis="0 0 1" range="-2 2"'):
    return compiled(ARM.format(root=root, j1=j1))


def test_two_joints_on_one_body_is_an_error():
    model = arm(j1='axis="0 0 1" range="-2 2"/><joint name="extra" axis="1 0 0" range="-1 1"')
    with pytest.raises(ValueError, match="b1.*2 joints"):
        derive.chain_joints(model, model.site("attachment_site").id, "two")


def test_a_ball_joint_is_an_error():
    model = arm(j1='type="ball"')
    with pytest.raises(ValueError, match="b1.*ball or free"):
        derive.chain_joints(model, model.site("attachment_site").id, "ball")


def test_a_free_root_is_an_error():
    model = arm(root='<freejoint name="loose"/><geom size="0.05"/>')
    with pytest.raises(ValueError, match="b0.*ball or free"):
        derive.chain_joints(model, model.site("attachment_site").id, "free")


def test_a_hinge_off_its_body_origin_is_an_error():
    model = arm(j1='axis="0 0 1" range="-2 2" pos="0.01 0 0"')
    with pytest.raises(ValueError, match="j1.*pos"):
        derive.chain_joints(model, model.site("attachment_site").id, "offset")


def test_a_flange_on_the_root_has_no_chain():
    model = compiled('<mujoco><worldbody><body name="b0"><geom size="0.1"/>'
                     '<site name="attachment_site"/></body></worldbody></mujoco>')
    with pytest.raises(ValueError, match="no hinge or slide joints"):
        derive.chain_joints(model, model.site("attachment_site").id, "none")
def derived(asset):
    model, site, base, flange = bundled(asset)
    joints = derive.chain_joints(model, site, asset)
    sva = derive.derive_sva(model, joints, site, base, flange, asset)
    kin = robots.parse_kinematics(json.dumps(sva).encode(), urdf=False, label=asset)
    return model, site, base, flange, joints, sva, kin


def mujoco_end(model, joints, site, base, flange, q):
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[joints]] = q
    mujoco.mj_kinematics(model, data)
    pose = robots.transform(data.site_xpos[site], robots.quat(data.site_xmat[site].reshape(3, 3)))
    return base @ pose @ flange


@pytest.mark.parametrize("asset", ["lite6", "ur5e"])
def test_derived_forward_kinematics_equal_mujocos(asset):
    model, site, base, flange, joints, _, kin = derived(asset)
    assert kin.joints == tuple(model.joint(j).name for j in joints)
    low, high = model.jnt_range[joints].T
    rng = np.random.default_rng(7)
    home = robots.home(model, "home", robots.arm_actuators(model, joints, asset))
    for q in [np.zeros(len(joints)), np.asarray(home, float), *rng.uniform(low, high, size=(20, len(joints)))]:
        metres, degrees = robots.pose_error(kin.forward(q)[kin.end], mujoco_end(model, joints, site, base, flange, q))
        assert metres < 1e-6 and degrees < 1e-4, (q, metres, degrees)


@pytest.mark.parametrize("asset", ["lite6", "ur5e"])
def test_derived_limits_are_the_mjcfs(asset):
    model, _, _, _, joints, sva, _ = derived(asset)
    for joint, j in zip(sva["joints"], joints):
        assert [joint["min"], joint["max"]] == pytest.approx(np.degrees(model.jnt_range[j]))


def inside(shape, point, slack=1e-9):
    """Whether a body-frame point lies in a robots.Shape (posed in its body's frame)."""
    local = np.linalg.inv(robots.transform(shape.pos, shape.quat)) @ [*point, 1.0]
    p = local[:3]
    if shape.kind == "box":
        return bool(np.all(np.abs(p) <= np.asarray(shape.size) + slack))
    if shape.kind == "capsule":
        radius, half = shape.size
        return math.dist(p, [0, 0, float(np.clip(p[2], -half, half))]) <= radius + slack
    return float(np.linalg.norm(p)) <= shape.size[0] + slack


@pytest.mark.parametrize("asset", ["lite6", "ur5e"])
def test_each_derived_shape_covers_the_geoms_of_the_body_it_rides(asset):
    model, _, base, _, joints, _, kin = derived(asset)
    root = int(model.body_rootid[model.jnt_bodyid[joints[0]]])
    shapes = robots.kinematics_shapes(model, kin, joints, root, base)
    covered = set()
    for shape in shapes:
        body = model.body(shape.body).id
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                 and (model.geom_contype[g] or model.geom_conaffinity[g])]
        assert geoms, f"{shape.link} rides {shape.body}, which has no collision geoms"
        for g in geoms:
            to_body = robots.transform(model.geom_pos[g], model.geom_quat[g])
            for point in derive.geom_points(model, g):
                assert inside(shape, (to_body @ [*point, 1.0])[:3], 1e-3), (shape.link, shape.body, model.geom(g).name)
        covered.add(body)
    chain = {int(model.jnt_bodyid[j]) for j in joints} | {root}
    with_geoms = {b for b in chain if any(model.geom_bodyid[g] == b and (model.geom_contype[g] or model.geom_conaffinity[g])
                                          for g in range(model.ngeom))}
    assert with_geoms <= covered


@pytest.mark.parametrize("asset", ["lite6", "ur5e"])
def test_derived_kinematics_are_close_to_the_vendors(asset):
    # a loose sanity check: vendor link lengths differ from Menagerie's by about a millimetre
    model, _, _, _, joints, _, kin = derived(asset)
    vendor = robots.parse_kinematics(
        robots.resolve_path(KNOWN_ASSETS[asset]["kinematics_url"], asset).read_bytes(), urdf=False, label=asset
    )
    rng = np.random.default_rng(3)
    low, high = model.jnt_range[joints].T
    for q in rng.uniform(low, high, size=(10, len(joints))):
        metres, degrees = robots.pose_error(kin.forward(q)[kin.end], vendor.forward(q)[vendor.end])
        assert metres < 0.005 and degrees < 2.0


def test_an_unlimited_hinge_gets_a_full_turn_and_a_warning(caplog):
    model = arm(j1='axis="0 0 1" limited="false"')
    site = model.site("attachment_site").id
    sva = derive.derive_sva(model, derive.chain_joints(model, site, "c"), site, np.eye(4), np.eye(4), "c")
    assert (sva["joints"][0]["min"], sva["joints"][0]["max"]) == (-360.0, 360.0)
    assert "j1" in caplog.text


def test_an_unlimited_slide_is_an_error():
    model = arm(j1='type="slide" axis="0 0 1" limited="false"')
    site = model.site("attachment_site").id
    with pytest.raises(ValueError, match="j1.*no range"):
        derive.derive_sva(model, derive.chain_joints(model, site, "s"), site, np.eye(4), np.eye(4), "s")


def test_a_capsule_along_x_is_fitted_as_that_capsule():
    model = arm()
    site = model.site("attachment_site").id
    sva = derive.derive_sva(model, derive.chain_joints(model, site, "x"), site, np.eye(4), np.eye(4), "x")
    geometry = sva["links"][-1]["geometry"]  # the flange link carries b2's capsule, which runs along x
    assert geometry["orientation"]["value"] == {"x": 1.0, "y": 0.0, "z": 0.0, "th": 0.0}
    assert geometry["r"] == pytest.approx(20, abs=0.5)  # the geom's own radius, not its AABB's corner
    assert geometry["l"] == pytest.approx(140, abs=1)   # 100 mm between centres + both caps
    assert geometry["translation"]["x"] == pytest.approx(50, abs=0.5)


def test_a_tilted_capsule_is_covered_by_its_fitted_shape():
    model = compiled("""<mujoco><worldbody><body name="b0"><body name="b1" pos="0 0 0.1">
      <joint name="j1" axis="0 0 1" range="-1 1"/>
      <geom type="capsule" size="0.05 0.1" euler="0 45 0"/><site name="attachment_site"/>
    </body></body></worldbody></mujoco>""")
    site = model.site("attachment_site").id
    joints = derive.chain_joints(model, site, "t")
    sva = derive.derive_sva(model, joints, site, np.eye(4), np.eye(4), "t")
    kin = robots.parse_kinematics(json.dumps(sva).encode(), urdf=False, label="t")
    (shape,) = [s for s in robots.kinematics_shapes(model, kin, joints, 0, np.eye(4)) if s.body == "b1"]
    g = 0
    centre, axis = model.geom_pos[g], model.geom_quat[g]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, axis)
    rot = rot.reshape(3, 3)
    r, half = model.geom_size[g][:2]
    # dense true surface: sphere caps at both ends plus the cylinder wall
    u, v = np.meshgrid(np.linspace(0, np.pi, 40), np.linspace(0, 2 * np.pi, 80))
    sphere = np.stack([r * np.sin(u) * np.cos(v), r * np.sin(u) * np.sin(v), r * np.cos(u)], -1).reshape(-1, 3)
    wall = np.stack([r * np.cos(v), r * np.sin(v), np.linspace(-half, half, 80)[:, None] * np.ones_like(v)], -1).reshape(-1, 3)
    for sign in (-1, 1):
        for p in sphere:
            assert inside(shape, centre + rot @ (p + [0, 0, sign * half]), 0.5e-3)
    for p in wall:
        assert inside(shape, centre + rot @ p, 0.5e-3)


SIDE = """<mujoco><worldbody><body name="b0"><geom size="0.05"/>
  <body name="b1" pos="0 0 0.1"><joint name="j1" axis="0 0 1" range="-1 1"/><geom size="0.02"/>
    <body name="b2" pos="0 0 0.1"><joint name="j2" axis="0 1 0" range="-1 1"/><geom size="0.02"/>
      <site name="attachment_site"/></body></body>
  <body name="side" pos="0.2 0 0"><joint name="js" axis="0 0 1" range="-1 1"/><geom size="0.02"/></body>
</body></worldbody></mujoco>"""


def test_given_joints_are_checked_against_the_chain():
    model = compiled(SIDE)
    site = model.site("attachment_site").id
    assert derive.given_joints(model, ["j1", "j2"], site, "g") == [model.joint("j1").id, model.joint("j2").id]
    with pytest.raises(ValueError, match="no joint 'nope'"):
        derive.given_joints(model, ["j1", "nope"], site, "g")
    for wrong in (["j1", "js"], ["j2", "j1"], ["j2"]):  # off the path, out of order, a skipped joint
        with pytest.raises(ValueError, match=r"must be the chain .*\['j1', 'j2'\]"):
            derive.given_joints(model, wrong, site, "g")


STATIC = """<mujoco><worldbody><body name="b0"><geom size="0.05"/>
  <body name="b1" pos="0 0 0.1"><joint name="j1" axis="0 0 1" range="-1 1"/><geom size="0.03"/>
    <body name="s" pos="0.05 0.02 0.1" euler="10 20 30"><geom type="capsule" size="0.02 0.06" pos="0.01 0 0.03" euler="0 70 0"/>
      <body name="b2" pos="0 0.03 0.1" euler="0 0 20"><joint name="j2" axis="0 1 0" range="-1 1" ref="0.4"/><geom size="0.02"/>
        <site name="attachment_site" pos="0.02 0 0.05"/></body></body></body></body></worldbody></mujoco>"""


def test_a_static_body_between_joints_and_a_ref_and_offsets():
    model = compiled(STATIC)
    site = model.site("attachment_site").id
    joints = derive.chain_joints(model, site, "m")
    base = robots.transform([0.1, -0.2, 0.3], [0.9, 0.1, 0.3, 0.2] / np.linalg.norm([0.9, 0.1, 0.3, 0.2]))
    flange = robots.transform([0.01, 0.02, 0.03], [0.8, 0.3, 0.1, 0.5] / np.linalg.norm([0.8, 0.3, 0.1, 0.5]))
    sva = derive.derive_sva(model, joints, site, base, flange, "m")
    kin = robots.parse_kinematics(json.dumps(sva).encode(), urdf=False, label="m")
    rng = np.random.default_rng(5)
    low, high = model.jnt_range[joints].T
    # joint values: q is MuJoCo's qpos (the hinge's ref is already in qpos0 = 0.4)
    for q in rng.uniform(low, high, size=(8, 2)):
        metres, degrees = robots.pose_error(kin.forward(q)[kin.end], mujoco_end(model, joints, site, base, flange, q))
        assert metres < 1e-6 and degrees < 1e-4, (q, metres, degrees)
    shapes = robots.kinematics_shapes(model, kin, joints, 0, base)
    above = [s for s in shapes if s.body == "b1"]
    assert above
    g = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == model.body("s").id)
    to_b1 = robots.transform(model.body("s").pos, model.body("s").quat) @ robots.transform(model.geom_pos[g], model.geom_quat[g])
    for point in derive.geom_points(model, g):
        assert any(inside(s, (to_b1 @ [*point, 1.0])[:3], 1e-3) for s in above)


def test_a_squat_cylinder_fits_as_its_own_box():
    # a disc 200 mm across and 40 mm tall: its box is 200 x 200 x 40, not stretched by its radius
    model = compiled("""<mujoco><worldbody><body name="b0"><body name="b1">
      <joint name="j1" axis="0 0 1" range="-1 1"/><geom type="cylinder" size="0.1 0.02"/>
      <site name="attachment_site"/></body></body></worldbody></mujoco>""")
    site = model.site("attachment_site").id
    sva = derive.derive_sva(model, derive.chain_joints(model, site, "c"), site, np.eye(4), np.eye(4), "c")
    geometry = sva["links"][-1]["geometry"]
    assert [geometry["x"], geometry["y"], geometry["z"]] == pytest.approx([200, 200, 40], abs=0.5)


def ply_vertices(geometry):
    """The vertices (m) of an SVA mesh geometry's ASCII PLY."""
    import base64

    assert geometry["type"] == "mesh" and geometry["mesh_content_type"] == "ply"
    lines = base64.b64decode(geometry["mesh_data"]).decode().splitlines()
    count = int(next(line for line in lines if line.startswith("element vertex")).split()[-1])
    start = lines.index("end_header") + 1
    return np.array([[float(v) for v in line.split()] for line in lines[start:start + count]])


def link_frames(sva, base):
    """Each SVA link with geometry: the MuJoCo-world pose its geometry is posed in, at joints zero."""
    kin = robots.parse_kinematics(json.dumps(sva).encode(), urdf=False, label="t")
    poses = kin.forward(np.zeros(len(kin.joints)))
    return {f.name: np.linalg.inv(base) @ poses[f.parent] for f in kin.frames.values() if f.geometry}


@pytest.mark.parametrize("asset", ["lite6", "ur5e"])
def test_mesh_planning_geometry_is_each_links_collision_hulls_where_the_shapes_were(asset):
    model, site, base, flange = bundled(asset)
    joints = derive.chain_joints(model, site, asset)
    shapes = derive.derive_sva(model, joints, site, base, flange, asset)
    meshes = derive.derive_sva(model, joints, site, base, flange, asset, meshes=True)
    # the same chain, the same links, a mesh wherever there was a shape
    strip = lambda sva: [{k: v for k, v in link.items() if k != "geometry"} for link in sva["links"]]
    assert strip(meshes) == strip(shapes) and meshes["joints"] == shapes["joints"]
    assert [bool(link.get("geometry")) for link in meshes["links"]] == [bool(link.get("geometry")) for link in shapes["links"]]
    # every link's PLY, back in MuJoCo's world, is exactly the chain's colliding hulls there
    served = []
    for link in meshes["links"]:
        if link.get("geometry"):
            ply = ply_vertices(link["geometry"])
            served.append((link_frames(meshes, base)[link["id"]] @ np.c_[ply, np.ones(len(ply))].T).T[:, :3])
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    path, body = set(), int(model.site_bodyid[site])
    while body:
        path.add(body)
        body = int(model.body_parentid[body])
    expected = np.vstack([derive.geom_piece(model, g, hull=True)[0] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                          for g in derive._colliding(model, path)])
    served = np.vstack(served)
    gap = lambda a, b: np.linalg.norm(a[:, None] - b[None], axis=2).min(axis=1).max()
    assert len(served) == len(expected) and gap(served, expected) < 1e-5 and gap(expected, served) < 1e-5


def test_mesh_planning_geometry_tessellates_primitives():
    model = arm()
    site = model.site("attachment_site").id
    joints = derive.chain_joints(model, site, "prims")
    sva = derive.derive_sva(model, joints, site, np.eye(4), np.eye(4), "prims", meshes=True)
    capsule = ply_vertices(next(link for link in sva["links"] if link.get("geometry"))["geometry"])
    # b1's capsule, radius 0.02 around 0..0.1 on z, rides the link after b1's joint, posed in b1's frame
    assert np.linalg.norm(capsule[:, :2], axis=1).max() == pytest.approx(0.02, abs=1e-3)
    assert capsule[:, 2].min() == pytest.approx(-0.02, abs=1e-3) and capsule[:, 2].max() == pytest.approx(0.12, abs=1e-3)


def test_visual_models_are_glbs_of_each_links_visible_geoms_posed_on_its_collider():
    import io

    import trimesh

    model = compiled(ARM.format(root="", j1='axis="0 0 1" range="-2 2"').replace(
        '<site name="attachment_site"',
        '<geom type="box" size="0.01 0.02 0.03" pos="0.05 0 0" contype="0" conaffinity="0" group="2" rgba="1 0 0 1"/>'
        '<geom type="sphere" size="0.5" contype="0" conaffinity="0" group="3"/>'
        '<site name="attachment_site"'))
    site = model.site("attachment_site").id
    joints = derive.chain_joints(model, site, "vis")
    models = {}
    sva = derive.derive_sva(model, joints, site, np.eye(4), np.eye(4), "vis", models=models)
    # b2's link carries b2's capsule (and now the red box); b1's has b1's capsule
    assert set(models) == {link["id"] for link in sva["links"] if link.get("geometry")}
    end = sva["links"][-1]  # rides b2, whose capsule runs along x: the collider is turned
    scene = trimesh.load(io.BytesIO(models[end["id"]]), file_type="glb", force="scene")
    meshes = list(scene.dump())
    assert len(meshes) == 2  # the capsule and the red box; group 3 is hidden
    red = next(m for m in meshes if np.allclose(m.visual.material.baseColorFactor[:3], [255, 0, 0]))
    # in the collider's frame: back to b2's frame (the flange link's geometry rides b2), the box at x 0.05
    collider = robots.sva_pose(end["geometry"], "end")
    in_b2 = (collider @ np.c_[red.vertices, np.ones(len(red.vertices))].T).T[:, :3]
    np.testing.assert_allclose(in_b2.min(axis=0), [0.04, -0.02, -0.03], atol=1e-6)
    np.testing.assert_allclose(in_b2.max(axis=0), [0.06, 0.02, 0.03], atol=1e-6)
