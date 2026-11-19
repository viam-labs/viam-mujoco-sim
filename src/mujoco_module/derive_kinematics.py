"""Viam SVA kinematics derived from a compiled MJCF arm, for an arm configured
with an .xml mjcf and no kinematics_url. The derived file matches MuJoCo's
forward kinematics exactly, so the configure-time frame check can only fail on
a bug here."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from logging import getLogger

import mujoco
import numpy as np
from viam.spatialmath import Quaternion

from .robots import quat, sva_pose

LOGGER = getLogger(__name__)
_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
UNLIMITED_DEG = 360.0  # a continuous joint's planning range


def check_joint(model: mujoco.MjModel, joint: int, label: str) -> None:
    """A chain joint must be a hinge or slide, alone on its body, a hinge through its body's origin."""
    body = int(model.jnt_bodyid[joint])
    name = model.body(body).name or f"body {body}"
    if int(model.body_jntnum[body]) > 1:
        raise ValueError(f"{label}: body {name!r} has {int(model.body_jntnum[body])} joints; an arm has one joint per link")
    if int(model.jnt_type[joint]) not in (_HINGE, _SLIDE):
        raise ValueError(f"{label}: body {name!r} has a ball or free joint; an arm's joints are hinges or slides")
    if int(model.jnt_type[joint]) == _HINGE and np.any(model.jnt_pos[joint]):
        raise ValueError(
            f"{label}: hinge {model.joint(joint).name!r} has a nonzero pos (its axis is off its "
            "body's origin), which derived kinematics don't support; set kinematics_url"
        )


def chain_joints(model: mujoco.MjModel, flange_site: int, label: str, base_body: int = 0) -> list[int]:
    """The arm's joints, base first: the hinge and slide joints on the flange
    site's body and its ancestors below base_body (the file's world, or a robot
    limb's chain base). Each chain body has at most one joint."""
    joints = []
    body = int(model.site_bodyid[flange_site])
    while body != base_body:
        if body == 0:
            raise ValueError(f"{label}: the flange site doesn't hang from {model.body(base_body).name!r}")
        if int(model.body_jntnum[body]):
            first = int(model.body_jntadr[body])
            check_joint(model, first, label)
            joints.append(first)
        body = int(model.body_parentid[body])
    if not joints:
        raise ValueError(f"{label}: no hinge or slide joints between the base and the flange site")
    return joints[::-1]


def given_joints(
    model: mujoco.MjModel, names: Sequence[str], flange_site: int, label: str, base_body: int = 0
) -> list[int]:
    """The configured `joints` names as ids. They must be exactly the flange
    site's chain (below base_body), base first: a skipped joint would be
    planned as fixed while the sim leaves it free."""
    chain = chain_joints(model, flange_site, label, base_body)
    ids = []
    for name in names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"{label}: no joint {name!r}")
        ids.append(joint)
    if ids != chain:
        expected = [model.joint(j).name for j in chain]
        raise ValueError(f"{label}: joints must be the chain from the base to the flange site, base first: {expected}")
    return ids


def _pose(pos: Sequence[float], mat: Sequence[float]) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = np.asarray(mat).reshape(3, 3)
    result[:3, 3] = pos
    return result


def _mm(v: float) -> float:
    return float(v) * 1000.0


def _sva_pose(matrix: np.ndarray) -> dict:
    """translation (mm) and ov_degrees orientation, as an SVA link or geometry carries them."""
    ov = Quaternion(*quat(matrix)).to_orientation_vector()
    t = matrix[:3, 3]
    return {
        "translation": {"x": _mm(t[0]), "y": _mm(t[1]), "z": _mm(t[2])},
        "orientation": {"type": "ov_degrees", "value": {
            "x": float(ov.o_x), "y": float(ov.o_y), "z": float(ov.o_z), "th": math.degrees(ov.theta),
        }},
    }


def _colliding(model: mujoco.MjModel, bodies: Sequence[int]) -> list[int]:
    wanted = set(bodies)
    return [g for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in wanted and (model.geom_contype[g] or model.geom_conaffinity[g])]


_RING = 16  # samples around a round geom's axis
_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
_SPHERE, _CAPSULE, _CYLINDER = (int(t) for t in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER))


def geom_points(model: mujoco.MjModel, geom: int) -> np.ndarray:
    """Points on a geom's surface, in its own frame: a mesh's vertices, rings
    around a round geom (ponytail: 16 samples under-cover a circle's radius by 2 %), else
    its AABB corners. AABB corners of a mesh or round geom would fatten the fit."""
    kind, size = int(model.geom_type[geom]), model.geom_size[geom]
    if kind == _MESH:
        mesh = int(model.geom_dataid[geom])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        return np.asarray(model.mesh_vert[start:start + count], float)
    if kind in (_SPHERE, _CAPSULE, _CYLINDER):
        radius = float(size[0])
        half = 0.0 if kind == _SPHERE else float(size[1])
        angles = np.linspace(0, 2 * np.pi, _RING, endpoint=False)
        ring = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
        points = [np.column_stack([ring, np.full(_RING, z)]) for z in (-half, half)]
        tip = half + (radius if kind != _CYLINDER else 0.0)
        points.append(np.array([[0, 0, -tip], [0, 0, tip]]))
        return np.vstack(points)
    centre, half = model.geom_aabb[geom][:3], model.geom_aabb[geom][3:]
    return np.array([centre + half * np.array(s) for s in itertools.product((-1, 1), repeat=3)])


def fit(model: mujoco.MjModel, data: mujoco.MjData, geoms: Sequence[int], frame: np.ndarray) -> dict | None:
    """The smaller of a box and a capsule covering the geoms' surface points,
    as SVA geometry posed in `frame` (a world pose). None without geoms."""
    if not geoms:
        return None
    world = [geom_points(model, g) @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] for g in geoms]
    for g in geoms:  # a round geom tilted against the frame: its extremes along each frame axis
        kind = int(model.geom_type[g])
        if kind in (_SPHERE, _CAPSULE, _CYLINDER):
            half = 0.0 if kind == _SPHERE else float(model.geom_size[g][1])
            z = data.geom_xmat[g].reshape(3, 3)[:, 2]
            ends = np.array([data.geom_xpos[g] + s * half * z for s in (-1, 1)])
            axes = frame[:3, :3].T  # rows: the frame's axes in world
            if kind == _CYLINDER:  # a flat cap's rim reaches furthest along the axis's part in its plane
                rim = axes - np.outer(axes @ z, z)
                norms = np.linalg.norm(rim, axis=1)
                axes = rim[norms > 1e-9] / norms[norms > 1e-9, None]
            reach = float(model.geom_size[g][0]) * axes
            world.append((ends[:, None, None, :] + np.array([-1, 1])[None, :, None, None] * reach).reshape(-1, 3))
    p = (np.vstack(world) - frame[:3, 3]) @ frame[:3, :3]  # world -> frame
    low, high = p.min(axis=0), p.max(axis=0)
    centre, extent = (low + high) / 2, high - low
    axis = int(np.argmax(extent))
    across = [i for i in range(3) if i != axis]
    radial = np.linalg.norm(p[:, across] - centre[across], axis=1)
    radius = float(radial.max())
    # the shortest segment along the axis that keeps every point within radius of it
    reach = np.sqrt(np.maximum(radius**2 - radial**2, 0.0))
    top, bottom = float((p[:, axis] - reach).max()), float((p[:, axis] + reach).min())
    if top < bottom:
        top = bottom = (top + bottom) / 2
    box_volume = float(np.prod(extent))
    capsule_volume = math.pi * radius**2 * (top - bottom) + 4 / 3 * math.pi * radius**3
    if box_volume <= capsule_volume:
        translation = {"x": _mm(centre[0]), "y": _mm(centre[1]), "z": _mm(centre[2])}
        return {"x": _mm(extent[0]), "y": _mm(extent[1]), "z": _mm(extent[2]), "translation": translation}
    middle = centre.copy()
    middle[axis] = (top + bottom) / 2
    geometry = {
        "r": _mm(radius), "l": _mm(top - bottom + 2 * radius),
        "translation": {"x": _mm(middle[0]), "y": _mm(middle[1]), "z": _mm(middle[2])},
    }
    if axis != 2:  # RDK capsules run along z; turn z onto the long axis
        geometry["orientation"] = {"type": "ov_degrees", "value": {
            "x": float(axis == 0), "y": float(axis == 1), "z": 0.0, "th": 0.0,
        }}
    return geometry


def geom_piece(model: mujoco.MjModel, geom: int, hull: bool) -> tuple[np.ndarray, np.ndarray]:
    """A geom as triangles in its own frame (m): a mesh's convex hull (what MuJoCo collides) or,
    with hull False, its full mesh; a primitive tessellated. ValueError for planes and height fields."""
    import trimesh

    from .scenes import _hull

    kind, size = int(model.geom_type[geom]), model.geom_size[geom]
    if kind == _MESH:
        mesh = int(model.geom_dataid[geom])
        if hull:
            return _hull(model, mesh)
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        first, faces = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        return np.asarray(model.mesh_vert[start:start + count], float), np.asarray(model.mesh_face[first:first + faces])
    # ponytail: coarse primitives (vertices on the surface, flats just inside it, ~2 % of the radius)
    if kind == _SPHERE:
        shape = trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    elif kind == _CAPSULE:
        shape = trimesh.creation.capsule(height=2 * float(size[1]), radius=float(size[0]), count=[16, 8])
    elif kind == _CYLINDER:
        shape = trimesh.creation.cylinder(radius=float(size[0]), height=2 * float(size[1]), sections=16)
    elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        shape = trimesh.creation.box(extents=2 * np.asarray(size, float))
    elif kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        shape = trimesh.creation.icosphere(subdivisions=2)
        shape.apply_scale(np.asarray(size, float))
    else:
        raise ValueError(f"no triangles for a {mujoco.mjtGeom(kind).name} geom")
    return np.asarray(shape.vertices), np.asarray(shape.faces)


def _in(model: mujoco.MjModel, data: mujoco.MjData, geom: int, frame: np.ndarray, hull: bool):
    """A geom's triangles moved into `frame` (a world pose)."""
    vertices, faces = geom_piece(model, geom, hull)
    world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    return (world - frame[:3, 3]) @ frame[:3, :3], faces


def mesh_geometry(model: mujoco.MjModel, data: mujoco.MjData, geoms: Sequence[int], frame: np.ndarray) -> dict | None:
    """The geoms' collision hulls merged into one PLY mesh, as SVA geometry posed in `frame`."""
    import base64

    from .meshes import pieces_ply

    geoms = [g for g in geoms if int(model.geom_type[g]) not in _FLAT]  # no hull; fit's AABB covers them in shapes
    if not geoms:
        return None
    ply = pieces_ply([_in(model, data, g, frame, hull=True) for g in geoms])
    return {"type": "mesh", "mesh_content_type": "ply", "mesh_data": base64.b64encode(ply).decode()}


_FLAT = (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD))
_DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)


def _rgba(model: mujoco.MjModel, geom: int) -> np.ndarray:
    """A geom's colour as MuJoCo draws it: its own rgba when set, else its material's."""
    mat = int(model.geom_matid[geom])
    own = model.geom_rgba[geom]
    return np.asarray(model.mat_rgba[mat] if mat >= 0 and np.allclose(own, _DEFAULT_RGBA) else own, float)


def _visible(model: mujoco.MjModel, bodies: Sequence[int]) -> list[int]:
    """The bodies' geoms MuJoCo's viewer shows by default: groups 0-2, not transparent."""
    wanted = set(bodies)
    return [g for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in wanted and model.geom_group[g] <= 2 and _rgba(model, g)[3] > 0
            and int(model.geom_type[g]) not in _FLAT]


def visual_glb(model: mujoco.MjModel, data: mujoco.MjData, geoms: Sequence[int], frame: np.ndarray) -> bytes:
    """The geoms' full meshes and primitives, each in its colour (textures are dropped), as one
    GLB (m) in `frame`."""
    import trimesh

    scene = trimesh.Scene()
    for g in geoms:
        vertices, faces = _in(model, data, g, frame, hull=False)
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        mesh.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(baseColorFactor=np.round(_rgba(model, g) * 255).astype(np.uint8)))
        scene.add_geometry(mesh, geom_name=f"geom{g}")
    return scene.export(file_type="glb")


def _joint(model: mujoco.MjModel, joint: int, parent: str, label: str) -> dict:
    name = model.joint(joint).name
    if not name:
        raise ValueError(f"{label}: an arm joint has no name; name the MJCF's arm joints")
    slide = int(model.jnt_type[joint]) == _SLIDE
    if model.jnt_limited[joint]:
        scale = 1000.0 if slide else 180.0 / math.pi
        low, high = (float(v) * scale for v in model.jnt_range[joint])
    elif slide:
        raise ValueError(f"{label}: slide joint {name!r} has no range; Viam's planner needs joint limits")
    else:
        LOGGER.warning("%s: hinge %s has no range; planning with -360..360 degrees", label, name)
        low, high = -UNLIMITED_DEG, UNLIMITED_DEG
    axis = model.jnt_axis[joint]
    return {
        "id": name, "type": "prismatic" if slide else "revolute", "parent": parent,
        "axis": {"x": float(axis[0]), "y": float(axis[1]), "z": float(axis[2])},
        "min": low, "max": high,
    }


def derive_sva(
    model: mujoco.MjModel,
    joints: Sequence[int],
    flange_site: int,
    base_offset: np.ndarray,
    flange_offset: np.ndarray,
    label: str,
    meshes: bool = False,
    models: dict[str, bytes] | None = None,
    base_body: int = 0,
) -> dict:
    """SVA kinematics for the chain `joints` (chain_joints) ending at
    flange_site, in the arm frame (base_body's pose times inverse(base_offset):
    the MJCF's world, or a robot limb's chain base), at all arm joints zero. The
    first link carries base_body and the bodies below it, not a parent limb's. Each body's collision shape goes on the link after its
    joint, since RDK poses a link's geometry in its parent's frame: a fitted box
    or capsule, or with `meshes` its collision hulls. With `models`, fills it
    with a GLB of each shaped link's visible geoms, in its shape's frame (where
    a viewer draws a link's 3D model), keyed by link id. A link with no colliding
    geoms has no shape for a viewer to draw a model on, so it gets none."""
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[list(joints)]] = 0.0  # 0, not qpos0: a joint ref is folded into the offsets
    mujoco.mj_kinematics(model, data)

    def world(body: int) -> np.ndarray:
        return _pose(data.xpos[body], data.xmat[body])

    bodies = [int(model.jnt_bodyid[j]) for j in joints]
    names = [model.joint(j).name for j in joints]
    # every body on the path from the root (or a limb's chain base) to the flange,
    # under the jointed body (None: the base) whose shape folds in its geoms
    path, body = [], int(model.site_bodyid[flange_site])
    while body != 0:
        path.append(body)
        if body == base_body:
            break
        body = int(model.body_parentid[body])
    carried: dict[int | None, list[int]] = {}
    owner: int | None = None
    for body in reversed(path):
        if body in bodies:
            owner = body
        carried.setdefault(owner, []).append(body)

    taken = set(names)

    def link_id(name: str) -> str:
        result = name if name and name not in taken else f"{name or 'link'}_link"
        while result in taken:
            result += "_"
        taken.add(result)
        return result

    def link(name: str, parent: str, offset: np.ndarray, owner: int | None, frame: np.ndarray) -> dict:
        entry = {"id": link_id(name), "parent": parent, **_sva_pose(offset)}
        geometry = (mesh_geometry if meshes else fit)(model, data, _colliding(model, carried.get(owner, [])), frame)
        if geometry:
            entry["geometry"] = geometry
            visible = _visible(model, carried.get(owner, []))
            if models is not None and visible:
                models[entry["id"]] = visual_glb(model, data, visible, frame @ sva_pose(geometry, entry["id"]))
        return entry

    to_mj = world(base_body) @ np.linalg.inv(base_offset)  # the arm frame's world pose (body 0: the identity)
    links, sva_joints, parent = [], [], "world"
    for i, (joint, body) in enumerate(zip(joints, bodies)):
        if i == 0:
            entry = link(model.body(body).name, parent, np.linalg.inv(to_mj) @ world(body), None, to_mj)
        else:
            above = bodies[i - 1]
            entry = link(model.body(body).name, parent, np.linalg.inv(world(above)) @ world(body), above, world(above))
        links.append(entry)
        sva_joints.append(_joint(model, joint, entry["id"], label))
        parent = names[i]
    site = _pose(data.site_xpos[flange_site], data.site_xmat[flange_site])
    end = np.linalg.inv(world(bodies[-1])) @ site @ flange_offset
    links.append(link(model.site(flange_site).name or "flange", parent, end, bodies[-1], world(bodies[-1])))
    return {"name": label, "kinematic_param_type": "SVA", "links": links, "joints": sva_joints}
