"""MJCF scene files: scenery attached into the world at a frame.

A scene file's fixed and passively jointed geometry is simulated and served to
the motion planner (models/scene.py); its free bodies become props the world's
prop commands move. Nothing here touches the running sim.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from viam.proto.common import Geometry, Mesh, Pose, RectangularPrism, Vector3
from viam.spatialmath import Quaternion

from .meshes import pieces_ply
from .robots import load_file, resolve_path
from .shapes import viewer_geometry
from .sim import FREE_CONTYPE, LOGGER, PROP_CONTACT_PRIORITY, PROP_SOLREF

# The world owns these; a scene file's values would make every attach warn.
SIZE_FIELDS = (
    "memory", "nuserdata", "nuser_body", "nuser_jnt", "nuser_geom", "nuser_site",
    "nuser_cam", "nuser_tendon", "nuser_actuator", "nuser_sensor", "nconmax", "njmax", "nkey",
)
# element lists a scene may not have: Viam couldn't drive them, or they'd
# couple the scene to things it doesn't own
REJECTED = ("actuators", "tendons", "equalities", "flexes", "skins", "plugins")

_FREE = mujoco.mjtJoint.mjJNT_FREE
_PLANE = mujoco.mjtGeom.mjGEOM_PLANE
_HFIELD = mujoco.mjtGeom.mjGEOM_HFIELD
_KINDS = {
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
}


@dataclass(frozen=True)
class SceneProp:
    """A scene file's free body, moved by the world's prop commands like a
    config prop (it duck-types PropSpec's name, position, orientation and fixed)."""

    name: str  # "<scene>/<body>", also its free joint's name
    position: tuple[float, float, float]  # spawn, world frame, meters
    orientation_wxyz: tuple[float, float, float, float]
    fixed: bool = False


@dataclass(frozen=True)
class Scene:
    spec: mujoco.MjSpec  # checked, free joints named, movables' contacts set
    has_plane: bool
    geometries: tuple[Geometry, ...]  # planner geometry, scene frame, mm
    props: tuple[SceneProp, ...]
    has_lights: bool = False  # any <light>: the world leaves its default sun out


def copy_world_settings(world: mujoco.MjSpec, child: mujoco.MjSpec) -> None:
    """The world's <option> and <size> onto a scene before attaching: the
    world's win anyway, and any difference makes MuJoCo warn on every rebuild."""
    for name in dir(world.option):
        if not name.startswith("_"):
            try:
                setattr(child.option, name, getattr(world.option, name))
            except (AttributeError, TypeError):
                pass  # read-only
    for name in SIZE_FIELDS:
        try:
            setattr(child, name, getattr(world, name))
        except (AttributeError, TypeError):
            pass  # not in this MuJoCo


def _check(name: str, spec: mujoco.MjSpec) -> None:
    for kind in REJECTED:
        found = list(getattr(spec, kind, ()))
        if found:
            first = getattr(found[0], "name", "") or "#0"
            raise ValueError(
                f"{name}: a scene file can't have {kind} (found {first!r}); "
                "robots come from the arm, gripper and base components"
            )
    for body in spec.bodies:
        if body.mocap:
            raise ValueError(f"{name}: a scene file can't have mocap bodies (found {body.name!r})")


def _free_bodies(name: str, spec: mujoco.MjSpec) -> list[Any]:
    """Bodies with a free joint, each joint renamed to its body's name (the
    prop commands look up the joint and the body by one name)."""
    free = []
    for index, body in enumerate(spec.bodies):
        joints = [j for j in body.joints if j.type == _FREE]
        if not joints:
            continue
        if not body.name:
            raise ValueError(
                f"{name}: free body #{index} needs a name: the prop commands address it "
                f"as {name}/<name>"
            )
        try:
            joints[0].name = body.name
        except ValueError:
            raise ValueError(
                f"{name}: free body {body.name!r} names its free joint, but another joint "
                f"is already called {body.name!r}; rename one"
            ) from None
        free.append(body)
    return free


def _collides(geom: Any) -> bool:
    return bool(geom.contype or geom.conaffinity)


def _quat(xmat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.asarray(xmat, float).ravel())
    return quat


def _pose_mm(position: np.ndarray, quat: np.ndarray) -> Pose:
    w, x, y, z = (float(v) for v in quat)
    return Quaternion(w, x, y, z).to_pose(*(float(v) * 1000.0 for v in position))


def _hull(model: mujoco.MjModel, mesh: int) -> tuple[np.ndarray, np.ndarray]:
    """A mesh's convex hull from the compiled model's mesh_graph, as (vertices
    in the mesh frame, faces). MuJoCo 4 builds a hull for every mesh."""
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    vertices = model.mesh_vert[start : start + count]
    graph = model.mesh_graph[model.mesh_graphadr[mesh] :]
    numvert, numface = int(graph[0]), int(graph[1])
    first = 2 + 3 * numvert + 3 * numface  # after vert_edgeadr, vert_globalid, edge_localid
    faces = graph[first : first + 3 * numface].reshape(-1, 3)
    used, faces = np.unique(faces, return_inverse=True)  # only the hull's vertices
    return vertices[used], faces.reshape(-1, 3)


def geom_geometry(
    model: mujoco.MjModel, data: mujoco.MjData | None, geom: int, label: str
) -> Geometry:
    """A compiled geom as a Viam geometry (mm): at its pose in data's world
    frame, or in the geom's own frame when data is None. Ellipsoids become a
    box of their semi-axes; meshes their convex hull, vertices moved into that
    frame. ValueError for planes and height fields."""
    kind = int(model.geom_type[geom])
    if data is None:
        position, xmat = np.zeros(3), np.eye(3)
    else:
        position, xmat = data.geom_xpos[geom], data.geom_xmat[geom].reshape(3, 3)
    size = model.geom_size[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        vertices, faces = _hull(model, int(model.geom_dataid[geom]))
        world = vertices @ xmat.T + position
        return Geometry(
            center=Pose(o_z=1),
            mesh=Mesh(content_type="ply", mesh=pieces_ply([(world, faces)])),
            label=label,
        )
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        geometry = Geometry(
            box=RectangularPrism(dims_mm=Vector3(x=size[0] * 2000, y=size[1] * 2000, z=size[2] * 2000)),
            label=label,
        )
    elif kind in _KINDS:
        geometry = viewer_geometry(_KINDS[kind], size, label)
    else:
        raise ValueError(f"no Viam geometry for {mujoco.mjtGeom(kind).name}")
    geometry.center.CopyFrom(_pose_mm(position, _quat(xmat)))
    return geometry


def _label(model: mujoco.MjModel, geom: int) -> str:
    """The geom's name, else "<body>:<index in body>"."""
    name = model.geom(geom).name
    if name:
        return name
    body = int(model.geom_bodyid[geom])
    owner = "world" if body == 0 else model.body(body).name or f"body{body}"
    index = geom - int(model.body_geomadr[body])
    return f"{owner}:{index}"


def subtree_geoms(model: mujoco.MjModel, body: int) -> list[int]:
    """The colliding geoms of a top-level body (a free body always is) and its
    descendants: the geoms whose body's root is it."""
    in_tree = model.body_rootid[model.geom_bodyid] == body
    colliding = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    return [int(g) for g in np.flatnonzero(in_tree & colliding)]


def load_scene(
    name: str,
    source: str,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> Scene:
    """Load, check and prepare a scene file placed at (position, quat) in the
    world. ValueError naming `name` for anything the scene can't take."""
    if source.lower().endswith(".urdf"):
        raise ValueError(f"{name}: a scene must be MJCF (.xml), got {source!r}")
    path = resolve_path(source, name)
    try:
        root = ET.parse(path).getroot().tag
    except (ET.ParseError, OSError):
        root = None  # MuJoCo's own load error names the problem
    if root == "robot":
        raise ValueError(f"{name}: a scene must be MJCF (.xml), got a URDF {source!r}")
    spec = load_file(path, name)
    _check(name, spec)
    for key in list(spec.keys):
        spec.delete(key)  # its qpos length wouldn't match the world's
        LOGGER.info("%s: dropped keyframe %r", name, key.name)
    free = _free_bodies(name, spec)
    for body in free:
        for geom in body.find_all(mujoco.mjtObj.mjOBJ_GEOM):
            if _collides(geom):
                geom.contype = FREE_CONTYPE
                geom.priority = PROP_CONTACT_PRIORITY
                geom.solref = list(PROP_SOLREF)
    has_plane = any(g.type == _PLANE and _collides(g) for g in spec.worldbody.geoms)

    try:
        model = spec.copy().compile()
    except ValueError as error:
        raise ValueError(f"{name}: cannot compile scene {source}: {error}") from None
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    free_ids = {model.body(body.name).id for body in free}
    movable = {g for body in free_ids for g in subtree_geoms(model, body)}
    geometries = []
    for geom in range(model.ngeom):
        if geom in movable or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        kind = int(model.geom_type[geom])
        if kind == _PLANE:
            continue
        if kind == _HFIELD:
            LOGGER.info("%s: the planner doesn't see height field %s", name, _label(model, geom))
            continue
        geometries.append(geom_geometry(model, data, geom, _label(model, geom)))
    if not geometries:
        LOGGER.info("%s: the planner sees nothing of this scene", name)

    origin = np.asarray(position, float)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quat, float))
    rotation = rotation.reshape(3, 3)
    props = []
    for body in free:
        compiled = model.body(body.name).id
        spawn = rotation @ data.xpos[compiled] + origin
        orientation = np.zeros(4)
        mujoco.mju_mulQuat(orientation, np.asarray(quat, float), data.xquat[compiled])
        props.append(SceneProp(
            name=f"{name}/{body.name}",
            position=tuple(float(v) for v in spawn),
            orientation_wxyz=tuple(float(v) for v in orientation),
        ))
    return Scene(spec, has_plane, tuple(geometries), tuple(props), has_lights=bool(spec.lights))
