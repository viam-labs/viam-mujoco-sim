"""Shape math shared by props, obstacles and the world state store.

A shape is (kind, size): kind is "box", "sphere", "cylinder" or "capsule",
and size is in MuJoCo's convention, in meters: box half lengths, sphere
(radius,), cylinder (radius, half height), capsule (radius, half the length of
its straight part). Viam geometries are in mm, and a Viam capsule's length is
tip to tip.
"""

import math
from collections.abc import Sequence

import mujoco
from viam.proto.common import Capsule, Geometry, Mesh, Pose, RectangularPrism, Sphere, Vector3

Shape = tuple[str, tuple[float, ...]]

GEOM_TYPES = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
}
PROP_TYPES = {"cube": "box", "sphere": "sphere", "cylinder": "cylinder", "capsule": "capsule"}
CYLINDER_SEGMENTS = 16


def prop_shape(name: str, kind: str, size: float, scale: Sequence[float]) -> Shape:
    """A prop fills a box of size x scale meters, whatever its type."""
    if kind not in PROP_TYPES:
        raise ValueError(f"prop {name}: type must be one of {sorted(PROP_TYPES)}, got {kind!r}")
    sx, sy, sz = (size * item for item in scale)
    if kind == "cube":
        return "box", (sx / 2, sy / 2, sz / 2)
    if kind == "sphere":
        if not scale[0] == scale[1] == scale[2]:
            raise ValueError(f"prop {name}: a sphere's scale must be uniform")
        return "sphere", (sx / 2,)
    if scale[0] != scale[1]:
        raise ValueError(f"prop {name}: a {kind}'s scale[0] and scale[1] must match")
    radius = sx / 2
    if kind == "cylinder":
        return "cylinder", (radius, sz / 2)
    if sz <= sx:
        raise ValueError(f"prop {name}: a capsule's height must be greater than its diameter")
    return "capsule", (radius, sz / 2 - radius)


def _positive_mm(values: Sequence[float], what: str) -> None:
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError(f"{what} must be positive")


def geometry_shape(geometry: Geometry) -> Shape:
    """A Viam frame geometry (mm) as a MuJoCo shape (m). Box, sphere, capsule."""
    which = geometry.WhichOneof("geometry_type")
    if which == "box":
        dims = geometry.box.dims_mm
        values = (dims.x, dims.y, dims.z)
        _positive_mm(values, "box dims_mm")
        return "box", tuple(v / 2000.0 for v in values)
    if which == "sphere":
        _positive_mm((geometry.sphere.radius_mm,), "sphere radius_mm")
        return "sphere", (geometry.sphere.radius_mm / 1000.0,)
    if which == "capsule":
        radius, length = geometry.capsule.radius_mm, geometry.capsule.length_mm
        _positive_mm((radius, length), "capsule radius_mm and length_mm")
        if length < 2 * radius:
            raise ValueError("a capsule's length_mm must be at least twice its radius_mm")
        if length == 2 * radius:
            return "sphere", (radius / 1000.0,)
        return "capsule", (radius / 1000.0, (length - 2 * radius) / 2000.0)
    if which == "mesh":
        raise ValueError("mesh obstacles aren't supported yet; use a box, sphere or capsule")
    raise ValueError("set frame.geometry to a box, sphere or capsule")


def mj_size(shape: Shape) -> list[float]:
    """MjSpec geom sizes are three numbers; unused ones are 0."""
    return [*shape[1], 0.0, 0.0][:3]


def cylinder_ply(radius: float, height: float, segments: int = CYLINDER_SEGMENTS) -> bytes:
    """A capped cylinder as ASCII PLY, in meters, centred, axis along Z.
    Viam reads PLY vertices as meters (RDK scales them to mm)."""
    half = height / 2
    ring = [
        (radius * math.cos(2 * math.pi * i / segments), radius * math.sin(2 * math.pi * i / segments))
        for i in range(segments)
    ]
    vertices = [(x, y, -half) for x, y in ring] + [(x, y, half) for x, y in ring]
    vertices += [(0.0, 0.0, -half), (0.0, 0.0, half)]
    bottom, top = 2 * segments, 2 * segments + 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces += [(i, j, segments + j), (i, segments + j, segments + i)]  # side
        faces += [(bottom, j, i), (top, segments + i, segments + j)]  # caps
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
        *(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices),
        *(f"3 {a} {b} {c}" for a, b, c in faces),
    ]
    return ("\n".join(lines) + "\n").encode()


def viewer_geometry(kind: str, size: Sequence[float], label: str) -> Geometry:
    """A MuJoCo shape as a Viam geometry (mm), centred on its frame. The API
    has no cylinder (viamrobotics/api#894 is a draft), so it goes as a mesh."""
    center = Pose(o_z=1)
    if kind == "box":
        dims = Vector3(x=size[0] * 2000, y=size[1] * 2000, z=size[2] * 2000)
        return Geometry(center=center, box=RectangularPrism(dims_mm=dims), label=label)
    if kind == "sphere":
        return Geometry(center=center, sphere=Sphere(radius_mm=size[0] * 1000), label=label)
    if kind == "capsule":
        radius, half = size[0], size[1]
        capsule = Capsule(radius_mm=radius * 1000, length_mm=(2 * half + 2 * radius) * 1000)
        return Geometry(center=center, capsule=capsule, label=label)
    if kind == "cylinder":
        mesh = Mesh(content_type="ply", mesh=cylinder_ply(size[0], 2 * size[1]))
        return Geometry(center=center, mesh=mesh, label=label)
    raise ValueError(f"unknown shape {kind!r}")
