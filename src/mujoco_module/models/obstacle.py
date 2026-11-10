"""viam-labs:mujoco-sim:obstacle - a fixed object the motion planner also sees.

Place it with the standard frame (parent "world") and give it a shape: either
a frame geometry (box, sphere or capsule) or a mesh file. RDK puts frame
geometries in the frame system; for a mesh, RDK asks the obstacle for its
geometries (the frame has none) and gets the mesh's convex pieces. Either way,
Viam's motion planner avoids the same shapes MuJoCo simulates. Cylinders
aren't in Viam's geometry API yet; use a capsule or box.

Attributes:
  world (string, required)  - name of the viam-labs:mujoco-sim:world component
  mesh (string)             - .stl, .obj or .ply: an absolute path, a path relative
                              to the module's assets/ (e.g. "meshes/bin.stl"), or an
                              https URL. Instead of frame.geometry.
  scale (number | [x,y,z])  - mesh vertex scale (default 1: meters; 0.001 for mm)
  threshold (float)         - CoACD concavity, 0.01 to 1 (default 0.05)
  color ([r, g, b])         - 0 to 1 (default [0.7, 0.7, 0.7])
  static_friction (float)   - sliding friction, >= 0 (default 1)
  texture (string)          - a .png or .jpg image (path, assets-relative path or
                              https URL) on each face. Cameras see it; the 3D
                              viewer uses color.
"""

import asyncio
import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self

import mujoco
from viam.components.generic import Generic
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, Pose, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.spatialmath import Quaternion

from .. import FAMILY, NAMESPACE
from ..meshes import (
    UNTINTED_RGBA,
    Piece,
    Texture,
    add_pieces,
    add_texture,
    load_pieces,
    load_texture,
    mesh_geometry,
    parse_mesh,
    parse_texture,
)
from ..shapes import GEOM_TYPES, geometry_shape, mj_size
from ..sim import PROP_CONTACT_PRIORITY, PROP_SOLREF, Contribution, SimManager
from .utils import finite_number, frame_pose, get_attrs, validate_sim_component


def _pose_quat(pose: Pose) -> list[float]:
    """A geometry center's orientation as (w, x, y, z); unset means identity."""
    if pose.o_x == pose.o_y == pose.o_z == 0:
        return [1.0, 0.0, 0.0, 0.0]
    q = Quaternion.from_pose(pose)
    return [q.w, q.i, q.j, q.k]


def _parse(config: ComponentConfig) -> dict[str, Any]:
    """Everything the obstacle needs from its config; ValueError if invalid."""
    name = config.name
    if not config.HasField("frame"):
        raise ValueError(f'{name}: set a frame with parent "world", and a geometry or a mesh')
    validate_sim_component(config, needs_asset=False)  # world attr; parent must be world
    attrs = get_attrs(config)
    has_geometry = config.frame.HasField("geometry")
    shape, mesh = None, None
    if "mesh" in attrs:
        if has_geometry:
            raise ValueError(
                f"{name}: set frame.geometry or mesh, not both: "
                "RDK would plan against the frame geometry"
            )
        mesh = parse_mesh(name, attrs["mesh"], attrs.get("scale", 1), attrs.get("threshold", 0.05))
    else:
        stray = [key for key in ("scale", "threshold") if key in attrs]
        if stray:
            raise ValueError(f"{name}: {' and '.join(stray)} only apply with mesh")
        if not has_geometry:
            raise ValueError(
                f"{name}: set frame.geometry to a box, sphere or capsule, or set mesh"
            )
        try:
            shape = geometry_shape(config.frame.geometry)
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from None
    color = attrs.get("color", [0.7, 0.7, 0.7])
    if not (
        isinstance(color, list)
        and len(color) == 3
        and all(isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) for c in color)
    ):
        raise ValueError(f"{name}: color must be a list of 3 numbers")
    friction = finite_number(name, attrs, "static_friction", 1.0)
    if friction < 0:
        raise ValueError(f"{name}: static_friction must be non-negative")
    texture = parse_texture(name, attrs["texture"]) if "texture" in attrs else None
    position, quat = frame_pose(config)
    center = config.frame.geometry.center
    return {
        "shape": shape,
        "mesh": mesh,
        "position": list(position or (0.0, 0.0, 0.0)),
        "quat": list(quat or (1.0, 0.0, 0.0, 0.0)),
        "center_pos": [center.x / 1000.0, center.y / 1000.0, center.z / 1000.0],
        "center_quat": _pose_quat(center),
        "color": [float(c) for c in color],
        "friction": friction,
        "texture": texture,
    }


def obstacle_contribution(
    config: ComponentConfig, pieces: list[Piece] | None = None, texture: Texture | None = None
) -> Contribution:
    """A fixed body named <name>/obstacle: one geom from the frame geometry, or
    one convex mesh geom per piece (<name>/obstacle/<i>) for a mesh obstacle.
    texture is the loaded image (meshes.load_texture) when the config has one."""
    spec = _parse(config)
    if spec["mesh"] is not None and not pieces:
        raise ValueError(f"{config.name}: a mesh obstacle needs its pieces loaded")
    if spec["texture"] is not None and texture is None:
        raise ValueError(f"{config.name}: a textured obstacle needs its texture loaded")
    body_name = f"{config.name}/obstacle"
    contact = {
        "rgba": [*spec["color"], 1.0],
        "friction": [spec["friction"], 0.005, 0.0001],
        "priority": PROP_CONTACT_PRIORITY,
        "solref": list(PROP_SOLREF),
    }

    def build(mj: mujoco.MjSpec) -> None:
        geom = dict(contact)
        if texture is not None:
            geom["material"] = add_texture(mj, f"{body_name}/texture", texture)
            geom["rgba"] = list(UNTINTED_RGBA)
        body = mj.worldbody.add_body(name=body_name, pos=spec["position"], quat=spec["quat"])
        if pieces:
            add_pieces(mj, body, body_name, pieces, **geom)
            return
        body.add_geom(
            name=body_name,
            type=GEOM_TYPES[spec["shape"][0]],
            size=mj_size(spec["shape"]),
            pos=spec["center_pos"],
            quat=spec["center_quat"],
            **geom,
        )

    return Contribution(build=build)


class MujocoObstacle(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "obstacle")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._geometries: list[Geometry] | None = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        obstacle = cls(config.name)
        obstacle.reconfigure(config, dependencies)
        return obstacle

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> tuple[Sequence[str], Sequence[str]]:
        _parse(config)  # validates the world attr and frame parent too
        return [str(get_attrs(config)["world"])], []

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    # A mesh is decomposed here, synchronously: RDK builds the frame system after new()
    # returns and asks for these geometries then, so the pieces must exist by now.
    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        spec = _parse(config)
        pieces = load_pieces(self.name, spec["mesh"]) if spec["mesh"] is not None else None
        texture = load_texture(self.name, spec["texture"]) if spec["texture"] is not None else None
        if pieces:
            # the frame has no geometry, so RDK plans against these (several become a
            # zero-DoF model labelled <name>:piece<i>)
            geometries = [mesh_geometry([piece], f"piece{i}") for i, piece in enumerate(pieces)]
        else:
            geometry = Geometry()
            geometry.CopyFrom(config.frame.geometry)
            geometry.label = self.name
            geometries = [geometry]  # RDK uses the frame config's geometry; for direct callers
        SimManager.get().set_contribution(self.name, obstacle_contribution(config, pieces, texture))
        self._geometries = geometries

    async def close(self) -> None:
        if self._geometries is not None:
            await asyncio.to_thread(SimManager.get().remove_contribution, self.name)
            self._geometries = None

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        return list(self._geometries or [])
