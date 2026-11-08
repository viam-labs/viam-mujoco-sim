"""viam-labs:mujoco-sim:obstacle - a fixed object the motion planner also sees.

Place it with the standard frame (parent "world") and give the frame a
geometry: a box, sphere or capsule. RDK puts frame geometries in the frame
system, so Viam's motion planner avoids the same shape MuJoCo simulates.
Cylinders aren't in Viam's geometry API yet; use a capsule or box.

Attributes:
  world (string, required)  - name of the viam-labs:mujoco-sim:world component
  color ([r, g, b])         - 0 to 1 (default [0.7, 0.7, 0.7])
  static_friction (float)   - sliding friction, >= 0 (default 1)
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
        raise ValueError(f'{name}: set a frame with parent "world" and a geometry')
    validate_sim_component(config, needs_asset=False)  # world attr; parent must be world
    if not config.frame.HasField("geometry"):
        raise ValueError(f"{name}: set frame.geometry to a box, sphere or capsule")
    try:
        shape = geometry_shape(config.frame.geometry)
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from None
    attrs = get_attrs(config)
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
    position, quat = frame_pose(config)
    center = config.frame.geometry.center
    return {
        "shape": shape,
        "position": list(position or (0.0, 0.0, 0.0)),
        "quat": list(quat or (1.0, 0.0, 0.0, 0.0)),
        "center_pos": [center.x / 1000.0, center.y / 1000.0, center.z / 1000.0],
        "center_quat": _pose_quat(center),
        "color": [float(c) for c in color],
        "friction": friction,
    }


def obstacle_contribution(config: ComponentConfig) -> Contribution:
    """A fixed body named <name>/obstacle with one geom from the frame geometry."""
    spec = _parse(config)
    body_name = f"{config.name}/obstacle"

    def build(mj: mujoco.MjSpec) -> None:
        body = mj.worldbody.add_body(name=body_name, pos=spec["position"], quat=spec["quat"])
        body.add_geom(
            name=body_name,
            type=GEOM_TYPES[spec["shape"][0]],
            size=mj_size(spec["shape"]),
            pos=spec["center_pos"],
            quat=spec["center_quat"],
            rgba=[*spec["color"], 1.0],
            friction=[spec["friction"], 0.005, 0.0001],
            priority=PROP_CONTACT_PRIORITY,
            solref=list(PROP_SOLREF),
        )

    return Contribution(build=build)


class MujocoObstacle(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "obstacle")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._geometry: Geometry | None = None

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
    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        SimManager.get().set_contribution(self.name, obstacle_contribution(config))
        geometry = Geometry()
        geometry.CopyFrom(config.frame.geometry)
        geometry.label = self.name
        self._geometry = geometry

    async def close(self) -> None:
        if self._geometry is not None:
            await asyncio.to_thread(SimManager.get().remove_contribution, self.name)
            self._geometry = None

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        # RDK reads the frame config's geometry itself; this is for direct callers.
        return [self._geometry] if self._geometry is not None else []
