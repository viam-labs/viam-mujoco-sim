"""viam-labs:mujoco-sim:scale - a platform scale, served as a sensor.

Attributes:
  world (string, required) - name of the viam-labs:mujoco-sim:world component
  size_m ([x, y, z], required) - the platform box, meters
  color ([r, g, b])        - 0 to 1

Place it with the standard frame, parent "world": the frame is the centre of
the platform's top face. The scale weighs the contact forces pressing on the
platform, filtered with a 0.1 s time constant.

Readings: weight_kg (minus the tare), stable (the filtered weight moved less
than 5 g over the last 0.3 s). DoCommand {"command": "tare"} sets the tare to
the current weight. RDK never asks a sensor for its shape: give the frame a
geometry box for the planner (the module logs the one to paste); the world
state store shows the platform in the 3D scene.
"""

import asyncio
import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self

from viam.components.sensor import Sensor
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, Pose, RectangularPrism, ResourceName, Vector3
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import SensorReading, ValueTypes

from .. import FAMILY, NAMESPACE
from ..handles import ScaleHandle
from ..sim import SimManager
from .utils import apply_frame_to_attrs, get_attrs, publish_shapes, unpublish_shapes, validate_sim_component


def _numbers(value: Any, size: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == size
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value)
    )


class MujocoScale(Sensor, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "scale")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: ScaleHandle | None = None

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        scale = cls(config.name)
        scale.reconfigure(config, dependencies)
        return scale

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        if not config.HasField("frame"):
            raise ValueError(f'{config.name}: set a frame with parent "world": the centre of the platform\'s top face')
        attrs = get_attrs(config)
        size = attrs.get("size_m")
        if not (_numbers(size, 3) and all(v > 0 for v in size)):
            raise ValueError(f"{config.name}: size_m must be a list of 3 positive numbers")
        if "color" in attrs and not _numbers(attrs["color"], 3):
            raise ValueError(f"{config.name}: color must be a list of 3 numbers")
        return validate_sim_component(config, needs_asset=False)

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        attrs: dict[str, Any] = apply_frame_to_attrs(config, get_attrs(config))
        self._handle = ScaleHandle(SimManager.get(), self.name, attrs)
        publish_shapes(config, f"{self.name}/platform", self._geometries(), self._handle.color)

    async def close(self) -> None:
        if self._handle is not None:
            await asyncio.to_thread(unpublish_shapes, self.name)
            await asyncio.to_thread(self._handle.close)
            self._handle = None

    def _h(self) -> ScaleHandle:
        if self._handle is None:
            raise RuntimeError(f"scale {self.name} is not attached to the sim")
        return self._handle

    async def get_readings(self, **kwargs) -> Mapping[str, SensorReading]:
        return await asyncio.to_thread(self._h().readings)

    async def do_command(
        self, command: Mapping[str, ValueTypes], *, timeout: float | None = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        if command.get("command") == "tare":
            return {"tare_kg": await asyncio.to_thread(self._h().tare)}
        raise ValueError(f"unknown command: {command}")

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        return self._geometries()

    def _geometries(self) -> list[Geometry]:
        """The platform box, in the scale's frame. RDK never calls
        get_geometries on a sensor: the world state store publishes it to the
        3D scene, and the planner sees the frame geometry."""
        x, y, z = (v * 1000.0 for v in self._h().size)
        return [
            Geometry(
                center=Pose(z=-z / 2, o_z=1),
                box=RectangularPrism(dims_mm=Vector3(x=x, y=y, z=z)),
                label=f"{self.name}-platform",
            )
        ]
