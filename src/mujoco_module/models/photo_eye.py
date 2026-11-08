"""viam-labs:mujoco-sim:photo_eye - a through-beam photo-eye, served as a sensor.

Attributes:
  world (string, required) - name of the viam-labs:mujoco-sim:world component
  range_m (float)          - beam length (default 2)

Place it with the standard frame: the beam starts at the frame origin and
runs along +X. frame.parent is "world" or another mujoco-sim component; the
eye never sees its parent or the bodies below it. To look across a conveyor,
parent it on the conveyor, so its rails and belt are left out (a
world-parented eye would hit the rails).

Readings: blocked (something within range), distance_m (to the hit, or
range_m), count (times the beam became blocked since configure or reset).
DoCommand {"command": "reset"} zeroes count; a world reset does too.
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self

from viam.components.sensor import Sensor
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import SensorReading, ValueTypes

from .. import FAMILY, NAMESPACE
from ..handles import PhotoEyeHandle
from ..sim import SimManager
from .utils import apply_frame_to_attrs, get_attrs, positive_number, validate_sim_component


class MujocoPhotoEye(Sensor, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "photo_eye")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: PhotoEyeHandle | None = None

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        eye = cls(config.name)
        eye.reconfigure(config, dependencies)
        return eye

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        positive_number(config.name, get_attrs(config), "range_m", 2.0)
        return validate_sim_component(config, needs_asset=False, world_frame_only=False)

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        attrs: dict[str, Any] = apply_frame_to_attrs(config, get_attrs(config))
        if config.HasField("frame") and config.frame.parent != "world":
            attrs["parent"] = config.frame.parent
        self._handle = PhotoEyeHandle(SimManager.get(), self.name, attrs)

    async def close(self) -> None:
        if self._handle is not None:
            await asyncio.to_thread(self._handle.close)
            self._handle = None

    def _h(self) -> PhotoEyeHandle:
        if self._handle is None:
            raise RuntimeError(f"photo-eye {self.name} is not attached to the sim")
        return self._handle

    async def get_readings(self, **kwargs) -> Mapping[str, SensorReading]:
        return await asyncio.to_thread(self._h().readings)

    async def do_command(
        self, command: Mapping[str, ValueTypes], *, timeout: float | None = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        if command.get("command") == "reset":
            await asyncio.to_thread(self._h().reset_count)
            return {"count": 0}
        raise ValueError(f"unknown command: {command}")

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        return []
