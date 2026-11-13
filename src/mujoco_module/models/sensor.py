"""viam-labs:mujoco-sim:sensor - the <sensor>s an MJCF file declares, served as a sensor.

Attributes:
  world (string, required)  - name of the viam-labs:mujoco-sim:world component
  source (string, required) - the component that loaded the file (a scene,
                              robot, arm, gripper or base with its own mjcf)
  sensors ([string])        - names as written in the file; omitted: all of
                              the file's named sensors

Readings: {name in the file: value}, a number for a one-value sensor, else a
list (framepos [x, y, z], framequat [w, x, y, z], force [fx, fy, fz]), in
MuJoCo's units (SI, angles in radians). The sensors sit where the file puts
them: a frame, if set, is ignored.
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
from ..sensors import MjcfSensorHandle
from ..sim import SimManager
from .utils import get_attrs, validate_sim_component


class MujocoSensor(Sensor, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "sensor")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: MjcfSensorHandle | None = None

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        sensor = cls(config.name)
        sensor.reconfigure(config, dependencies)
        return sensor

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        deps = validate_sim_component(config, needs_asset=False, world_frame_only=False)
        attrs: dict[str, Any] = get_attrs(config)
        source = attrs.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError(
                f"{config.name}: source must be the name of the component whose MJCF file declares the sensors"
            )
        if "sensors" in attrs:
            sensors = attrs["sensors"]
            if not (
                isinstance(sensors, list)
                and sensors
                and all(isinstance(s, str) and s for s in sensors)
                and len(set(sensors)) == len(sensors)
            ):
                raise ValueError(f"{config.name}: sensors must be a non-empty list of distinct sensor names")
        return deps

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        attrs = get_attrs(config)
        self._handle = MjcfSensorHandle(SimManager.get(), self.name, attrs["source"], attrs.get("sensors"))

    async def close(self) -> None:
        self._handle = None

    async def get_readings(self, **kwargs) -> Mapping[str, SensorReading]:
        if self._handle is None:
            raise RuntimeError(f"sensor {self.name} is not attached to the sim")
        return await asyncio.to_thread(self._handle.read)

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        return []

    async def do_command(
        self, command: Mapping[str, ValueTypes], *, timeout: float | None = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        raise ValueError(f"unknown command: {command}")
