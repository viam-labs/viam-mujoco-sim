"""viam-labs:mujoco-sim:movement_sensor - wheel odometry for a mujoco-sim base.

Attributes:
  world (string, required)    - name of the viam-labs:mujoco-sim:world component
  base (string, required)     - the viam-labs:mujoco-sim:base it rides on
  linear_scale_error (float)  - reported speed = true speed x (1 + e) (default 0)
  heading_drift_dps (float)   - constant yaw-rate bias, deg/s (default 0)

Odometry starts at zero where the base is when the sensor starts, a world reset,
or reset_odometry, with +Y forward. Readings are exact unless drift is set.
"""

import asyncio
import math
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.movement_sensor import GeoPoint, MovementSensor, Orientation, Vector3
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName
from viam.proto.component.movementsensor import GetAccuracyResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import SensorReading, ValueTypes

from .. import FAMILY, NAMESPACE
from ..odometry import OdometryHandle, OdometryReading, odometry_geo_point
from ..sim import SimManager
from .utils import finite_number, get_attrs, validate_sim_component


def _geo(r: OdometryReading) -> GeoPoint:
    lat, lng = odometry_geo_point(r.x, r.y)
    return GeoPoint(latitude=lat, longitude=lng)


def _linear(r: OdometryReading) -> Vector3:
    return Vector3(x=r.lateral, y=r.forward, z=0.0)


def _angular(r: OdometryReading) -> Vector3:
    return Vector3(x=0.0, y=0.0, z=math.degrees(r.yaw_rate))


def _orientation(r: OdometryReading) -> Orientation:
    theta = math.degrees(math.remainder(r.yaw, 2 * math.pi))
    return Orientation(o_x=0.0, o_y=0.0, o_z=1.0, theta=theta)


def _compass(r: OdometryReading) -> float:
    return (-math.degrees(r.yaw)) % 360.0


class MujocoMovementSensor(MovementSensor, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "movement_sensor")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[OdometryHandle] = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        sensor = cls(config.name)
        sensor.reconfigure(config, dependencies)
        return sensor

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = get_attrs(config)
        deps, optional = validate_sim_component(
            config, needs_asset=False, world_frame_only=False
        )
        base = attrs.get("base")
        if not base or not isinstance(base, str):
            raise ValueError(
                f'{config.name}: set "base" to the viam-labs:mujoco-sim:base it rides on'
            )
        if finite_number(config.name, attrs, "linear_scale_error", 0.0) <= -1:
            raise ValueError(f"{config.name}: linear_scale_error must be greater than -1")
        finite_number(config.name, attrs, "heading_drift_dps", 0.0)
        return [*deps, base], optional

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        self._handle = OdometryHandle(SimManager.get(), self.name, get_attrs(config))

    async def close(self) -> None:
        if self._handle is not None:
            await asyncio.to_thread(self._handle.close)
            self._handle = None

    def _h(self) -> OdometryHandle:
        if self._handle is None:
            raise RuntimeError(f"movement sensor {self.name} is not attached to the sim")
        return self._handle

    async def _read(self) -> OdometryReading:
        return await asyncio.to_thread(self._h().read)

    async def get_position(
        self, *, extra: Optional[Dict[str, Any]] = None, **kwargs
    ) -> Tuple[GeoPoint, float]:
        reading = await self._read()
        if extra and extra.get("return_relative_pos_m"):
            # RDK wheeled-odometry convention: latitude = y m, longitude = x m
            return GeoPoint(latitude=reading.y, longitude=reading.x), 0.0
        return _geo(reading), 0.0

    async def get_linear_velocity(self, **kwargs) -> Vector3:
        return _linear(await self._read())

    async def get_angular_velocity(self, **kwargs) -> Vector3:
        return _angular(await self._read())

    async def get_orientation(self, **kwargs) -> Orientation:
        return _orientation(await self._read())

    async def get_compass_heading(self, **kwargs) -> float:
        return _compass(await self._read())

    async def get_linear_acceleration(self, **kwargs) -> Vector3:
        raise NotImplementedError("odometry has no linear acceleration")

    async def get_properties(self, **kwargs) -> MovementSensor.Properties:
        return MovementSensor.Properties(
            linear_acceleration_supported=False,
            angular_velocity_supported=True,
            orientation_supported=True,
            position_supported=True,
            compass_heading_supported=True,
            linear_velocity_supported=True,
        )

    async def get_accuracy(self, **kwargs) -> GetAccuracyResponse:
        return GetAccuracyResponse()

    async def get_readings(self, **kwargs) -> Mapping[str, SensorReading]:
        # The SDK's MovementSensor.get_readings is an empty stub; build every value
        # from one snapshot so they come from the same physics step.
        reading = await self._read()
        return {
            "position": _geo(reading),
            "altitude": 0.0,
            "linear_velocity": _linear(reading),
            "angular_velocity": _angular(reading),
            "compass": _compass(reading),
            "orientation": _orientation(reading),
        }

    async def do_command(
        self, command: Mapping[str, ValueTypes], **kwargs
    ) -> Mapping[str, ValueTypes]:
        if command.get("command") == "reset_odometry":
            await asyncio.to_thread(self._h().reset_odometry)
            return {}
        raise ValueError(f"unknown command {command.get('command')!r}; try reset_odometry")

    async def get_geometries(self, **kwargs) -> List[Geometry]:
        return []
