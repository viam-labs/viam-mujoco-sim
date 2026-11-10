"""viam-labs:mujoco-sim:imu - an inertial measurement unit, served as a movement sensor.

Attributes:
  world (string, required)        - name of the viam-labs:mujoco-sim:world component
  orientation_noise_deg (float)   - std of a random rotation on each orientation (default 0)
  gyro_noise_dps (float)          - angular-velocity noise std, deg/s, per axis (default 0)
  gyro_bias_dps ([x, y, z])       - constant angular-velocity bias, deg/s (default 0)
  accel_noise_mps2 (float)        - linear-acceleration noise std, m/s^2, per axis (default 0)
  accel_bias_mps2 ([x, y, z])     - constant acceleration bias, m/s^2 (default 0)
  seed (int)                      - noise generator seed, 0 or more (default 0)

Place it with the standard frame: frame.parent is "world", a mujoco-sim base
or a mujoco-sim arm (its end effector). Orientation is in the world frame;
angular velocity and linear acceleration are in the sensor's frame, and the
acceleration includes gravity as a real accelerometer reports it (+9.81 m/s^2
on the axis pointing up at rest). Readings are exact unless noise is set.
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
from viam.spatialmath import Quaternion
from viam.utils import SensorReading

from .. import FAMILY, NAMESPACE
from ..sensors import ImuHandle, ImuReading
from ..sim import SimManager
from .utils import (
    apply_frame_to_attrs,
    get_attrs,
    non_negative_number,
    validate_sim_component,
    vector3,
    whole_number,
)


def _orientation(r: ImuReading) -> Orientation:
    w, x, y, z = (float(v) for v in r.orientation_wxyz)
    ov = Quaternion(w, x, y, z).to_orientation_vector()
    return Orientation(o_x=ov.o_x, o_y=ov.o_y, o_z=ov.o_z, theta=math.degrees(ov.theta))


def _vector(values) -> Vector3:
    x, y, z = (float(v) for v in values)
    return Vector3(x=x, y=y, z=z)


class MujocoImu(MovementSensor, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "imu")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[ImuHandle] = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        imu = cls(config.name)
        imu.reconfigure(config, dependencies)
        return imu

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = get_attrs(config)
        for key in ("orientation_noise_deg", "gyro_noise_dps", "accel_noise_mps2"):
            non_negative_number(config.name, attrs, key, 0.0)
        for key in ("gyro_bias_dps", "accel_bias_mps2"):
            vector3(config.name, attrs, key)
        whole_number(config.name, attrs, "seed", 0, minimum=0)  # numpy seeds are non-negative
        return validate_sim_component(config, needs_asset=False, world_frame_only=False)

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs: Dict[str, Any] = apply_frame_to_attrs(config, get_attrs(config))
        if config.HasField("frame") and config.frame.parent != "world":
            attrs["parent"] = config.frame.parent
        self._handle = ImuHandle(SimManager.get(), self.name, attrs)

    async def close(self) -> None:
        if self._handle is not None:
            await asyncio.to_thread(self._handle.close)
            self._handle = None

    def _h(self) -> ImuHandle:
        if self._handle is None:
            raise RuntimeError(f"imu {self.name} is not attached to the sim")
        return self._handle

    async def _read(self) -> ImuReading:
        return await asyncio.to_thread(self._h().read)

    async def get_orientation(self, **kwargs) -> Orientation:
        return _orientation(await self._read())

    async def get_angular_velocity(self, **kwargs) -> Vector3:
        return _vector((await self._read()).angular_velocity_dps)

    async def get_linear_acceleration(self, **kwargs) -> Vector3:
        return _vector((await self._read()).linear_acceleration_mps2)

    async def get_position(self, **kwargs) -> Tuple[GeoPoint, float]:
        raise NotImplementedError("an imu has no position")

    async def get_linear_velocity(self, **kwargs) -> Vector3:
        raise NotImplementedError("an imu has no linear velocity")

    async def get_compass_heading(self, **kwargs) -> float:
        raise NotImplementedError("an imu has no compass heading")

    async def get_properties(self, **kwargs) -> MovementSensor.Properties:
        return MovementSensor.Properties(
            linear_acceleration_supported=True,
            angular_velocity_supported=True,
            orientation_supported=True,
            position_supported=False,
            compass_heading_supported=False,
            linear_velocity_supported=False,
        )

    async def get_accuracy(self, **kwargs) -> GetAccuracyResponse:
        return GetAccuracyResponse()

    async def get_readings(self, **kwargs) -> Mapping[str, SensorReading]:
        # The SDK's MovementSensor.get_readings is an empty stub; every value
        # comes from one snapshot so they share a physics step.
        reading = await self._read()
        return {
            "orientation": _orientation(reading),
            "angular_velocity": _vector(reading.angular_velocity_dps),
            "linear_acceleration": _vector(reading.linear_acceleration_mps2),
        }

    async def get_geometries(self, **kwargs) -> List[Geometry]:
        return []
