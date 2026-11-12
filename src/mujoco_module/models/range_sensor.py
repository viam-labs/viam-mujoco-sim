"""viam-labs:mujoco-sim:range_sensor - an ultrasonic or time-of-flight range
sensor, served as a sensor.

Attributes:
  world (string, required) - name of the viam-labs:mujoco-sim:world component
  cone_deg (float)         - full cone angle (default 20; 0 is a single ray,
                             and rings and rays_per_ring go unused)
  rings (int)              - rings of rays around the axis ray (default 2)
  rays_per_ring (int)      - rays per ring (default 8)
  range_min_m (float)      - nearer hits read as range_min_m: too close to
                             measure (default 0.02)
  range_max_m (float)      - farther hits are ignored; the reading when
                             nothing is hit (default 4)

Place it with the standard frame: the beam's axis is the frame's +X, as the
photo-eye's is (to face a base's forward, +Y, turn it 90 degrees about Z).
frame.parent is "world" or another mujoco-sim component; the sensor never
sees its parent or the bodies below it.

Readings: distance (meters, the nearest hit across the cone; range_min_m if
something is nearer than that; range_max_m if nothing is hit), as RDK's
ultrasonic sensor reports it.
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
from ..handles import (
    DEFAULT_RANGE_SENSOR_CONE_DEG,
    DEFAULT_RANGE_SENSOR_MAX_M,
    DEFAULT_RANGE_SENSOR_MIN_M,
    DEFAULT_RANGE_SENSOR_RAYS_PER_RING,
    DEFAULT_RANGE_SENSOR_RINGS,
    RangeSensorHandle,
)
from ..sim import SimManager
from .utils import (
    apply_frame_to_attrs,
    finite_number,
    get_attrs,
    non_negative_number,
    positive_number,
    validate_sim_component,
    whole_number,
)


class MujocoRangeSensor(Sensor, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "range_sensor")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: RangeSensorHandle | None = None

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        sensor = cls(config.name)
        sensor.reconfigure(config, dependencies)
        return sensor

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        name, attrs = config.name, get_attrs(config)
        cone = finite_number(name, attrs, "cone_deg", DEFAULT_RANGE_SENSOR_CONE_DEG)
        if not 0 <= cone < 180:
            raise ValueError(f"{name}: cone_deg must be at least 0 and less than 180")
        whole_number(name, attrs, "rings", DEFAULT_RANGE_SENSOR_RINGS, minimum=1)
        whole_number(name, attrs, "rays_per_ring", DEFAULT_RANGE_SENSOR_RAYS_PER_RING, minimum=1)
        range_min = non_negative_number(name, attrs, "range_min_m", DEFAULT_RANGE_SENSOR_MIN_M)
        range_max = positive_number(name, attrs, "range_max_m", DEFAULT_RANGE_SENSOR_MAX_M)
        if range_max <= range_min:
            raise ValueError(f"{name}: range_max_m must be greater than range_min_m")
        return validate_sim_component(config, needs_asset=False, world_frame_only=False)

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        attrs: dict[str, Any] = apply_frame_to_attrs(config, get_attrs(config))
        if config.HasField("frame") and config.frame.parent != "world":
            attrs["parent"] = config.frame.parent
        self._handle = RangeSensorHandle(SimManager.get(), self.name, attrs)

    async def close(self) -> None:
        if self._handle is not None:
            await asyncio.to_thread(self._handle.close)
            self._handle = None

    def _h(self) -> RangeSensorHandle:
        if self._handle is None:
            raise RuntimeError(f"range sensor {self.name} is not attached to the sim")
        return self._handle

    async def get_readings(self, **kwargs) -> Mapping[str, SensorReading]:
        return await asyncio.to_thread(self._h().readings)

    async def do_command(
        self, command: Mapping[str, ValueTypes], *, timeout: float | None = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        return {}

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        return []
