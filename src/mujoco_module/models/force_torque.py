"""viam-labs:mujoco-sim:force_torque - a wrist force-torque sensor, served as a sensor.

Attributes:
  world (string, required) - name of the viam-labs:mujoco-sim:world component
  filter_s (float)         - low-pass time constant, s; 0 is the raw per-step value (default 0.02)
  force_noise_n (float)    - force noise std, N, per axis (default 0)
  torque_noise_nm (float)  - torque noise std, N m, per axis (default 0)
  seed (int)               - noise generator seed, 0 or more (default 0)

Set frame.parent to a mujoco-sim arm, with no translation or orientation: the
sensor measures at the arm's flange, adding no length or mass, and its frame
is the arm's end-effector frame. Readings: force {x, y, z} (N) and torque
{x, y, z} (N m), the wrench the tool (gripper, what it holds and what it
touches) exerts on the flange, in that frame and about its origin, so a
hanging tool reads its weight along +Z; and tared. DoCommand {"command":
"tare"} subtracts the current wrench from later readings; {"command":
"untare"} stops.
"""

import asyncio
import math
from collections.abc import Mapping, Sequence
from typing import ClassVar, Self

from viam.components.sensor import Sensor
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import SensorReading, ValueTypes

from .. import FAMILY, NAMESPACE
from ..sensors import DEFAULT_FT_FILTER_S, ForceTorqueHandle
from ..sim import SimManager
from .utils import frame_pose, get_attrs, non_negative_number, validate_sim_component, whole_number


class MujocoForceTorque(Sensor, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "force_torque")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: ForceTorqueHandle | None = None

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        sensor = cls(config.name)
        sensor.reconfigure(config, dependencies)
        return sensor

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        name = config.name
        if not config.HasField("frame") or config.frame.parent in ("", "world"):
            raise ValueError(f"{name}: set frame.parent to the mujoco-sim arm whose flange it measures at")
        position, quat = frame_pose(config)
        if any(abs(v) > 1e-9 for v in position or ()) or (
            quat is not None and not math.isclose(abs(quat[0]), 1.0, abs_tol=1e-9)
        ):
            raise ValueError(
                f"{name}: remove the frame's translation and orientation: the sensor measures at the arm's flange"
            )
        attrs = get_attrs(config)
        for key, default in (("filter_s", DEFAULT_FT_FILTER_S), ("force_noise_n", 0.0), ("torque_noise_nm", 0.0)):
            non_negative_number(name, attrs, key, default)
        whole_number(name, attrs, "seed", 0, minimum=0)  # numpy seeds are non-negative
        deps, optional = validate_sim_component(config, needs_asset=False, world_frame_only=False)
        return [*deps, config.frame.parent], optional

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        self._handle = ForceTorqueHandle(SimManager.get(), self.name, config.frame.parent, get_attrs(config))

    async def close(self) -> None:
        if self._handle is not None:
            await asyncio.to_thread(self._handle.close)
            self._handle = None

    def _h(self) -> ForceTorqueHandle:
        if self._handle is None:
            raise RuntimeError(f"force-torque sensor {self.name} is not attached to the sim")
        return self._handle

    async def get_readings(self, **kwargs) -> Mapping[str, SensorReading]:
        return await asyncio.to_thread(self._h().readings)

    async def do_command(
        self, command: Mapping[str, ValueTypes], *, timeout: float | None = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        what = command.get("command")
        if what == "tare":
            await asyncio.to_thread(self._h().tare)
            return {"tared": True}
        if what == "untare":
            await asyncio.to_thread(self._h().untare)
            return {"tared": False}
        raise ValueError(f"unknown command {what!r}; try tare or untare")

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        return []
