"""viam-labs:mujoco-sim:gripper - the Lite6's integrated gripper.

Attributes:
  world (string, required)  - name of the viam-labs:mujoco-sim:world component
  arm (string, required)    - the viam-labs:mujoco-sim:arm it is part of
  asset (string)            - "lite6_gripper" (the default and only option)
  sensorless (bool)         - IsHoldingSomething reports the latched grip
                              instead of contacts, like the production
                              gripper (default false)
  grip_force_n (float)      - closing/opening force, (0, 10] N (default 5)

The gripper is binary: Open, Grab and GoToInputs latch a state and return
immediately, matching the viam-labs:isaac-sim Lite6 gripper. Grab and Open
latch the command and return before the jaws move, so read IsMoving or
IsHoldingSomething after the next physics step, or poll as scripted_pick.py does.
"""

import asyncio
import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Optional, Tuple

from grpclib import Status
from typing_extensions import Self
from viam.components.gripper import Gripper
from viam.errors import ViamGRPCError
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, KinematicsFileFormat, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes

from .. import FAMILY, NAMESPACE
from ..handles import GripperHandle
from ..sim import SimManager
from .utils import get_attrs, validate_sim_component

ASSET = "lite6_gripper"
DEFAULT_GRIP_FORCE_N = 5.0
MAX_GRIP_FORCE_N = 10.0  # the Menagerie actuator's ctrlrange


def _grip_force(attrs: Mapping[str, Any], name: str) -> float:
    raw = attrs.get("grip_force_n", DEFAULT_GRIP_FORCE_N)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{name}: grip_force_n must be numeric")
    force = float(raw)
    if not math.isfinite(force) or not 0.0 < force <= MAX_GRIP_FORCE_N:
        raise ValueError(f"{name}: grip_force_n must be in (0, {MAX_GRIP_FORCE_N}]")
    return force


class MujocoGripper(Gripper, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "gripper")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[GripperHandle] = None
        self._sensorless = False
        self._closed = False

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        gripper = cls(config.name)
        gripper.reconfigure(config, dependencies)
        return gripper

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        world, _ = validate_sim_component(config, needs_asset=False)
        attrs = get_attrs(config)
        arm = attrs.get("arm")
        if not arm or not isinstance(arm, str):
            raise ValueError(f'{config.name}: set "arm" to the arm this gripper is part of')
        if attrs.get("asset", ASSET) != ASSET:
            raise ValueError(f"{config.name}: asset must be {ASSET!r}")
        if not isinstance(attrs.get("sensorless", False), bool):
            raise ValueError(f"{config.name}: sensorless must be a boolean")
        _grip_force(attrs, config.name)
        return [*world, arm], []

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = get_attrs(config)
        self._sensorless = bool(attrs.get("sensorless", False))
        self._closed = False
        self._handle = GripperHandle(
            SimManager.get(), self.name, str(attrs["arm"]), _grip_force(attrs, config.name)
        )

    async def close(self) -> None:
        if self._handle is not None:
            await asyncio.to_thread(self._handle.close)
            self._handle = None

    def _h(self) -> GripperHandle:
        if self._handle is None:
            raise RuntimeError(f"gripper {self.name} is not attached to the sim")
        return self._handle

    async def _latch(self, closed: bool) -> None:
        handle = self._h()
        await asyncio.to_thread(handle.close_jaws if closed else handle.open_jaws)
        self._closed = closed

    async def open(self, **kwargs) -> None:
        await self._latch(False)

    async def grab(self, **kwargs) -> bool:
        await self._latch(True)
        return True

    async def stop(self, **kwargs) -> None:
        await asyncio.to_thread(self._h().stop)
        self._closed = False

    async def is_moving(self) -> bool:
        return await asyncio.to_thread(self._h().is_moving)

    async def is_holding_something(self, **kwargs) -> Gripper.HoldingStatus:
        handle = self._h()
        firm_grasp = await asyncio.to_thread(handle.firm_grasp)
        meta = {
            "input": 1.0 if self._closed else 0.0,
            "jaw_m": await asyncio.to_thread(handle.jaw_m),
            "open_m": GripperHandle.OPEN_M,
            "closed_m": GripperHandle.CLOSED_M,
            "firm_grasp": firm_grasp,
        }
        holding = self._closed if self._sensorless else firm_grasp
        return Gripper.HoldingStatus(is_holding_something=holding, meta=meta)

    async def get_current_inputs(self, **kwargs) -> list[float]:
        return [1.0 if self._closed else 0.0]

    async def go_to_inputs(self, values: list[float], **kwargs) -> None:
        if len(values) != 1 or float(values[0]) not in (0.0, 1.0):
            raise ViamGRPCError(
                f"gripper {self.name}: input must be a single 0 (open) or 1 (closed)",
                Status.INVALID_ARGUMENT,
            )
        await self._latch(float(values[0]) == 1.0)

    async def get_kinematics(
        self, **kwargs
    ) -> Tuple[KinematicsFileFormat.ValueType, bytes]:
        # Deliberate for the PoC: no jaw collision box, so the planner is
        # more permissive, not wrong.
        raise NotImplementedError("the simulated gripper serves no kinematics")

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        return []

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        raise ValueError(f"unknown command: {command}")
