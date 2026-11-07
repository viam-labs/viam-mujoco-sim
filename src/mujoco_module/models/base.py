"""viam-labs:mujoco-sim:base - a simulated Viam Rover.

Attributes:
  world (string, required)        - name of the viam-labs:mujoco-sim:world component
  asset (string, required)        - "viam_rover"
  max_linear_mps (float)          - top speed; scales SetPower (default 0.762)
  max_angular_dps (float)         - top yaw rate; scales SetPower (default 245)
  move_timeout_margin_sec (float) - extra sim time MoveStraight/Spin get before
                                    they fail as stalled (default 2)

Place it with the standard frame; frame.parent must be "world". Chassis +Y is
forward, as in Viam's base frame. Wheels are visual and motion is ideal.
"""

import asyncio
import math
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from typing_extensions import Self
from viam.components.base import Base
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, Pose, RectangularPrism, ResourceName, Vector3
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily

from .. import FAMILY, NAMESPACE
from ..handles import BASE_ASSETS, DEFAULT_BASE_MOVE_MARGIN_S, BaseHandle
from ..sim import SimManager
from .utils import apply_frame_to_attrs, get_attrs, positive_number, validate_sim_component

# Wall-clock backstop beyond the sim-time deadline, so a stopped sim thread
# can't hang a call. The stall timeout itself is sim time (see BaseHandle).
_BACKSTOP_S = 30.0


class MujocoBase(Base, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "base")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[BaseHandle] = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        base = cls(config.name)
        base.reconfigure(config, dependencies)
        return base

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = get_attrs(config)
        if attrs.get("asset") not in BASE_ASSETS:
            raise ValueError(
                f"{config.name}: asset must be one of {sorted(BASE_ASSETS)}, "
                f"got {attrs.get('asset')!r}"
            )
        meta = BASE_ASSETS[attrs["asset"]]
        positive_number(config.name, attrs, "max_linear_mps", meta["max_linear_mps"])
        positive_number(config.name, attrs, "max_angular_dps", meta["max_angular_dps"])
        positive_number(
            config.name, attrs, "move_timeout_margin_sec", DEFAULT_BASE_MOVE_MARGIN_S
        )
        return validate_sim_component(config, needs_asset=False)

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs: Dict[str, Any] = apply_frame_to_attrs(config, get_attrs(config))
        self._handle = BaseHandle(SimManager.get(), self.name, attrs)

    async def close(self) -> None:
        if self._handle is not None:
            await asyncio.to_thread(self._handle.close)
            self._handle = None

    def _h(self) -> BaseHandle:
        if self._handle is None:
            raise RuntimeError(f"base {self.name} is not attached to the sim")
        return self._handle

    async def _drive(self, start, expected_s: float) -> None:
        """Start a goal off-thread and await it. If cancelled (or timed out)
        the goal is cancelled too, so the rover doesn't keep driving."""
        handle = self._h()
        start_task = asyncio.ensure_future(asyncio.to_thread(start))
        try:
            future = await asyncio.shield(start_task)
        except asyncio.CancelledError:
            start_task.add_done_callback(
                lambda task: None if task.cancelled() or task.exception() else task.result().cancel()
            )
            raise
        # ponytail: wall-clock backstop; a sim slower than half real time can still trip it
        backstop = 2 * (expected_s + handle.move_margin_s) + _BACKSTOP_S
        try:
            await asyncio.wait_for(asyncio.wrap_future(future), timeout=backstop)
        except TimeoutError:
            raise TimeoutError(
                f"base {self.name!r} move did not finish in time (is the sim running?)"
            ) from None

    async def move_straight(self, distance: int, velocity: float, **kwargs) -> None:
        handle = self._h()
        speed = min(abs(velocity), handle.max_linear_mps * 1000.0)
        await self._drive(
            lambda: handle.move_straight(distance / 1000.0, velocity / 1000.0),
            abs(distance) / speed if speed else 0.0,
        )

    async def spin(self, angle: float, velocity: float, **kwargs) -> None:
        handle = self._h()
        speed = min(abs(velocity), math.degrees(handle.max_angular_rad_s))
        await self._drive(
            lambda: handle.spin(math.radians(angle), math.radians(velocity)),
            abs(angle) / speed if speed else 0.0,
        )

    async def set_power(self, linear: Vector3, angular: Vector3, **kwargs) -> None:
        handle = self._h()
        forward = float(np.clip(linear.y, -1.0, 1.0)) * handle.max_linear_mps
        turn = float(np.clip(angular.z, -1.0, 1.0)) * handle.max_angular_rad_s
        await asyncio.to_thread(handle.set_velocity, forward, turn)

    async def set_velocity(self, linear: Vector3, angular: Vector3, **kwargs) -> None:
        await asyncio.to_thread(
            self._h().set_velocity, linear.y / 1000.0, math.radians(angular.z)
        )

    async def stop(self, **kwargs) -> None:
        await asyncio.to_thread(self._h().stop)

    async def is_moving(self) -> bool:
        return await asyncio.to_thread(self._h().is_moving)

    async def get_properties(self, **kwargs) -> Base.Properties:
        meta = self._h().meta
        return Base.Properties(
            width_meters=meta["width_m"],
            turning_radius_meters=0.0,
            wheel_circumference_meters=meta["wheel_circumference_m"],
        )

    async def get_geometries(self, **kwargs) -> List[Geometry]:
        handle = self._h()
        sx, sy, sz = handle.chassis_size_m
        cx, cy, cz = handle.chassis_center_m
        return [
            Geometry(
                center=Pose(x=cx * 1000, y=cy * 1000, z=cz * 1000, o_z=1, theta=0),
                box=RectangularPrism(dims_mm=Vector3(x=sx * 1000, y=sy * 1000, z=sz * 1000)),
                label=f"{self.name}-chassis",
            )
        ]
