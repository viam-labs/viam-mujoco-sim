"""viam-labs:mujoco-sim:conveyor - a belt conveyor, served as a motor.

Attributes:
  world (string, required)  - name of the viam-labs:mujoco-sim:world component
  length_m (float, required) - belt length, along +Y
  width_m (float, required)  - belt width
  roller_diameter_m (float) - converts RPM to belt speed (default 0.05)
  max_speed_mps (float)     - belt speed at power 1 (default 0.5)
  friction (float)          - the belt's sliding friction, >= 0 (default 1)
  rails_m (float)           - side-rail height above the belt, >= 0 (default 0.04; 0 = none)
  color ([r, g, b])         - the frame's colour, 0 to 1

Place it with the standard frame, parent "world": the frame is the centre of
the belt's top surface, and +Y is the direction of travel at positive power.
Legs reach down to the floor. The infeed is the belt's -Y end, the outfeed
its +Y end. Parcels ride on the belt by friction, so they slip, pile up
against an obstacle and collide naturally. RDK never asks a motor for its
shape: give the frame a geometry box from the floor to the belt's top for the
planner (the module logs the one to paste); the world state store shows the
exact shapes, rails included, in the 3D scene.

Motor API: SetPower (belt speed = power x max_speed_mps), SetRPM and GoFor
(RPM x pi x roller_diameter_m / 60 m/s, capped at max_speed_mps), GoTo,
GetPosition (belt travel in roller revolutions), ResetZeroPosition, Stop,
IsPowered, IsMoving. Speed changes ramp at 2 m/s^2. A world reset stops the
belt and zeroes its position.
"""

import asyncio
import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self

from viam.components.motor import Motor
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, Pose, RectangularPrism, ResourceName, Vector3
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily

from .. import FAMILY, NAMESPACE
from ..handles import CONVEYOR_ACCEL_MPS2, CONVEYOR_RAIL_M, ConveyorHandle
from ..sim import SimManager
from .utils import (
    apply_frame_to_attrs,
    finite_number,
    get_attrs,
    positive_number,
    publish_shapes,
    unpublish_shapes,
    validate_sim_component,
)

# Wall-clock backstop beyond the expected GoFor time, so a stopped sim thread
# can't hang a call (as the base's moves).
_BACKSTOP_S = 30.0


def _check(config: ComponentConfig) -> None:
    name = config.name
    if not config.HasField("frame"):
        raise ValueError(f'{name}: set a frame with parent "world": the centre of the belt\'s top surface')
    if config.frame.translation.z <= 0:
        raise ValueError(f"{name}: frame.translation.z is the belt's height above the floor; it must be positive")
    attrs = get_attrs(config)
    for key in ("length_m", "width_m"):
        if key not in attrs:
            raise ValueError(f"{name}: set {key}")
        positive_number(name, attrs, key, 1.0)
    positive_number(name, attrs, "roller_diameter_m", 0.05)
    positive_number(name, attrs, "max_speed_mps", 0.5)
    for key, default in (("friction", 1.0), ("rails_m", 0.04)):
        if finite_number(name, attrs, key, default) < 0:
            raise ValueError(f"{name}: {key} must be >= 0")
    color = attrs.get("color", [0.25, 0.27, 0.3])
    if not (
        isinstance(color, list)
        and len(color) == 3
        and all(isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) for c in color)
    ):
        raise ValueError(f"{name}: color must be a list of 3 numbers")


class MujocoConveyor(Motor, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "conveyor")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: ConveyorHandle | None = None

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        conveyor = cls(config.name)
        conveyor.reconfigure(config, dependencies)
        return conveyor

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        _check(config)
        return validate_sim_component(config, needs_asset=False)

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        attrs: dict[str, Any] = apply_frame_to_attrs(config, get_attrs(config))
        self._handle = ConveyorHandle(SimManager.get(), self.name, attrs)
        publish_shapes(config, f"{self.name}/frame", self._geometries(), self._handle.color)

    async def close(self) -> None:
        if self._handle is not None:
            await asyncio.to_thread(unpublish_shapes, self.name)
            await asyncio.to_thread(self._handle.close)
            self._handle = None

    def _h(self) -> ConveyorHandle:
        if self._handle is None:
            raise RuntimeError(f"conveyor {self.name} is not attached to the sim")
        return self._handle

    async def set_power(self, power: float, **kwargs) -> None:
        handle = self._h()
        await asyncio.to_thread(handle.set_speed, max(-1.0, min(1.0, power)) * handle.max_speed)

    async def set_rpm(self, rpm: float, **kwargs) -> None:
        handle = self._h()
        await asyncio.to_thread(handle.set_speed, handle.rpm_to_mps(rpm))

    async def go_for(self, rpm: float, revolutions: float, **kwargs) -> None:
        handle = self._h()
        future = await asyncio.to_thread(handle.go_for, rpm, revolutions)
        speed = min(abs(handle.rpm_to_mps(rpm)), handle.max_speed)
        expected = abs(revolutions) * math.pi * handle.diameter / speed + speed / CONVEYOR_ACCEL_MPS2
        backstop = 2 * expected + _BACKSTOP_S
        try:
            await asyncio.wait_for(asyncio.wrap_future(future), timeout=backstop)
        except TimeoutError:
            raise TimeoutError(
                f"conveyor {self.name!r} move did not finish within {backstop:.0f} s of wall time (is the sim running?)"
            ) from None

    async def go_to(self, rpm: float, position_revolutions: float, **kwargs) -> None:
        position = await asyncio.to_thread(self._h().position)
        await self.go_for(abs(rpm), position_revolutions - position)

    async def reset_zero_position(self, offset: float, **kwargs) -> None:
        await asyncio.to_thread(self._h().reset_zero_position, offset)

    async def get_position(self, **kwargs) -> float:
        return await asyncio.to_thread(self._h().position)

    async def get_properties(self, **kwargs) -> Motor.Properties:
        return Motor.Properties(position_reporting=True)

    async def stop(self, **kwargs) -> None:
        await asyncio.to_thread(self._h().stop)

    async def is_powered(self, **kwargs) -> tuple[bool, float]:
        power = await asyncio.to_thread(self._h().power)
        return power != 0.0, power

    async def is_moving(self) -> bool:
        return await asyncio.to_thread(self._h().is_moving)

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        return self._geometries()

    def _geometries(self) -> list[Geometry]:
        """The frame, belt and legs as one box from the floor to the belt's
        top, and one box per rail, in the conveyor's frame. RDK never calls
        get_geometries on a motor: the world state store publishes these to
        the 3D scene, and the planner sees the frame geometry (the first box)."""
        handle = self._h()
        mm = 1000.0
        boxes = [
            (f"{self.name}-body", (0.0, 0.0, -handle.height / 2), (handle.width, handle.length, handle.height))
        ]
        if handle.rails > 0:
            for i, side in enumerate((-1, 1)):
                x = side * (handle.width + CONVEYOR_RAIL_M) / 2
                boxes.append(
                    (f"{self.name}-rail-{i}", (x, 0.0, handle.rails / 2), (CONVEYOR_RAIL_M, handle.length, handle.rails))
                )
        return [
            Geometry(
                center=Pose(x=cx * mm, y=cy * mm, z=cz * mm, o_z=1),
                box=RectangularPrism(dims_mm=Vector3(x=sx * mm, y=sy * mm, z=sz * mm)),
                label=label,
            )
            for label, (cx, cy, cz), (sx, sy, sz) in boxes
        ]
