"""viam-labs:mujoco-sim:spawner - feeds box parcels onto a conveyor, served as a sensor.

Attributes:
  world (string, required)    - name of the viam-labs:mujoco-sim:world component
  conveyor (string, required) - the viam-labs:mujoco-sim:conveyor it feeds
  classes (list, required)    - parcel classes: {name, weight (default 1),
                                size_m: [low, high] for every axis or one
                                [low, high] per axis, mass_kg: [low, high],
                                color: [r, g, b] or {"min": [r, g, b], "max": [r, g, b]},
                                friction (default 1)}
  interval_s (float)          - sim seconds between parcels (default 4)
  pool_size (int)             - most parcels live at once (default 12)
  seed (int)                  - repeat the same parcels (default: random)
  start_paused (bool)         - wait for resume (default false)
  despawn (list)              - boxes in world coordinates, {"center": [x, y, z],
                                "size": [x, y, z]} m: a parcel whose centre enters
                                one returns to the pool (as one that falls below
                                z = -0.5 m does)

Parcels spawn at rest on the conveyor's infeed spot (its -Y end), centred
across the belt and turned up to 10 degrees, every interval_s of sim time, a
class picked by weight and each size, mass and colour drawn from its ranges.
A spawn waits while a parcel is on the spot; with every parcel live it is
skipped (pool_exhausted). Reconfiguring the conveyor returns every parcel to
the pool. A world reset does too, and restarts the timer and the random
stream.

Readings: spawned, active, paused, pool_exhausted, seconds_until_next, and
viam:workcell-components box-detect's enabled, box_present (a parcel on the
infeed spot) and interval_seconds.

DoCommand: {"command": "spawn", "force": false} (one now; force skips the
occupancy check), "pause", "resume", "clear", "get_parcels"; and box-detect's
{"take": true} (despawn the parcel on the infeed spot) and {"reset": true}
(spawn one now).
"""

import asyncio
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
from ..handles import SpawnerHandle, despawn_boxes, parcel_classes
from ..sim import SimManager
from .utils import get_attrs, positive_number, validate_sim_component


class MujocoSpawner(Sensor, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "spawner")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: SpawnerHandle | None = None

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        spawner = cls(config.name)
        spawner.reconfigure(config, dependencies)
        return spawner

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        name, attrs = config.name, get_attrs(config)
        conveyor = attrs.get("conveyor")
        if not conveyor or not isinstance(conveyor, str):
            raise ValueError(f"{name}: set conveyor to the viam-labs:mujoco-sim:conveyor it feeds")
        parcel_classes(name, attrs.get("classes"))
        despawn_boxes(name, attrs.get("despawn", []))
        positive_number(name, attrs, "interval_s", 4.0)
        pool = attrs.get("pool_size", 12)
        if isinstance(pool, bool) or not isinstance(pool, (int, float)) or pool != int(pool) or pool < 1:
            raise ValueError(f"{name}: pool_size must be a whole number >= 1")
        seed = attrs.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, float)) or seed != int(seed)):
            raise ValueError(f"{name}: seed must be a whole number")
        if not isinstance(attrs.get("start_paused", False), bool):
            raise ValueError(f"{name}: start_paused must be true or false")
        deps, _ = validate_sim_component(config, needs_asset=False)
        return [*deps, conveyor], []

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        attrs = get_attrs(config)
        attrs["pool_size"] = int(attrs.get("pool_size", 12))  # Struct numbers are floats
        if attrs.get("seed") is not None:
            attrs["seed"] = int(attrs["seed"])
        self._handle = SpawnerHandle(SimManager.get(), self.name, attrs)

    async def close(self) -> None:
        if self._handle is not None:
            await asyncio.to_thread(self._handle.close)
            self._handle = None

    def _h(self) -> SpawnerHandle:
        if self._handle is None:
            raise RuntimeError(f"spawner {self.name} is not attached to the sim")
        return self._handle

    async def get_readings(self, **kwargs) -> Mapping[str, SensorReading]:
        return await asyncio.to_thread(self._h().readings)

    async def do_command(
        self, command: Mapping[str, ValueTypes], *, timeout: float | None = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        handle = self._h()
        if command.get("take") is True:
            return {"taken": await asyncio.to_thread(handle.take)}
        cmd = "spawn" if command.get("reset") is True else command.get("command")
        if cmd == "spawn":
            name, reason = await asyncio.to_thread(handle.spawn_now, command.get("force") is True)
            return {"spawned": name, "reason": reason} if name is None else {"spawned": name}
        if cmd in ("pause", "resume"):
            await asyncio.to_thread(handle.set_paused, cmd == "pause")
            return {"paused": cmd == "pause"}
        if cmd == "clear":
            return {"cleared": await asyncio.to_thread(handle.clear)}
        if cmd == "get_parcels":
            return {"parcels": await asyncio.to_thread(handle.parcels)}
        raise ValueError(f"unknown command: {command}")

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        return []
