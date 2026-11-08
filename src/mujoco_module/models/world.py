"""viam-labs:mujoco-sim:world - owns the MuJoCo scene.

Attributes:
  physics_dt (float)  - physics step in seconds (default 0.002)
  real_time (bool)    - pace sim time to wall time (default true)
  props (list)        - movable objects: {name, type: "cube" | "sphere" |
                        "cylinder" | "capsule", size, scale, position,
                        orientation_wxyz, mass, color, fixed, static_friction,
                        dynamic_friction (accepted, ignored)}. Each shape fills
                        a box of size x scale.

DoCommand: status, play, pause, reset, set_prop_pose, get_prop_pose,
park_props. Shapes match the viam-labs:isaac-sim world.
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import ClassVar, Self

from viam.components.generic import Generic
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes, struct_to_dict

from .. import FAMILY, NAMESPACE
from ..sim import SimManager, WorldConfig


class MujocoWorld(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "world")

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        world = cls(config.name)
        world.reconfigure(config, dependencies)
        return world

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> tuple[Sequence[str], Sequence[str]]:
        WorldConfig.from_attrs(struct_to_dict(config.attributes))
        return [], []

    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        cfg = WorldConfig.from_attrs(struct_to_dict(config.attributes))
        SimManager.get().configure_world(cfg)

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: float | None = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        return await asyncio.to_thread(self._command, command)

    @staticmethod
    def _command(command: Mapping[str, ValueTypes]) -> Mapping[str, ValueTypes]:
        sim = SimManager.get()
        cmd = str(command.get("command", ""))
        if cmd == "status":
            return sim.status()
        if cmd == "play":
            sim.play()
            return {"ok": True}
        if cmd == "pause":
            sim.pause()
            return {"ok": True}
        if cmd == "reset":
            sim.reset()
            return {"ok": True}
        if cmd == "park_props":
            props = command.get("props") or []
            if not isinstance(props, list) or not all(isinstance(p, str) for p in props):
                raise ValueError("park_props requires props: a list of prop names")
            return sim.park_props(props)
        prop = str(command.get("prop", ""))
        if cmd == "get_prop_pose":
            if not prop:
                raise ValueError("get_prop_pose requires prop")
            return {"prop": prop, "pose": sim.get_prop_pose(prop)}
        if cmd == "set_prop_pose":
            if not prop:
                raise ValueError("set_prop_pose requires prop")
            return sim.set_prop_pose(
                prop, command.get("position"), command.get("orientation_wxyz")
            )
        raise ValueError(f"unknown command: {command}")
