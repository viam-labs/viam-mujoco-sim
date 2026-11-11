"""viam-labs:mujoco-sim:robot - a robot file whose limbs are separate arms.

The robot loads an MJCF file once and attaches it under "<name>/" at its
frame (parent "world"). Arms bind to it with `robot` and their `joints`
(each its own chain, with its own kinematics_url); grippers bind to those
arms as usual. Each arm's frame must be its chain base's pose in the file,
relative to the robot (the arm's config error says what to paste). The robot
sets the joints no arm or gripper drives from its home keyframe. RDK asks it
for geometries: the file's fixed parts other than the limbs' chain bases.

Attributes:
  world (string, required) - name of the viam-labs:mujoco-sim:world component
  asset (string)           - a bundled robot: "aloha"
  mjcf (string)            - your own .xml robot file (path, assets-relative); one of asset, mjcf
  home_keyframe (string)   - the keyframe for undriven joints (default: the file's first)
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self

from viam.components.generic import Generic
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily

from .. import FAMILY, NAMESPACE, robots
from ..robot_file import ROBOT_ASSETS, RobotHandle
from ..sim import SimManager
from .utils import apply_frame_to_attrs, get_attrs, validate_sim_component


def _source(config: ComponentConfig, attrs: Mapping[str, Any]) -> str:
    name = config.name
    asset, mjcf = attrs.get("asset"), attrs.get("mjcf")
    if (asset is None) == (mjcf is None):
        raise ValueError(f"{name}: set asset (one of {sorted(ROBOT_ASSETS)}) or mjcf, not both")
    if asset is not None and asset not in ROBOT_ASSETS:
        raise ValueError(f"{name}: asset must be one of {sorted(ROBOT_ASSETS)}, got {asset!r}")
    source = ROBOT_ASSETS[asset]["mjcf"] if asset is not None else mjcf
    if not isinstance(source, str) or not source.lower().endswith(".xml"):
        raise ValueError(f"{name}: mjcf must be a .xml MJCF file path (URDF robot files aren't supported yet), got {source!r}")
    robots.resolve_path(source, f"{name}: mjcf")
    keyframe = attrs.get("home_keyframe")
    if keyframe is not None and (not isinstance(keyframe, str) or not keyframe):
        raise ValueError(f"{name}: home_keyframe must be a keyframe name")
    return source


class MujocoRobot(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "robot")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: RobotHandle | None = None

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        robot = cls(config.name)
        robot.reconfigure(config, dependencies)
        return robot

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        if config.HasField("frame") and config.frame.parent not in ("", "world"):
            raise ValueError(
                f'{config.name}: a robot is placed on the world (frame.parent "world"); '
                "on a base is not supported yet"
            )
        _source(config, get_attrs(config))
        return validate_sim_component(config, needs_asset=False)

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        attrs = apply_frame_to_attrs(config, get_attrs(config))
        self._handle = RobotHandle(SimManager.get(), self.name, attrs)

    async def close(self) -> None:
        if self._handle is not None:
            await asyncio.to_thread(self._handle.close)
            self._handle = None

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        return self._handle.geometries() if self._handle is not None else []
