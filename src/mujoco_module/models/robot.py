"""viam-labs:mujoco-sim:robot - a robot file whose limbs are separate arms.

The robot loads an MJCF file once and attaches it under "<name>/" at its
frame: parent "world", or a mujoco-sim component with a viam_frame (a base:
the robot rides it, and drops out with its arms while the base is away). Arms bind to it with `robot` and their `joints`
(each its own chain, with its own kinematics_url); grippers bind to those
arms as usual. Each arm's frame must be its chain base's pose in the file,
relative to the robot (the arm's config error says what to paste). The robot
sets the joints no arm or gripper drives from its home keyframe. RDK asks it
for geometries: the file's fixed parts other than the limbs' chain bases.

Attributes:
  world (string, required) - name of the viam-labs:mujoco-sim:world component
  asset (string)           - a bundled robot: "aloha", or "tiago" (on a pal_tiago_base base)
  mjcf (string)            - your own .xml robot file (path, assets-relative); one of asset, mjcf
  home_keyframe (string)   - the keyframe for undriven joints (default: the file's first)
  actuators (object)       - {"model": "bam", ...}: BAM servo physics for the matching actuators
                             (motor, bam_model, kp_fw, vin, vin_drop_gain, vin_min, joints); see README
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
from viam.utils import ValueTypes

from .. import FAMILY, NAMESPACE, robots
from ..bam import BamConfig
from ..robot_file import ROBOT_ASSETS, RobotHandle
from ..sim import SimManager
from .utils import apply_frame_to_attrs, frame_parent, get_attrs, validate_sim_component


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
        attrs = get_attrs(config)
        _source(config, attrs)
        if attrs.get("actuators") is not None:
            BamConfig.from_attrs(attrs["actuators"], config.name)
        # a robot is placed on the world or on a mujoco-sim component with a
        # viam_frame (a base); another parent never appears, so the robot waits
        deps, _ = validate_sim_component(config, needs_asset=False, world_frame_only=False)
        parent = frame_parent(config)
        return ([*deps, parent] if parent else deps), []

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        attrs = apply_frame_to_attrs(config, get_attrs(config))
        if parent := frame_parent(config):
            attrs["parent"] = parent
        self._handle = RobotHandle(SimManager.get(), self.name, attrs)

    async def close(self) -> None:
        if self._handle is not None:
            await asyncio.to_thread(self._handle.close)
            self._handle = None

    async def get_geometries(self, **kwargs) -> list[Geometry]:
        return self._handle.geometries() if self._handle is not None else []

    async def do_command(
        self, command: Mapping[str, ValueTypes], *, timeout: float | None = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        if "io" in command:
            if self._handle is None:
                raise RuntimeError(f"robot {self.name} is closed")
            # viam-sdk's struct_to_dict decodes a wire null as int 0, while real numbers arrive as
            # float, so an exact int 0 is a null (bool is not exactly int, so False stays False).
            def value(v):
                return None if v is None or (type(v) is int and v == 0) else v

            request = value(command["io"])
            if request is None:
                request = {}
            if not isinstance(request, Mapping):
                raise ValueError("io must be an object")
            torque, targets = value(request.get("torque")), value(request.get("targets"))
            if torque is not None and not isinstance(torque, bool):
                raise ValueError("io.torque must be true, false or null")
            if targets is not None and not isinstance(targets, list):
                raise ValueError("io.targets must be a list of numbers or null")
            return await asyncio.to_thread(self._handle.io, targets, torque)
        raise ValueError(f"robot {self.name}: unknown command {sorted(command)}; known: ['io']")
