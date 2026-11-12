"""viam-labs:mujoco-sim:gripper - a gripper on a mujoco-sim arm or gantry.

Attributes:
  world (string, required)  - name of the viam-labs:mujoco-sim:world component
  mount (string, required)  - the viam-labs:mujoco-sim arm or gantry it is on
                              ("arm" is accepted as its older name)
  asset (string)            - "lite6_gripper": the arm MJCF's own gripper (default),
                              or "robotiq_2f85" / "robotiq_epick" (suction), a
                              bundled MJCF attached like mjcf
  mjcf (string)             - instead: a gripper MJCF (.xml) or URDF (.urdf) to attach
                              at the arm's flange site; absolute, ~, or relative to
                              assets/. A URDF's one movable joint that isn't a
                              <mimic> gets a position servo sized from its
                              <limit effort>
  mesh_dir (string)         - a .urdf mjcf's mesh folder, absolute, ~ or relative to
                              the file (default meshes/ beside it if it exists,
                              else its folder)
  actuator (string)         - the gripper's actuator (default inferred)
  open, closed (float)      - actuator values for open and closed (default the
                              low and high ends of its ctrlrange)
  sensorless (bool)         - IsHoldingSomething reports the latched grip
                              instead of contacts, like the production
                              gripper (default false)
  grip_force_n (float)      - force grippers: closing/opening force, > 0 and
                              within the actuator's ctrlrange (default 5);
                              suction grippers: the adhesion force (default the
                              actuator's maximum)

The gripper is binary: Open, Grab and GoToInputs latch a state and return
immediately, matching the viam-labs:isaac-sim Lite6 gripper. The jaws move
after the return, so read IsMoving or IsHoldingSomething after the next physics
step, or poll as scripted_pick.py (in viam-modules/viam-isaac-sim) does.
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import ClassVar, Optional, Tuple

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

from .. import FAMILY, NAMESPACE, robots
from ..handles import DEFAULT_GRIP_FORCE_N, GripperHandle
from ..sim import GRIPPER_ASSETS, SimManager
from .utils import apply_frame_to_attrs, finite_number, get_attrs, positive_number, validate_sim_component


def _mount(name: str, attrs: dict) -> str:
    """The arm or gantry the gripper is on: `mount`, or `arm` (its older name)."""
    if "mount" in attrs and "arm" in attrs:
        raise ValueError(f'{name}: set "mount" or its older name "arm", not both')
    if "mount" not in attrs and "arm" not in attrs:
        raise ValueError(f'{name}: set "mount" to the mujoco-sim arm or gantry this gripper is on')
    mount = attrs.get("mount", attrs.get("arm"))
    if not mount or not isinstance(mount, str):
        raise ValueError(f'{name}: "mount" must be the name of a mujoco-sim arm or gantry')
    return mount


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
        world, _ = validate_sim_component(
            config, needs_asset=False, world_frame_only=False
        )
        attrs = get_attrs(config)
        mount = _mount(config.name, attrs)
        if attrs.get("asset", "lite6_gripper") not in GRIPPER_ASSETS:
            raise ValueError(f"{config.name}: asset must be one of {sorted(GRIPPER_ASSETS)}")
        if "mjcf" in attrs:
            if "asset" in attrs:
                raise ValueError(f"{config.name}: set asset or mjcf, not both")
            if not isinstance(attrs["mjcf"], str) or not attrs["mjcf"]:
                raise ValueError(f"{config.name}: mjcf must be a file path")
            robots.resolve_path(attrs["mjcf"], f"{config.name}: mjcf")
        if "mesh_dir" in attrs and (not isinstance(attrs["mesh_dir"], str) or not attrs["mesh_dir"]):
            raise ValueError(f"{config.name}: mesh_dir must be a folder path")
        if not isinstance(attrs.get("sensorless", False), bool):
            raise ValueError(f"{config.name}: sensorless must be a boolean")
        if "actuator" in attrs and (not isinstance(attrs["actuator"], str) or not attrs["actuator"]):
            raise ValueError(f"{config.name}: actuator must be an actuator name")
        for key in ("open", "closed"):
            if key in attrs:
                finite_number(config.name, attrs, key, 0.0)
        positive_number(config.name, attrs, "grip_force_n", DEFAULT_GRIP_FORCE_N)
        if config.HasField("frame") and config.frame.parent != mount:
            raise ValueError(
                f'{config.name}: frame.parent must be its mount "{mount}": the gripper serves its '
                "shape in its frame relative to the mount's end effector"
            )
        return [*world, mount], []

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = apply_frame_to_attrs(config, get_attrs(config))  # its frame: where it serves its shape
        self._sensorless = bool(attrs.get("sensorless", False))
        self._closed = False
        self._handle = GripperHandle(SimManager.get(), self.name, _mount(self.name, attrs), attrs)

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
        handle = self._h()
        await asyncio.to_thread(handle.stop)
        if not handle.suction:  # suction keeps its vacuum on Stop
            self._closed = False

    async def is_moving(self) -> bool:
        return await asyncio.to_thread(self._h().is_moving)

    async def is_holding_something(self, **kwargs) -> Gripper.HoldingStatus:
        handle = self._h()
        firm_grasp = await asyncio.to_thread(handle.firm_grasp)
        meta = {"input": 1.0 if self._closed else 0.0, "firm_grasp": firm_grasp}
        if handle.suction:
            meta["vacuum"] = await asyncio.to_thread(handle.vacuum)
        else:
            jaw = await asyncio.to_thread(handle.jaw)
            meta.update(jaw=jaw, open=handle.open_q, closed=handle.closed_q)
            if handle.slide:  # the names Lite6 clients read
                meta.update(jaw_m=jaw, open_m=handle.open_q, closed_m=handle.closed_q)
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
        """The gripper's collision shape, open and closed, for the motion planner
        (RDK falls back to it when get_kinematics raises); none for a gripper in
        its arm's own file, whose kinematics carry it."""
        return list(self._handle.geometries) if self._handle is not None else []

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        raise ValueError(f"unknown command: {command}")
