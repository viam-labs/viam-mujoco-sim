"""viam-labs:mujoco-sim:base - a differential-drive or skid-steer base.

Attributes:
  world (string, required)        - name of the viam-labs:mujoco-sim:world component
  asset (string)                  - a bundled base, "viam_rover" or "clearpath_husky";
                                    its defaults apply under the attributes set here
  mjcf (string)                   - instead of asset: the base's .xml (MJCF) or .urdf
                                    file, absolute, ~, or relative to assets/
  mesh_dir (string)               - a URDF's mesh folder, absolute, ~ or relative to the file; its
                                    package:// meshes are found there by file name
                                    (default meshes/ beside the file if it exists,
                                    else the file's folder)
  drive (string)                  - "ideal" (default): exact motion on slides, the
                                    wheels only turn; "wheels": wheel actuators drive
                                    the base through floor friction
  chassis (string)                - the chassis body (default: the file's one top-level body)
  wheels ({"left": [..], "right": [..]}) - the driven wheel joints (default inferred:
                                    horizontal hinges on bodies fixed to the chassis)
  forward_axis (string)           - "+x", "-x", "+y" or "-y" in the chassis frame
                                    (default from the wheel axles: along y means +x)
  wheel_radius_m (float)          - wheel radius (default from the wheel geoms)
  max_linear_mps (float)          - top speed; scales SetPower (default the asset's, else
                                    from the wheels' velocity actuators' ctrlrange)
  max_angular_dps (float)         - top yaw rate; scales SetPower (default as above)
  move_timeout_margin_sec (float) - sim time MoveStraight/Spin may go without
                                    progress before they fail as stalled (default 2)

Place it with the standard frame; frame.parent must be "world". The frame is
on the floor under the chassis, +Y forward, as in Viam's base frame.
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

from .. import FAMILY, NAMESPACE, robots
from ..bases import FORWARD_AXES
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
        name = config.name
        asset, mjcf = attrs.get("asset"), attrs.get("mjcf")
        if asset is not None and mjcf is not None:
            raise ValueError(f"{name}: set asset or mjcf, not both")
        if mjcf is None and not (isinstance(asset, str) and asset in BASE_ASSETS):
            raise ValueError(
                f"{name}: asset must be one of {sorted(BASE_ASSETS)}, got {asset!r}; "
                "or set mjcf to your base's MJCF or URDF file"
            )
        if mjcf is not None:
            if not isinstance(mjcf, str) or not mjcf:
                raise ValueError(f"{name}: mjcf must be a file path")
            path = robots.resolve_path(mjcf, f"{name}: mjcf")
            if path.suffix.lower() not in (".xml", ".urdf"):
                raise ValueError(f"{name}: mjcf must be an .xml (MJCF) or .urdf file, got {path.name}")
        if attrs.get("drive", "ideal") not in ("ideal", "wheels"):
            raise ValueError(f'{name}: drive must be "ideal" or "wheels"')
        if "chassis" in attrs and (not isinstance(attrs["chassis"], str) or not attrs["chassis"]):
            raise ValueError(f"{name}: chassis must be a body name")
        if "mesh_dir" in attrs and (not isinstance(attrs["mesh_dir"], str) or not attrs["mesh_dir"]):
            raise ValueError(f"{name}: mesh_dir must be a folder path")
        if "forward_axis" in attrs and attrs["forward_axis"] not in FORWARD_AXES:
            raise ValueError(f"{name}: forward_axis must be one of {list(FORWARD_AXES)}")
        wheels = attrs.get("wheels")
        if wheels is not None and not (
            isinstance(wheels, dict)
            and set(wheels) == {"left", "right"}
            and all(
                isinstance(side, list) and side and all(isinstance(j, str) and j for j in side)
                for side in wheels.values()
            )
        ):
            raise ValueError(f'{name}: wheels must be {{"left": [joint, ...], "right": [joint, ...]}}')
        if wheels is not None:
            for joint in sorted(set(wheels["left"]) & set(wheels["right"])):
                raise ValueError(f"{name}: wheels joint {joint!r} is on both sides")
        for key in ("wheel_radius_m", "max_linear_mps", "max_angular_dps"):
            if key in attrs:
                positive_number(name, attrs, key, 0.0)
        positive_number(name, attrs, "move_timeout_margin_sec", DEFAULT_BASE_MOVE_MARGIN_S)
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
                f"base {self.name!r} move did not finish within {backstop:.0f} s of wall time (is the sim running?)"
            ) from None

    async def move_straight(self, distance: int, velocity: float, **kwargs) -> None:
        handle = self._h()
        speed = min(abs(velocity), handle.max_linear_mps * 1000.0)
        await self._drive(
            lambda: handle.move_straight(distance / 1000.0, velocity / 1000.0),
            (abs(distance) / speed if speed else 0.0) + speed / (handle._linear_accel * 1000.0),
        )

    async def spin(self, angle: float, velocity: float, **kwargs) -> None:
        handle = self._h()
        speed = min(abs(velocity), math.degrees(handle.max_angular_rad_s))
        await self._drive(
            lambda: handle.spin(math.radians(angle), math.radians(velocity)),
            (abs(angle) / speed if speed else 0.0) + math.radians(speed) / handle._angular_accel,
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
        base = self._h().base
        return Base.Properties(
            width_meters=base.track,
            turning_radius_meters=0.0,
            wheel_circumference_meters=2 * math.pi * base.radius,
        )

    async def get_geometries(self, **kwargs) -> List[Geometry]:
        """A box around the chassis's collision geoms (not the wheels or
        casters), in the base's frame."""
        box = self._h().base.box
        if box is None:
            return []
        (cx, cy, cz), (sx, sy, sz) = (value * 1000 for value in box)
        return [
            Geometry(
                center=Pose(x=cx, y=cy, z=cz, o_z=1, theta=0),
                box=RectangularPrism(dims_mm=Vector3(x=sx, y=sy, z=sz)),
                label=f"{self.name}-chassis",
            )
        ]
