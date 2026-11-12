"""viam-labs:mujoco-sim:gantry - an XYZ Cartesian gantry.

Attributes:
  world (string, required)        - the viam-labs:mujoco-sim:world
  lengths_mm ([x, y, z], required) - each axis's travel, mm
  max_speed_mm_s (number)          - each axis's top speed (default 500)
  max_accel_mm_s2 (number)         - each axis's acceleration (default 2000)
  move_timeout_sec (number)        - wall-clock time a move may take to settle
                                     after its trajectory ends (default 30)

The frame (required, parent "world", no rotation) is the tool point with
every axis at 0: the near corner of the travel with Z fully raised.
Positions are mm from there, each in [0, length]; Z grows downward. The tool
frame points down (orientation vector (0, 0, -1), theta 0), so a gripper
mounts on the gantry ("mount": "<gantry>", frame parent the gantry) as it
would on an arm. The module generates the gantry's MJCF and kinematics
(gantry_file.py) and runs them with the arm machinery (ArmHandle), so
anything that finds an arm by name finds the gantry.
"""

import asyncio
import math
from concurrent.futures import Future
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from typing_extensions import Self
from viam.components import KinematicsReturn
from viam.components.arm import KinematicsFileFormat
from viam.components.gantry import Gantry
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes

from .. import FAMILY, NAMESPACE
from ..gantry_file import gantry_files
from ..handles import ArmHandle, JointTrajectoryPoint
from ..sim import SimManager
from .utils import apply_frame_to_attrs, frame_pose, get_attrs, positive_number, validate_sim_component

DEFAULT_MAX_SPEED_MM_S = 500.0
DEFAULT_MAX_ACCEL_MM_S2 = 2000.0
DEFAULT_MOVE_TIMEOUT_S = 30.0
TOLERANCE_MM = 1.0  # arrived: every axis within this, and still (as an arm's slide joints)
STEP_S = 0.01  # trajectory point spacing, sim seconds
FLOOR_CLEARANCE_MM = 100  # the tool at full Z stays this far above world z = 0


def _lengths(name: str, attrs: Dict[str, Any]) -> Tuple[float, float, float]:
    raw = attrs.get("lengths_mm")
    if (
        not isinstance(raw, (list, tuple))
        or len(raw) != 3
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw)
        or not all(math.isfinite(v) and v > 0 for v in raw)
    ):
        raise ValueError(f"{name}: lengths_mm must be three positive numbers, the X, Y and Z travel in mm")
    return float(raw[0]), float(raw[1]), float(raw[2])


def profile(start: np.ndarray, goal: np.ndarray, speeds: np.ndarray, accel: float) -> List[JointTrajectoryPoint]:
    """A straight move from start to goal (same units as speeds, per second),
    every axis within its speed and `accel`, as trajectory points STEP_S
    apart: trapezoidal along the path, triangular when too short to cruise."""
    delta = goal - start
    distance = float(np.linalg.norm(delta))
    share = np.abs(delta) / distance  # each axis's part of the path
    moving = share > 0
    speed = float(np.min(speeds[moving] / share[moving]))
    acceleration = float(np.min(accel / share[moving]))
    ramp = speed / acceleration
    if acceleration * ramp * ramp >= distance:  # never reaches cruise
        ramp = math.sqrt(distance / acceleration)
        speed = acceleration * ramp
    cruise = (distance - acceleration * ramp * ramp) / speed
    total = 2 * ramp + cruise

    def along(t: float) -> float:
        if t < ramp:
            return 0.5 * acceleration * t * t
        if t < ramp + cruise:
            return 0.5 * acceleration * ramp * ramp + speed * (t - ramp)
        rest = total - t
        return distance - 0.5 * acceleration * rest * rest

    times = [*np.arange(STEP_S, total, STEP_S), total]
    return [
        JointTrajectoryPoint(float(t), tuple(start + delta * (min(along(t), distance) / distance)))
        for t in times
    ]


class MujocoGantry(Gantry, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "gantry")
    AXES: ClassVar[str] = "xyz"  # one letter per axis, in joint order

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[ArmHandle] = None
        self._lengths = (0.0, 0.0, 0.0)
        self._max_speed = DEFAULT_MAX_SPEED_MM_S
        self._max_accel = DEFAULT_MAX_ACCEL_MM_S2
        self._move_timeout = DEFAULT_MOVE_TIMEOUT_S
        self._moving_count = 0
        self._settle: Optional[Future] = None  # the running move's settle wait, which a newer move or stop ends
        self._move: Optional[object] = None  # the running move's token: a newer move or stop replaces it

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        gantry = cls(config.name)
        gantry.reconfigure(config, dependencies)
        return gantry

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        name = config.name
        if not config.HasField("frame"):
            raise ValueError(f'{name}: set a frame (parent "world"): its translation is the tool point at zero')
        deps, _ = validate_sim_component(config, needs_asset=False, world_frame_only=True)
        attrs = get_attrs(config)
        lengths = _lengths(name, attrs)
        position, quat = frame_pose(config)
        if quat is not None and not np.allclose(quat, (1, 0, 0, 0), atol=1e-6) and not np.allclose(
            quat, (-1, 0, 0, 0), atol=1e-6
        ):
            # ponytail: world-aligned only; an oriented gantry needs its posts generated in its frame
            raise ValueError(f"{name}: the gantry's frame can't be rotated yet; leave its orientation unset")
        if position is None or position[2] * 1000 < lengths[2] + FLOOR_CLEARANCE_MM:
            raise ValueError(
                f"{name}: frame z must be at least lengths_mm z + {FLOOR_CLEARANCE_MM} "
                f"({lengths[2] + FLOOR_CLEARANCE_MM:g} mm), so the column stays off the floor"
            )
        positive_number(name, attrs, "max_speed_mm_s", DEFAULT_MAX_SPEED_MM_S)
        positive_number(name, attrs, "max_accel_mm_s2", DEFAULT_MAX_ACCEL_MM_S2)
        positive_number(name, attrs, "move_timeout_sec", DEFAULT_MOVE_TIMEOUT_S)
        return deps, []

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        attrs = apply_frame_to_attrs(config, get_attrs(config))
        self._lengths = _lengths(self.name, attrs)
        self._max_speed = float(attrs.get("max_speed_mm_s", DEFAULT_MAX_SPEED_MM_S))
        self._max_accel = float(attrs.get("max_accel_mm_s2", DEFAULT_MAX_ACCEL_MM_S2))
        self._move_timeout = float(attrs.get("move_timeout_sec", DEFAULT_MOVE_TIMEOUT_S))
        mjcf, kinematics = gantry_files(self._lengths, attrs["position"][2] * 1000)
        arm_attrs = {
            "world": attrs["world"],
            "mjcf": str(mjcf),
            "kinematics_url": str(kinematics),
            "collision": "mesh",  # the MJCF's own geoms and contact bits
            "position": attrs["position"],
        }
        self._handle = ArmHandle(SimManager.get(), self.name, arm_attrs)

    async def close(self) -> None:
        if self._handle is not None:
            await asyncio.to_thread(self._handle.close)
            self._handle = None

    def _h(self) -> ArmHandle:
        if self._handle is None:
            raise RuntimeError(f"{self.name} is not attached to the sim")
        return self._handle

    async def get_position(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None,
                           **kwargs) -> List[float]:
        metres = await asyncio.to_thread(self._h().get_joint_positions)
        return [value * 1000 for value in metres]

    async def move_to_position(self, positions: List[float], speeds: List[float], *,
                               extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None,
                               **kwargs) -> None:
        handle = self._h()
        n = len(self.AXES)
        if len(positions) != n:
            raise ValueError(f"{self.name}: expected {n} positions ({', '.join(self.AXES)} mm), got {len(positions)}")
        for axis, value, length in zip(self.AXES, positions, self._lengths):
            # within the settle tolerance of an end counts as the end: RDK sends back the
            # position the sim reports, which can sit a hair past it after settling
            if not math.isfinite(value) or not -TOLERANCE_MM < value < length + TOLERANCE_MM:
                raise ValueError(f"{self.name}: {axis} {value} mm is outside its travel [0, {length:g}]")
        positions = [min(max(value, 0.0), length) for value, length in zip(positions, self._lengths)]
        if speeds and len(speeds) != n:
            raise ValueError(f"{self.name}: expected {n} speeds (mm/s) or none, got {len(speeds)}")
        if any(not math.isfinite(s) or s <= 0 for s in speeds):
            raise ValueError(f"{self.name}: speeds must be positive")
        limits = np.minimum(np.asarray(speeds, float), self._max_speed) if speeds else np.full(n, self._max_speed)
        start = np.asarray(await asyncio.to_thread(handle.get_commanded_positions)) * 1000
        goal = np.asarray(positions, float)
        if np.all(np.abs(goal - start) < TOLERANCE_MM):
            return  # RDK's plans often start with the current position
        points = [
            JointTrajectoryPoint(p.time_s, tuple(v / 1000 for v in p.positions))
            for p in profile(start, goal, limits, self._max_accel)
        ]
        self._end_settle()  # a move still settling is superseded too
        token = self._move = object()
        self._moving_count += 1
        try:
            # a new trajectory cancels the last one's future (ArmHandle._drop_trajectory)
            await self._follow(await self._current(token, await asyncio.to_thread(handle.start_trajectory, points)))
            targets = [v / 1000 for v in positions]
            # ponytail: a cancel during the wait_for_joint_targets hop leaves that wait registered until it resolves or the gantry closes
            wait = await asyncio.to_thread(handle.wait_for_joint_targets, targets, TOLERANCE_MM / 1000)
            self._settle = await self._current(token, wait)
            try:
                await asyncio.wait_for(self._follow(wait), timeout=self._move_timeout)
            except TimeoutError:
                now = await self.get_position()
                raise RuntimeError(
                    f"{self.name}: didn't settle at {list(positions)} mm within {self._move_timeout:g} s; "
                    f"at {[round(v, 1) for v in now]}"
                ) from None
        except asyncio.CancelledError:
            if self._move is token:  # this call was cancelled (not superseded): stop and hold
                self._end_settle()
                await asyncio.to_thread(handle.stop)
            raise
        finally:
            self._moving_count -= 1
            if self._move is token:
                self._move, self._settle = None, None

    async def _follow(self, future: Future) -> None:
        """Await one phase of a move (its trajectory, then its settling). A
        newer move or stop cancels the phase's future: fail as superseded,
        leaving the newer move alone. When this call itself is cancelled (the
        client went away, or the settle timed out), stop and hold here."""
        try:
            # shielded: cancelling this call must not cancel the future, so the two cases stay apart
            await asyncio.shield(asyncio.wrap_future(future))
        except asyncio.CancelledError:
            if future.cancelled():
                raise RuntimeError(f"{self.name}: move superseded by a newer move or stop") from None
            future.cancel()
            await asyncio.to_thread(self._h().stop)
            raise

    async def _current(self, token: object, future: Future) -> Future:
        """The future, if no newer move or stop came while it was being set
        up on the sim thread; else cancel it and fail as superseded. After a
        stop, stop again: the trajectory may have started after it."""
        # ponytail: two moves racing through setup can still finish in either order; RDK sends one at a time
        if self._move is not token:
            future.cancel()
            if self._move is None:
                await asyncio.to_thread(self._h().stop)
            raise RuntimeError(f"{self.name}: move superseded by a newer move or stop")
        return future

    def _end_settle(self) -> None:
        if self._settle is not None:
            self._settle.cancel()  # removes its check from the sim thread
            self._settle = None

    async def get_lengths(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None,
                          **kwargs) -> List[float]:
        return list(self._lengths)

    async def home(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None,
                   **kwargs) -> bool:
        return True  # the sim always knows where the axes are

    async def stop(self, *, extra: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None, **kwargs):
        self._end_settle()
        self._move = None
        await asyncio.to_thread(self._h().stop)

    async def is_moving(self) -> bool:
        return self._moving_count > 0

    async def get_kinematics(self, **kwargs) -> KinematicsReturn:
        return KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA, self._h().kinematics_data

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None,
                         **kwargs) -> Mapping[str, ValueTypes]:
        raise NotImplementedError(f"{self.name}: no DoCommands")
