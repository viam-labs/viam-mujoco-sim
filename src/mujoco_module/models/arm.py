"""viam-labs:mujoco-sim:arm - a simulated arm.

Adapted from viam-modules/viam-isaac-sim (Apache-2.0); modified for MuJoCo.

Attributes:
  world (string, required)         - name of the viam-labs:mujoco-sim:world component
  asset (string)                   - a bundled arm, "lite6" (with its wide gripper)
                                     or "ur5e";
                                     its defaults apply under the attributes set here
  mjcf (string)                    - instead of asset: the arm's MJCF (.xml, needs
                                     kinematics_url) or URDF (.urdf) file, absolute,
                                     ~, or relative to assets/. A URDF's joints need
                                     <limit effort>: the module adds a position servo
                                     on each, sized from it
  mesh_dir (string)                - a .urdf mjcf's mesh folder, absolute, ~ or relative
                                     to the file (default meshes/ beside it if it
                                     exists, else its folder)
  joints ([string])                - MJCF joint names in chain order (default inferred)
  flange_site (string)             - MJCF site at the flange (default attachment_site)
  flange_link (string)             - .urdf only: the link at the flange (default the
                                     file's one end link)
  home_keyframe (string)           - MJCF keyframe with the home ctrl (default home)
  base_offset, flange_offset       - {translation: {x,y,z} mm, orientation: {o_x,o_y,
                                     o_z,theta deg}}: arm frame to MJCF world, and
                                     flange site to Viam's end effector
  collision (string)               - "kinematics" (default with an SVA file that has
                                     geometries): MuJoCo collides the arm links with
                                     the planner's shapes; "mesh": the MJCF's own
                                     collision geoms. On a body that gets a shape of
                                     at least 5 mm, all of that body's own colliding
                                     geoms are turned off (so a gripper's geoms
                                     sitting directly on the last link's body stop
                                     colliding; use "mesh" for such MJCFs)
  position ([x,y,z] meters)        - fallback placement when no frame is set
  orientation_wxyz ([w,x,y,z])     - fallback orientation when no frame is set
  orientation_rpy_deg ([r,p,y])    - fallback orientation when no frame is set
  move_timeout_sec (float)         - max time to wait for a move (default 30)
  max_joint_deg_s (float)          - per-joint velocity limit on the commanded
                                     target (default 180)
  max_joint_deg_s2 (float)         - per-joint acceleration limit reported for
                                     trajectory parameterization
  is_moving_velocity_threshold_rad_s (float) - settle threshold (default 0.01)
  robot_control_freq_hz (float)    - MoveThroughJointPositions push rate (default 50)
  path_tolerance_delta_deg (float) - convergence tolerance (default 0.5)
  path_max_step_deg (float)        - max per-joint delta between pushed targets
                                     (default 2.0)
  kinematics_url (string)          - the arm's Viam kinematics file (.json = SVA,
                                     .urdf = URDF): http(s)://, file://, or a path
                                     (absolute, ~, or relative to assets/). Read
                                     when the arm is configured. A .urdf mjcf's
                                     default: the file from its root to
                                     flange_link, served with its collision meshes

Place it with the standard frame. frame.parent is "world" or another
mujoco-sim component with a Viam frame, usually a base: the arm then rides
it, mounted at the frame's offset from the base's frame, and is built after
it (frame.parent is a dependency).
"""

import asyncio
import math
from datetime import timedelta
from typing import (
    Any,
    AsyncIterator,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from typing_extensions import Self
from viam.components import KinematicsReturn
from viam.components.arm import Arm, JointPositions, KinematicsFileFormat, Mesh, Pose
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.utils import ValueTypes

from .. import FAMILY, NAMESPACE, robots
from ..handles import (
    DEFAULT_IS_MOVING_VELOCITY_THRESHOLD_RAD_S,
    ArmHandle,
    JointTrajectoryPoint,
)
from ..sim import SimManager
from viam.spatialmath import Quaternion
from .utils import apply_frame_to_attrs, get_attrs, validate_sim_component

_DEFAULT_TOLERANCE_DEG = 0.5
_DEFAULT_CONTROL_FREQ_HZ = 50.0
_MAX_CONTROL_FREQ_HZ = 1000.0
_DEFAULT_MAX_STEP_DEG = 2.0
_MAX_MAX_STEP_DEG = 30.0


def _finite_radians(values_deg: Sequence[float], what: str) -> List[float]:
    radians = [math.radians(float(v)) for v in values_deg]
    if not all(math.isfinite(v) for v in radians):
        raise ValueError(f"{what} must be finite")
    return radians


def _densify(
    start: Sequence[float],
    waypoints: Sequence[Sequence[float]],
    max_step_rad: float,
) -> List[List[float]]:
    """Linearly interpolate targets between consecutive waypoints so no single
    step moves any joint by more than `max_step_rad`. First segment runs from
    `start` (the arm's current joint positions) to `waypoints[0]`; the last
    emitted target equals `waypoints[-1]` exactly."""
    out: List[List[float]] = []
    prev: List[float] = list(start)
    for wp in waypoints:
        target = list(wp)
        max_delta = max((abs(a - b) for a, b in zip(prev, target)), default=0.0)
        n = max(1, math.ceil(max_delta / max_step_rad))
        for k in range(1, n + 1):
            t = k / n
            out.append(target[:] if k == n else [a + t * (b - a) for a, b in zip(prev, target)])
        prev = target
    return out


def _max_joint_deg_s(
    attrs: Mapping[str, Any], component_name: str
) -> float | None:
    raw = attrs.get("max_joint_deg_s")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{component_name}: max_joint_deg_s must be numeric")
    limit = float(raw)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError(
            f"{component_name}: max_joint_deg_s must be finite and positive"
        )
    return limit


def _max_joint_deg_s2(
    attrs: Mapping[str, Any], component_name: str
) -> float | None:
    raw = attrs.get("max_joint_deg_s2")
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{component_name}: max_joint_deg_s2 must be numeric")
    limit = float(raw)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError(
            f"{component_name}: max_joint_deg_s2 must be finite and positive"
        )
    return limit


def _number(
    attrs: Mapping[str, Any],
    key: str,
    default: float,
    name: str,
    low: float,
    high: float = math.inf,
) -> float:
    """A finite, non-bool number in (low, high]."""
    raw = attrs.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{name}: {key} must be numeric")
    value = float(raw)
    if not math.isfinite(value) or not low < value <= high:
        raise ValueError(f"{name}: {key} must be finite and in ({low}, {high}]")
    return value


def _is_moving_velocity_threshold(
    attrs: Mapping[str, Any], component_name: str
) -> float:
    raw = attrs.get(
        "is_moving_velocity_threshold_rad_s",
        DEFAULT_IS_MOVING_VELOCITY_THRESHOLD_RAD_S,
    )
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(
            f"{component_name}: is_moving_velocity_threshold_rad_s must be numeric"
        )
    threshold = float(raw)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            f"{component_name}: is_moving_velocity_threshold_rad_s must be finite "
            "and non-negative"
        )
    return threshold


def _frame_parent(config: ComponentConfig) -> str | None:
    """The component the arm is framed on, None for the world."""
    if config.HasField("frame") and config.frame.parent not in ("", "world"):
        return config.frame.parent
    return None


class MujocoArm(Arm, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "arm")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[ArmHandle] = None
        self._attrs: Dict[str, Any] = {}
        self._move_timeout = 30.0
        self._tolerance_rad = math.radians(_DEFAULT_TOLERANCE_DEG)
        self._control_period = 1.0 / _DEFAULT_CONTROL_FREQ_HZ
        self._max_step_rad = math.radians(_DEFAULT_MAX_STEP_DEG)
        self._moving_count = 0

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        arm = cls(config.name)
        arm.reconfigure(config, dependencies)
        return arm

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        deps, _ = validate_sim_component(config, world_frame_only=False)
        attrs = get_attrs(config)
        url = attrs.get("kinematics_url")
        if url is not None:
            if not isinstance(url, str) or not url:
                raise ValueError(f"{config.name}: kinematics_url must be a URL or a file path")
            if "://" in url and url.split("://", 1)[0] not in ("file", "http", "https"):
                raise ValueError(
                    f"{config.name}: kinematics_url scheme must be file, http or https"
                )
            if "://" not in url:
                robots.resolve_path(url, f"{config.name}: kinematics_url")
        if isinstance(attrs.get("mjcf"), str):
            path = robots.resolve_path(attrs["mjcf"], f"{config.name}: mjcf")
            robots.check_flange_attrs(attrs, path.suffix.lower() == ".urdf", config.name)
        joints = attrs.get("joints")
        if joints is not None and (
            not isinstance(joints, list)
            or not joints
            or not all(isinstance(joint, str) and joint for joint in joints)
        ):
            raise ValueError(f"{config.name}: joints must be a list of MJCF joint names")
        for key in ("flange_site", "flange_link", "home_keyframe"):
            if key in attrs and (not isinstance(attrs[key], str) or not attrs[key]):
                raise ValueError(f"{config.name}: {key} must be a name")
        if "mesh_dir" in attrs and (not isinstance(attrs["mesh_dir"], str) or not attrs["mesh_dir"]):
            raise ValueError(f"{config.name}: mesh_dir must be a folder path")
        if attrs.get("collision", "kinematics") not in ("kinematics", "mesh"):
            raise ValueError(f'{config.name}: collision must be "kinematics" or "mesh"')
        for key in ("base_offset", "flange_offset"):
            robots.parse_pose(attrs.get(key), f"{config.name}: {key}")
        _is_moving_velocity_threshold(attrs, config.name)
        _max_joint_deg_s(attrs, config.name)
        _max_joint_deg_s2(attrs, config.name)
        _number(attrs, "move_timeout_sec", 30.0, config.name, 0.0)
        _number(attrs, "path_tolerance_delta_deg", _DEFAULT_TOLERANCE_DEG, config.name, 0.0)
        _number(attrs, "robot_control_freq_hz", _DEFAULT_CONTROL_FREQ_HZ, config.name, 0.0, _MAX_CONTROL_FREQ_HZ)
        _number(attrs, "path_max_step_deg", _DEFAULT_MAX_STEP_DEG, config.name, 0.0, _MAX_MAX_STEP_DEG)
        parent = _frame_parent(config)
        return ([*deps, parent] if parent else deps), []

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs = apply_frame_to_attrs(config, get_attrs(config))
        if parent := _frame_parent(config):
            attrs["parent"] = parent
        self._move_timeout = float(attrs.get("move_timeout_sec", 30.0))
        self._tolerance_rad = math.radians(
            float(attrs.get("path_tolerance_delta_deg", _DEFAULT_TOLERANCE_DEG))
        )
        self._control_period = 1.0 / float(
            attrs.get("robot_control_freq_hz", _DEFAULT_CONTROL_FREQ_HZ)
        )
        self._max_step_rad = math.radians(
            float(attrs.get("path_max_step_deg", _DEFAULT_MAX_STEP_DEG))
        )
        self._attrs = attrs
        self._handle = ArmHandle(SimManager.get(), self.name, attrs)

    async def close(self) -> None:
        if self._handle is not None:
            await asyncio.to_thread(self._handle.close)
            self._handle = None

    def _h(self) -> ArmHandle:
        if self._handle is None:
            raise RuntimeError(f"arm {self.name} is not attached to the sim")
        return self._handle

    async def get_end_position(self, **kwargs) -> Pose:
        (x, y, z), quat = await asyncio.to_thread(self._h().get_end_pose)
        return Quaternion(*quat).to_pose(x * 1000.0, y * 1000.0, z * 1000.0)

    async def move_to_position(self, pose: Pose, **kwargs) -> None:
        raise NotImplementedError(
            "IK and motion planning are Viam's job, not the module's: use the "
            "motion service or move_to_joint_positions"
        )

    async def _wait_for_joint_targets(
        self,
        handle: ArmHandle,
        targets: Sequence[float],
        tolerance_rad: float,
        timeout_s: float,
    ) -> bool:
        """Block until the sim thread reports the joints within tolerance.

        The convergence check runs once per simulator frame on the sim thread,
        so there is no RPC polling or wall-clock sleep between checks; the
        wall-clock timeout only bounds a stalled move. Returns False on
        timeout."""
        future = await asyncio.to_thread(
            handle.wait_for_joint_targets, targets, tolerance_rad
        )
        try:
            await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout_s)
        except TimeoutError:
            future.cancel()
            return False
        return True

    @staticmethod
    def _stuck_joint_detail(
        current: Sequence[float], targets: Sequence[float], tolerance_rad: float
    ) -> str:
        return ", ".join(
            f"j{index}: at {math.degrees(actual):.1f} want {math.degrees(target):.1f}"
            for index, (actual, target) in enumerate(zip(current, targets))
            if abs(actual - target) > tolerance_rad
        )

    async def move_to_joint_positions(
        self, positions: JointPositions, **kwargs
    ) -> None:
        targets = _finite_radians(positions.values, "joint positions")
        handle = self._h()
        self._moving_count += 1
        try:
            await asyncio.to_thread(handle.set_joint_targets, targets)
            if await self._wait_for_joint_targets(
                handle, targets, self._tolerance_rad, self._move_timeout
            ):
                return
            current = await asyncio.to_thread(handle.get_joint_positions)
            detail = self._stuck_joint_detail(
                current, targets, self._tolerance_rad
            )
            raise TimeoutError(
                f"arm {self.name} did not reach target within {self._move_timeout}s"
                f" (stuck joints: {detail or 'within tolerance but still moving'})"
            )
        finally:
            self._moving_count -= 1

    async def move_through_joint_positions(
        self, positions: Sequence[JointPositions], *args, **kwargs
    ) -> None:
        """Stream waypoint targets at `robot_control_freq_hz`. Sparse waypoint
        pairs are linearly interpolated so no single pushed target moves any
        joint by more than `path_max_step_deg`, or than `max_joint_deg_s`
        allows in one tick; otherwise drives would cut corners past
        intermediate waypoints they cannot reach in one tick."""
        handle = self._h()
        raw = [_finite_radians(wp.values, "joint positions") for wp in positions]
        if not raw:
            return

        self._moving_count += 1
        try:
            start = await asyncio.to_thread(handle.get_joint_positions)
            # never push targets faster than the joint limit: the rate-limited
            # servo would chase the newest target and cut corners off the plan
            limit = min(handle.dynamics_metadata()["max_velocity_rad_s"])
            stream = _densify(start, raw, min(self._max_step_rad, limit * self._control_period))

            loop = asyncio.get_running_loop()
            push_start = loop.time()
            for i, targets in enumerate(stream):
                await asyncio.to_thread(handle.set_joint_targets, targets)
                target_time = push_start + (i + 1) * self._control_period
                sleep_needed = target_time - loop.time()
                if sleep_needed > 0:
                    await asyncio.sleep(sleep_needed)

            final_targets = raw[-1]
            if await self._wait_for_joint_targets(
                handle, final_targets, self._tolerance_rad, self._move_timeout
            ):
                return
            current = await asyncio.to_thread(handle.get_joint_positions)
            detail = self._stuck_joint_detail(
                current, final_targets, self._tolerance_rad
            )
            raise TimeoutError(
                f"arm {self.name} did not converge on final waypoint "
                f"{len(raw)}/{len(raw)} within {self._move_timeout}s "
                f"(stuck joints: {detail or 'within tolerance but still moving'})"
            )
        finally:
            self._moving_count -= 1

    async def move_through_joint_positions_streamed(  # type: ignore
        self,
        batches: AsyncIterator[List[Arm.TrajectoryPoint]],
        **kwargs,
    ) -> AsyncIterator[Arm.TrajectoryUpdate]:
        last_time = timedelta(0)
        first = True
        handle = self._h()
        async for batch in batches:
            trajectory: list[JointTrajectoryPoint] = []
            batch_start = last_time
            for point in batch:
                if first:
                    if point.time != timedelta(0):
                        raise ValueError("first trajectory point must have time zero")
                    first = False
                elif point.time <= last_time:
                    raise ValueError("trajectory point times must strictly increase")
                trajectory.append(
                    JointTrajectoryPoint(
                        time_s=(point.time - batch_start).total_seconds(),
                        positions=tuple(
                            _finite_radians(point.positions, "joint positions")
                        ),
                    )
                )
                last_time = point.time

            if trajectory:
                expected = trajectory[-1].positions
                start_task = asyncio.create_task(
                    asyncio.to_thread(handle.start_trajectory, trajectory)
                )
                started = False
                try:
                    future = await asyncio.shield(start_task)
                    started = True
                    await asyncio.wrap_future(future)
                finally:
                    if not started:
                        try:
                            await start_task
                        except BaseException:
                            started = False
                        else:
                            started = True
                    if started:
                        await asyncio.to_thread(handle.stop_trajectory)
                actual = await asyncio.to_thread(handle.get_joint_positions)
                error_deg = max(
                    (
                        abs(math.degrees(want - got))
                        for want, got in zip(expected, actual)
                    ),
                    default=0.0,
                )
                self.logger.info(
                    "%s: trajectory complete expected=%s actual=%s max_error_deg=%.2f",
                    self.name,
                    [round(math.degrees(value), 2) for value in expected],
                    [round(math.degrees(value), 2) for value in actual],
                    error_deg,
                )
            yield Arm.TrajectoryUpdate()

    async def get_joint_positions(self, **kwargs) -> JointPositions:
        radians = await asyncio.to_thread(self._h().get_joint_positions)
        return JointPositions(values=[math.degrees(r) for r in radians])

    async def stop(self, **kwargs) -> None:
        await asyncio.to_thread(self._h().stop)

    async def is_moving(self) -> bool:
        """True while a MoveTo* RPC is in flight."""
        return self._moving_count > 0

    async def get_kinematics(self, **kwargs) -> KinematicsReturn:
        handle = self._h()
        fmt = (
            KinematicsFileFormat.KINEMATICS_FILE_FORMAT_URDF
            if robots.is_urdf(handle.kinematics_url)
            else KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA
        )
        if not handle.kinematics_meshes:
            return fmt, handle.kinematics_data
        meshes = {
            path: Mesh(content_type=kind, mesh=data)
            for path, (kind, data) in handle.kinematics_meshes.items()
        }
        return fmt, handle.kinematics_data, meshes

    async def get_3d_models(self, **kwargs) -> Mapping[str, Mesh]:
        return {}

    async def set_manual_mode(
        self, manual_mode: bool, enabled_for: int = 0, **kwargs
    ) -> None:
        raise NotImplementedError("manual mode is not supported by the simulated arm")

    async def get_manual_mode(self, **kwargs) -> bool:
        return False

    async def get_properties(self, **kwargs) -> Arm.Properties:
        return Arm.Properties(
            support_manual_mode=False,
            support_cartesian_commands=False,
        )

    async def get_geometries(self, **kwargs) -> List[Geometry]:
        return []

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        cmd = command.get("command")
        if cmd == "get_joint_positions_radians":
            return cast(
                Mapping[str, ValueTypes],
                {"values": await asyncio.to_thread(self._h().get_joint_positions)},
            )
        if cmd == "dynamics_metadata":
            return await asyncio.to_thread(self._h().dynamics_metadata)
        if cmd == "set_joint_targets_radians":
            raw = command.get("values")
            count = len(self._h().joint_names())
            if not isinstance(raw, (list, tuple)) or len(raw) != count:
                raise ValueError(f"set_joint_targets_radians requires {count} values")
            targets = [float(value) for value in cast(Any, raw)]
            if not all(math.isfinite(value) for value in targets):
                raise ValueError("set_joint_targets_radians values must be finite")
            await asyncio.to_thread(self._h().set_joint_targets, targets)
            return {"accepted": True}
        raise ValueError(f"unknown command: {command}")
