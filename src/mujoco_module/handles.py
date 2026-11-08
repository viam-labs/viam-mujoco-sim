"""Engine-side handles that the Viam component models call.

Each handle registers a Contribution (what it adds to the scene) and, where it
drives actuators, a step callback. Only the sim thread writes data.ctrl (step
callbacks, init, holds and close). The arm resolves ids again when the sim's
generation changes (every rebuild); the gripper looks names up on each step.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass
from typing import Any, Literal

import mujoco
import numpy as np

from . import bases, robots
from .sim import GRIPPER_ASSETS, KNOWN_ASSETS, LOGGER, Contribution, SimManager
from viam.spatialmath import Quaternion

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (w, x, y, z), MuJoCo's order

DEFAULT_MAX_JOINT_DEG_S = 180.0
# Each component that others can be framed on adds "{name}/viam_frame": its
# frame in Viam's frame system (an arm's end effector, a base's origin).
VIAM_FRAME_SITE = "viam_frame"
DEFAULT_IS_MOVING_VELOCITY_THRESHOLD_RAD_S = 0.01
DEFAULT_GRIP_FORCE_N = 5.0

# Bundled bases: default attributes for `asset` (config attributes win).
# Paths are relative to assets/.
BASE_ASSETS: dict[str, dict[str, Any]] = {
    # written here from Viam's ViamRover2-2024-rpi5 fragment (motors max_rpm 120)
    "viam_rover": {
        "mjcf": "viam_rover/rover.xml",
        "max_linear_mps": 0.762,  # 120 rpm x 0.381 m
        "max_angular_dps": 245.0,  # 2 x 0.762 / 0.356 rad/s
    },
    # written here from husky/husky@41e15d2 husky_description; speeds and
    # acceleration limits from husky_control/config/control.yaml
    "clearpath_husky": {
        "mjcf": "clearpath_husky/husky.xml",
        "max_linear_mps": 1.0,
        "max_angular_dps": 115.0,  # 2.0 rad/s
        "max_linear_accel_mps2": 3.0,
        "max_angular_accel_rad_s2": 6.0,
    },
}
BASE_JOINTS = ("x", "y", "yaw")
BASE_TAU_S = 0.005  # velocity actuators settle on a new command in ~5 ms
BASE_MAX_LINEAR_ACCEL = 50.0  # m/s^2, caps actuator force (no runaway push into a wall)
BASE_MAX_YAW_ACCEL = 200.0  # rad/s^2
BASE_MOVING_MPS = 0.001
BASE_MOVING_RAD_S = math.radians(0.5)
BASE_PROGRESS = 1e-3  # m or rad: less is no progress (the stall check)
DEFAULT_BASE_MOVE_MARGIN_S = 2.0
# drive "wheels"
GRAVITY = 9.81
BASE_WHEEL_SPEED_ERROR = 1.0  # rad/s at which an added wheel actuator reaches its force cap (stiffer chatters)
DEFAULT_BASE_ACCEL_MPS2 = 1.0  # braking ramp; the angular one is 2 a / track
BASE_MIN_MPS = 0.02  # the braking ramp's least speed (or 10 % of the command)
BASE_MIN_RAD_S = math.radians(2)
BASE_HEADING_GAIN = 3.0  # 1/s: MoveStraight's yaw rate per radian off the start heading
BASE_YAW_RATE_GAIN = 8.0  # 1/s: integral feedback on the measured yaw rate


@dataclass(frozen=True)
class JointTrajectoryPoint:
    time_s: float
    positions: tuple[float, ...]


@dataclass
class _Trajectory:
    points: tuple[JointTrajectoryPoint, ...]
    start_s: float
    future: Future


@dataclass
class _BaseGoal:
    kind: Literal["straight", "spin"]  # meters along the start heading, or radians of yaw
    target: float  # signed
    start_xy: np.ndarray
    start_yaw: float
    future: Future
    best: float  # the least remaining distance or angle so far
    best_s: float  # sim time when it was reached


@dataclass(frozen=True)
class _BaseIds:
    site: int  # the base's viam_frame
    ctrl: np.ndarray  # ideal: the x, y and yaw actuators; wheels: each wheel's
    wheel_qpos: np.ndarray
    wheel_dof: np.ndarray


@dataclass(frozen=True)
class _ArmIds:
    qpos: np.ndarray
    dof: np.ndarray
    ctrl: np.ndarray
    end_site: int  # the arm's viam_frame site


def _interpolate(points: Sequence[JointTrajectoryPoint], t: float) -> np.ndarray:
    for before, after in itertools.pairwise(points):
        if t <= after.time_s:
            span = after.time_s - before.time_s
            alpha = (t - before.time_s) / span if span > 0 else 1.0
            a, b = np.asarray(before.positions), np.asarray(after.positions)
            return a + alpha * (b - a)
    return np.asarray(points[-1].positions)


class ArmHandle:
    """An arm from an MJCF file and a Viam kinematics file. Its joints,
    actuators and home are inferred from the MJCF (robots.py); `attrs` are
    the arm's attributes, over KNOWN_ASSETS[asset] when asset is set."""

    def __init__(self, sim: SimManager, name: str, attrs: dict[str, Any]) -> None:
        attrs = {**KNOWN_ASSETS.get(str(attrs.get("asset")), {}), **attrs}
        label = f"arm {name}"
        self._sim = sim
        self.name = name
        self._prefix = f"{name}/"
        self.mjcf = robots.resolve_path(str(attrs["mjcf"]), f"{label}: mjcf")
        where = str(self.mjcf)
        urdf = self.mjcf.suffix.lower() == ".urdf"
        robots.check_flange_attrs(attrs, urdf, label)
        mesh_dir = attrs.get("mesh_dir")
        spec = robots.load_mjcf(self.mjcf, mesh_dir)
        if urdf:
            flange = robots.flange_link(self.mjcf.read_bytes(), attrs.get("flange_link"), where)
        # served by GetKinematics with the kinematics: RDK reads a URDF's mesh collisions from them
        self.kinematics_meshes: dict[str, tuple[str, bytes]] = {}
        if urdf and "kinematics_url" not in attrs:
            self.kinematics_url = where
            self.kinematics_data = robots.prune_urdf(self.mjcf.read_bytes(), flange)
            self.kinematics_meshes = robots.kinematics_meshes(
                self.kinematics_data, robots.mesh_folder(self.mjcf, mesh_dir), where
            )
        else:
            # read here, not on first GetKinematics: building the arm needs it
            # ponytail: a cold-cache http(s) fetch blocks the event loop (up to the download timeout); fetch in validate_config or make reconfigure async if that bites
            self.kinematics_url = str(attrs["kinematics_url"])
            self.kinematics_data = robots.load_kinematics(self.kinematics_url, f"{label}: kinematics_url")
        urdf_kinematics = robots.is_urdf(self.kinematics_url)
        self.kinematics = robots.parse_kinematics(self.kinematics_data, urdf_kinematics, self.kinematics_url)
        names = attrs.get("joints")
        # an MJCF arm's kinematics_url isn't checked: it never was (backward compatibility)
        if urdf and urdf_kinematics and "kinematics_url" in attrs and robots.has_mesh_collisions(
            self.kinematics_data
        ):
            raise ValueError(
                f"{label}: kinematics_url {self.kinematics_url} has mesh collisions, which Viam's "
                "planner can't read from it; leave kinematics_url unset to use the mjcf's own "
                "(served with its meshes), or use an SVA file"
            )
        if urdf and names is None:
            if not urdf_kinematics:
                raise ValueError(
                    f"{label}: a .urdf arm with an SVA kinematics_url needs joints, the URDF's "
                    "arm joint names in chain order"
                )
            names = list(self.kinematics.joints)
        # The arm on its own, as the file describes it. Ids below index this
        # model; names carry over into the scene with the prefix.
        self.model = spec.compile()
        self.joints = robots.arm_joints(self.model, len(self.kinematics.joints), where, names)
        if urdf:
            robots.prepare_urdf_arm(spec, [self.model.joint(j).name for j in self.joints], flange, where)
            self.model = spec.compile()
        self.actuators = robots.arm_actuators(self.model, self.joints, where)
        self._joint_names = tuple(self._prefix + self.model.joint(j).name for j in self.joints)
        self._actuator_names = tuple(self._prefix + self.model.actuator(a).name for a in self.actuators)
        self._home = tuple(
            float(value)
            for value in robots.home(self.model, str(attrs.get("home_keyframe", "home")), self.actuators)
        )
        self.root_body = self.model.body(
            int(self.model.body_rootid[self.model.jnt_bodyid[self.joints[0]]])
        ).name
        self.flange_site = "attachment_site" if urdf else str(attrs.get("flange_site", "attachment_site"))
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.flange_site) < 0:
            raise ValueError(f"{where}: has no site {self.flange_site!r}; set flange_site")
        site = self.model.site(self.flange_site)
        base_offset = robots.parse_pose(attrs.get("base_offset"), f"{label}: base_offset")
        flange_offset = robots.parse_pose(attrs.get("flange_offset"), f"{label}: flange_offset")
        robots.check_frames(
            self.model, self.kinematics, self.joints, self.model.site(self.flange_site).id,
            self._home, base_offset, flange_offset, label,
        )
        self._max_rad_s = math.radians(
            float(attrs.get("max_joint_deg_s", DEFAULT_MAX_JOINT_DEG_S))
        )
        acceleration = attrs.get("max_joint_deg_s2")
        self._max_rad_s2 = math.radians(float(acceleration)) if acceleration else None
        self._velocity_threshold = float(
            attrs.get(
                "is_moving_velocity_threshold_rad_s",
                DEFAULT_IS_MOVING_VELOCITY_THRESHOLD_RAD_S,
            )
        )
        # None until init (new arm) or the first step (replaced arm) sets it
        self._target: np.ndarray | None = None
        self._trajectory: _Trajectory | None = None
        self._waits: dict[Future, str] = {}
        self._ids: _ArmIds | None = None
        self._ids_generation = -1
        self._callback_key = f"arm:{name}"
        # The arm's Viam frame (its configured pose) and, base_offset on from
        # it, where the MJCF's world goes.
        self._frame = robots.transform(
            [float(value) for value in attrs.get("position", (0.0, 0.0, 0.0))],
            [float(value) for value in attrs.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0))],
        )
        mount = self._frame @ base_offset
        excluded: list[tuple[str, str]] = []
        has_shapes = any(frame.geometry for frame in self.kinematics.frames.values())
        collision = attrs.get("collision") or ("kinematics" if has_shapes else "mesh")
        shapes: list[robots.Shape] = []
        if collision == "kinematics":
            if not has_shapes:
                raise ValueError(
                    f'{label}: collision "kinematics" needs an SVA kinematics file with '
                    'geometries; use collision "mesh"'
                )
            shapes = robots.kinematics_shapes(
                self.model, self.kinematics, self.joints, self.model.body(self.root_body).id, base_offset
            )
            # Like RDK, ignore collisions the arm is already in at home.
            child = spec.copy()
            robots.apply_shapes(child, shapes, label)
            excluded += robots.home_contacts(child, self.joint_names(), self._home)

        arm_spec = spec

        def build(spec: mujoco.MjSpec) -> None:
            child = arm_spec.copy()  # the file isn't read again
            robots.apply_shapes(child, shapes, label)
            frame = spec.worldbody.add_frame(pos=list(mount[:3, 3]), quat=list(robots.quat(mount)))
            spec.attach(child, prefix=self._prefix, frame=frame)
            # Viam's end-effector frame, which children of the arm's frame
            # (a wrist camera) are placed relative to.
            flange = spec.site(self._prefix + self.flange_site)
            # from the compiled model: the spec's quat ignores euler/axisangle/xyaxes/zaxis
            end = robots.transform(site.pos, site.quat) @ flange_offset
            flange.parent.add_site(
                name=self._prefix + VIAM_FRAME_SITE, pos=list(end[:3, 3]), quat=list(robots.quat(end))
            )
            for first, second in excluded:
                spec.add_exclude(bodyname1=self._prefix + first, bodyname2=self._prefix + second)

        def init(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            ids = self._resolve(model)
            data.qpos[ids.qpos] = self._home
            data.qvel[ids.dof] = 0.0
            data.ctrl[ids.ctrl] = self._home
            self._target = np.asarray(self._home)
            self._drop_trajectory()

        def register() -> None:
            sim.step_callbacks[self._callback_key] = self._on_step
            sim.arms[name] = self

        sim.set_contribution(name, Contribution(build=build, init=init))
        sim.run(register)

    def close(self) -> None:
        def detach() -> None:
            self._sim.step_callbacks.pop(self._callback_key, None)
            if self._sim.arms.get(self.name) is self:
                del self._sim.arms[self.name]
            error = RuntimeError(f"arm {self.name!r} was closed (removed or reconfigured)")
            trajectory, self._trajectory = self._trajectory, None
            pending = [trajectory.future] if trajectory is not None else []
            for future, key in list(self._waits.items()):
                self._sim.step_callbacks.pop(key, None)
                pending.append(future)
            self._waits.clear()
            for future in pending:
                if not future.done():
                    try:
                        future.set_exception(error)
                    except InvalidStateError:
                        pass

        self._sim.run(detach)
        self._sim.remove_contribution(self.name)

    # -- ids -------------------------------------------------------------

    def _resolve(self, model: mujoco.MjModel) -> _ArmIds:
        joints = [model.joint(name) for name in self._joint_names]
        return _ArmIds(
            qpos=np.array([joint.qposadr[0] for joint in joints]),
            dof=np.array([joint.dofadr[0] for joint in joints]),
            ctrl=np.array([model.actuator(name).id for name in self._actuator_names]),
            end_site=model.site(self._prefix + VIAM_FRAME_SITE).id,
        )

    def _now(self) -> tuple[mujoco.MjData, _ArmIds]:
        sim = self._sim
        assert sim.model is not None and sim.data is not None
        if self._ids is None or self._ids_generation != sim.generation:
            self._ids = self._resolve(sim.model)
            self._ids_generation = sim.generation
        return sim.data, self._ids

    # -- control (sim thread) ---------------------------------------------

    def _on_step(self, dt: float) -> None:
        data, ids = self._now()
        trajectory = self._trajectory
        try:
            self._step(dt, data, ids, trajectory)
        except Exception as exc:
            # keep the callback alive so the arm keeps holding position
            self._trajectory = None
            self._target = data.ctrl[ids.ctrl].copy()
            if trajectory is None:
                raise
            if not trajectory.future.done():
                try:
                    trajectory.future.set_exception(exc)
                except InvalidStateError:
                    pass

    def _step(
        self,
        dt: float,
        data: mujoco.MjData,
        ids: _ArmIds,
        trajectory: _Trajectory | None,
    ) -> None:
        if trajectory is not None:
            t = data.time - trajectory.start_s
            final = trajectory.points[-1]
            if t >= final.time_s:
                self._target = np.asarray(final.positions)
                data.ctrl[ids.ctrl] = self._target
                self._trajectory = None
                if not trajectory.future.done():
                    try:
                        trajectory.future.set_result(None)
                    except InvalidStateError:
                        pass
            else:
                data.ctrl[ids.ctrl] = _interpolate(trajectory.points, t)
            return
        current = data.ctrl[ids.ctrl]
        if self._target is None:
            self._target = current.copy()
        # Without a velocity limit the kp=2000 servo snaps to far targets.
        step = self._max_rad_s * dt
        data.ctrl[ids.ctrl] = current + np.clip(self._target - current, -step, step)

    def _drop_trajectory(self) -> _Trajectory | None:
        trajectory, self._trajectory = self._trajectory, None
        if trajectory is not None and not trajectory.future.done():
            trajectory.future.cancel()
        return trajectory

    def _hold_measured(self) -> None:
        data, ids = self._now()
        measured = data.qpos[ids.qpos].copy()
        self._target = measured
        data.ctrl[ids.ctrl] = measured

    # -- public API ---------------------------------------------------------

    def joint_names(self) -> tuple[str, ...]:
        """The arm's MJCF joint names, in chain order."""
        return tuple(name.removeprefix(self._prefix) for name in self._joint_names)

    def get_joint_positions(self) -> list[float]:
        def read() -> list[float]:
            data, ids = self._now()
            return [float(value) for value in data.qpos[ids.qpos]]

        return self._sim.run(read)

    def set_joint_targets(self, positions: Sequence[float]) -> None:
        targets = np.asarray([float(value) for value in positions])
        if targets.shape != (len(self._joint_names),):
            raise ValueError(f"expected {len(self._joint_names)} joint targets")

        def apply() -> None:
            if self._drop_trajectory() is not None:
                self._hold_measured()
            self._target = targets

        self._sim.run(apply)

    def wait_for_joint_targets(
        self, targets: Sequence[float], tolerance_rad: float
    ) -> Future:
        """Resolve once every joint is within tolerance of its target and slower
        than the is-moving threshold. Checked once per physics step on the sim
        thread; cancelling the future removes the check."""
        future: Future = Future()
        key = f"arm-wait:{self.name}:{id(future)}"
        wanted = np.asarray([float(value) for value in targets])

        def check(_dt: float) -> None:
            if future.done():
                self._sim.step_callbacks.pop(key, None)
                self._waits.pop(future, None)
                return
            try:
                data, ids = self._now()
                arrived = bool(
                    np.all(np.abs(data.qpos[ids.qpos] - wanted) <= tolerance_rad)
                    and np.all(np.abs(data.qvel[ids.dof]) <= self._velocity_threshold)
                )
            except BaseException as exc:
                self._sim.step_callbacks.pop(key, None)
                self._waits.pop(future, None)
                future.set_exception(exc)
                return
            if arrived:
                self._sim.step_callbacks.pop(key, None)
                self._waits.pop(future, None)
                future.set_result(None)

        def attach() -> None:
            self._sim.step_callbacks[key] = check
            self._waits[future] = key
            check(0.0)

        self._sim.run(attach)
        return future

    def start_trajectory(self, points: Sequence[JointTrajectoryPoint]) -> Future:
        """Follow timestamped points in sim time. The future resolves when the
        last point's time passes; the arm then holds that point."""
        if not points:
            raise ValueError("trajectory needs at least one point")
        count = len(self._joint_names)
        if any(len(point.positions) != count for point in points):
            raise ValueError(f"every trajectory point needs {count} positions")
        times = [point.time_s for point in points]
        if times[0] < 0 or any(b < a for a, b in itertools.pairwise(times)):
            raise ValueError("trajectory times must be non-negative and non-decreasing")
        future: Future = Future()

        def start() -> None:
            self._drop_trajectory()
            data, ids = self._now()
            pts = tuple(points)
            if pts[0].time_s > 0:  # start from the commanded pose, don't extrapolate back
                here = tuple(float(v) for v in data.ctrl[ids.ctrl])
                pts = (JointTrajectoryPoint(0.0, here), *pts)
            self._trajectory = _Trajectory(pts, data.time, future)

        self._sim.run(start)
        return future

    def stop_trajectory(self) -> None:
        def stop() -> None:
            if self._drop_trajectory() is not None:
                self._hold_measured()

        self._sim.run(stop)

    def stop(self) -> None:
        def stop() -> None:
            if self._drop_trajectory() is not None:
                self._hold_measured()
            else:  # measured qpos includes gravity sag; holding it would ratchet down
                data, ids = self._now()
                self._target = data.ctrl[ids.ctrl].copy()

        self._sim.run(stop)

    def dynamics_metadata(self) -> dict[str, Any]:
        count = len(self._joint_names)
        result: dict[str, Any] = {
            "source": "mujoco-rate-limited-position-servo",
            "joint_names": list(self.joint_names()),
            "max_velocity_rad_s": [self._max_rad_s] * count,
        }
        if self._max_rad_s2 is not None:
            result["max_acceleration_rad_s2"] = [self._max_rad_s2] * count
        return result

    def get_end_pose(self) -> tuple[Vec3, Quat]:
        """Viam's end-effector frame (the flange with flange_offset) in the
        arm's frame (metres, w-first quaternion)."""

        def read() -> tuple[Vec3, Quat]:
            data, ids = self._now()
            rotation = self._frame[:3, :3]
            position = rotation.T @ (data.site_xpos[ids.end_site] - self._frame[:3, 3])
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (rotation.T @ data.site_xmat[ids.end_site].reshape(3, 3)).ravel())
            return (
                (float(position[0]), float(position[1]), float(position[2])),
                (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
            )

        return self._sim.run(read)


def _fail(future: Future | None, error: BaseException) -> None:
    if future is not None and not future.done():
        try:
            future.set_exception(error)
        except InvalidStateError:
            pass


class BaseHandle:
    """A differential-drive or skid-steer base from an MJCF or URDF file; its
    chassis and wheels are inferred (bases.py). The viam_frame site, on the
    floor under the chassis origin with +Y forward, is the base's frame.
    drive "ideal": the chassis rides spawn-frame x/y slides and a yaw hinge
    driven by velocity actuators, so motion is exact; the wheels only turn
    to match it and collide with nothing. drive "wheels": the chassis is
    free and velocity actuators on the wheels drive it through floor
    friction; MoveStraight and Spin close the loop on the measured pose."""

    def __init__(self, sim: SimManager, name: str, attrs: dict[str, Any]) -> None:
        meta = BASE_ASSETS.get(str(attrs.get("asset")), {})
        attrs = {**meta, **attrs}
        label = f"base {name}"
        self._sim = sim
        self.name = name
        self._prefix = f"{name}/"
        self.mjcf = robots.resolve_path(str(attrs["mjcf"]), f"{label}: mjcf")
        where = str(self.mjcf)
        mesh_dir = attrs.get("mesh_dir")
        spec = robots.load_file(self.mjcf, where, mesh_dir)
        chassis = bases.take_chassis(spec, attrs.get("chassis"), where)
        try:
            model = spec.compile()
        except ValueError as e:
            raise ValueError(f"{where}: cannot load mjcf {self.mjcf}: {e}") from e
        self.base = base = bases.infer(
            model, where, attrs.get("wheels"), attrs.get("forward_axis"), attrs.get("wheel_radius_m")
        )
        limits = bases.speed_limits(model, base)
        linear = attrs.get("max_linear_mps", limits and limits[0])
        angular = attrs.get("max_angular_dps", limits and limits[1])
        if linear is None or angular is None:
            raise ValueError(
                f"{label}: set max_linear_mps and max_angular_dps; {where} has no velocity "
                "actuators with a ctrlrange on its wheels to take them from"
            )
        self.max_linear_mps = float(linear)
        self.max_angular_rad_s = math.radians(float(angular))
        self.move_margin_s = float(
            attrs.get("move_timeout_margin_sec", DEFAULT_BASE_MOVE_MARGIN_S)
        )
        self.drive = str(attrs.get("drive", "ideal"))
        self._linear_accel = float(meta.get("max_linear_accel_mps2", DEFAULT_BASE_ACCEL_MPS2))
        self._angular_accel = float(
            meta.get("max_angular_accel_rad_s2", 2 * self._linear_accel / base.track)
        )
        # wheels: each driven wheel's actuator, the file's or one added at build
        own = bases.wheel_actuators(model, base, where) if self.drive == "wheels" else {}
        self._wheel_actuators = {w.joint: own.get(w.joint, f"{w.joint}_drive") for w in base.wheels}
        caps = {
            w.joint: bases.traction(model, w.joint) * base.mass * GRAVITY * base.radius / len(base.wheels)
            for w in base.wheels if w.joint not in own
        }
        self._integral = 0.0  # of the yaw-rate error: rad (MoveStraight), or relative (Spin), s
        self._signs = np.array([wheel.sign for wheel in base.wheels])
        self._sides = np.array([-1.0 if wheel.left else 1.0 for wheel in base.wheels])
        self._v = 0.0  # latched forward speed, m/s
        self._w = 0.0  # latched yaw rate, rad/s
        self._yaw = 0.0  # unwrapped yaw of the viam frame in the spawn frame
        self._goal: _BaseGoal | None = None
        self._ids: _BaseIds | None = None
        self._ids_generation = -1
        self._callback_key = f"base:{name}"
        position = [float(v) for v in attrs.get("position", (0.0, 0.0, 0.0))]
        orientation = [float(v) for v in attrs.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0))]
        self._spawn = robots.transform(position, orientation)
        LOGGER.info(
            "%s: %.2f kg, wheels left %s right %s, radius %.4f m, track %.4f m, forward %s",
            label, base.mass, [w.joint for w in base.wheels if w.left],
            [w.joint for w in base.wheels if not w.left], base.radius, base.track, base.forward,
        )

        spec_cache = spec

        def build(spec: mujoco.MjSpec) -> None:
            child = spec_cache.copy()  # the file isn't read again
            if self.drive == "ideal":
                for actuator in list(child.actuators):
                    child.delete(actuator)
                for body in _jointed(child.body(chassis)):  # wheels and casters
                    for geom in body.find_all(mujoco.mjtObj.mjOBJ_GEOM):
                        geom.contype = geom.conaffinity = 0
            frame = spec.worldbody.add_frame(pos=position, quat=orientation)
            spec.attach(child, prefix=self._prefix, frame=frame)
            body = spec.body(self._prefix + chassis)
            half = base.yaw / 2
            body.pos = [0.0, 0.0, base.lift]
            body.quat = [math.cos(half), 0.0, 0.0, math.sin(half)]
            # the base's frame: on the floor under the chassis origin, +Y forward
            body.add_site(
                name=self._prefix + VIAM_FRAME_SITE,
                pos=[0.0, 0.0, -base.lift],
                quat=[math.cos(half), 0.0, 0.0, -math.sin(half)],
            )
            if self.drive == "wheels":
                body.add_freejoint(name=self._prefix + "free")
                for joint, cap in caps.items():
                    actuator = spec.add_actuator(
                        name=self._prefix + self._wheel_actuators[joint],
                        target=self._prefix + joint,
                        trntype=mujoco.mjtTrn.mjTRN_JOINT,
                    )
                    actuator.set_to_velocity(kv=cap / BASE_WHEEL_SPEED_ERROR)
                    actuator.forcelimited = True
                    actuator.forcerange = [-cap, cap]
                return
            c, s = math.cos(base.yaw), math.sin(base.yaw)
            # the spawn frame's x and y in the (turned) chassis frame
            axes = ([c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0])
            kinds = (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE)
            limits = (
                (self.max_linear_mps, base.mass / BASE_TAU_S, base.mass * BASE_MAX_LINEAR_ACCEL),
                (self.max_linear_mps, base.mass / BASE_TAU_S, base.mass * BASE_MAX_LINEAR_ACCEL),
                (self.max_angular_rad_s, base.izz / BASE_TAU_S, base.izz * BASE_MAX_YAW_ACCEL),
            )
            for joint, axis, kind, (speed, gain, cap) in zip(BASE_JOINTS, axes, kinds, limits):
                body.add_joint(name=self._prefix + joint, type=kind, axis=axis)
                actuator = spec.add_actuator(
                    name=self._prefix + joint,
                    target=self._prefix + joint,
                    trntype=mujoco.mjtTrn.mjTRN_JOINT,
                )
                actuator.set_to_velocity(kv=gain)
                actuator.ctrllimited = True
                actuator.ctrlrange = [-1.1 * speed, 1.1 * speed]
                actuator.forcelimited = True
                actuator.forcerange = [-cap, cap]

        def init(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            data.ctrl[self._resolve(model).ctrl] = 0.0
            self._v = self._w = self._yaw = self._integral = 0.0
            self._drop_goal(RuntimeError(f"base {name!r} was reset"))

        sim.set_contribution(name, Contribution(build=build, init=init))
        sim.run(lambda: sim.step_callbacks.__setitem__(self._callback_key, self._on_step))

    def close(self) -> None:
        def detach() -> None:
            self._sim.step_callbacks.pop(self._callback_key, None)
            self._v = self._w = 0.0
            self._drop_goal(
                RuntimeError(f"base {self.name!r} was closed (removed or reconfigured)")
            )
            if self._sim.model is not None and self._sim.data is not None:
                data, ids = self._now()
                data.ctrl[ids.ctrl] = 0.0

        self._sim.run(detach)
        self._sim.remove_contribution(self.name)

    # -- ids -------------------------------------------------------------

    def _resolve(self, model: mujoco.MjModel) -> _BaseIds:
        wheels = [model.joint(self._prefix + wheel.joint) for wheel in self.base.wheels]
        if self.drive == "wheels":
            actuators = [self._wheel_actuators[wheel.joint] for wheel in self.base.wheels]
        else:
            actuators = list(BASE_JOINTS)
        return _BaseIds(
            site=model.site(self._prefix + VIAM_FRAME_SITE).id,
            ctrl=np.array([model.actuator(self._prefix + name).id for name in actuators]),
            wheel_qpos=np.array([joint.qposadr[0] for joint in wheels]),
            wheel_dof=np.array([joint.dofadr[0] for joint in wheels]),
        )

    def _now(self) -> tuple[mujoco.MjData, _BaseIds]:
        sim = self._sim
        assert sim.model is not None and sim.data is not None
        if self._ids is None or self._ids_generation != sim.generation:
            self._ids = self._resolve(sim.model)
            self._ids_generation = sim.generation
        return sim.data, self._ids

    def _pose(self, data: mujoco.MjData, ids: _BaseIds) -> tuple[float, float, float]:
        """The viam frame's x, y and yaw in the spawn frame; yaw is unwrapped
        from the previous call, so goals see continuous angles."""
        rotation = self._spawn[:3, :3]
        x, y, _ = rotation.T @ (data.site_xpos[ids.site] - self._spawn[:3, 3])
        forward = rotation.T @ data.site_xmat[ids.site].reshape(3, 3)[:, 1]
        yaw = math.atan2(-forward[0], forward[1])
        self._yaw += math.remainder(yaw - self._yaw, 2 * math.pi)
        return float(x), float(y), self._yaw

    def _velocity(self, data: mujoco.MjData, ids: _BaseIds) -> np.ndarray:
        """(angular, linear) velocity of the viam frame in its own axes."""
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self._sim.model, data, mujoco.mjtObj.mjOBJ_SITE, ids.site, velocity, 1)
        return velocity

    # -- control (sim thread) ---------------------------------------------

    def _on_step(self, dt: float) -> None:
        data, ids = self._now()
        x, y, yaw = self._pose(data, ids)
        velocity = self._velocity(data, ids)
        goal = self._goal
        remaining = 0.0
        if goal is not None:
            try:
                if goal.future.done():  # cancelled by the caller
                    self._drop_goal()
                else:
                    progress = self._progress(goal, x, y, yaw)
                    remaining = abs(goal.target) - progress * math.copysign(1.0, goal.target)
                    if remaining <= 0.0:
                        self._drop_goal()
                        try:
                            goal.future.set_result(None)
                        except InvalidStateError:
                            pass
                    elif remaining < goal.best - BASE_PROGRESS:
                        goal.best, goal.best_s = remaining, data.time
                    elif data.time - goal.best_s >= self.move_margin_s:
                        self._drop_goal(RuntimeError(self._stalled(goal, progress)))
            except Exception as exc:
                # halt, fail the move, and keep the callback registered
                self._drop_goal(exc)
        if self.drive == "wheels":
            v, w = self._wheel_command(remaining, yaw, float(velocity[2]), dt)
            sides = v + self._sides * w * self.base.track / 2
            data.ctrl[ids.ctrl] = self._signs * sides / self.base.radius
            return
        data.ctrl[ids.ctrl] = (-self._v * math.sin(yaw), self._v * math.cos(yaw), self._w)
        # turn the wheels by the distance each side travelled
        speeds = velocity[4] + self._sides * velocity[2] * self.base.track / 2
        data.qpos[ids.wheel_qpos] += self._signs * speeds * dt / self.base.radius
        data.qvel[ids.wheel_dof] = 0.0

    def _wheel_command(self, remaining: float, yaw: float, yaw_rate: float, dt: float) -> tuple[float, float]:
        """(forward speed, yaw rate) for the wheels. Open loop without a goal
        (a skid-steer under-turns, as the real one does). A goal brakes on a
        ramp to its target, MoveStraight holds the start heading, and the yaw
        rate gets integral feedback so a slipping base still turns."""
        goal = self._goal
        if goal is None:
            return self._v, self._w
        spin = goal.kind == "spin"
        rate = self._w if spin else self._v
        accel = self._angular_accel if spin else self._linear_accel
        least = min(abs(rate), max(0.1 * abs(rate), BASE_MIN_RAD_S if spin else BASE_MIN_MPS))
        speed = math.copysign(max(least, min(abs(rate), math.sqrt(2 * accel * remaining))), rate)
        if spin:
            # integral feedback as a factor on the ramp: it learns how much the
            # base under-turns, and brakes with the ramp instead of winding up
            self._integral += (1.0 - yaw_rate / speed) * dt
            return 0.0, speed * max(0.0, 1.0 + BASE_YAW_RATE_GAIN * self._integral)
        w = BASE_HEADING_GAIN * (goal.start_yaw - yaw)
        self._integral += (w - yaw_rate) * dt
        return speed, w + BASE_YAW_RATE_GAIN * self._integral

    @staticmethod
    def _progress(goal: _BaseGoal, x: float, y: float, yaw: float) -> float:
        if goal.kind == "spin":
            return yaw - goal.start_yaw
        heading = np.array([-math.sin(goal.start_yaw), math.cos(goal.start_yaw)])
        return float(np.dot(np.array([x, y]) - goal.start_xy, heading))

    def _stalled(self, goal: _BaseGoal, progress: float) -> str:
        if goal.kind == "spin":
            done, wanted, unit = math.degrees(progress), math.degrees(goal.target), "deg"
        else:
            done, wanted, unit = 1000 * progress, 1000 * goal.target, "mm"
        return (
            f"base {self.name!r} stalled: moved {done:.0f} of {wanted:.0f} {unit}, then made no "
            f"progress for {self.move_margin_s:g} s (blocked?)"
        )

    def _drop_goal(self, error: BaseException | None = None) -> None:
        """Clear the goal and, only when there was one, the latched command
        (stop, init and close zero it themselves); fail the goal's future with
        `error` when given. Sim thread only."""
        goal, self._goal = self._goal, None
        if goal is not None:
            self._v = self._w = 0.0
            if error is not None:
                _fail(goal.future, error)

    def _replace_goal(self, reason: str) -> None:
        self._drop_goal(RuntimeError(f"base {self.name!r} {reason}"))

    # -- public API ---------------------------------------------------------

    def set_velocity(self, linear_mps: float, angular_rad_s: float) -> None:
        """Latch forward speed and yaw rate (positive = left), clamped to the maxima."""
        if not (math.isfinite(linear_mps) and math.isfinite(angular_rad_s)):
            raise ValueError("velocity must be finite")
        v = float(np.clip(linear_mps, -self.max_linear_mps, self.max_linear_mps))
        w = float(np.clip(angular_rad_s, -self.max_angular_rad_s, self.max_angular_rad_s))

        def apply() -> None:
            self._replace_goal("move was superseded by a new command")
            self._v, self._w = v, w

        self._sim.run(apply)

    def stop(self) -> None:
        def apply() -> None:
            self._replace_goal("move was stopped")
            self._v = self._w = 0.0

        self._sim.run(apply)

    def move_straight(self, distance_m: float, speed_mps: float) -> Future:
        """Drive distance_m along the current heading. The sign of
        distance x speed picks forward or back. The future resolves on arrival
        and fails when the remaining distance hasn't shrunk for
        move_timeout_margin_sec of sim time (stalled), or on stop, a new
        command, reset or close."""
        return self._start_goal("straight", distance_m, speed_mps, self.max_linear_mps)

    def spin(self, angle_rad: float, speed_rad_s: float) -> Future:
        """Turn in place by angle_rad (positive = left); same rules as
        move_straight."""
        return self._start_goal("spin", angle_rad, speed_rad_s, self.max_angular_rad_s)

    def _start_goal(
        self, kind: Literal["straight", "spin"], amount: float, speed: float, max_speed: float
    ) -> Future:
        if not math.isfinite(amount) or not math.isfinite(speed):
            raise ValueError("distance, angle and velocity must be finite")
        if speed == 0.0:
            raise ValueError("velocity must be non-zero")
        future: Future = Future()
        if amount == 0.0:
            future.set_result(None)
            return future
        target = abs(amount) * math.copysign(1.0, amount * speed)
        rate = math.copysign(min(abs(speed), max_speed), target)

        def start() -> None:
            self._replace_goal("move was superseded by a new command")
            data, ids = self._now()
            x, y, yaw = self._pose(data, ids)
            self._goal = _BaseGoal(
                kind=kind,
                target=target,
                start_xy=np.array([x, y]),
                start_yaw=yaw,
                future=future,
                best=abs(target),
                best_s=data.time,
            )
            self._integral = 0.0
            if kind == "spin":
                self._v, self._w = 0.0, rate
            else:
                self._v, self._w = rate, 0.0

        self._sim.run(start)
        return future

    def is_moving(self) -> bool:
        def read() -> bool:
            data, ids = self._now()
            velocity = self._velocity(data, ids)
            return bool(
                self._v != 0.0
                or self._w != 0.0
                or math.hypot(velocity[3], velocity[4]) > BASE_MOVING_MPS
                or abs(velocity[2]) > BASE_MOVING_RAD_S
            )

        return self._sim.run(read)


def _jointed(body: Any) -> list[Any]:
    """The jointed bodies nearest below `body` (each with its subtree): a
    base's wheels and casters."""
    found = []
    for child in body.bodies:
        found += [child] if child.joints else _jointed(child)
    return found


class GripperHandle:
    """A gripper driven by one actuator: the arm MJCF's own (the Lite6's), or
    with attrs["mjcf"] a gripper MJCF attached at the arm's flange site.
    Position control writes the open/closed values to
    ctrl; force control writes their signs times grip_force_n. The jaw is the
    actuator's joint (a tendon's first), in its own units (m or rad). Suction
    (<adhesion>) has no jaw: grab writes the vacuum level to ctrl, open 0."""

    MOVING = 1e-3  # jaw joint units per second

    def __init__(self, sim: SimManager, name: str, arm: str, attrs: dict[str, Any]) -> None:
        attrs = {**GRIPPER_ASSETS.get(str(attrs.get("asset")), {}), **attrs}
        self._sim = sim
        self.name = name
        label = f"gripper {name}"
        handle = sim.run(lambda: sim.arms.get(arm))
        if handle is None:
            raise RuntimeError(f"arm {arm!r} is not in the scene; configure the arm first")
        mjcf = attrs.get("mjcf")
        if mjcf is None:
            model, prefix, where = handle.model, f"{arm}/", f"arm {arm} ({handle.mjcf})"
            actuator = robots.gripper_actuator(
                model, where, exclude=handle.actuators, name=attrs.get("actuator")
            )
        else:
            own = [handle.model.actuator(a).name for a in range(handle.model.nu) if a not in handle.actuators]
            if own:
                raise ValueError(
                    f"{label}: arm {arm!r} already has a gripper in its MJCF (actuators {own}); "
                    "drop mjcf to use the arm's own gripper, or use an arm MJCF without one"
                )
            path = robots.resolve_path(str(mjcf), f"{label}: mjcf")
            gripper_spec = robots.load_mjcf(path, attrs.get("mesh_dir"))
            prefix, where = f"{name}/", str(path)
            if path.suffix.lower() == ".urdf":
                robots.add_servos(gripper_spec, [robots.driven_joint(gripper_spec.compile(), where)], where)
            model = gripper_spec.compile()
            actuator = robots.gripper_actuator(model, where, name=attrs.get("actuator"))
        kind = robots.control_type(model, actuator, where)
        self._position = kind == "position"
        self.suction = kind == "suction"
        low, high = (float(value) for value in model.actuator_ctrlrange[actuator])
        self._force = float(attrs.get("grip_force_n", DEFAULT_GRIP_FORCE_N))
        if self.suction:
            stray = [key for key in ("open", "closed") if key in attrs]
            if stray:
                raise ValueError(f"{label}: {' and '.join(stray)} do not apply to a suction gripper")
            # adhesion force is gain x ctrl, shared among the cup's contacts
            gain = float(model.actuator_gainprm[actuator, 0])
            self._max = high
            self._open, self._closed = 0.0, high
            if "grip_force_n" in attrs:
                self._closed = self._force / gain
                if not low <= self._closed <= high:
                    raise ValueError(
                        f"{label}: grip_force_n {self._force} is outside the suction actuator's "
                        f"range [{low * gain:g}, {high * gain:g}] N"
                    )
            gripper_body = int(model.actuator_trnid[actuator, 0])
            self._cup = prefix + model.body(gripper_body).name
        else:
            if not model.actuator_ctrllimited[actuator] and not ("open" in attrs and "closed" in attrs):
                raise ValueError(f"{where}: the gripper actuator has no ctrlrange; set open and closed")
            self._open = float(attrs.get("open", low))
            self._closed = float(attrs.get("closed", high))
            jaw = robots.jaw(model, actuator, self._open, self._closed, where)
            self.open_q, self.closed_q = jaw.open, jaw.closed
            self.slide = int(model.jnt_type[jaw.joint]) == int(mujoco.mjtJoint.mjJNT_SLIDE)
            if not model.joint(jaw.joint).name:
                raise ValueError(f"{where}: the gripper's jaw joint has no name; name the MJCF's jaw joint")
            self._joint = prefix + model.joint(jaw.joint).name
            gripper_body = int(model.jnt_bodyid[jaw.joint])
        self._actuator = prefix + model.actuator(actuator).name
        if mjcf is None:
            roots = robots.integrated_gripper_bodies(model, handle.joints[-1], gripper_body)[:1]
        else:
            roots = [b for b in range(1, model.nbody) if model.body_parentid[b] == 0]
        if not all(model.body(root).name for root in roots):
            raise ValueError(f"{where}: a gripper root body has no name; name the MJCF's gripper bodies")
        # Only roots are named; their subtrees are resolved in the current model.
        self._roots = tuple(prefix + model.body(root).name for root in roots)
        # Captures the arm's flange site, integrated-gripper conflict and root at build;
        # viam-server rebuilds dependents when the arm is reconfigured.
        self._arm_root = f"{arm}/{handle.root_body}"
        if self._position:
            if "grip_force_n" in attrs:
                LOGGER.debug("%s: grip_force_n is ignored; the gripper is position controlled", label)
        elif not self.suction:
            if self._open * self._closed >= 0:
                raise ValueError(
                    f"{where}: a force gripper needs open and closed of opposite signs, "
                    f"got {self._open} and {self._closed}"
                )
            if model.actuator_ctrllimited[actuator] and not all(
                low <= math.copysign(self._force, value) <= high for value in (self._open, self._closed)
            ):
                hint = "" if "grip_force_n" in attrs else " (the default); set grip_force_n within it"
                raise ValueError(
                    f"{label}: grip_force_n {self._force} is outside the actuator's "
                    f"ctrlrange [{low}, {high}]{hint}"
                )
        self._ctrl = self._command(self._open)
        self._callback_key = f"gripper:{name}"
        self._attached = mjcf is not None
        if self._attached:
            flange = f"{arm}/{handle.flange_site}"

            def build(spec: mujoco.MjSpec) -> None:
                site = spec.site(flange)
                if site is None:
                    return  # arm removed; the gripper comes back with it
                child = gripper_spec.copy()  # the file isn't read again
                # attach keeps the scene's options; take the gripper's contact
                # settings (the 2F-85's elliptic cones) where it sets them
                defaults = mujoco.MjSpec().option
                for key in ("cone", "impratio"):
                    if getattr(child.option, key) != getattr(defaults, key):
                        setattr(spec.option, key, getattr(child.option, key))
                spec.attach(child, prefix=prefix, site=site)

            sim.set_contribution(name, Contribution(build=build, late=True))
        sim.run(lambda: sim.step_callbacks.__setitem__(self._callback_key, self._on_step))

    def close(self) -> None:
        def detach() -> None:
            self._sim.step_callbacks.pop(self._callback_key, None)
            model, data = self._sim.model, self._sim.data
            if model is not None and data is not None:
                actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self._actuator)
                if actuator >= 0:
                    data.ctrl[actuator] = 0.0

        self._sim.run(detach)
        if self._attached:
            self._sim.remove_contribution(self.name)

    def _command(self, value: float) -> float:
        if self._position or self.suction:
            return value
        return math.copysign(self._force, value)

    def _on_step(self, _dt: float) -> None:
        model, data = self._sim.model, self._sim.data
        assert model is not None and data is not None
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self._actuator)
        if actuator < 0:
            return  # arm absent; resume when it is re-added
        data.ctrl[actuator] = self._ctrl

    def _require_scene(self) -> None:
        model = self._sim.model
        if model is None or mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self._actuator) < 0:
            raise RuntimeError(f"the arm for gripper {self.name!r} is not in the scene")

    def _set(self, ctrl: float) -> None:
        def apply() -> None:
            self._ctrl = ctrl

        self._sim.run(apply)

    def open_jaws(self) -> None:
        self._set(self._command(self._open))

    def close_jaws(self) -> None:
        self._set(self._command(self._closed))

    def stop(self) -> None:
        """Force: no force. Position: hold the jaw where it is. Suction: keep
        the vacuum, so a held object stays held."""

        def apply() -> None:
            if self.suction:
                return
            if not self._position:
                self._ctrl = 0.0
                return
            jaw = self._read_jaw()
            span = self.closed_q - self.open_q
            fraction = min(max((jaw - self.open_q) / span, 0.0), 1.0) if span else 0.0
            # linear between open and closed ctrl: approximate for tendon/coupled grippers
            self._ctrl = self._open + fraction * (self._closed - self._open)

        self._sim.run(apply)

    def _read_jaw(self) -> float:
        self._require_scene()
        assert self._sim.model is not None and self._sim.data is not None
        return float(self._sim.data.qpos[self._sim.model.joint(self._joint).qposadr[0]])

    def jaw(self) -> float:
        """The jaw joint's position, from open_q to closed_q."""
        return self._sim.run(self._read_jaw)

    def vacuum(self) -> float:
        """Suction: the commanded vacuum, 0 to 1 of the actuator's maximum."""
        return self._sim.run(lambda: self._ctrl / self._max)

    def is_moving(self) -> bool:
        def read() -> bool:
            self._require_scene()
            if self.suction:
                return False
            assert self._sim.model is not None and self._sim.data is not None
            joint = self._sim.model.joint(self._joint)
            return bool(abs(self._sim.data.qvel[joint.dofadr[0]]) > self.MOVING)

        return self._sim.run(read)

    def firm_grasp(self) -> bool:
        """Some body, neither the world nor part of the arm or gripper, touches
        the gripper at two contacts that push from opposite sides. Suction:
        the vacuum is on and the cup touches a body that can move (not an
        obstacle) and is not part of the arm or gripper."""

        def read() -> bool:
            self._require_scene()
            model, data = self._sim.model, self._sim.data
            assert model is not None and data is not None
            arm_root = model.body_rootid[model.body(self._arm_root).id]
            if self.suction:
                cup = model.body(self._cup).id
                return self._ctrl > 0.0 and any(
                    model.geom_bodyid[mine] == cup
                    and model.body_weldid[model.geom_bodyid[other]] != 0  # 0: welded to the world
                    and model.body_rootid[model.geom_bodyid[other]] != arm_root
                    for contact in data.contact[: data.ncon]
                    for mine, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1))
                )
            gripper = {
                b for name in self._roots for b in robots.subtree(model, model.body(name).id)
            }
            pushes: dict[int, list[np.ndarray]] = {}
            for contact in data.contact[: data.ncon]:
                for mine, other, sign in (
                    (contact.geom1, contact.geom2, 1.0), (contact.geom2, contact.geom1, -1.0)
                ):
                    body = int(model.geom_bodyid[other])
                    if (
                        model.geom_bodyid[mine] in gripper
                        and body != 0
                        and model.body_rootid[body] != arm_root
                    ):
                        # the normal points from geom1 to geom2; this one points into `other`
                        pushes.setdefault(body, []).append(sign * contact.frame[:3])
            return any(
                float(np.dot(a, b)) < -0.5
                for normals in pushes.values()
                for a, b in itertools.combinations(normals, 2)
            )

        return self._sim.run(read)




def _look_at(position: Sequence[float], target: Sequence[float]) -> Quat:
    """MuJoCo camera quaternion (looks along -Z, +Y up) aimed at target."""
    forward = np.asarray(target, float) - np.asarray(position, float)
    norm = np.linalg.norm(forward)
    if norm == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    z_axis = -forward / norm
    x_axis = np.cross((0.0, 0.0, 1.0), z_axis)
    if np.linalg.norm(x_axis) < 1e-6:  # looking straight up or down
        x_axis = np.cross((0.0, 1.0, 0.0), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x_axis, y_axis, z_axis]).ravel())
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


# Viam's camera frame looks along +Z with +Y down; MuJoCo's looks along -Z
# with +Y up. A half turn about X maps one to the other.
_VIAM_TO_MUJOCO_CAMERA = Quaternion(0.0, 1.0, 0.0, 0.0)


def _mount(
    spec: mujoco.MjSpec, parent: str | None, position: Sequence[float], quat: Sequence[float]
) -> tuple[Any, np.ndarray, np.ndarray] | None:
    """Where something framed on `parent` goes in the spec: (body, pos, quat).
    The offset is relative to the parent's viam_frame site; with no parent it
    is on the world body. None when the parent isn't in the spec (yet)."""
    if not parent:
        return spec.worldbody, np.asarray(position, float), np.asarray(quat, float)
    site = spec.site(f"{parent}/{VIAM_FRAME_SITE}")
    if site is None:
        return None
    pos = np.zeros(3)
    mujoco.mju_rotVecQuat(pos, np.asarray(position, float), site.quat)
    pos += site.pos
    rot = np.zeros(4)
    mujoco.mju_mulQuat(rot, site.quat, np.asarray(quat, float))
    return site.parent, pos, rot


class CameraHandle:
    """A camera on the world, or, with a parent, on that component's Viam
    frame (an arm's end effector, a base). A mounted camera's position and
    orientation are relative to that frame, as the frame system reads them."""

    def __init__(self, sim: SimManager, name: str, attrs: dict[str, Any]) -> None:
        self._sim = sim
        self.name = name
        self.parent: str | None = attrs.get("parent")
        self.width = int(attrs.get("width", 640))
        self.height = int(attrs.get("height", 480))
        self.fov_deg = float(attrs.get("fov_deg", 60.0))
        # depth defaults to half the color size: raw depth is 2 bytes a pixel, uncompressed
        self.depth_width = int(attrs.get("depth_width", max(1, self.width // 2)))
        self.depth_height = int(attrs.get("depth_height", max(1, self.height // 2)))
        default_position = (0.0, 0.0, 0.0) if self.parent else (1.0, 1.0, 1.0)
        position = tuple(float(value) for value in attrs.get("position", default_position))
        if attrs.get("target") is not None:
            quat = _look_at(position, attrs["target"])
        elif attrs.get("orientation_wxyz") is not None or self.parent:
            viam = Quaternion(*[float(v) for v in attrs.get("orientation_wxyz", (1, 0, 0, 0))])
            mj = viam * _VIAM_TO_MUJOCO_CAMERA
            quat = (mj.w, mj.i, mj.j, mj.k)
        else:
            quat = _look_at(position, (0.0, 0.0, 0.0))

        def build(spec: mujoco.MjSpec) -> None:
            mount = _mount(spec, self.parent, position, quat)
            if mount is None:
                return  # parent not configured (yet); render() says so
            body, pos, rot = mount
            body.add_camera(name=name, pos=list(pos), quat=list(rot), fovy=self.fov_deg)
            spec.visual.global_.offwidth = max(
                spec.visual.global_.offwidth, self.width, self.depth_width
            )
            spec.visual.global_.offheight = max(
                spec.visual.global_.offheight, self.height, self.depth_height
            )

        sizes = ((self.width, self.height), (self.depth_width, self.depth_height))
        sim.set_contribution(
            name, Contribution(build=build, late=bool(self.parent), render_sizes=sizes)
        )

    def close(self) -> None:
        self._sim.remove_contribution(self.name)

    def get_rgb(self) -> np.ndarray:
        return self._sim.render(self.name, self.width, self.height)[0]

    def get_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        """RGB and depth in meters (0 = nothing in view) from one sim state."""
        rgb, depth = self._sim.render(
            self.name, self.width, self.height, depth_size=(self.depth_width, self.depth_height)
        )
        assert depth is not None
        return rgb, depth

    def intrinsics(self, width: int | None = None, height: int | None = None) -> dict[str, float]:
        """Pinhole intrinsics of the color image, or of an image of the given
        size from the same camera (the depth image)."""
        width, height = width or self.width, height or self.height
        focal = (height / 2.0) / math.tan(math.radians(self.fov_deg) / 2.0)
        return {
            "width_px": width,
            "height_px": height,
            "focal_x_px": focal,
            "focal_y_px": focal,
            "center_x_px": width / 2.0,
            "center_y_px": height / 2.0,
        }

    def point_cloud(self) -> tuple[np.ndarray, np.ndarray]:
        """Points in Viam's camera frame (+Z forward, +Y down), meters, with
        their RGB colors. Pixels with nothing in view are dropped."""
        rgb, depth = self.get_rgbd()
        k = self.intrinsics(self.depth_width, self.depth_height)
        rows, cols = np.nonzero(depth)
        z = depth[rows, cols]
        x = (cols + 0.5 - k["center_x_px"]) * z / k["focal_x_px"]
        y = (rows + 0.5 - k["center_y_px"]) * z / k["focal_y_px"]
        # the color pixel each depth pixel's ray passes through
        c = self.intrinsics()
        u = np.clip(c["center_x_px"] + x / z * c["focal_x_px"], 0, self.width - 1).astype(int)
        v = np.clip(c["center_y_px"] + y / z * c["focal_y_px"], 0, self.height - 1).astype(int)
        return np.column_stack([x, y, z]).astype(np.float32), rgb[v, u]


DEFAULT_LIDAR_RANGE_MIN_M = 0.15
DEFAULT_LIDAR_RANGE_MAX_M = 12.0
DEFAULT_LIDAR_RESOLUTION_DEG = 1.0
MAX_LIDAR_RAYS = 3600
LIDAR_RECAST_STEP_M = 1e-4  # a re-cast starts this far past the hit it skips


class LidarHandle:
    """A 2D scanning LIDAR: one ring of rays in the sensor frame's XY plane,
    ray 0 along +X, counterclockwise about +Z. A scan samples one sim
    instant. Mounted on the world or on a parent's viam_frame, like a camera;
    a mounted lidar never sees its parent's body or the bodies below it (a
    base's wheels and casters)."""

    def __init__(self, sim: SimManager, name: str, attrs: dict[str, Any]) -> None:
        self._sim = sim
        self.name = name
        self.parent: str | None = attrs.get("parent")
        self.range_min_m = float(attrs.get("range_min_m", DEFAULT_LIDAR_RANGE_MIN_M))
        self.range_max_m = float(attrs.get("range_max_m", DEFAULT_LIDAR_RANGE_MAX_M))
        resolution = float(attrs.get("angular_resolution_deg", DEFAULT_LIDAR_RESOLUTION_DEG))
        count = max(1, round(360.0 / resolution))  # evenly spaced around the full circle
        theta = 2.0 * math.pi * np.arange(count) / count
        self._rays = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(count)])
        default_position = (0.0, 0.0, 0.0) if self.parent else (0.0, 0.0, 0.2)
        position = [float(v) for v in attrs.get("position", default_position)]
        quat = [float(v) for v in attrs.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0))]

        def build(spec: mujoco.MjSpec) -> None:
            mount = _mount(spec, self.parent, position, quat)
            if mount is None:
                return  # parent not configured (yet); scan() says so
            body, pos, rot = mount
            body.add_site(name=name, pos=list(pos), quat=list(rot))

        sim.set_contribution(name, Contribution(build=build, late=bool(self.parent)))

    def close(self) -> None:
        self._sim.remove_contribution(self.name)

    def scan(self) -> np.ndarray:
        """Hit points in the sensor frame, meters, float32, z = 0."""

        def cast() -> np.ndarray:
            self._sim.require_world()
            model, data = self._sim.model, self._sim.data
            assert model is not None and data is not None
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.name)
            if site < 0:
                raise RuntimeError(
                    f"lidar {self.name!r} is not in the scene; configure its frame "
                    f"parent {self.parent!r}"
                )
            count = len(self._rays)
            directions = (data.site_xmat[site].reshape(3, 3) @ self._rays.T).T
            geomid = np.zeros(count, np.int32)
            dist = np.zeros(count)
            exclude = (int(model.site_bodyid[site]) or -1) if self.parent else -1  # body 0 is the world: excluding it would hide every static geom
            mujoco.mj_multiRay(
                model, data, data.site_xpos[site].copy(), directions.ravel().copy(),
                None, 1, exclude, geomid, dist, None, count, self.range_max_m,
            )
            if self.parent:
                # mj_multiRay excludes one body; re-cast past hits on the rest of the parent's subtree
                own = set(robots.subtree(model, int(model.site_bodyid[site])))
                origin = data.site_xpos[site]
                found = np.zeros(1, np.int32)
                for ray in np.flatnonzero(geomid >= 0):
                    total, geom = float(dist[ray]), int(geomid[ray])
                    while geom >= 0 and model.geom_bodyid[geom] in own and total <= self.range_max_m:
                        total += LIDAR_RECAST_STEP_M
                        step = mujoco.mj_ray(
                            model, data, origin + total * directions[ray], directions[ray],
                            None, 1, int(model.geom_bodyid[geom]), found,
                        )
                        geom = int(found[0])
                        total = total + step if geom >= 0 else -1.0
                    dist[ray] = total
            return dist

        dist = self._sim.run(cast)
        keep = (dist >= self.range_min_m) & (dist <= self.range_max_m)  # misses are -1
        return (self._rays[keep] * dist[keep, None]).astype(np.float32)


def encode_depth(depth_m: np.ndarray) -> bytes:
    """Viam's raw depth format (image/vnd.viam.dep): "DEPTHMAP", width and
    height as big-endian uint64, then big-endian uint16 millimeters."""
    height, width = depth_m.shape
    mm = np.clip(np.rint(depth_m * 1000.0), 0, 65535).astype(">u2")
    return b"DEPTHMAP" + width.to_bytes(8, "big") + height.to_bytes(8, "big") + mm.tobytes()


def encode_pcd(points: np.ndarray, colors: np.ndarray | None = None) -> bytes:
    """Binary PCD in meters, as rdk's pointcloud reader expects: packed rgb
    when colors are given, plain x y z otherwise."""
    count = len(points)
    if colors is None:
        fields = "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        body = np.asarray(points, "<f4").tobytes()
    else:
        fields = "FIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F I\nCOUNT 1 1 1 1\n"
        record = np.zeros(count, dtype=[("xyz", "<f4", 3), ("rgb", "<u4")])
        record["xyz"] = points
        c = colors.astype(np.uint32)
        record["rgb"] = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
        body = record.tobytes()
    header = (
        f"VERSION .7\n{fields}WIDTH {count}\nHEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {count}\nDATA binary\n"
    )
    return header.encode() + body
