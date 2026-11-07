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

from .sim import KNOWN_ASSETS, Contribution, SimManager
from viam.spatialmath import Quaternion

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (w, x, y, z), MuJoCo's order

DEFAULT_MAX_JOINT_DEG_S = 180.0
# Each component that others can be framed on adds "{name}/viam_frame": its
# frame in Viam's frame system (an arm's end effector, a base's origin).
VIAM_FRAME_SITE = "viam_frame"
DEFAULT_IS_MOVING_VELOCITY_THRESHOLD_RAD_S = 0.01

BASE_ASSETS: dict[str, dict[str, Any]] = {
    # Viam Rover 2, from Viam's public ViamRover2-2024-rpi5 fragment
    # (wheel_circumference_mm 381, width_mm 356, motors max_rpm 120).
    "viam_rover": {
        "wheel_circumference_m": 0.381,
        "width_m": 0.356,
        "max_linear_mps": 0.762,  # 120 rpm x 0.381 m
        "max_angular_dps": 245.0,  # 2 x 0.762 / 0.356 rad/s
        # approximate shell: x across, y forward, z up; bottom above the floor
        "chassis_size_m": (0.26, 0.30, 0.10),
        "chassis_bottom_m": 0.03,
        "mass_kg": 2.5,
    },
}
BASE_JOINTS = ("x", "y", "yaw")
BASE_TAU_S = 0.005  # velocity actuators settle on a new command in ~5 ms
BASE_MAX_LINEAR_ACCEL = 50.0  # m/s^2, caps actuator force (no runaway push into a wall)
BASE_MAX_YAW_ACCEL = 200.0  # rad/s^2
BASE_MOVING_MPS = 0.001
BASE_MOVING_RAD_S = math.radians(0.5)
DEFAULT_BASE_MOVE_MARGIN_S = 2.0


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
    deadline_s: float  # sim time
    future: Future


@dataclass(frozen=True)
class _ArmIds:
    qpos: np.ndarray
    dof: np.ndarray
    ctrl: np.ndarray
    base_body: int
    flange_site: int


def _interpolate(points: Sequence[JointTrajectoryPoint], t: float) -> np.ndarray:
    for before, after in itertools.pairwise(points):
        if t <= after.time_s:
            span = after.time_s - before.time_s
            alpha = (t - before.time_s) / span if span > 0 else 1.0
            a, b = np.asarray(before.positions), np.asarray(after.positions)
            return a + alpha * (b - a)
    return np.asarray(points[-1].positions)


class ArmHandle:
    def __init__(self, sim: SimManager, name: str, attrs: dict[str, Any]) -> None:
        meta = KNOWN_ASSETS[str(attrs["asset"])]
        self._sim = sim
        self.name = name
        self._meta = meta
        self._prefix = f"{name}/"
        self._joint_names = tuple(self._prefix + joint for joint in meta["joint_names"])
        self._home = tuple(float(value) for value in meta["home"])
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
        position = [float(value) for value in attrs.get("position", (0.0, 0.0, 0.0))]
        orientation = [
            float(value) for value in attrs.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0))
        ]
        mjcf = str(meta["mjcf"])

        def build(spec: mujoco.MjSpec) -> None:
            child = mujoco.MjSpec.from_file(mjcf)
            for actuator in child.actuators:
                if not actuator.name:  # Menagerie leaves the arm actuators unnamed
                    actuator.name = actuator.target
            frame = spec.worldbody.add_frame(pos=position, quat=orientation)
            spec.attach(child, prefix=self._prefix, frame=frame)
            # Viam's end-effector frame, which children of the arm's frame
            # (a wrist camera) are placed relative to.
            flange = spec.site(self._prefix + meta["flange_site"])
            end_quat = np.zeros(4)
            mujoco.mju_mulQuat(end_quat, flange.quat, np.asarray(meta["flange_offset_wxyz"], float))
            flange.parent.add_site(
                name=self._prefix + VIAM_FRAME_SITE, pos=list(flange.pos), quat=list(end_quat)
            )
            for first, second in meta.get("self_collision_excluded_pairs", ()):
                spec.add_exclude(bodyname1=self._prefix + first, bodyname2=self._prefix + second)

        def init(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            ids = self._resolve(model)
            data.qpos[ids.qpos] = self._home
            data.qvel[ids.dof] = 0.0
            data.ctrl[ids.ctrl] = self._home
            self._target = np.asarray(self._home)
            self._drop_trajectory()

        sim.set_contribution(name, Contribution(build=build, init=init))
        sim.run(lambda: sim.step_callbacks.__setitem__(self._callback_key, self._on_step))

    def close(self) -> None:
        def detach() -> None:
            self._sim.step_callbacks.pop(self._callback_key, None)
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
            ctrl=np.array([model.actuator(name).id for name in self._joint_names]),
            base_body=model.body(self._prefix + self._meta["base_body"]).id,
            flange_site=model.site(self._prefix + self._meta["flange_site"]).id,
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
        return tuple(self._meta["joint_names"])

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
        """Flange pose in the arm's base frame (metres, w-first quaternion)."""

        def read() -> tuple[Vec3, Quat]:
            data, ids = self._now()
            base_rotation = data.xmat[ids.base_body].reshape(3, 3)
            offset = data.site_xpos[ids.flange_site] - data.xpos[ids.base_body]
            position = base_rotation.T @ offset
            rotation = base_rotation.T @ data.site_xmat[ids.flange_site].reshape(3, 3)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, rotation.ravel())
            viam = Quaternion(*quat) * Quaternion(*self._meta["flange_offset_wxyz"])
            quat = (viam.w, viam.i, viam.j, viam.k)
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
    """A differential-drive rover: a chassis on spawn-frame x/y slides and a yaw
    hinge, driven by velocity actuators. Chassis +Y is forward, as in Viam's
    base frame. Wheels are visual only; motion is ideal (no slip)."""

    def __init__(self, sim: SimManager, name: str, attrs: dict[str, Any]) -> None:
        meta = BASE_ASSETS[str(attrs["asset"])]
        self._sim = sim
        self.name = name
        self.meta = meta
        self.max_linear_mps = float(attrs.get("max_linear_mps", meta["max_linear_mps"]))
        self.max_angular_rad_s = math.radians(
            float(attrs.get("max_angular_dps", meta["max_angular_dps"]))
        )
        self.move_margin_s = float(
            attrs.get("move_timeout_margin_sec", DEFAULT_BASE_MOVE_MARGIN_S)
        )
        self._v = 0.0  # latched forward speed, m/s
        self._w = 0.0  # latched yaw rate, rad/s
        self._goal: _BaseGoal | None = None
        self._ids: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._ids_generation = -1
        self._callback_key = f"base:{name}"

        sx, sy, sz = meta["chassis_size_m"]
        mass = float(meta["mass_kg"])
        self.chassis_size_m = (sx, sy, sz)
        self.chassis_center_m = (0.0, 0.0, meta["chassis_bottom_m"] + sz / 2.0)
        izz = mass * (sx * sx + sy * sy) / 12.0
        wheel_radius = meta["wheel_circumference_m"] / (2.0 * math.pi)
        position = [float(v) for v in attrs.get("position", (0.0, 0.0, 0.0))]
        orientation = [float(v) for v in attrs.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0))]
        center = list(self.chassis_center_m)

        def build(spec: mujoco.MjSpec) -> None:
            frame = spec.worldbody.add_frame(pos=position, quat=orientation)
            body = frame.add_body(name=f"{name}/chassis")
            body.explicitinertial = True
            body.mass = mass
            body.ipos = center
            body.inertia = [
                mass * (sy * sy + sz * sz) / 12.0,
                mass * (sx * sx + sz * sz) / 12.0,
                izz,
            ]
            body.add_joint(name=f"{name}/x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0])
            body.add_joint(name=f"{name}/y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])
            body.add_joint(name=f"{name}/yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
            body.add_geom(
                name=f"{name}/chassis",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[sx / 2, sy / 2, sz / 2],
                pos=center,
                density=0,
                rgba=[0.15, 0.15, 0.17, 1],
            )
            # On the chassis body so a rover-mounted lidar's body exclusion skips them.
            for side in (-1, 1):
                body.add_geom(
                    name=f"{name}/wheel_{'left' if side < 0 else 'right'}",
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[wheel_radius, 0.0125, 0],
                    pos=[side * meta["width_m"] / 2, 0, wheel_radius],
                    quat=[math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0],  # axle along X
                    density=0,
                    contype=0,
                    conaffinity=0,
                    rgba=[0.05, 0.05, 0.05, 1],
                )
            body.add_site(name=f"{name}/{VIAM_FRAME_SITE}", pos=[0, 0, 0])
            limits = (
                (self.max_linear_mps, mass / BASE_TAU_S, mass * BASE_MAX_LINEAR_ACCEL),
                (self.max_linear_mps, mass / BASE_TAU_S, mass * BASE_MAX_LINEAR_ACCEL),
                (self.max_angular_rad_s, izz / BASE_TAU_S, izz * BASE_MAX_YAW_ACCEL),
            )
            for joint, (speed, gain, cap) in zip(BASE_JOINTS, limits):
                actuator = spec.add_actuator(
                    name=f"{name}/{joint}",
                    target=f"{name}/{joint}",
                    trntype=mujoco.mjtTrn.mjTRN_JOINT,
                )
                actuator.set_to_velocity(kv=gain)
                actuator.ctrllimited = True
                actuator.ctrlrange = [-1.1 * speed, 1.1 * speed]
                actuator.forcelimited = True
                actuator.forcerange = [-cap, cap]

        def init(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            qpos, dof, ctrl = self._resolve(model)
            data.qpos[qpos] = 0.0
            data.qvel[dof] = 0.0
            data.ctrl[ctrl] = 0.0
            self._v = self._w = 0.0
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
                data, (_q, _d, ctrl) = self._now()
                data.ctrl[ctrl] = 0.0

        self._sim.run(detach)
        self._sim.remove_contribution(self.name)

    # -- ids -------------------------------------------------------------

    def _resolve(self, model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        joints = [model.joint(f"{self.name}/{joint}") for joint in BASE_JOINTS]
        return (
            np.array([joint.qposadr[0] for joint in joints]),
            np.array([joint.dofadr[0] for joint in joints]),
            np.array([model.actuator(f"{self.name}/{joint}").id for joint in BASE_JOINTS]),
        )

    def _now(self) -> tuple[mujoco.MjData, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        sim = self._sim
        assert sim.model is not None and sim.data is not None
        if self._ids is None or self._ids_generation != sim.generation:
            self._ids = self._resolve(sim.model)
            self._ids_generation = sim.generation
        return sim.data, self._ids

    # -- control (sim thread) ---------------------------------------------

    def _on_step(self, _dt: float) -> None:
        data, (qpos, _dof, ctrl) = self._now()
        x, y, yaw = (float(value) for value in data.qpos[qpos])
        goal = self._goal
        if goal is not None:
            try:
                if goal.future.done():  # cancelled by the caller
                    self._drop_goal()
                else:
                    progress = self._progress(goal, x, y, yaw)
                    if progress * math.copysign(1.0, goal.target) >= abs(goal.target):
                        self._drop_goal()
                        try:
                            goal.future.set_result(None)
                        except InvalidStateError:
                            pass
                    elif data.time >= goal.deadline_s:
                        self._drop_goal(RuntimeError(self._stalled(goal, progress)))
            except Exception as exc:
                # halt, fail the move, and keep the callback registered
                self._drop_goal(exc)
        data.ctrl[ctrl] = (-self._v * math.sin(yaw), self._v * math.cos(yaw), self._w)

    @staticmethod
    def _progress(goal: _BaseGoal, x: float, y: float, yaw: float) -> float:
        if goal.kind == "spin":
            return yaw - goal.start_yaw  # hinge yaw is unwrapped
        heading = np.array([-math.sin(goal.start_yaw), math.cos(goal.start_yaw)])
        return float(np.dot(np.array([x, y]) - goal.start_xy, heading))

    def _stalled(self, goal: _BaseGoal, progress: float) -> str:
        if goal.kind == "spin":
            done, wanted, unit = math.degrees(progress), math.degrees(goal.target), "deg"
        else:
            done, wanted, unit = 1000 * progress, 1000 * goal.target, "mm"
        return (
            f"base {self.name!r} stalled: moved {done:.0f} of {wanted:.0f} {unit} "
            "before the timeout (blocked?)"
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
        and fails when the move takes longer, in sim time, than
        distance / speed + move_timeout_margin_sec (stalled), or on stop,
        a new command, reset or close."""
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
            data, (qpos, _dof, _ctrl) = self._now()
            x, y, yaw = (float(value) for value in data.qpos[qpos])
            self._goal = _BaseGoal(
                kind=kind,
                target=target,
                start_xy=np.array([x, y]),
                start_yaw=yaw,
                deadline_s=data.time + abs(target) / abs(rate) + self.move_margin_s,
                future=future,
            )
            if kind == "spin":
                self._v, self._w = 0.0, rate
            else:
                self._v, self._w = rate, 0.0

        self._sim.run(start)
        return future

    def is_moving(self) -> bool:
        def read() -> bool:
            data, (_qpos, dof, _ctrl) = self._now()
            vx, vy, wz = (float(value) for value in data.qvel[dof])
            return bool(
                self._v != 0.0
                or self._w != 0.0
                or math.hypot(vx, vy) > BASE_MOVING_MPS
                or abs(wz) > BASE_MOVING_RAD_S
            )

        return self._sim.run(read)


class GripperHandle:
    """The Lite6 gripper already attached with its arm: a force motor on the
    left finger, the right finger mirrored by an equality constraint. Positive
    force closes (q -> 0); negative opens (q -> -0.0081)."""

    OPEN_M = -0.0081
    CLOSED_M = 0.0
    MOVING_M_S = 1e-3

    def __init__(
        self, sim: SimManager, name: str, arm: str, grip_force_n: float
    ) -> None:
        self._sim = sim
        self.name = name
        prefix = f"{arm}/"
        self._actuator = prefix + "gripper"
        self._joint = prefix + "gripper_left_finger"
        self._tips = (prefix + "gripper_left_finger", prefix + "gripper_right_finger")
        self._arm_base = prefix + "link_base"
        self._force = float(grip_force_n)
        self._command = -1.0  # +1 close, -1 open, 0 no force
        self._callback_key = f"gripper:{name}"

        def attach() -> None:
            sim.require_world()
            assert sim.model is not None
            if mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self._actuator) < 0:
                raise RuntimeError(
                    f"arm {arm!r} has no Lite6 gripper; configure the arm first"
                )
            sim.step_callbacks[self._callback_key] = self._on_step

        sim.run(attach)

    def close(self) -> None:
        def detach() -> None:
            self._sim.step_callbacks.pop(self._callback_key, None)
            model, data = self._sim.model, self._sim.data
            if model is not None and data is not None:
                actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self._actuator)
                if actuator >= 0:
                    data.ctrl[actuator] = 0.0

        self._sim.run(detach)

    def _on_step(self, _dt: float) -> None:
        model, data = self._sim.model, self._sim.data
        assert model is not None and data is not None
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, self._actuator)
        if actuator < 0:
            return  # arm absent; resume when it is re-added
        data.ctrl[actuator] = self._command * self._force

    def _require_joint(self) -> None:
        model = self._sim.model
        if model is None or mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self._joint) < 0:
            raise RuntimeError(f"the arm for gripper {self.name!r} is not in the scene")

    def _set(self, command: float) -> None:
        def apply() -> None:
            self._command = command

        self._sim.run(apply)

    def open_jaws(self) -> None:
        self._set(-1.0)

    def close_jaws(self) -> None:
        self._set(1.0)

    def stop(self) -> None:
        self._set(0.0)

    def jaw_m(self) -> float:
        def read() -> float:
            self._require_joint()
            assert self._sim.model is not None and self._sim.data is not None
            joint = self._sim.model.joint(self._joint)
            return float(self._sim.data.qpos[joint.qposadr[0]])

        return self._sim.run(read)

    def is_moving(self) -> bool:
        def read() -> bool:
            self._require_joint()
            assert self._sim.model is not None and self._sim.data is not None
            joint = self._sim.model.joint(self._joint)
            return bool(abs(self._sim.data.qvel[joint.dofadr[0]]) > self.MOVING_M_S)

        return self._sim.run(read)

    def firm_grasp(self) -> bool:
        """Both fingertips touch the same body, which is neither the world nor
        part of the arm."""

        def read() -> bool:
            self._require_joint()
            model, data = self._sim.model, self._sim.data
            assert model is not None and data is not None
            left, right = (model.geom(tip).id for tip in self._tips)
            arm_root = model.body_rootid[model.body(self._arm_base).id]
            touching: dict[int, set[int]] = {left: set(), right: set()}
            for contact in data.contact[: data.ncon]:
                for tip, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                    if tip in touching:
                        body = int(model.geom_bodyid[other])
                        if body != 0 and model.body_rootid[body] != arm_root:
                            touching[tip].add(body)
            return bool(touching[left] & touching[right])

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


class LidarHandle:
    """A 2D scanning LIDAR: one ring of rays in the sensor frame's XY plane,
    ray 0 along +X, counterclockwise about +Z. A scan samples one sim
    instant. Mounted on the world or on a parent's viam_frame, like a camera;
    a mounted lidar never sees its parent's body."""

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
