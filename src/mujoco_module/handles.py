"""Engine-side handles that the Viam component models call.

Each handle registers a Contribution (what it adds to the scene) and, where it
drives actuators, a step callback. Step callbacks are the only writers of
data.ctrl. Handles store MuJoCo names, not ids, and resolve ids again whenever
the sim's generation changes (every rebuild).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .sim import KNOWN_ASSETS, Contribution, SimManager
from viam.spatialmath import Quaternion

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (w, x, y, z), MuJoCo's order

DEFAULT_MAX_JOINT_DEG_S = 180.0
DEFAULT_IS_MOVING_VELOCITY_THRESHOLD_RAD_S = 0.01


@dataclass(frozen=True)
class JointTrajectoryPoint:
    time_s: float
    positions: tuple[float, ...]


@dataclass
class _Trajectory:
    points: tuple[JointTrajectoryPoint, ...]
    start_s: float
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
        self._sim.run(lambda: self._sim.step_callbacks.pop(self._callback_key, None))
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
        if trajectory is not None:
            t = data.time - trajectory.start_s
            final = trajectory.points[-1]
            if t >= final.time_s:
                self._target = np.asarray(final.positions)
                data.ctrl[ids.ctrl] = self._target
                self._trajectory = None
                if not trajectory.future.done():
                    trajectory.future.set_result(None)
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
                return
            try:
                data, ids = self._now()
                arrived = bool(
                    np.all(np.abs(data.qpos[ids.qpos] - wanted) <= tolerance_rad)
                    and np.all(np.abs(data.qvel[ids.dof]) <= self._velocity_threshold)
                )
            except BaseException as exc:
                self._sim.step_callbacks.pop(key, None)
                future.set_exception(exc)
                return
            if arrived:
                self._sim.step_callbacks.pop(key, None)
                future.set_result(None)

        def attach() -> None:
            self._sim.step_callbacks[key] = check
            check(0.0)

        self._sim.run(attach)
        return future

    def start_trajectory(self, points: Sequence[JointTrajectoryPoint]) -> Future:
        """Follow timestamped points in sim time. The future resolves when the
        last point's time passes; the arm then holds that point."""
        if not points:
            raise ValueError("trajectory needs at least one point")
        future: Future = Future()

        def start() -> None:
            self._drop_trajectory()
            assert self._sim.data is not None
            self._trajectory = _Trajectory(tuple(points), self._sim.data.time, future)

        self._sim.run(start)
        return future

    def stop_trajectory(self) -> None:
        def stop() -> None:
            if self._drop_trajectory() is not None:
                self._hold_measured()

        self._sim.run(stop)

    def stop(self) -> None:
        def stop() -> None:
            self._drop_trajectory()
            self._hold_measured()

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
