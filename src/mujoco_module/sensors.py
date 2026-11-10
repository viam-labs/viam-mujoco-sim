"""An IMU and a wrist force-torque sensor, read from MuJoCo on the sim thread.

Both are exact unless noise is configured: a reading is truth plus a constant
bias plus Gaussian noise, from a generator seeded per sensor, so runs repeat.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from . import robots
from .handles import VIAM_FRAME_SITE, _mount
from .sim import Contribution, SimManager


class Noise:
    """Seeded Gaussian noise. Draws nothing while every std is 0, so exact
    readings stay exact."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def vector(self, truth: np.ndarray, std: float, bias: Sequence[float] = (0.0, 0.0, 0.0)) -> np.ndarray:
        out = np.asarray(truth, float) + np.asarray(bias, float)
        if std > 0:
            out = out + self._rng.normal(0.0, std, 3)
        return out

    def rotation(self, quat: np.ndarray, std_deg: float) -> np.ndarray:
        """quat (w, x, y, z) turned by a random axis-angle rotation whose angle
        is N(0, std_deg)."""
        if std_deg <= 0:
            return np.array(quat, float)
        axis = self._rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        turn = np.zeros(4)
        mujoco.mju_axisAngle2Quat(turn, axis, math.radians(self._rng.normal(0.0, std_deg)))
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, turn, np.asarray(quat, float))
        return out


@dataclass(frozen=True)
class ImuReading:
    orientation_wxyz: np.ndarray  # world frame
    angular_velocity_dps: np.ndarray  # sensor frame
    linear_acceleration_mps2: np.ndarray  # sensor frame, +g up at rest


class ImuHandle:
    """A site mounted like the lidar (world, or a base's or arm's viam_frame)
    with MuJoCo gyro and accelerometer sensors on it."""

    def __init__(self, sim: SimManager, name: str, attrs: dict[str, Any]) -> None:
        self._sim = sim
        self.name = name
        self.parent: str | None = attrs.get("parent")
        self.orientation_noise_deg = float(attrs.get("orientation_noise_deg", 0.0))
        self.gyro_noise_dps = float(attrs.get("gyro_noise_dps", 0.0))
        self.gyro_bias_dps = [float(v) for v in attrs.get("gyro_bias_dps", (0.0, 0.0, 0.0))]
        self.accel_noise_mps2 = float(attrs.get("accel_noise_mps2", 0.0))
        self.accel_bias_mps2 = [float(v) for v in attrs.get("accel_bias_mps2", (0.0, 0.0, 0.0))]
        self._noise = Noise(int(attrs.get("seed", 0)))
        self._generation = -1
        self._site = self._gyro = self._accel = -1
        position = [float(v) for v in attrs.get("position", (0.0, 0.0, 0.0))]
        quat = [float(v) for v in attrs.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0))]

        def build(spec: mujoco.MjSpec) -> bool | None:
            mount = _mount(spec, self.parent, position, quat)
            if mount is None:
                return False  # parent not configured (yet); read() says so
            body, pos, rot = mount
            body.add_site(name=name, pos=list(pos), quat=list(rot))
            for suffix, kind in (("gyro", mujoco.mjtSensor.mjSENS_GYRO), ("accel", mujoco.mjtSensor.mjSENS_ACCELEROMETER)):
                spec.add_sensor(name=f"{name}/{suffix}", type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=name)

        sim.set_contribution(name, Contribution(build=build))

    def close(self) -> None:
        self._sim.remove_contribution(self.name)

    def _resolve(self) -> None:
        """The site and the sensors' sensordata addresses, once per model."""
        model = self._sim.model
        assert model is not None
        if self._generation != self._sim.generation:
            self._generation = self._sim.generation
            self._site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.name)
            self._gyro = self._accel = -1
            if self._site >= 0:
                self._gyro = int(model.sensor(f"{self.name}/gyro").adr[0])
                self._accel = int(model.sensor(f"{self.name}/accel").adr[0])

    def read(self) -> ImuReading:
        def snapshot() -> ImuReading:
            self._sim.require_world()
            self._resolve()
            if self._site < 0:
                raise RuntimeError(
                    f"imu {self.name!r} is not in the scene; configure its frame parent {self.parent!r}"
                )
            data = self._sim.data
            assert data is not None
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, data.site_xmat[self._site])
            gyro = np.degrees(data.sensordata[self._gyro : self._gyro + 3])
            accel = data.sensordata[self._accel : self._accel + 3]
            return ImuReading(
                orientation_wxyz=self._noise.rotation(quat, self.orientation_noise_deg),
                angular_velocity_dps=self._noise.vector(gyro, self.gyro_noise_dps, self.gyro_bias_dps),
                linear_acceleration_mps2=self._noise.vector(accel, self.accel_noise_mps2, self.accel_bias_mps2),
            )

        return self._sim.run(snapshot)


DEFAULT_FT_FILTER_S = 0.02


@dataclass(frozen=True)
class _ToolIds:
    site: int  # the arm's viam_frame site: the reading's frame and torque origin
    roots: tuple[int, ...]  # bodies on the flange body that carry the tool
    tree: int  # their kinematic tree's root body: cfrc_int is about its subtree_com
    tool: frozenset[int]  # every body of the tool
    adhesion: tuple[tuple[int, int], ...]  # (actuator, body) of each suction actuator on the tool


class ForceTorqueHandle:
    """The wrench the tool mounted on an arm's flange (its gripper and
    whatever that holds or touches) exerts on the flange, in the arm's
    viam_frame site's axes and about its origin: a hanging tool reads its
    weight along +Z. Virtual: nothing is added to the scene. Each step sums
    MuJoCo's internal forces (cfrc_int) on the flange body's child bodies
    that hold none of the arm's joints, low-pass filtered with filter_s.
    A suction cup's adhesion is an actuator force, which cfrc_int leaves
    out: its pull on the tool is added back, split evenly over the cup's
    contacts as MuJoCo applies it."""

    def __init__(self, sim: SimManager, name: str, arm: str, attrs: dict[str, Any]) -> None:
        self._sim = sim
        self.name = name
        self.arm = arm
        self.filter_s = float(attrs.get("filter_s", DEFAULT_FT_FILTER_S))
        self.force_noise_n = float(attrs.get("force_noise_n", 0.0))
        self.torque_noise_nm = float(attrs.get("torque_noise_nm", 0.0))
        self._noise = Noise(int(attrs.get("seed", 0)))
        self._generation = -1
        self._ids: _ToolIds | None = None
        self._wrench: np.ndarray | None = None  # filtered [force, torque]
        self._tare = np.zeros(6)
        self._tared = False
        self._callback_key = f"force_torque:{name}"

        def init(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            self._wrench = None  # the filter restarts from the next raw value

        sim.set_contribution(name, Contribution(build=lambda spec: None, init=init))
        sim.run(lambda: sim.step_callbacks.__setitem__(self._callback_key, self._on_step))

    def close(self) -> None:
        self._sim.run(lambda: self._sim.step_callbacks.pop(self._callback_key, None))
        self._sim.remove_contribution(self.name)

    def _resolve(self) -> None:
        model = self._sim.model
        assert model is not None
        if self._generation == self._sim.generation:
            return
        self._generation = self._sim.generation
        self._wrench = None
        self._ids = None
        handle = self._sim.arms.get(self.arm)
        if handle is None:
            return
        prefix = f"{self.arm}/"
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + VIAM_FRAME_SITE)
        flange = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, prefix + handle.flange_site)
        if site < 0 or flange < 0:
            return
        joint_bodies = {int(model.jnt_bodyid[model.joint(name).id]) for name in handle._joint_names}
        body = int(model.site_bodyid[flange])
        roots = tuple(
            child for child in range(1, model.nbody)
            if model.body_parentid[child] == body
            and not joint_bodies & set(robots.subtree(model, child))
        )
        tool = frozenset(b for root in roots for b in robots.subtree(model, root))
        adhesion = tuple(
            (a, int(model.actuator_trnid[a, 0])) for a in range(model.nu)
            if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_BODY and int(model.actuator_trnid[a, 0]) in tool
        )
        self._ids = _ToolIds(site, roots, int(model.body_rootid[body]), tool, adhesion)

    def _raw(self) -> np.ndarray | None:
        """This instant's [force, torque] in the site frame, or None."""
        model, data = self._sim.model, self._sim.data
        assert model is not None and data is not None
        self._resolve()
        ids = self._ids
        if ids is None:
            return None
        mujoco.mj_rnePostConstraint(model, data)
        total = data.cfrc_int[list(ids.roots)].sum(axis=0) if ids.roots else np.zeros(6)
        # what the flange exerts on the tool, world axes, about the tree's subtree_com
        torque, force = total[:3].copy(), total[3:].copy()
        centre = data.subtree_com[ids.tree]
        if ids.adhesion and data.ncon:
            contact = data.contact
            geoms = contact.geom[: data.ncon]
            bodies = np.where(geoms >= 0, model.geom_bodyid[geoms], -1)  # -1: a flex contact's missing geom
            for actuator, body in ids.adhesion:
                pull = abs(float(data.actuator_force[actuator]))
                if pull == 0.0:
                    continue  # vacuum off
                mine = (bodies == body).any(axis=1)
                if not mine.any():
                    continue
                # the cup is pulled toward what it touches, split evenly; the frame's normal points from geom1 to geom2
                toward = np.where(bodies[mine, 0] == body, 1.0, -1.0)
                f = toward[:, None] * contact.frame[: data.ncon][mine, :3] * (pull / int(mine.sum()))
                force -= f.sum(axis=0)  # RNE counted none of it: the flange carries it instead
                torque -= np.cross(contact.pos[: data.ncon][mine] - centre, f).sum(axis=0)
        lever = data.site_xpos[ids.site] - centre
        torque = torque - np.cross(lever, force)
        rotation = data.site_xmat[ids.site].reshape(3, 3)
        return -np.concatenate([rotation.T @ force, rotation.T @ torque])  # the tool on the flange

    def _on_step(self, dt: float) -> None:
        raw = self._raw()
        if raw is None:
            return
        if self._wrench is None or self.filter_s <= 0:
            self._wrench = raw
        else:
            self._wrench = self._wrench + (raw - self._wrench) * (dt / (self.filter_s + dt))

    def _current(self) -> np.ndarray:
        self._sim.require_world()
        self._resolve()
        if self._ids is None:
            raise RuntimeError(
                f"force-torque sensor {self.name!r}: {self.arm!r} is not a mujoco-sim arm in the scene; "
                "set its frame parent to one"
            )
        if self._wrench is None:  # no step since the model was built
            self._wrench = self._raw()
        assert self._wrench is not None
        return self._wrench

    def readings(self) -> dict[str, Any]:
        def read() -> dict[str, Any]:
            wrench = self._current() - self._tare
            force = self._noise.vector(wrench[:3], self.force_noise_n)
            torque = self._noise.vector(wrench[3:], self.torque_noise_nm)
            return {
                "force": dict(zip("xyz", map(float, force))),
                "torque": dict(zip("xyz", map(float, torque))),
                "tared": self._tared,
            }

        return self._sim.run(read)

    def tare(self) -> None:
        def store() -> None:
            self._tare = self._current().copy()
            self._tared = True

        self._sim.run(store)

    def untare(self) -> None:
        def clear() -> None:
            self._tare = np.zeros(6)
            self._tared = False

        self._sim.run(clear)
