"""Engine-side handles that the Viam component models call.

Each handle registers a Contribution (what it adds to the scene) and, where it
drives actuators, a step callback. Only the sim thread writes data.ctrl (step
callbacks, init, holds and close). The arm resolves ids again when the sim's
generation changes (every rebuild); the gripper looks names up on each step.
"""

from __future__ import annotations

import contextlib
import itertools
import math
import random
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass
from typing import Any, Literal

import mujoco
import numpy as np

from . import bases, robots
from .sim import (
    BELT_CONAFFINITY,
    FREE_CONTYPE,
    GRIPPER_ASSETS,
    KNOWN_ASSETS,
    LOGGER,
    ROBOT_LIMB_ASSETS,
    PROP_CONTACT_PRIORITY,
    PROP_SOLREF,
    Contribution,
    DynamicProp,
    SimManager,
)
from viam.spatialmath import Quaternion

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (w, x, y, z), MuJoCo's order

DEFAULT_MAX_JOINT_DEG_S = 180.0
# ponytail: a fixed slide-joint speed cap; a max_joint_mm_s attribute when someone asks
SLIDE_MAX_M_S = 0.1
SLIDE_MOVING_M_S = 0.001  # a slide joint is still moving above this
# an arm in a robot file must be framed at its chain base within these
ROBOT_FRAME_TOLERANCE_M = 0.001
ROBOT_FRAME_TOLERANCE_DEG = 0.1
# Each component that others can be framed on adds "{name}/viam_frame": its
# frame in Viam's frame system (an arm's end effector, a base's origin).
VIAM_FRAME_SITE = "viam_frame"
# "{arm}/arm_frame": the arm's own Viam frame, where its kinematic base sits
# (base_offset is measured from it).
ARM_FRAME_SITE = "arm_frame"
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
    # TIAGo++'s base, made by pal_tiago_dual/make_tiago.py from mujoco_menagerie@370c439;
    # its omni wheels drive as a diff-drive. Speeds from the wheels' ctrlrange.
    "pal_tiago_base": {"mjcf": "pal_tiago_dual/tiago_base.xml", "drive": "ideal"},
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
    arm_frame: int  # the arm's arm_frame site


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
    the arm's attributes, over KNOWN_ASSETS[asset] when asset is set. With
    attrs["parent"] (a base) the arm rides that component's viam_frame, and
    position/orientation_wxyz are relative to it.

    With attrs["robot"] the arm is a limb of that robot's file
    (robot_file.RobotHandle): it attaches nothing, reads the robot's model,
    and drives `joints`; its frame (parent the robot, or another arm bound to
    it: a torso) must be its chain base's pose in the file, relative to that
    parent's end effector for an arm.
    File names (joints, actuators, bodies, flange_site) carry file_prefix
    (the robot's "<robot>/", else the arm's own "<arm>/"); the Viam-facing
    sites keep the arm's name ("<arm>/viam_frame", "<arm>/arm_frame")."""

    def __init__(self, sim: SimManager, name: str, attrs: dict[str, Any]) -> None:
        self.robot: str | None = str(attrs["robot"]) if attrs.get("robot") else None
        assets = ROBOT_LIMB_ASSETS if self.robot else KNOWN_ASSETS
        attrs = {**assets.get(str(attrs.get("asset")), {}), **attrs}
        label = f"arm {name}"
        self._sim = sim
        self.name = name
        self._prefix = f"{name}/"
        if self.robot:
            holder = sim.run(lambda: sim.robots.get(self.robot))
            if holder is None:
                raise RuntimeError(f"{label}: robot {self.robot!r} is not in the scene; configure the robot first")
            if attrs.get("collision") == "kinematics":
                raise ValueError(
                    f'{label}: collision "kinematics" is not supported for an arm in a robot file yet; '
                    "the file's own geoms are used"
                )
            for key in ("joints", "kinematics_url"):
                if not attrs.get(key):
                    raise ValueError(f"{label}: an arm in a robot file needs {key}")
            self.file_prefix = holder.prefix
            self.mjcf = holder.mjcf
            where = f"robot {self.robot} ({holder.mjcf})"
            urdf = False
        else:
            holder = None
            self.file_prefix = self._prefix
            self.mjcf = robots.resolve_path(str(attrs["mjcf"]), f"{label}: mjcf")
            where = str(self.mjcf)
            urdf = self.mjcf.suffix.lower() == ".urdf"
            robots.check_flange_attrs(attrs, urdf, label)
        mesh_dir = attrs.get("mesh_dir")
        spec = None if holder is not None else robots.load_mjcf(self.mjcf, mesh_dir)
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
        # The arm on its own (or its robot's file), as the file describes it.
        # Ids below index this model; names carry over into the scene with
        # file_prefix.
        self.model = holder.model if holder is not None else spec.compile()
        self.joints = robots.arm_joints(self.model, len(self.kinematics.joints), where, names)
        if urdf:
            robots.prepare_urdf_arm(spec, [self.model.joint(j).name for j in self.joints], flange, where)
            self.model = spec.compile()
        self.actuators = robots.arm_actuators(self.model, self.joints, where)
        self._joint_names = tuple(self.file_prefix + self.model.joint(j).name for j in self.joints)
        self._actuator_names = tuple(self.file_prefix + self.model.actuator(a).name for a in self.actuators)
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
        # The arm's Viam frame (its configured pose, relative to the parent's
        # viam_frame or the world) and, base_offset on from it, where the
        # MJCF's world goes.
        self._frame = robots.transform(
            [float(value) for value in attrs.get("position", (0.0, 0.0, 0.0))],
            [float(value) for value in attrs.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0))],
        )
        # Viam's end effector in the flange site's body: the flange site with flange_offset
        # (from the compiled model: the spec's quat ignores euler/axisangle/xyaxes/zaxis)
        self.end_offset = robots.transform(site.pos, site.quat) @ flange_offset
        # an arm in a robot file: its frame is its chain base's (the first joint's parent body)
        chain_base = 0
        if holder is not None:
            chain_base = int(self.model.body_parentid[self.model.jnt_bodyid[self.joints[0]]])
            self.chain_base = self.model.body(chain_base).name
            parent = attrs.get("parent")
            # framed on the robot (its root) or on another of its arms (that arm's end effector)
            above = sim.run(lambda: sim.arms.get(parent)) if parent != self.robot else None
            if parent and parent != self.robot and above is None:
                raise ValueError(f"{label}: arm {parent!r} is not in the scene; configure it first")
            if parent != self.robot and (above is None or above.robot != self.robot):
                raise ValueError(
                    f'{label}: an arm in robot {self.robot} needs frame.parent "{self.robot}" or another of its arms'
                )
            data = mujoco.MjData(self.model)
            mujoco.mj_kinematics(self.model, data)
            reference, parent_end = 0, np.eye(4)
            if above is not None:
                reference = int(self.model.site(above.flange_site).bodyid[0])
                parent_end = robots.transform(data.xpos[reference], data.xquat[reference]) @ above.end_offset
            if self.model.body_weldid[chain_base] != self.model.body_weldid[reference]:
                if above is None:
                    raise ValueError(
                        f"{label}: {self.chain_base} moves relative to robot {self.robot}; an arm's base "
                        "must be fixed to the robot (no joint between them)"
                    )
                raise ValueError(
                    f"{label}: {self.chain_base} moves relative to {self.model.body(reference).name} (where arm "
                    f"{parent}'s end effector is); an arm's base must be fixed to its parent arm's last link"
                )
            base = robots.transform(data.xpos[chain_base], data.xquat[chain_base])
            expected = np.linalg.inv(parent_end) @ base @ np.linalg.inv(base_offset)
            metres, degrees = robots.pose_error(self._frame, expected)
            if metres > ROBOT_FRAME_TOLERANCE_M or degrees > ROBOT_FRAME_TOLERANCE_DEG:
                from .robot_file import frame_error

                raise ValueError(frame_error(
                    expected, label, has_offset="base_offset" in attrs, parent=parent if above else None
                ))
        robots.check_frames(
            self.model, self.kinematics, self.joints, self.model.site(self.flange_site).id,
            self._home, base_offset, flange_offset, label, base_body=chain_base,
        )
        # per joint, SI: rad for hinges, m for slides (a torso lift)
        self.slides = self.model.jnt_type[self.joints] == mujoco.mjtJoint.mjJNT_SLIDE
        self._max_rad_s = np.where(
            self.slides, SLIDE_MAX_M_S, math.radians(float(attrs.get("max_joint_deg_s", DEFAULT_MAX_JOINT_DEG_S)))
        )
        acceleration = attrs.get("max_joint_deg_s2")
        self._max_rad_s2 = math.radians(float(acceleration)) if acceleration else None
        self._velocity_threshold = np.where(
            self.slides,
            SLIDE_MOVING_M_S,
            float(attrs.get("is_moving_velocity_threshold_rad_s", DEFAULT_IS_MOVING_VELOCITY_THRESHOLD_RAD_S)),
        )
        # None until init (new arm) or the first step (replaced arm) sets it
        self._target: np.ndarray | None = None
        self._trajectory: _Trajectory | None = None
        self._waits: dict[Future, str] = {}
        self._ids: _ArmIds | None = None
        self._ids_generation = -1
        self._callback_key = f"arm:{name}"
        self._parent: str | None = attrs.get("parent")
        excluded: list[tuple[str, str]] = []
        has_shapes = any(frame.geometry for frame in self.kinematics.frames.values())
        # an arm in a robot file uses the file's own geoms
        collision = attrs.get("collision") or ("kinematics" if has_shapes and holder is None else "mesh")
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
        robot_site = f"{self.robot}/{VIAM_FRAME_SITE}"
        arm_site = np.linalg.inv(base_offset)  # the arm's frame on its chain base

        def build_in_robot(spec: mujoco.MjSpec) -> bool | None:
            if spec.site(robot_site) is None:
                return False  # robot not in the scene; calls say so
            base = spec.body(self.file_prefix + self.chain_base)
            base.add_site(name=self._prefix + ARM_FRAME_SITE, pos=list(arm_site[:3, 3]), quat=list(robots.quat(arm_site)))
            flange = spec.site(self.file_prefix + self.flange_site)
            end = self.end_offset
            flange.parent.add_site(
                name=self._prefix + VIAM_FRAME_SITE, pos=list(end[:3, 3]), quat=list(robots.quat(end))
            )

        def build(spec: mujoco.MjSpec) -> bool | None:
            placed = _mount(spec, self._parent, self._frame[:3, 3], robots.quat(self._frame))
            if placed is None:
                return False  # parent not configured (yet); calls say so
            body, pos, quat = placed
            at_frame = robots.transform(pos, quat)  # the arm's Viam frame, in the parent's body
            at_mount = at_frame @ base_offset
            child = arm_spec.copy()  # the file isn't read again
            robots.apply_shapes(child, shapes, label)
            frame = body.add_frame(pos=list(at_mount[:3, 3]), quat=list(robots.quat(at_mount)))
            spec.attach(child, prefix=self.file_prefix, frame=frame)
            # the arm's Viam frame, which end_pose is measured from
            body.add_site(name=self._prefix + ARM_FRAME_SITE, pos=list(pos), quat=list(quat))
            # Viam's end-effector frame, which children of the arm's frame
            # (a wrist camera) are placed relative to.
            flange = spec.site(self.file_prefix + self.flange_site)
            end = self.end_offset
            flange.parent.add_site(
                name=self._prefix + VIAM_FRAME_SITE, pos=list(end[:3, 3]), quat=list(robots.quat(end))
            )
            for first, second in excluded:
                spec.add_exclude(bodyname1=self.file_prefix + first, bodyname2=self.file_prefix + second)

        def init(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            ids = self._resolve(model)
            data.qpos[ids.qpos] = self._home
            data.qvel[ids.dof] = 0.0
            data.ctrl[ids.ctrl] = self._home
            self._target = np.asarray(self._home)
            self._drop_trajectory()

        claims = [f"joint:{self.model.joint(j).name}" for j in self.joints] + [
            f"actuator:{self.model.actuator(a).name}" for a in self.actuators
        ]
        contribution = Contribution(build=build_in_robot if holder is not None else build, init=init)

        def add() -> None:
            # one sim-thread task, so two arms built at once can't both pass the
            # claim check; a failed claim or contribution leaves nothing behind
            if holder is not None:
                from .robot_file import check_claims, claim

                check_claims(sim, self.robot, claims, name)
            sim.set_contribution(name, contribution)  # inline on the sim thread
            if holder is not None:
                claim(sim, self.robot, claims, name)
            sim.step_callbacks[self._callback_key] = self._on_step
            sim.arms[name] = self

        sim.run(add)

    def close(self) -> None:
        def detach() -> None:
            self._sim.step_callbacks.pop(self._callback_key, None)
            if self._sim.arms.get(self.name) is self:
                del self._sim.arms[self.name]
            if self.robot is not None:
                from .robot_file import release_claims

                release_claims(self._sim, self.robot, self.name)
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
            arm_frame=model.site(self._prefix + ARM_FRAME_SITE).id,
        )

    def _in_scene(self) -> bool:
        # the arm's own site: a robot's joints exist while the arm's build still waits
        assert self._sim.model is not None
        return mujoco.mj_name2id(self._sim.model, mujoco.mjtObj.mjOBJ_SITE, self._prefix + VIAM_FRAME_SITE) >= 0

    def _absent(self) -> RuntimeError:
        if self.robot is not None:
            return RuntimeError(f"arm {self.name!r}: robot {self.robot!r} is not in the scene")
        return RuntimeError(f"arm {self.name!r}: parent {self._parent!r} has no viam_frame in the scene")

    @property
    def flange_site_name(self) -> str:
        """The flange site's name in the scene."""
        return self.file_prefix + self.flange_site

    @property
    def root_body_name(self) -> str:
        """The arm's root body's name in the scene (its kinematic tree's root)."""
        return self.file_prefix + self.root_body

    def _now(self) -> tuple[mujoco.MjData, _ArmIds]:
        sim = self._sim
        assert sim.model is not None and sim.data is not None
        if self._ids is None or self._ids_generation != sim.generation:
            if not self._in_scene():
                raise self._absent()
            self._ids = self._resolve(sim.model)
            self._ids_generation = sim.generation
        return sim.data, self._ids

    # -- control (sim thread) ---------------------------------------------

    def _on_step(self, dt: float) -> None:
        # ids are stale from the moment the model changes, so the name lookup
        # only runs after a rebuild (and every step while the arm is absent)
        if self._ids_generation != self._sim.generation and not self._in_scene():
            # parent not in the scene; init re-homes the arm when it's back
            trajectory, self._trajectory = self._trajectory, None
            if trajectory is not None:
                _fail(trajectory.future, self._absent())
            return
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
        return tuple(name.removeprefix(self.file_prefix) for name in self._joint_names)

    def get_joint_positions(self) -> list[float]:
        def read() -> list[float]:
            data, ids = self._now()
            return [float(value) for value in data.qpos[ids.qpos]]

        return self._sim.run(read)

    def get_commanded_positions(self) -> list[float]:
        """Where the servos are being driven now (on a trajectory, its current
        point): measured positions trail it."""

        def read() -> list[float]:
            data, ids = self._now()
            return [float(value) for value in data.ctrl[ids.ctrl]]

        return self._sim.run(read)

    def set_joint_targets(self, positions: Sequence[float]) -> None:
        targets = np.asarray([float(value) for value in positions])
        if targets.shape != (len(self._joint_names),):
            raise ValueError(f"expected {len(self._joint_names)} joint targets")

        def apply() -> None:
            self._now()  # raises when the arm isn't in the scene
            if self._drop_trajectory() is not None:
                self._hold_measured()
            self._target = targets

        self._sim.run(apply)

    def wait_for_joint_targets(
        self, targets: Sequence[float], tolerance_rad: float | np.ndarray
    ) -> Future:
        """Resolve once every joint is within tolerance (one value, or one per
        joint) of its target and slower than the is-moving threshold. Checked
        once per physics step on the sim thread; cancelling the future removes
        the check."""
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

    def extend_trajectory(self, points: Sequence[JointTrajectoryPoint]) -> Future:
        """Append points, timed from the end of what came before, to the running
        trajectory (same future), or start one from the commanded pose when none is
        running (the first batch of a stream, or after it ran dry)."""
        if not points:
            raise ValueError("trajectory needs at least one point")
        if any(b.time_s < a.time_s for a, b in itertools.pairwise(points)) or points[0].time_s < 0:
            raise ValueError("trajectory times must be non-negative and non-decreasing")
        running = self._sim.run(lambda: self._trajectory)
        if running is None:
            return self.start_trajectory(points)

        def extend() -> Future:
            trajectory = self._trajectory
            if trajectory is None:  # it finished between the check and now: start afresh
                return None
            end = trajectory.points[-1].time_s
            more = tuple(JointTrajectoryPoint(end + p.time_s, tuple(p.positions)) for p in points)
            self._trajectory = _Trajectory(trajectory.points + more, trajectory.start_s, trajectory.future)
            return trajectory.future

        future = self._sim.run(extend)
        return future if future is not None else self.start_trajectory(points)

    def stop_trajectory(self) -> None:
        def stop() -> None:
            if self._drop_trajectory() is not None:
                self._hold_measured()

        self._sim.run(stop)

    def stop(self) -> None:
        def stop() -> None:
            if not self._in_scene():
                return  # nothing to hold; Stop (session end, e-stop) must not fail
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
            "max_velocity_rad_s": self._max_rad_s.tolist(),  # m/s for a slide
        }
        if self._max_rad_s2 is not None:
            # ponytail: slides share the hinge value (metadata only, nothing enforces it); split it if anything starts to
            result["max_acceleration_rad_s2"] = [self._max_rad_s2] * count
        return result

    def get_end_pose(self) -> tuple[Vec3, Quat]:
        """Viam's end-effector frame (the flange with flange_offset) in the
        arm's frame (metres, w-first quaternion), from the live arm_frame
        site, so it holds while a base carries the arm."""

        def read() -> tuple[Vec3, Quat]:
            data, ids = self._now()
            rotation = data.site_xmat[ids.arm_frame].reshape(3, 3)
            position = rotation.T @ (data.site_xpos[ids.end_site] - data.site_xpos[ids.arm_frame])
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
        # an arm's own gripper drives a body below its last joint; another limb's
        # actuators don't count. An arm on its own file falls back to every
        # non-arm actuator when none maps to a body (a site or slider-crank drive).
        below: list[int] | None = robots.actuators_below(handle.model, handle.joints[-1], handle.actuators)
        if not below and handle.robot is None:
            below = None
        self._robot: str | None = handle.robot
        if mjcf is None:
            model, prefix = handle.model, handle.file_prefix
            where = f"arm {arm} ({f'robot {handle.robot}' if handle.robot else handle.mjcf})"
            named = attrs.get("actuator")
            actuator = robots.gripper_actuator(
                model, where, exclude=handle.actuators, name=named,
                # a named actuator must be this arm's only in a robot file (another limb's otherwise)
                candidates=below if named is None or handle.robot is not None else None,
            )
        else:
            pool = below if below is not None else [a for a in range(handle.model.nu) if a not in handle.actuators]
            own = [handle.model.actuator(a).name for a in pool]
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
        self._arm_root = handle.root_body_name
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
            flange = handle.flange_site_name

            def build(spec: mujoco.MjSpec) -> bool | None:
                site = spec.site(flange)
                if site is None:
                    return False  # arm not in the scene; the gripper comes back with it
                child = gripper_spec.copy()  # the file isn't read again
                # attach keeps the scene's options; take the gripper's contact
                # settings (the 2F-85's elliptic cones) where it sets them
                defaults = mujoco.MjSpec().option
                for key in ("cone", "impratio"):
                    if getattr(child.option, key) != getattr(defaults, key):
                        setattr(spec.option, key, getattr(child.option, key))
                spec.attach(child, prefix=prefix, site=site)

            sim.set_contribution(name, Contribution(build=build))
        claims = []
        if self._robot is not None and mjcf is None:
            # the robot's init then leaves the gripper's actuator alone (the gripper
            # drives it); the jaw joints keep the keyframe, so coupled fingers agree
            from .robot_file import claim

            claims = [f"actuator:{model.actuator(actuator).name}"]
            sim.run(lambda: claim(sim, self._robot, claims, name))
        self._claims = claims
        sim.run(lambda: sim.step_callbacks.__setitem__(self._callback_key, self._on_step))

    def close(self) -> None:
        def detach() -> None:
            self._sim.step_callbacks.pop(self._callback_key, None)
            if self._claims:
                from .robot_file import release_claims

                release_claims(self._sim, self._robot, self.name)
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
            if self._robot is not None:
                raise RuntimeError(f"gripper {self.name!r}: robot {self._robot!r} is not in the scene")
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

        def build(spec: mujoco.MjSpec) -> bool | None:
            mount = _mount(spec, self.parent, position, quat)
            if mount is None:
                return False  # parent not configured (yet); render() says so
            body, pos, rot = mount
            body.add_camera(name=name, pos=list(pos), quat=list(rot), fovy=self.fov_deg)
            spec.visual.global_.offwidth = max(
                spec.visual.global_.offwidth, self.width, self.depth_width
            )
            spec.visual.global_.offheight = max(
                spec.visual.global_.offheight, self.height, self.depth_height
            )

        sizes = ((self.width, self.height), (self.depth_width, self.depth_height))
        sim.set_contribution(name, Contribution(build=build, render_sizes=sizes))

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
MAX_LIDAR_RAYS = 3600  # per channel
MAX_LIDAR_CHANNELS = 128
MAX_LIDAR_TOTAL_RAYS = 65536  # channels x rays per channel; ~3.4 ms per 65,000-ray scan on a rover
# Geom group 5 holds what no sensor should see (parked parcels); renderers
# hide groups 3 to 5 by default, and rays get this mask.
HIDDEN_GROUP = 5
RAY_GROUPS = np.array([1, 1, 1, 1, 1, 0], np.uint8)


def subtree_geoms(model: mujoco.MjModel, root: int) -> np.ndarray:
    """The geoms on `root`'s body and every body below it."""
    bodies = robots.subtree(model, root)
    return np.flatnonzero(np.isin(model.geom_bodyid, bodies))


@contextlib.contextmanager
def hidden_geoms(model: mujoco.MjModel, geoms: np.ndarray):
    """Rays skip `geoms` inside the block: they move to HIDDEN_GROUP, which
    RAY_GROUPS leaves out, and get their groups back afterwards. A mounted
    sensor never sees its own parent this way, in one ray call. Only for the
    sim thread, which owns the model (and renders on the same thread)."""
    saved = model.geom_group[geoms].copy()
    model.geom_group[geoms] = HIDDEN_GROUP
    try:
        yield
    finally:
        model.geom_group[geoms] = saved


class LidarHandle:
    """A scanning LIDAR: `channels` rings of rays, ring elevations spread
    evenly over vertical_fov_deg (one ring: their midpoint; the default is a
    2D scan in the sensor frame's XY plane), ray 0 of each ring along +X,
    counterclockwise about +Z. A scan samples one sim instant. Mounted on the
    world or on a parent's viam_frame, like a camera; a mounted lidar never
    sees its parent's body or the bodies below it (a base's wheels and
    casters)."""

    def __init__(self, sim: SimManager, name: str, attrs: dict[str, Any]) -> None:
        self._sim = sim
        self.name = name
        self.parent: str | None = attrs.get("parent")
        self.range_min_m = float(attrs.get("range_min_m", DEFAULT_LIDAR_RANGE_MIN_M))
        self.range_max_m = float(attrs.get("range_max_m", DEFAULT_LIDAR_RANGE_MAX_M))
        resolution = float(attrs.get("angular_resolution_deg", DEFAULT_LIDAR_RESOLUTION_DEG))
        count = max(1, round(360.0 / resolution))  # evenly spaced around the full circle
        theta = 2.0 * math.pi * np.arange(count) / count
        channels = int(attrs.get("channels", 1))
        low, high = (float(v) for v in attrs.get("vertical_fov_deg", (0.0, 0.0)))
        elevations = np.radians(np.linspace(low, high, channels) if channels > 1 else [(low + high) / 2])
        el, az = np.meshgrid(elevations, theta, indexing="ij")  # rows are rings
        self._rays = np.column_stack(
            [(np.cos(el) * np.cos(az)).ravel(), (np.cos(el) * np.sin(az)).ravel(), np.sin(el).ravel()]
        )
        default_position = (0.0, 0.0, 0.0) if self.parent else (0.0, 0.0, 0.2)
        position = [float(v) for v in attrs.get("position", default_position)]
        quat = [float(v) for v in attrs.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0))]

        def build(spec: mujoco.MjSpec) -> bool | None:
            mount = _mount(spec, self.parent, position, quat)
            if mount is None:
                return False  # parent not configured (yet); scan() says so
            body, pos, rot = mount
            body.add_site(name=name, pos=list(pos), quat=list(rot))

        sim.set_contribution(name, Contribution(build=build))

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
            # the parent's subtree is hidden; body 0 is the world, whose subtree is everything
            own = subtree_geoms(model, int(model.site_bodyid[site])) if self.parent else np.zeros(0, int)
            with hidden_geoms(model, own):
                mujoco.mj_multiRay(
                    model, data, data.site_xpos[site].copy(), directions.ravel().copy(),
                    RAY_GROUPS, 1, -1, geomid, dist, None, count, self.range_max_m,
                )
            return dist

        dist = self._sim.run(cast)
        keep = (dist >= self.range_min_m) & (dist <= self.range_max_m)  # misses are -1
        return (self._rays[keep] * dist[keep, None]).astype(np.float32)


CONVEYOR_PERIOD_M = 0.05  # the belt shifts back by this much; it overhangs each end by up to this
CONVEYOR_ACCEL_MPS2 = 2.0  # speed changes ramp at this rate, so parcels aren't flung
CONVEYOR_MIN_MPS = 0.01  # GoFor's braking ramp's least speed
CONVEYOR_MOVING_MPS = 1e-3
CONVEYOR_BELT_KG = 2.0
CONVEYOR_KV = 2000.0  # N per m/s: the belt settles in ~1 ms and droops 5 % under a 3.5 kg pile held still
CONVEYOR_FORCE_N = 500.0
CONVEYOR_BELT_M = 0.01  # thickness
CONVEYOR_BED_M = 0.04  # the frame box under the belt
CONVEYOR_RAIL_M = 0.02  # side-rail thickness
CONVEYOR_LEG_M = 0.04  # leg width


@dataclass
class _ConveyorGoal:
    end: float  # belt travel, m, at which the move ends
    direction: float  # +1 or -1
    future: Future


@dataclass(frozen=True)
class _ConveyorIds:
    qpos: int
    dof: int
    ctrl: int


class ConveyorHandle:
    """A belt conveyor. The fixed frame body (bed, legs, rails) sits at the
    frame, the centre of the belt's top surface; the belt is a thin box on a
    slide joint along +Y, one period longer than the conveyor, driven by a
    velocity actuator. Whenever the slide passes a period it shifts back by
    one (no velocity change), so the belt always covers the conveyor and
    parcels ride on friction. The belt collides only with free bodies (see
    sim.FREE_CONTYPE)."""

    def __init__(self, sim: SimManager, name: str, attrs: dict[str, Any]) -> None:
        self._sim = sim
        self.name = name
        prefix = f"{name}/"
        self.length = float(attrs["length_m"])
        self.width = float(attrs["width_m"])
        self.diameter = float(attrs.get("roller_diameter_m", 0.05))
        self.max_speed = float(attrs.get("max_speed_mps", 0.5))
        self.rails = float(attrs.get("rails_m", 0.04))
        friction = float(attrs.get("friction", 1.0))
        self.color = color = [float(c) for c in attrs.get("color", (0.25, 0.27, 0.3))]
        position = [float(v) for v in attrs.get("position", (0.0, 0.0, 0.0))]
        quat = [float(v) for v in attrs.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0))]
        self.height = position[2]  # legs reach world z = 0 (for a level conveyor)
        self._target = 0.0  # commanded belt speed, m/s
        self._speed = 0.0  # the command ramped at CONVEYOR_ACCEL_MPS2
        self._wraps = 0.0  # travel taken out of the slide by the shifts, m
        self._zero = 0.0  # travel at position zero, m
        self._goal: _ConveyorGoal | None = None
        self._ids: _ConveyorIds | None = None
        self._ids_generation = -1
        self._callback_key = f"conveyor:{name}"
        half_l, half_w, period = self.length / 2, self.width / 2, CONVEYOR_PERIOD_M

        def build(spec: mujoco.MjSpec) -> None:
            box = mujoco.mjtGeom.mjGEOM_BOX
            frame = spec.worldbody.add_body(name=prefix + "frame", pos=position, quat=quat)
            frame.add_site(name=prefix + VIAM_FRAME_SITE)
            rgba = [*color, 1.0]
            under = CONVEYOR_BELT_M + CONVEYOR_BED_M
            frame.add_geom(
                name=prefix + "bed", type=box, size=[half_w, half_l, CONVEYOR_BED_M / 2],
                pos=[0, 0, -CONVEYOR_BELT_M - CONVEYOR_BED_M / 2], rgba=rgba,
            )
            leg = self.height - under
            if leg > 0:
                inset = CONVEYOR_LEG_M / 2
                for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (-1, 1), (1, 1))):
                    frame.add_geom(
                        name=f"{prefix}leg_{i}", type=box, size=[inset, inset, leg / 2],
                        pos=[sx * (half_w - inset), sy * (half_l - inset), -under - leg / 2], rgba=rgba,
                    )
            if self.rails > 0:
                tall = self.rails + CONVEYOR_BELT_M
                for i, side in enumerate((-1, 1)):
                    frame.add_geom(
                        name=f"{prefix}rail_{i}", type=box,
                        size=[CONVEYOR_RAIL_M / 2, half_l, tall / 2],
                        pos=[side * (half_w + CONVEYOR_RAIL_M / 2), 0, self.rails - tall / 2], rgba=rgba,
                    )
            belt = frame.add_body(name=prefix + "belt")
            belt.add_joint(name=prefix + "belt", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0])
            # at slide q in [0, period) the belt spans [-L/2 - period + q, L/2 + q]
            belt.add_geom(
                name=prefix + "belt", type=box,
                size=[half_w, half_l + period / 2, CONVEYOR_BELT_M / 2],
                pos=[0, -period / 2, -CONVEYOR_BELT_M / 2],
                mass=CONVEYOR_BELT_KG, rgba=[0.1, 0.1, 0.1, 1.0],
                friction=[friction, 0.005, 0.0001], priority=PROP_CONTACT_PRIORITY + 1,
                contype=0, conaffinity=BELT_CONAFFINITY,
            )
            actuator = spec.add_actuator(
                name=prefix + "belt", target=prefix + "belt", trntype=mujoco.mjtTrn.mjTRN_JOINT
            )
            actuator.set_to_velocity(kv=CONVEYOR_KV)
            actuator.forcelimited = True
            actuator.forcerange = [-CONVEYOR_FORCE_N, CONVEYOR_FORCE_N]

        def init(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            data.ctrl[self._resolve(model).ctrl] = 0.0
            self._target = self._speed = self._wraps = self._zero = 0.0
            self._drop_goal(RuntimeError(f"conveyor {name!r} was reset"))

        sim.set_contribution(name, Contribution(build=build, init=init))

        def register() -> None:
            sim.step_callbacks[self._callback_key] = self._on_step
            sim.conveyors[name] = self

        sim.run(register)

    def close(self) -> None:
        def detach() -> None:
            self._sim.step_callbacks.pop(self._callback_key, None)
            if self._sim.conveyors.get(self.name) is self:
                del self._sim.conveyors[self.name]
            self._drop_goal(RuntimeError(f"conveyor {self.name!r} was closed (removed or reconfigured)"))

        self._sim.run(detach)
        self._sim.remove_contribution(self.name)

    def _resolve(self, model: mujoco.MjModel) -> _ConveyorIds:
        joint = model.joint(f"{self.name}/belt")
        return _ConveyorIds(
            qpos=int(joint.qposadr[0]), dof=int(joint.dofadr[0]), ctrl=model.actuator(f"{self.name}/belt").id
        )

    def _now(self) -> tuple[mujoco.MjData, _ConveyorIds]:
        sim = self._sim
        assert sim.model is not None and sim.data is not None
        if self._ids is None or self._ids_generation != sim.generation:
            self._ids = self._resolve(sim.model)
            self._ids_generation = sim.generation
        return sim.data, self._ids

    def _travel(self, data: mujoco.MjData, ids: _ConveyorIds) -> float:
        return self._wraps + float(data.qpos[ids.qpos])

    def _on_step(self, dt: float) -> None:
        data, ids = self._now()
        q = data.qpos[ids.qpos]
        if q > CONVEYOR_PERIOD_M:
            data.qpos[ids.qpos] -= CONVEYOR_PERIOD_M
            self._wraps += CONVEYOR_PERIOD_M
        elif q < 0.0:
            data.qpos[ids.qpos] += CONVEYOR_PERIOD_M
            self._wraps -= CONVEYOR_PERIOD_M
        target = self._target
        goal = self._goal
        if goal is not None:
            remaining = (goal.end - self._travel(data, ids)) * goal.direction
            if goal.future.done():  # cancelled by the caller
                self._drop_goal()
                target = 0.0
            elif remaining <= 0.0:
                self._drop_goal()
                target = self._speed = 0.0
                try:
                    goal.future.set_result(None)
                except InvalidStateError:
                    pass
            else:
                braking = max(CONVEYOR_MIN_MPS, math.sqrt(2 * CONVEYOR_ACCEL_MPS2 * remaining))
                target = goal.direction * min(abs(target), braking)
        step = CONVEYOR_ACCEL_MPS2 * dt
        self._speed += float(np.clip(target - self._speed, -step, step))
        data.ctrl[ids.ctrl] = self._speed

    def _drop_goal(self, error: BaseException | None = None) -> None:
        """Clear the goal and, when there was one, the speed command. Sim thread only."""
        goal, self._goal = self._goal, None
        if goal is not None:
            self._target = 0.0
            if error is not None:
                _fail(goal.future, error)

    # -- public API ---------------------------------------------------------

    def rpm_to_mps(self, rpm: float) -> float:
        return rpm * math.pi * self.diameter / 60.0

    def set_speed(self, mps: float) -> None:
        """Run the belt at mps (clamped to max_speed_mps) until told otherwise."""
        if not math.isfinite(mps):
            raise ValueError("speed must be finite")
        speed = float(np.clip(mps, -self.max_speed, self.max_speed))

        def apply() -> None:
            self._drop_goal(RuntimeError(f"conveyor {self.name!r} move was superseded by a new command"))
            self._target = speed

        self._sim.run(apply)

    def stop(self) -> None:
        def apply() -> None:
            self._drop_goal(RuntimeError(f"conveyor {self.name!r} move was stopped"))
            self._target = 0.0

        self._sim.run(apply)

    def go_for(self, rpm: float, revolutions: float) -> Future:
        """Run until the belt has travelled `revolutions` roller turns, then
        stop. The sign of rpm x revolutions picks the direction. The future
        resolves on arrival and fails on stop, a new command, reset or close."""
        if not math.isfinite(rpm) or not math.isfinite(revolutions):
            raise ValueError("rpm and revolutions must be finite")
        if rpm == 0.0:
            raise ValueError("rpm must be non-zero")
        future: Future = Future()
        if revolutions == 0.0:
            future.set_result(None)
            return future
        direction = math.copysign(1.0, rpm * revolutions)
        speed = min(abs(self.rpm_to_mps(rpm)), self.max_speed)
        distance = abs(revolutions) * math.pi * self.diameter

        def start() -> None:
            self._drop_goal(RuntimeError(f"conveyor {self.name!r} move was superseded by a new command"))
            data, ids = self._now()
            self._goal = _ConveyorGoal(self._travel(data, ids) + direction * distance, direction, future)
            self._target = direction * speed

        self._sim.run(start)
        return future

    def position(self) -> float:
        """Belt travel since the last reset_zero_position, in roller revolutions."""

        def read() -> float:
            data, ids = self._now()
            return (self._travel(data, ids) - self._zero) / (math.pi * self.diameter)

        return self._sim.run(read)

    def reset_zero_position(self, offset: float) -> None:
        """As RDK's encoded motors: the position reads -offset afterwards."""
        if not math.isfinite(offset):
            raise ValueError("offset must be finite")

        def apply() -> None:
            data, ids = self._now()
            self._zero = self._travel(data, ids) + offset * math.pi * self.diameter

        self._sim.run(apply)

    def power(self) -> float:
        """The commanded speed as a fraction of max_speed_mps."""
        return self._sim.run(lambda: self._target / self.max_speed)

    def is_moving(self) -> bool:
        def read() -> bool:
            data, ids = self._now()
            return bool(self._target != 0.0 or abs(data.qvel[ids.dof]) > CONVEYOR_MOVING_MPS)

        return self._sim.run(read)


class PhotoEyeHandle:
    """A through-beam photo-eye: one ray per step from its frame along +X,
    mounted like the lidar (on the world or a parent's viam_frame, never
    seeing the parent's subtree). Counts rising edges of `blocked`."""

    def __init__(self, sim: SimManager, name: str, attrs: dict[str, Any]) -> None:
        self._sim = sim
        self.name = name
        self.parent: str | None = attrs.get("parent")
        self.range_m = float(attrs.get("range_m", 2.0))
        self.blocked = False
        self.distance_m = self.range_m
        self.count = 0
        self._site = -1
        self._own = np.zeros(0, int)  # the parent's subtree geoms, hidden from the beam
        self._generation = -1
        self._callback_key = f"photo_eye:{name}"
        position = [float(v) for v in attrs.get("position", (0.0, 0.0, 0.0))]
        quat = [float(v) for v in attrs.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0))]

        def build(spec: mujoco.MjSpec) -> bool | None:
            mount = _mount(spec, self.parent, position, quat)
            if mount is None:
                return False  # parent not configured (yet); readings() says so
            body, pos, rot = mount
            body.add_site(name=name, pos=list(pos), quat=list(rot))

        def init(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            self.blocked, self.distance_m, self.count = False, self.range_m, 0

        sim.set_contribution(name, Contribution(build=build, init=init))
        sim.run(lambda: sim.step_callbacks.__setitem__(self._callback_key, self._on_step))

    def close(self) -> None:
        self._sim.run(lambda: self._sim.step_callbacks.pop(self._callback_key, None))
        self._sim.remove_contribution(self.name)

    def _resolve(self) -> None:
        """The beam's site and its parent's bodies, once per model."""
        model = self._sim.model
        assert model is not None
        if self._generation != self._sim.generation:
            self._generation = self._sim.generation
            self._site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.name)
            parent = int(model.site_bodyid[self._site]) if self._site >= 0 and self.parent else 0
            self._own = subtree_geoms(model, parent) if parent else np.zeros(0, int)

    def _on_step(self, _dt: float) -> None:
        model, data = self._sim.model, self._sim.data
        assert model is not None and data is not None
        self._resolve()
        if self._site < 0:
            return
        origin = data.site_xpos[self._site].copy()
        direction = data.site_xmat[self._site].reshape(3, 3)[:, 0].copy()
        found = np.zeros(1, np.int32)
        with hidden_geoms(model, self._own):
            dist = mujoco.mj_ray(model, data, origin, direction, RAY_GROUPS, 1, -1, found)
        blocked = 0.0 <= dist <= self.range_m
        if blocked and not self.blocked:
            self.count += 1
        self.blocked = blocked
        self.distance_m = float(dist) if blocked else self.range_m

    def readings(self) -> dict[str, Any]:
        def read() -> dict[str, Any]:
            self._sim.require_world()
            self._resolve()
            if self._site < 0:
                why = f"configure its frame parent {self.parent!r}" if self.parent else "the world has not rebuilt yet"
                raise RuntimeError(f"photo-eye {self.name!r} is not in the scene; {why}")
            return {"blocked": self.blocked, "distance_m": self.distance_m, "count": self.count}

        return self._sim.run(read)

    def reset_count(self) -> None:
        self._sim.run(lambda: setattr(self, "count", 0))


SCALE_TAU_S = 0.1  # low-pass filter time constant
SCALE_STABLE_S = 0.3
SCALE_STABLE_KG = 0.005


class ScaleHandle:
    """A fixed platform box whose frame is the centre of its top face. Each
    step it weighs the normal contact forces on the platform (static geoms
    never touch it), low-pass filtered; stable when the filtered weight has
    moved less than 5 g over the last 0.3 s."""

    def __init__(self, sim: SimManager, name: str, attrs: dict[str, Any]) -> None:
        self._sim = sim
        self.name = name
        self.size = [float(v) for v in attrs["size_m"]]
        self.color = color = [float(c) for c in attrs.get("color", (0.6, 0.6, 0.62))]
        position = [float(v) for v in attrs.get("position", (0.0, 0.0, 0.0))]
        quat = [float(v) for v in attrs.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0))]
        self.weight_kg = 0.0  # filtered
        self.tare_kg = 0.0
        self.stable = False
        self._history: deque[tuple[float, float]] = deque()
        self._geom = -1
        self._generation = -1
        self._callback_key = f"scale:{name}"
        self._geom_name = f"{name}/platform"

        def build(spec: mujoco.MjSpec) -> None:
            body = spec.worldbody.add_body(name=self._geom_name, pos=position, quat=quat)
            body.add_geom(
                name=self._geom_name, type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[v / 2 for v in self.size], pos=[0, 0, -self.size[2] / 2], rgba=[*color, 1.0],
            )

        def init(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            self.weight_kg, self.stable = 0.0, False
            self._history.clear()

        sim.set_contribution(name, Contribution(build=build, init=init))
        sim.run(lambda: sim.step_callbacks.__setitem__(self._callback_key, self._on_step))

    def close(self) -> None:
        self._sim.run(lambda: self._sim.step_callbacks.pop(self._callback_key, None))
        self._sim.remove_contribution(self.name)

    def _on_step(self, dt: float) -> None:
        model, data = self._sim.model, self._sim.data
        assert model is not None and data is not None
        if self._generation != self._sim.generation:
            self._generation = self._sim.generation
            self._geom = model.geom(self._geom_name).id
        force = np.zeros(6)
        load = 0.0
        for i in range(data.ncon):
            contact = data.contact[i]
            if self._geom not in (contact.geom1, contact.geom2):
                continue
            mujoco.mj_contactForce(model, data, i, force)
            # the normal (frame row 0) points from geom1 to geom2; force[0] pushes them apart
            push = force[0] * contact.frame[2]
            load += push if contact.geom1 == self._geom else -push
        self.weight_kg += dt / SCALE_TAU_S * (load / GRAVITY - self.weight_kg)
        now = float(data.time)
        self._history.append((now, self.weight_kg))
        while self._history[0][0] < now - SCALE_STABLE_S - 1e-9:
            self._history.popleft()
        values = [value for _, value in self._history]
        self.stable = bool(  # numpy's bool serializes as 1
            now - self._history[0][0] >= SCALE_STABLE_S - 1e-6 and max(values) - min(values) < SCALE_STABLE_KG
        )

    def readings(self) -> dict[str, Any]:
        # to 0.1 g: the filter decays to denormals (1e-322) on an empty platform; + 0.0 drops a -0.0
        return self._sim.run(lambda: {
            "weight_kg": round(float(self.weight_kg - self.tare_kg), 4) + 0.0, "stable": self.stable,
        })

    def tare(self) -> float:
        def apply() -> float:
            self.tare_kg = self.weight_kg
            return self.tare_kg

        return self._sim.run(apply)


PARK_POSITION = (0.0, 0.0, -2.0)  # parked parcels wait here, collisions off, out of sight
PARKED_HALF_M = (0.05, 0.05, 0.05)
PARKED_KG = 0.1
DESPAWN_BELOW_M = -0.5
INFEED_GAP_M = 0.01  # between the belt's -Y end and the infeed spot
MAX_SPAWN_YAW_RAD = math.radians(10)


@dataclass(frozen=True)
class Parcel:
    cls: str
    half: tuple[float, float, float]  # m
    mass: float
    color: tuple[float, float, float]
    friction: float


@dataclass(frozen=True)
class ParcelClass:
    name: str
    weight: float
    size: tuple[tuple[float, float], ...]  # (low, high) full size per axis, m
    mass: tuple[float, float]
    color: tuple[tuple[float, ...], tuple[float, ...]]  # (min, max) rgb
    friction: float


def _numbers(raw: Any, count: int, label: str) -> tuple[float, ...]:
    if not (
        isinstance(raw, list)
        and len(raw) == count
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in raw)
    ):
        raise ValueError(f"{label} must be a list of {count} numbers")
    return tuple(float(v) for v in raw)


def _span(raw: Any, label: str) -> tuple[float, float]:
    low, high = _numbers(raw, 2, label)
    if not 0 < low <= high:
        raise ValueError(f"{label} must be [low, high] with 0 < low <= high")
    return low, high


def parcel_classes(label: str, raw: Any) -> tuple[ParcelClass, ...]:
    """A spawner's classes attribute; ValueError if invalid."""
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label}: classes must be a list of at least one parcel class")
    classes = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"]:
            raise ValueError(f"{label}: every class needs a name")
        where = f"{label}: class {item['name']}"
        size = item.get("size_m")
        if isinstance(size, list) and len(size) == 3 and all(isinstance(axis, list) for axis in size):
            spans = tuple(_span(axis, f"{where}: size_m") for axis in size)
        else:
            spans = (_span(size, f"{where}: size_m"),) * 3
        color = item.get("color", [0.6, 0.45, 0.3])
        if isinstance(color, dict):
            low, high = (_numbers(color.get(key), 3, f"{where}: color {key}") for key in ("min", "max"))
        else:
            low = high = _numbers(color, 3, f"{where}: color")
        (weight,) = _numbers([item.get("weight", 1)], 1, f"{where}: weight")
        (friction,) = _numbers([item.get("friction", 1)], 1, f"{where}: friction")
        if weight <= 0 or friction < 0:
            raise ValueError(f"{where}: weight must be positive and friction >= 0")
        classes.append(ParcelClass(
            item["name"], weight, spans, _span(item.get("mass_kg"), f"{where}: mass_kg"), (low, high), friction
        ))
    return tuple(classes)


def despawn_boxes(label: str, raw: Any) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """A spawner's despawn attribute as (center, half size) pairs, world m."""
    if not isinstance(raw, list):
        raise ValueError(f"{label}: despawn must be a list of boxes")
    boxes = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f'{label}: a despawn box is {{"center": [x, y, z], "size": [x, y, z]}}')
        center = _numbers(item.get("center"), 3, f"{label}: despawn center")
        size = _numbers(item.get("size"), 3, f"{label}: despawn size")
        if min(size) <= 0:
            raise ValueError(f"{label}: despawn size must be positive")
        boxes.append((np.array(center), np.array(size) / 2))
    return tuple(boxes)


_PARCEL_SERIALS = itertools.count(1)  # process-wide: world_state keys parcels on name and serial


@dataclass(frozen=True)
class _PoolIds:
    geom: np.ndarray
    body: np.ndarray
    qpos: np.ndarray
    dof: np.ndarray


class SpawnerHandle:
    """A pool of free box bodies, {name}/parcel_{i}. A parked parcel waits
    below the floor with collisions off, in the hidden geom group, frozen
    each step. Spawning resizes one in place (no rebuild): its size, bounds,
    collision masks, mass, inertia and colour, then mj_setConst. A rebuild
    keeps live parcels (build writes their shape into the spec; state carry
    restores their pose); reset parks them all and restarts the timer and
    the random stream. Every interval_s of sim time a parcel of a random
    class spawns at rest on the conveyor's infeed spot, unless one is
    there already; a live parcel that enters a despawn box or falls below
    z = -0.5 m returns to the pool."""

    def __init__(self, sim: SimManager, name: str, attrs: dict[str, Any]) -> None:
        self._sim = sim
        self.name = name
        label = f"spawner {name}"
        self.conveyor = str(attrs["conveyor"])
        self.classes = parcel_classes(label, attrs.get("classes"))
        self.interval_s = float(attrs.get("interval_s", 4.0))
        self.pool_size = int(attrs.get("pool_size", 12))
        self._seed = attrs.get("seed")
        self._start_paused = bool(attrs.get("start_paused", False))
        self._despawn = despawn_boxes(label, attrs.get("despawn", []))
        # the infeed spot holds the largest parcel: its footprint's half diagonal
        self._spot_radius = max(math.hypot(c.size[0][1], c.size[1][1]) / 2 for c in self.classes)
        self._live: dict[int, Parcel] = {}
        self._misfits: set[str] = set()  # classes refused (and logged) for not fitting the conveyor
        self._ids: _PoolIds | None = None
        self._ids_generation = -1
        self._callback_key = f"spawner:{name}"
        self._restart()

        def build(spec: mujoco.MjSpec) -> None:
            for i in range(self.pool_size):
                body_name = self.body_name(i)
                body = spec.worldbody.add_body(name=body_name, pos=list(PARK_POSITION))
                body.add_freejoint(name=body_name)
                parcel = self._live.get(i)
                body.add_geom(
                    name=body_name, type=mujoco.mjtGeom.mjGEOM_BOX, priority=PROP_CONTACT_PRIORITY,
                    solref=list(PROP_SOLREF), **self._geom_fields(parcel),
                )

        def init(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            self._park(model, data, range(self.pool_size))
            self._restart()

        sim.set_contribution(name, Contribution(build=build, init=init))
        sim.run(lambda: sim.step_callbacks.__setitem__(self._callback_key, self._on_step))

    def _restart(self) -> None:
        self._random = random.Random(self._seed)
        self.paused = self._start_paused
        self._next_s = 0.0  # the first parcel comes at once
        self.spawned = 0
        self.pool_exhausted = False
        self._pending: tuple[Parcel, float] | None = None  # drawn, waiting for the infeed

    def close(self) -> None:
        def detach() -> None:
            self._sim.step_callbacks.pop(self._callback_key, None)
            for i in self._live:
                self._sim.dynamic_props.pop(self.body_name(i), None)
            self._live.clear()

        self._sim.run(detach)
        self._sim.remove_contribution(self.name)

    def body_name(self, i: int) -> str:
        return f"{self.name}/parcel_{i}"

    @staticmethod
    def _geom_fields(parcel: Parcel | None) -> dict[str, Any]:
        if parcel is None:
            return {
                "size": list(PARKED_HALF_M), "mass": PARKED_KG, "contype": 0, "conaffinity": 0,
                "group": HIDDEN_GROUP, "rgba": [0.5, 0.5, 0.5, 1.0],
            }
        return {
            "size": list(parcel.half), "mass": parcel.mass, "contype": FREE_CONTYPE, "conaffinity": 1,
            "group": 0, "rgba": [*parcel.color, 1.0], "friction": [parcel.friction, 0.005, 0.0001],
        }

    def _resolve(self, model: mujoco.MjModel) -> _PoolIds:
        joints = [model.joint(self.body_name(i)) for i in range(self.pool_size)]
        return _PoolIds(
            geom=np.array([model.geom(self.body_name(i)).id for i in range(self.pool_size)]),
            body=np.array([model.body(self.body_name(i)).id for i in range(self.pool_size)]),
            qpos=np.array([joint.qposadr[0] for joint in joints]),
            dof=np.array([joint.dofadr[0] for joint in joints]),
        )

    def _now(self) -> _PoolIds:
        sim = self._sim
        assert sim.model is not None
        if self._ids is None or self._ids_generation != sim.generation:
            self._ids = self._resolve(sim.model)
            self._ids_generation = sim.generation
        return self._ids

    def _set_fields(self, model: mujoco.MjModel, ids: _PoolIds, i: int, parcel: Parcel | None) -> None:
        """What compiling the spec with this parcel would give, in place."""
        fields = self._geom_fields(parcel)
        geom, body = ids.geom[i], ids.body[i]
        half = np.asarray(fields["size"])
        model.geom_size[geom] = half
        model.geom_rbound[geom] = np.linalg.norm(half)
        model.geom_aabb[geom] = [0.0, 0.0, 0.0, *half]
        # the compiled body masks are what the broad phase checks
        model.geom_contype[geom] = model.body_contype[body] = fields["contype"]
        model.geom_conaffinity[geom] = model.body_conaffinity[body] = fields["conaffinity"]
        model.geom_group[geom] = fields["group"]
        model.geom_rgba[geom] = fields["rgba"]
        if parcel is not None:
            model.geom_friction[geom, 0] = parcel.friction
        mass = fields["mass"]
        hx, hy, hz = half
        model.body_mass[body] = mass
        model.body_inertia[body] = mass / 3 * np.array([hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy])

    @staticmethod
    def _settle(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """mj_setConst after changing masses (without it a 3 kg parcel goes
        unstable); it resets every qpos to qpos0, so keep them. It also
        recomputes bvh_aabb, which collisions read: load-bearing for a
        resized parcel's contacts, not just its mass."""
        qpos = data.qpos.copy()
        mujoco.mj_setConst(model, data)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

    def _park(self, model: mujoco.MjModel, data: mujoco.MjData, indices: Sequence[int]) -> None:
        ids = self._resolve(model) if model is not self._sim.model else self._now()
        for i in indices:
            self._live.pop(i, None)
            self._sim.dynamic_props.pop(self.body_name(i), None)
            self._set_fields(model, ids, i, None)
            data.qpos[ids.qpos[i] : ids.qpos[i] + 7] = [*PARK_POSITION, 1.0, 0.0, 0.0, 0.0]
            data.qvel[ids.dof[i] : ids.dof[i] + 6] = 0.0
        self._settle(model, data)

    def _on_step(self, dt: float) -> None:
        sim = self._sim
        assert sim.model is not None and sim.data is not None
        ids = self._now()
        for i in range(self.pool_size):
            if i not in self._live:
                sim.data.qpos[ids.qpos[i] : ids.qpos[i] + 7] = [*PARK_POSITION, 1.0, 0.0, 0.0, 0.0]
                sim.data.qvel[ids.dof[i] : ids.dof[i] + 6] = 0.0
        gone = [i for i in self._live if self._despawned(sim.data.xpos[ids.body[i]])]
        if gone:
            self._park(sim.model, sim.data, gone)
        if self.paused or self.conveyor not in sim.conveyors:
            return
        self._next_s -= dt
        if self._next_s <= 0.0 and self._spawn_next(force=False)[1] != "infeed occupied":
            self._next_s = self.interval_s  # spawned, or skipped for want of a free parcel

    def _despawned(self, position: np.ndarray) -> bool:
        return position[2] < DESPAWN_BELOW_M or any(
            np.all(np.abs(position - center) <= half) for center, half in self._despawn
        )

    def _draw(self) -> tuple[Parcel, float]:
        """A random parcel and its yaw."""
        rng = self._random
        cls = rng.choices(self.classes, weights=[c.weight for c in self.classes])[0]
        half = tuple(rng.uniform(low, high) / 2 for low, high in cls.size)
        mass = rng.uniform(*cls.mass)
        color = tuple(rng.uniform(low, high) for low, high in zip(*cls.color))
        yaw = rng.uniform(-MAX_SPAWN_YAW_RAD, MAX_SPAWN_YAW_RAD)
        return Parcel(cls.name, half, mass, color, cls.friction), yaw

    def _infeed(self) -> tuple[np.ndarray, np.ndarray, float]:
        """The conveyor frame's world position and rotation, and the infeed
        spot's y in that frame."""
        sim = self._sim
        conveyor = sim.conveyors.get(self.conveyor)
        if conveyor is None:
            raise RuntimeError(f"spawner {self.name!r}: conveyor {self.conveyor!r} is not in the scene")
        assert sim.data is not None
        site = sim.data.site(f"{self.conveyor}/{VIAM_FRAME_SITE}")
        spot_y = -conveyor.length / 2 + INFEED_GAP_M + self._spot_radius
        return site.xpos.copy(), site.xmat.reshape(3, 3).copy(), spot_y

    def _on_infeed(self, radius: float) -> int | None:
        """A live parcel on the belt whose footprint overlaps a footprint of
        `radius` (a half diagonal) on the infeed spot."""
        assert self._sim.data is not None
        origin, rotation, spot_y = self._infeed()
        ids = self._now()
        for i, parcel in self._live.items():
            x, y, z = rotation.T @ (self._sim.data.xpos[ids.body[i]] - origin)
            if z > 0.0 and math.hypot(x, y - spot_y) < radius + math.hypot(parcel.half[0], parcel.half[1]):
                return i
        return None

    def _fits(self, cls_name: str) -> bool:
        """Whether a class's largest parcel fits between the conveyor's rails
        and on its belt from the infeed spot; logs once per class if not."""
        conveyor = self._sim.conveyors[self.conveyor]
        cls = next(c for c in self.classes if c.name == cls_name)
        (_, size_x), (_, size_y), _ = cls.size
        _, _, spot_y = self._infeed()
        if size_x < conveyor.width and spot_y + math.hypot(size_x, size_y) / 2 <= conveyor.length / 2:
            return True
        if cls_name not in self._misfits:
            self._misfits.add(cls_name)
            LOGGER.error(
                "spawner %r: class %r (up to %g x %g m) does not fit conveyor %r (%g m wide, %g m long);"
                " not spawning it",
                self.name, cls_name, size_x, size_y, self.conveyor, conveyor.width, conveyor.length,
            )
        return False

    def _spawn_next(self, force: bool) -> tuple[str | None, str]:
        """Spawn the next parcel on the infeed spot: (its name, "") or (None,
        why not). force skips the occupancy check."""
        if len(self._live) == self.pool_size:
            self.pool_exhausted = True
            return None, "pool exhausted"
        if self._pending is None:
            self._pending = self._draw()
        parcel, yaw = self._pending
        if not self._fits(parcel.cls):
            self._pending = None
            return None, f"class {parcel.cls} does not fit the conveyor"
        if not force and self._on_infeed(math.hypot(parcel.half[0], parcel.half[1])) is not None:
            return None, "infeed occupied"
        origin, rotation, spot_y = self._infeed()
        position = origin + rotation @ [0.0, spot_y, parcel.half[2] + 0.0005]
        frame, quat = np.zeros(4), np.zeros(4)
        mujoco.mju_mat2Quat(frame, rotation.ravel())
        mujoco.mju_mulQuat(quat, frame, np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]))
        name = self._spawn(parcel, position, quat)
        self._pending = None
        self.spawned += 1
        self.pool_exhausted = False
        return name, ""

    # -- sim thread -----------------------------------------------------------

    def _spawn(self, parcel: Parcel, position: Sequence[float], quat: Sequence[float]) -> str | None:
        """Put a free pool parcel at rest at the pose; its body name, or None
        when every parcel is live."""
        sim = self._sim
        assert sim.model is not None and sim.data is not None
        free = [i for i in range(self.pool_size) if i not in self._live]
        if not free:
            return None
        i, ids = free[0], self._now()
        self._live[i] = parcel
        serial = next(_PARCEL_SERIALS)
        self._set_fields(sim.model, ids, i, parcel)
        sim.data.qpos[ids.qpos[i] : ids.qpos[i] + 7] = [*position, *quat]
        sim.data.qvel[ids.dof[i] : ids.dof[i] + 6] = 0.0
        self._settle(sim.model, sim.data)
        sim.dynamic_props[self.body_name(i)] = DynamicProp(serial, parcel.half, parcel.color)
        return self.body_name(i)

    # -- public API -------------------------------------------------------------

    def spawn_at(self, parcel: Parcel, position: Sequence[float], quat: Sequence[float]) -> str | None:
        return self._sim.run(lambda: self._spawn(parcel, position, quat))

    def spawn_now(self, force: bool = False) -> tuple[str | None, str]:
        """Spawn the next parcel now, whatever the timer says."""
        return self._sim.run(lambda: self._spawn_next(force))

    def set_paused(self, paused: bool) -> None:
        self._sim.run(lambda: setattr(self, "paused", paused))

    def take(self) -> bool:
        """Despawn the parcel on the infeed spot, if there is one."""

        def apply() -> bool:
            i = self._on_infeed(self._spot_radius)
            if i is not None:
                self._park(self._sim.model, self._sim.data, [i])
            return i is not None

        return self._sim.run(apply)

    def readings(self) -> dict[str, Any]:
        def read() -> dict[str, Any]:
            self._sim.require_world()
            present = self.conveyor in self._sim.conveyors and self._on_infeed(self._spot_radius) is not None
            return {
                "spawned": self.spawned,
                "active": len(self._live),
                "paused": self.paused,
                "pool_exhausted": self.pool_exhausted,
                "seconds_until_next": max(0.0, self._next_s),
                # viam:workcell-components box-detect's names
                "enabled": not self.paused,
                "box_present": present,
                "interval_seconds": self.interval_s,
            }

        return self._sim.run(read)

    def parcels(self) -> list[dict[str, Any]]:
        """Every live parcel: name, class, world pose, full size (m) and mass."""

        def read() -> list[dict[str, Any]]:
            assert self._sim.data is not None
            out = []
            for i, parcel in sorted(self._live.items()):
                body = self._sim.data.body(self.body_name(i))
                out.append({
                    "name": self.body_name(i),
                    "class": parcel.cls,
                    "pose": {
                        "position": [float(v) for v in body.xpos],
                        "orientation_wxyz": [float(v) for v in body.xquat],
                    },
                    "size": [2 * v for v in parcel.half],
                    "mass_kg": parcel.mass,
                })
            return out

        return self._sim.run(read)

    def clear(self) -> int:
        """Park every live parcel; how many there were."""

        def apply() -> int:
            count = len(self._live)
            if count:
                self._park(self._sim.model, self._sim.data, list(self._live))
            return count

        return self._sim.run(apply)


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
