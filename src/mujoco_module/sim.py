"""The singleton that owns MuJoCo.

One thread (the process main thread, see main.py) owns the MjSpec, MjModel,
MjData and every renderer. The Viam module server runs on a side thread and
hands work to the sim thread through run(). MuJoCo has no thread affinity of
its own, but MjData is not thread-safe and GL contexts belong to the thread
that created them, so a single owner needs no locks.

The scene is rebuilt from scratch on every structural change: the world's
floor, light and props, then each component's contribution. State carries
across a rebuild by joint and actuator name.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from viam.logging import getLogger

from .meshes import MeshSource, Piece, add_pieces, parse_mesh
from .shapes import GEOM_TYPES, mj_size, prop_shape

LOGGER = getLogger("viam-mujoco-sim")


# Bundled arms: default attributes for `asset` (config attributes win).
# Paths are relative to assets/.
KNOWN_ASSETS: dict[str, dict[str, Any]] = {
    "lite6": {
        "mjcf": "ufactory_lite6/lite6_gripper_wide.xml",
        # vendored from viam-modules/viam-ufactory-xarm@73551eac (arm/lite6_kinematics.json)
        "kinematics_url": "ufactory_lite6/lite6_kinematics.json",
        # Viam's gripper_mount frame is link6 turned half a turn about the
        # tool axis. Symmetric for a parallel gripper, but GetEndPosition
        # reports in Viam's convention.
        "flange_offset": {"orientation": {"o_x": 0, "o_y": 0, "o_z": 1, "theta": 180}},
    },
    "ur5e": {
        # mujoco_menagerie@98a70c56 universal_robots_ur5e; its base body turns
        # half a turn, so its world already matches Viam's base frame
        "mjcf": "universal_robots_ur5e/ur5e.xml",
        # vendored from viam-modules/universal-robots@3baa567d (src/kinematics/ur5e.json)
        "kinematics_url": "universal_robots_ur5e/ur5e_kinematics.json",
    },
}
# Bundled grippers: default attributes for the gripper's `asset`. No mjcf
# means the arm MJCF's own gripper.
GRIPPER_ASSETS: dict[str, dict[str, Any]] = {
    "lite6_gripper": {},
    # mujoco_menagerie@98a70c56 robotiq_2f85
    "robotiq_2f85": {"mjcf": "robotiq_2f85/2f85.xml"},
    # written here; mesh from ValerioMa/robotiq_tcp@45891ab, decimated
    "robotiq_epick": {"mjcf": "robotiq_epick/epick.xml"},
}

MAX_SUBSTEPS = 50
SLIP_THRESHOLD_S = 0.1
RTF_WINDOW_S = 2.0
# Props win contact-parameter mixing and use a stiffer solref than MuJoCo's
# default. Without this the Menagerie fingertips' solimp averages with the
# prop's into a soft contact and the jaws sink ~9 mm into a block.
PROP_CONTACT_PRIORITY = 1
PROP_SOLREF = (0.004, 1.0)

_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_BALL = int(mujoco.mjtJoint.mjJNT_BALL)
_QPOS_WIDTH = {_FREE: 7, _BALL: 4}
_DOF_WIDTH = {_FREE: 6, _BALL: 3}


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _positive(value: Any, label: str) -> float:
    number = _finite(value, label)
    if number <= 0.0:
        raise ValueError(f"{label} must be positive")
    return number


def _vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} must be a list of {size} numbers")
    return tuple(_finite(item, label) for item in value)


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false, got {value!r}")
    return value


def _unit_quaternion(value: Any, label: str) -> tuple[float, float, float, float]:
    quat = _vector(value, 4, label)
    norm = math.sqrt(sum(item * item for item in quat))
    if norm == 0.0:
        raise ValueError(f"{label} must not be all zeros")
    w, x, y, z = (item / norm for item in quat)
    return (w, x, y, z)


@dataclass(frozen=True)
class PropSpec:
    name: str
    shape: str  # "box", "sphere", "cylinder" or "capsule"
    geom_size: tuple[float, ...]  # MuJoCo size, meters (see shapes.py)
    position: tuple[float, ...]
    orientation_wxyz: tuple[float, float, float, float]
    mass: float
    color: tuple[float, ...]
    fixed: bool
    friction: float
    mesh: MeshSource | None = None  # set for type "mesh"; shape is "mesh", geom_size ()

    @classmethod
    def parse(cls, raw: Any) -> "PropSpec":
        """Isaac's prop format plus sphere, cylinder and capsule types. dynamic_friction is accepted so
        Isaac configs carry over; MuJoCo has one sliding coefficient."""
        if not isinstance(raw, Mapping) or not raw.get("name"):
            raise ValueError(f"every prop needs a name: {raw!r}")
        name = str(raw["name"])
        size = _positive(raw.get("size", 0.05), f"prop {name}: size")
        kind = str(raw.get("type", "cube"))
        mesh = None
        if kind == "mesh":
            if "mesh" not in raw:
                raise ValueError(f"prop {name}: a mesh prop needs mesh, a file path or https URL")
            mesh = parse_mesh(
                f"prop {name}", raw["mesh"], raw.get("scale", 1), raw.get("threshold", 0.05)
            )
            shape, geom_size = "mesh", ()
        else:
            for key in ("mesh", "threshold"):
                if key in raw:
                    raise ValueError(f'prop {name}: {key} only applies to type "mesh"')
            scale = _vector(raw.get("scale", [1, 1, 1]), 3, f"prop {name}: scale")
            if any(item <= 0.0 for item in scale):
                raise ValueError(f"prop {name}: scale must be positive")
            shape, geom_size = prop_shape(name, kind, size, scale)
        friction = _finite(raw.get("static_friction", 1.0), f"prop {name}: static_friction")
        if friction < 0.0:
            raise ValueError(f"prop {name}: static_friction must be non-negative")
        if "dynamic_friction" in raw and (
            _finite(raw["dynamic_friction"], f"prop {name}: dynamic_friction") < 0.0
        ):
            raise ValueError(f"prop {name}: dynamic_friction must be non-negative")
        return cls(
            name=name,
            shape=shape,
            geom_size=geom_size,
            position=_vector(raw.get("position", [0, 0, 0]), 3, f"prop {name}: position"),
            orientation_wxyz=_unit_quaternion(
                raw.get("orientation_wxyz", [1, 0, 0, 0]), f"prop {name}: orientation_wxyz"
            ),
            mass=_positive(raw.get("mass", 0.05), f"prop {name}: mass"),
            color=_vector(raw.get("color", [0.7, 0.7, 0.7]), 3, f"prop {name}: color"),
            fixed=_strict_bool(raw.get("fixed", False), f"prop {name}: fixed"),
            friction=friction,
            mesh=mesh,
        )


@dataclass(frozen=True)
class WorldConfig:
    physics_dt: float = 0.002
    real_time: bool = True
    props: tuple[PropSpec, ...] = ()

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, Any]) -> "WorldConfig":
        props = tuple(PropSpec.parse(raw) for raw in attrs.get("props") or [])
        names = [prop.name for prop in props]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate prop names in {names}")
        return cls(
            physics_dt=_positive(attrs.get("physics_dt", 0.002), "physics_dt"),
            real_time=_strict_bool(attrs.get("real_time", True), "real_time"),
            props=props,
        )


@dataclass
class Contribution:
    """What one component adds to the scene. build edits a fresh spec on every
    rebuild; init runs on the first model that contains the contribution, and
    again on reset(). late contributions build after the rest, so they can
    attach to bodies another contribution added (a wrist camera on its arm).
    render_sizes are (width, height) renderers built and warmed with each
    model, so a camera's first frame doesn't stall the step loop."""

    build: Callable[[mujoco.MjSpec], None]
    init: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None
    late: bool = False
    render_sizes: tuple[tuple[int, int], ...] = ()


def _copy_state(
    old_model: mujoco.MjModel,
    old_data: mujoco.MjData,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    data.time = old_data.time
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        old = mujoco.mj_name2id(old_model, mujoco.mjtObj.mjOBJ_JOINT, name) if name else -1
        if old < 0 or old_model.jnt_type[old] != model.jnt_type[joint]:
            continue
        kind = int(model.jnt_type[joint])
        qpos_width = _QPOS_WIDTH.get(kind, 1)
        dof_width = _DOF_WIDTH.get(kind, 1)
        qpos, old_qpos = model.jnt_qposadr[joint], old_model.jnt_qposadr[old]
        dof, old_dof = model.jnt_dofadr[joint], old_model.jnt_dofadr[old]
        data.qpos[qpos : qpos + qpos_width] = old_data.qpos[old_qpos : old_qpos + qpos_width]
        data.qvel[dof : dof + dof_width] = old_data.qvel[old_dof : old_dof + dof_width]
    for actuator in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
        old = mujoco.mj_name2id(old_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) if name else -1
        if old >= 0:
            data.ctrl[actuator] = old_data.ctrl[old]


class SimManager:
    """Owns the sim thread. Get the process-wide instance via SimManager.get()."""

    _instance: "SimManager | None" = None
    _instance_lock = threading.Lock()

    @classmethod
    def get(cls) -> "SimManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = SimManager()
            return cls._instance

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._tasks: queue.Queue[tuple[Callable[[], Any], Future]] = queue.Queue()
        self._stop = threading.Event()
        self._stopped = False
        self._sim_thread_id: int | None = None
        self.cfg: WorldConfig | None = None
        self.prop_pieces: dict[str, list[Piece]] = {}
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.generation = 0
        self._contributions: dict[str, Contribution] = {}
        self.step_callbacks: dict[str, Callable[[float], None]] = {}
        self.arms: dict[str, Any] = {}  # ArmHandle by component name, for its gripper
        self.playing = True
        self.slips = 0
        self._anchor_wall = 0.0
        self._anchor_sim = 0.0
        self._rtf_samples: deque[tuple[float, float]] = deque()
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._warned: set[str] = set()

    # ------------------------------------------------------------------
    # threading
    # ------------------------------------------------------------------

    def bind_current_thread(self) -> None:
        """Make the calling thread the sim thread; run() then executes inline
        on it. main_loop does this; tests call it to drive the sim directly."""
        self._sim_thread_id = threading.get_ident()

    def run(self, fn: Callable[[], Any], timeout: float = 30.0) -> Any:
        """Run fn on the sim thread and return its result."""
        if threading.get_ident() == self._sim_thread_id:
            return fn()
        if self._stopped:
            raise RuntimeError("the simulator has stopped")
        future: Future = Future()
        self._tasks.put((fn, future))
        if self._stopped and future.cancel():
            raise RuntimeError("the simulator has stopped")
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            future.cancel()
            raise

    def request_stop(self) -> None:
        self._stop.set()

    def main_loop(self) -> None:
        self.bind_current_thread()
        try:
            while not self._stop.is_set():
                self._drain_tasks()
                stepped = self.tick()
                if not stepped or (self.cfg is not None and self.cfg.real_time):
                    time.sleep(0.001)
        finally:
            self._stopped = True
            while True:
                try:
                    _, future = self._tasks.get_nowait()
                except queue.Empty:
                    break
                if future.set_running_or_notify_cancel():
                    future.set_exception(RuntimeError("the simulator has stopped"))
            self._close_renderers()

    def _drain_tasks(self) -> None:
        while True:
            try:
                fn, future = self._tasks.get_nowait()
            except queue.Empty:
                return
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(fn())
                except BaseException as exc:
                    future.set_exception(exc)

    # ------------------------------------------------------------------
    # scene
    # ------------------------------------------------------------------

    def require_world(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "the world component has not started; check the component's "
                '"world" attribute'
            )

    def configure_world(
        self, cfg: WorldConfig, pieces: Mapping[str, list[Piece]] | None = None
    ) -> None:
        """pieces: each mesh prop's convex pieces (meshes.load_pieces), by prop name."""
        pieces = dict(pieces or {})
        missing = [p.name for p in cfg.props if p.mesh is not None and p.name not in pieces]
        if missing:
            raise ValueError(f"mesh pieces not loaded for props {missing}")

        def configure() -> None:
            old, old_pieces = self.cfg, self.prop_pieces
            self.cfg, self.prop_pieces = cfg, pieces
            try:
                self._rebuild(fresh=set())
            except BaseException:
                self.cfg, self.prop_pieces = old, old_pieces
                raise

        self.run(configure)

    def set_contribution(self, name: str, contribution: Contribution) -> None:
        def apply() -> None:
            self.require_world()
            old = self._contributions.get(name)
            self._contributions[name] = contribution
            try:
                # a replaced contribution keeps its state; only new ones init
                self._rebuild(fresh={name} if old is None else set())
            except BaseException:
                if old is None:
                    del self._contributions[name]
                else:
                    self._contributions[name] = old
                raise

        self.run(apply)

    def remove_contribution(self, name: str) -> None:
        def remove() -> None:
            old = self._contributions.pop(name, None)
            if old is not None and self.cfg is not None:
                try:
                    self._rebuild(fresh=set())
                except BaseException:
                    self._contributions[name] = old
                    raise

        self.run(remove)

    def _build_spec(self) -> mujoco.MjSpec:
        assert self.cfg is not None
        spec = mujoco.MjSpec()
        spec.option.timestep = self.cfg.physics_dt
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.worldbody.add_light(
            name="sun",
            pos=[0, 0, 3],
            dir=[0, 0, -1],
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        )
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 0.05],
            rgba=[0.3, 0.35, 0.4, 1],
        )
        for prop in self.cfg.props:
            body = spec.worldbody.add_body(
                name=prop.name,
                pos=list(prop.position),
                quat=list(prop.orientation_wxyz),
            )
            if not prop.fixed:
                body.add_freejoint(name=prop.name)
            contact = {
                "rgba": [*prop.color, 1.0],
                "friction": [prop.friction, 0.005, 0.0001],
                "priority": PROP_CONTACT_PRIORITY,
                "solref": list(PROP_SOLREF),
            }
            if prop.mesh is not None:
                add_pieces(spec, body, prop.name, self.prop_pieces[prop.name], mass=prop.mass, **contact)
                continue
            body.add_geom(
                name=prop.name,
                type=GEOM_TYPES[prop.shape],
                size=mj_size((prop.shape, prop.geom_size)),
                mass=prop.mass,
                **contact,
            )
        for contribution in sorted(self._contributions.values(), key=lambda c: c.late):
            contribution.build(spec)
        return spec

    def _rebuild(self, fresh: set[str]) -> None:
        """Compile a new model; swap it in only if compilation succeeds."""
        model = self._build_spec().compile()
        data = mujoco.MjData(model)
        if self.model is not None and self.data is not None:
            _copy_state(self.model, self.data, model, data)
        for name in fresh:
            init = self._contributions[name].init
            if init is not None:
                init(model, data)
        mujoco.mj_forward(model, data)
        self._close_renderers()
        self.model, self.data = model, data
        self.generation += 1
        for size in {size for c in self._contributions.values() for size in c.render_sizes}:
            self._renderer(*size).render()  # the first draw uploads the scene to the GPU
        self._reanchor()

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------

    def step_once(self) -> None:
        assert self.model is not None and self.data is not None
        for key, callback in list(self.step_callbacks.items()):
            try:
                callback(self.model.opt.timestep)
            except Exception:
                LOGGER.exception("step callback %s failed; removing it", key)
                self.step_callbacks.pop(key, None)
        mujoco.mj_step(self.model, self.data)
        self._check_warnings()

    def advance(self, seconds: float) -> None:
        """Step until sim time has moved by `seconds`. Sim thread only."""
        self.require_world()
        assert self.data is not None
        end = self.data.time + seconds
        while self.data.time < end - 1e-9:
            self.step_once()

    def tick(self) -> bool:
        """One main-loop iteration of stepping. True if it ran at least one
        step, False if it did nothing (no world, paused, or already caught up)."""
        if self.model is None or self.data is None or not self.playing:
            return False
        assert self.cfg is not None
        substeps = 0
        if not self.cfg.real_time:
            for _ in range(MAX_SUBSTEPS):
                self.step_once()
                substeps += 1
        else:
            target = self._anchor_sim + (self._clock() - self._anchor_wall)
            while self.data.time < target - 1e-9 and substeps < MAX_SUBSTEPS:
                self.step_once()
                substeps += 1
            if target - self.data.time > SLIP_THRESHOLD_S:
                self.slips += 1
                self._reanchor()
        self._sample_rtf()
        return substeps > 0

    def _reanchor(self) -> None:
        self._anchor_wall = self._clock()
        self._anchor_sim = self.data.time if self.data is not None else 0.0

    def _sample_rtf(self) -> None:
        assert self.data is not None
        now = self._clock()
        self._rtf_samples.append((now, self.data.time))
        while len(self._rtf_samples) > 2 and now - self._rtf_samples[0][0] > RTF_WINDOW_S:
            self._rtf_samples.popleft()

    def _real_time_factor(self) -> float:
        if len(self._rtf_samples) < 2:
            return 0.0
        (wall0, sim0), (wall1, sim1) = self._rtf_samples[0], self._rtf_samples[-1]
        return (sim1 - sim0) / (wall1 - wall0) if wall1 > wall0 else 0.0

    def _check_warnings(self) -> None:
        assert self.data is not None
        for index, warning in enumerate(self.data.warning):
            if warning.number:
                name = mujoco.mjtWarning(index).name
                if name not in self._warned:
                    self._warned.add(name)
                    LOGGER.warning("mujoco warning %s (first occurrence)", name)

    # ------------------------------------------------------------------
    # world commands
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        def read() -> dict[str, Any]:
            warnings: dict[str, int] = {}
            if self.data is not None:
                for index, warning in enumerate(self.data.warning):
                    if warning.number:
                        warnings[mujoco.mjtWarning(index).name] = int(warning.number)
            return {
                "playing": self.playing,
                "sim_time": float(self.data.time) if self.data is not None else 0.0,
                "real_time_factor": self._real_time_factor(),
                "slips": self.slips,
                "warnings": warnings,
            }

        return self.run(read)

    def play(self) -> None:
        def play() -> None:
            self.playing = True
            self._reanchor()

        self.run(play)

    def pause(self) -> None:
        def pause() -> None:
            self.playing = False
            self._rtf_samples.clear()

        self.run(pause)

    def reset(self) -> None:
        def reset() -> None:
            self.require_world()
            assert self.model is not None and self.data is not None
            mujoco.mj_resetData(self.model, self.data)
            for contribution in self._contributions.values():
                if contribution.init is not None:
                    contribution.init(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.slips = 0
            self._rtf_samples.clear()
            self._reanchor()

        self.run(reset)

    def _prop(self, name: str) -> PropSpec:
        for prop in self.cfg.props if self.cfg is not None else ():
            if prop.name == name:
                return prop
        raise ValueError(f"unknown prop {name!r}")

    def _prop_pose(self, name: str) -> dict[str, list[float]]:
        assert self.data is not None
        body = self.data.body(name)
        return {
            "position": [float(value) for value in body.xpos],
            "orientation_wxyz": [float(value) for value in body.xquat],
        }

    def _place_prop(
        self, name: str, position: Sequence[float], orientation: Sequence[float]
    ) -> None:
        assert self.model is not None and self.data is not None
        if self._prop(name).fixed:
            raise ValueError(f"prop {name} is fixed and cannot be moved")
        joint = self.model.joint(name)
        qpos, dof = joint.qposadr[0], joint.dofadr[0]
        self.data.qpos[qpos : qpos + 3] = position
        self.data.qpos[qpos + 3 : qpos + 7] = orientation
        self.data.qvel[dof : dof + 6] = 0.0

    def set_prop_pose(
        self, name: str, position: Any, orientation_wxyz: Any
    ) -> dict[str, Any]:
        position = _vector(position, 3, "position")
        orientation = _unit_quaternion(orientation_wxyz, "orientation_wxyz")

        def apply() -> dict[str, Any]:
            self.require_world()
            self._place_prop(name, position, orientation)
            mujoco.mj_forward(self.model, self.data)
            return {"prop": name, **self._prop_pose(name)}

        return self.run(apply)

    def get_prop_pose(self, name: str) -> dict[str, list[float]]:
        def read() -> dict[str, list[float]]:
            self.require_world()
            self._prop(name)
            return self._prop_pose(name)

        return self.run(read)

    def park_props(self, names: Sequence[str]) -> dict[str, Any]:
        def park() -> dict[str, Any]:
            self.require_world()
            for name in names:
                prop = self._prop(name)
                self._place_prop(name, prop.position, prop.orientation_wxyz)
            mujoco.mj_forward(self.model, self.data)
            return {"poses": {name: self._prop_pose(name) for name in names}}

        return self.run(park)

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------

    def _renderer(self, width: int, height: int) -> mujoco.Renderer:
        assert self.model is not None
        renderer = self._renderers.get((width, height))
        if renderer is None:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._renderers[(width, height)] = renderer
        return renderer

    def render(
        self,
        camera: str,
        width: int,
        height: int,
        depth_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """RGB at width x height and, with depth_size, depth in meters along
        the optical axis at that size, both from the same sim state. Depth is
        0 where nothing is in view."""

        def draw() -> tuple[np.ndarray, np.ndarray | None]:
            self.require_world()
            assert self.model is not None
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
                raise RuntimeError(
                    f"camera {camera!r} is not in the scene; if its frame parent is "
                    "another mujoco-sim component, configure that component"
                )
            renderer = self._renderer(width, height)
            renderer.update_scene(self.data, camera=camera)
            rgb = renderer.render().copy()
            if depth_size is None:
                return rgb, None
            renderer = self._renderer(*depth_size)
            renderer.update_scene(self.data, camera=camera)
            renderer.enable_depth_rendering()
            try:
                meters = renderer.render().copy()
            finally:
                renderer.disable_depth_rendering()
            far = self.model.vis.map.zfar * self.model.stat.extent
            meters[meters >= far * 0.999] = 0.0
            return rgb, meters

        return self.run(draw)

    def _close_renderers(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
