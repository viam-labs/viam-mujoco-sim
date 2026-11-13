"""The singleton that owns MuJoCo.

One thread (the process main thread, see main.py) owns the MjSpec, MjModel,
MjData and every renderer. The Viam module server runs on a side thread and
hands work to the sim thread through run(). MuJoCo has no thread affinity of
its own, but MjData is not thread-safe and GL contexts belong to the thread
that created them, so a single owner needs no locks.

The scene is rebuilt from scratch on every structural change: the world's
floor, lighting and props, then each component's contribution. State carries
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
from viam.proto.common import Geometry

from . import lighting
from .meshes import (
    UNTINTED_RGBA,
    MeshSource,
    Piece,
    Texture,
    add_pieces,
    add_texture,
    parse_mesh,
    parse_texture,
)
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
# Bundled limbs of a bundled robot file (viam-labs:mujoco-sim:robot): default
# attributes for an arm with `robot` set. Joint and site names are the file's.
_ALOHA_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")
ROBOT_LIMB_ASSETS: dict[str, dict[str, Any]] = {
    f"aloha_{side}": {
        "joints": [f"{side}/{joint}" for joint in _ALOHA_JOINTS],
        "flange_site": f"{side}/gripper",
        # written here from the vx300s MJCF; both arms share it
        "kinematics_url": "trossen_aloha/vx300s_kinematics.json",
        "home_keyframe": "neutral_pose",
        # Viam's end-effector z points out of the fingers: the site's +x
        "flange_offset": {"orientation": {"o_x": 1, "o_y": 0, "o_z": 0, "theta": 0}},
    }
    for side in ("left", "right")
} | {
    # TIAGo++ (robot asset "tiago"): the torso is an arm, framed on the robot;
    # the arms and head are framed on the torso. Kinematics written here by
    # pal_tiago_dual/make_kinematics.py
    "tiago_torso": {
        "joints": ["torso_lift_joint"],
        "flange_site": "torso_frame",
        "kinematics_url": "pal_tiago_dual/tiago_torso_kinematics.json",
        "home_keyframe": "home",
    },
    "tiago_head": {
        "joints": ["head_1_joint", "head_2_joint"],
        "flange_site": "head_camera",
        "kinematics_url": "pal_tiago_dual/tiago_head_kinematics.json",
        "home_keyframe": "home",
    },
} | {
    f"tiago_arm_{side}": {
        "joints": [f"arm_{side}_{i}_joint" for i in range(1, 8)],
        "flange_site": f"gripper_{side}_grasp",
        "kinematics_url": f"pal_tiago_dual/tiago_arm_{side}_kinematics.json",
        "home_keyframe": "home",
    }
    for side in ("left", "right")
}
# Bundled grippers: default attributes for the gripper's `asset`. No mjcf
# means the arm MJCF's own gripper.
GRIPPER_ASSETS: dict[str, dict[str, Any]] = {
    "lite6_gripper": {},
    # mujoco_menagerie@98a70c56 robotiq_2f85
    "robotiq_2f85": {"mjcf": "robotiq_2f85/2f85.xml"},
    # written here; mesh from ValerioMa/robotiq_tcp@45891ab, decimated
    "robotiq_epick": {"mjcf": "robotiq_epick/epick.xml"},
    # ALOHA's integrated fingers: ctrl 0.037 opens them, 0.002 closes them
    "aloha": {"open": 0.037, "closed": 0.002},
    # TIAGo's coupled fingers: ctrl 0 closes them (the default open), 0.045 opens them
    "tiago": {"open": 0.045, "closed": 0.0},
}

MAX_SUBSTEPS = 50
SLIP_THRESHOLD_S = 0.1
RTF_WINDOW_S = 2.0
# Props win contact-parameter mixing and use a stiffer solref than MuJoCo's
# default. Without this the Menagerie fingertips' solimp averages with the
# prop's into a soft contact and the jaws sink ~9 mm into a block.
PROP_CONTACT_PRIORITY = 1
PROP_SOLREF = (0.004, 1.0)
# Collision bits. Every geom collides on bit 1, MuJoCo's default. A conveyor
# belt takes only bit 2, which free bodies (movable props, parcels) add, so
# the belt never touches the static geoms around it: they are welded to the
# world, so no parent filter keeps them apart, and a touching one stalls it.
FREE_CONTYPE = 3
BELT_CONAFFINITY = 2

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
    texture: str | None = None  # image source; the world loads it (meshes.load_texture)

    @classmethod
    def parse(cls, raw: Any) -> "PropSpec":
        """Isaac's prop format plus sphere, cylinder and capsule types. dynamic_friction is accepted so
        Isaac configs carry over; MuJoCo has one sliding coefficient."""
        if not isinstance(raw, Mapping) or not raw.get("name"):
            raise ValueError(f"every prop needs a name: {raw!r}")
        name = str(raw["name"])
        if name == "floor":
            raise ValueError("prop floor: the name floor is taken by the floor")
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
            texture=parse_texture(f"prop {name}", raw["texture"]) if "texture" in raw else None,
        )


MAX_SHADOW_SIZE = 16384  # 32768 makes mujoco.Renderer fail (incomplete shadow framebuffer)
MAX_ANTIALIAS = 16
DEFAULT_LIGHT_HEIGHT_M = 3.0


def _color(value: Any, label: str) -> tuple[float, ...]:
    color = _vector(value, 3, label)
    if any(not 0.0 <= item <= 1.0 for item in color):
        raise ValueError(f"{label} must be 3 numbers from 0 to 1")
    return color


def _int_in(value: Any, low: int, high: int, label: str) -> int:
    number = _finite(value, label)
    if number != int(number):
        raise ValueError(f"{label} must be a whole number")
    if not low <= number <= high:
        raise ValueError(f"{label} must be from {low} to {high}")
    return int(number)


@dataclass(frozen=True)
class LightSpec:
    type: str = "directional"  # a key of lighting.LIGHT_TYPES
    position: tuple[float, ...] = (0.0, 0.0, DEFAULT_LIGHT_HEIGHT_M)
    direction: tuple[float, ...] = (0.0, 0.0, -1.0)  # MuJoCo normalizes it
    color: tuple[float, ...] = (0.7, 0.7, 0.7)  # diffuse; MuJoCo's default
    shadows: bool = True
    cutoff_deg: float = 45.0  # spot only

    @classmethod
    def parse(cls, raw: Any, label: str) -> "LightSpec":
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label} must be an object")
        kind = raw.get("type", "directional")
        if kind not in lighting.LIGHT_TYPES:
            raise ValueError(f'{label}.type must be "directional", "spot" or "point", got {kind!r}')
        direction = _vector(raw.get("direction", [0, 0, -1]), 3, f"{label}.direction")
        if not any(direction):
            raise ValueError(f"{label}.direction must not be all zeros")
        cutoff = 45.0
        if "cutoff_deg" in raw:
            if kind != "spot":
                raise ValueError(f'{label}.cutoff_deg only applies to type "spot"')
            cutoff = _finite(raw["cutoff_deg"], f"{label}.cutoff_deg")
            if not 0.0 < cutoff <= 90.0:
                raise ValueError(f"{label}.cutoff_deg must be in (0, 90]")
        if "position" in raw:
            position = _vector(raw["position"], 3, f"{label}.position")
        elif kind == "directional" and direction[2] < 0.0:
            # a directional light's shadow map is centred on its position: put
            # it 3 m up the ray that reaches the world origin
            scale = DEFAULT_LIGHT_HEIGHT_M / -direction[2]
            position = tuple(-item * scale for item in direction)
            if not all(math.isfinite(item) for item in position):
                position = (0.0, 0.0, DEFAULT_LIGHT_HEIGHT_M)
        else:
            position = (0.0, 0.0, DEFAULT_LIGHT_HEIGHT_M)
        return cls(
            type=kind,
            position=position,
            direction=direction,
            color=_color(raw.get("color", [0.7, 0.7, 0.7]), f"{label}.color"),
            shadows=_strict_bool(raw.get("shadows", True), f"{label}.shadows"),
            cutoff_deg=cutoff,
        )


@dataclass(frozen=True)
class LightingConfig:
    """The world's `lighting` attribute (applied by lighting.py). Defaults
    render as the world did before it existed."""

    lights: tuple[LightSpec, ...] | None = None  # None: the default sun
    keep_default_light: bool = False
    headlight: float = 1.0  # scales MuJoCo's headlight; 0 turns it off
    skybox: tuple[tuple[float, ...], tuple[float, ...]] | None = None  # (top, bottom) rgb
    shadows: bool = True
    shadow_size: int = 4096
    reflections: bool = True
    antialias: int = 4

    @classmethod
    def parse(cls, raw: Any) -> "LightingConfig":
        if not isinstance(raw, Mapping):
            raise ValueError("lighting must be an object")
        lights = None
        if "lights" in raw:
            if not isinstance(raw["lights"], (list, tuple)):
                raise ValueError("lighting.lights must be a list")
            if "keep_default_light" in raw:
                raise ValueError("lighting.keep_default_light only applies without lighting.lights")
            lights = tuple(
                LightSpec.parse(light, f"lighting.lights[{index}]")
                for index, light in enumerate(raw["lights"])
            )
        skybox = None
        if "skybox" in raw:
            value = raw["skybox"]
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("lighting.skybox must be two colors, [[r, g, b], [r, g, b]] (top, bottom)")
            skybox = (_color(value[0], "lighting.skybox[0]"), _color(value[1], "lighting.skybox[1]"))
        headlight = _finite(raw.get("headlight", 1.0), "lighting.headlight")
        if headlight < 0.0:
            raise ValueError("lighting.headlight must be non-negative")
        return cls(
            lights=lights,
            keep_default_light=_strict_bool(raw.get("keep_default_light", False), "lighting.keep_default_light"),
            headlight=headlight,
            skybox=skybox,
            shadows=_strict_bool(raw.get("shadows", True), "lighting.shadows"),
            shadow_size=_int_in(raw.get("shadow_size", 4096), 1, MAX_SHADOW_SIZE, "lighting.shadow_size"),
            reflections=_strict_bool(raw.get("reflections", True), "lighting.reflections"),
            antialias=_int_in(raw.get("antialias", 4), 0, MAX_ANTIALIAS, "lighting.antialias"),
        )


@dataclass(frozen=True)
class WorldConfig:
    physics_dt: float = 0.002
    real_time: bool = True
    props: tuple[PropSpec, ...] = ()
    floor_texture: str | None = None  # image source; the world loads it
    floor_texture_size: float = 1.0  # one tile's edge, meters
    lighting: LightingConfig = LightingConfig()

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, Any]) -> "WorldConfig":
        props = tuple(PropSpec.parse(raw) for raw in attrs.get("props") or [])
        names = [prop.name for prop in props]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate prop names in {names}")
        floor_texture = None
        if "floor_texture" in attrs:
            floor_texture = parse_texture("floor_texture", attrs["floor_texture"])
        elif "floor_texture_size" in attrs:
            raise ValueError("floor_texture_size only applies with floor_texture")
        return cls(
            physics_dt=_positive(attrs.get("physics_dt", 0.002), "physics_dt"),
            real_time=_strict_bool(attrs.get("real_time", True), "real_time"),
            props=props,
            floor_texture=floor_texture,
            floor_texture_size=_positive(
                attrs.get("floor_texture_size", 1.0), "floor_texture_size"
            ),
            lighting=LightingConfig.parse(
                {} if attrs.get("lighting") is None else attrs["lighting"]
            ),
        )


@dataclass(frozen=True)
class DynamicProp:
    """A box a component adds and removes without a rebuild (a spawner's
    parcel), for the world state store. serial is new on every spawn."""

    serial: int
    half_size: tuple[float, float, float]
    color: tuple[float, float, float]


@dataclass(frozen=True)
class StaticShape:
    """A component's fixed shapes, for the world state store: RDK never asks a
    motor or sensor for its geometries, so the 3D scene can't get them from
    the frame system. geometries are in mm, in body's frame."""

    body: str
    geometries: tuple[Geometry, ...]
    color: tuple[float, ...]


@dataclass
class Contribution:
    """What one component adds to the scene. build edits a fresh spec on every
    rebuild and returns False, before changing the spec, when what it attaches
    to isn't in the spec yet (a wrist camera's arm): False waits for its
    parent, None or True means it built. A waiting build is tried again after
    the others, and left out if that never appears. init runs on the first
    model the contribution is built into (again after it was left out), and
    on reset(). render_sizes are (width, height) renderers built and warmed
    with each model, so a camera's first frame doesn't stall the step loop.
    replaces_floor leaves the world's floor out (a scene file brings its own);
    brings_lights leaves the default sun out (a scene file with lights);
    props are movables the prop commands address by name (a scene's free
    bodies: name, position, orientation_wxyz, fixed)."""

    build: Callable[[mujoco.MjSpec], bool | None]
    init: Callable[[mujoco.MjModel, mujoco.MjData], None] | None = None
    render_sizes: tuple[tuple[int, int], ...] = ()
    replaces_floor: bool = False
    brings_lights: bool = False
    props: tuple[Any, ...] = ()


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
        self.textures: dict[str, Texture] = {}  # by prop name, and "floor"
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.generation = 0
        self._contributions: dict[str, Contribution] = {}
        self._built: set[str] = set()  # the contributions in the current model
        self.step_callbacks: dict[str, Callable[[float], None]] = {}
        self.arms: dict[str, Any] = {}  # ArmHandle by component name, for its gripper
        self.robots: dict[str, Any] = {}  # RobotHandle by component name, for its arms
        # by robot name: the file's joint and actuator names (unprefixed) an arm
        # or gripper drives, to its component name. Kept here, not in the robot
        # handle, so claims survive the robot being removed or reconfigured.
        self.robot_claims: dict[str, dict[str, str]] = {}
        self.conveyors: dict[str, Any] = {}  # ConveyorHandle by component name, for its spawner
        self.dynamic_props: dict[str, DynamicProp] = {}  # live ones, by body name
        self.static_shapes: dict[str, StaticShape] = {}  # by component name
        self.playing = True
        self.slips = 0
        self._anchor_wall = 0.0
        self._anchor_sim = 0.0
        self._rtf_samples: deque[tuple[float, float]] = deque()
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._warned: set[str] = set()
        self._floor_warned: list[str] = []  # the floor-replacing scenes last warned about
        self._skybox_warned: list[str] = []  # the skyboxes last warned about

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

    def built(self, name: str) -> bool:
        """Whether the contribution `name` is in the current model."""
        return name in self._built

    def configure_world(
        self,
        cfg: WorldConfig,
        pieces: Mapping[str, list[Piece]] | None = None,
        textures: Mapping[str, Texture] | None = None,
    ) -> None:
        """pieces: each mesh prop's convex pieces (meshes.load_pieces), by prop name.
        textures: each textured prop's image (meshes.load_texture), by prop name,
        and the floor's as "floor"."""
        pieces = dict(pieces or {})
        textures = dict(textures or {})
        missing = [p.name for p in cfg.props if p.mesh is not None and p.name not in pieces]
        if missing:
            raise ValueError(f"mesh pieces not loaded for props {missing}")
        wanted = [p.name for p in cfg.props if p.texture is not None]
        if cfg.floor_texture is not None:
            wanted.append("floor")
        unloaded = [name for name in wanted if name not in textures]
        if unloaded:
            raise ValueError(f"textures not loaded for {unloaded}")

        def configure() -> None:
            old = self.cfg, self.prop_pieces, self.textures
            self.cfg, self.prop_pieces, self.textures = cfg, pieces, textures
            try:
                self._rebuild()
            except BaseException:
                self.cfg, self.prop_pieces, self.textures = old
                raise

        self.run(configure)

    def set_contribution(self, name: str, contribution: Contribution) -> None:
        def apply() -> None:
            self.require_world()
            old = self._contributions.get(name)
            self._contributions[name] = contribution
            try:
                self._rebuild()
            except BaseException:
                if old is None:
                    del self._contributions[name]
                else:
                    self._contributions[name] = old
                raise

        self.run(apply)

    def contributes(self, name: str) -> bool:
        """Whether `name` is a component of this module (built, or waiting for its parent)."""
        return name in self._contributions

    def remove_contribution(self, name: str) -> None:
        def remove() -> None:
            old = self._contributions.pop(name, None)
            if old is not None and self.cfg is not None:
                try:
                    self._rebuild()
                except BaseException:
                    self._contributions[name] = old
                    raise

        self.run(remove)

    def _build_spec(self) -> tuple[mujoco.MjSpec, set[str]]:
        """The scene's spec and the contributions built into it."""
        assert self.cfg is not None
        spec = mujoco.MjSpec()
        spec.option.timestep = self.cfg.physics_dt
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        lighting.apply(
            spec, self.cfg.lighting, any(c.brings_lights for c in self._contributions.values())
        )
        floored = sorted(n for n, c in self._contributions.items() if c.replaces_floor)
        if floored and self.cfg.floor_texture is not None and self._floor_warned != floored:
            LOGGER.warning("floor_texture is ignored: scene %s brings its own floor", ", ".join(floored))
        self._floor_warned = floored if self.cfg.floor_texture is not None else []
        if not floored:
            floor = spec.worldbody.add_geom(
                name="floor",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[0, 0, 0.05],
                rgba=[0.3, 0.35, 0.4, 1],
            )
            if self.cfg.floor_texture is not None:
                floor.material = add_texture(
                    spec, "floor/texture", self.textures["floor"],
                    repeat_m=self.cfg.floor_texture_size,
                )
                floor.rgba = list(UNTINTED_RGBA)
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
            if not prop.fixed:
                contact["contype"] = FREE_CONTYPE
            if prop.texture is not None:
                contact["material"] = add_texture(spec, f"{prop.name}/texture", self.textures[prop.name])
                contact["rgba"] = list(UNTINTED_RGBA)
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
        # passes until one builds nothing new: a chain (base, arm, gripper,
        # wrist camera) builds in any insertion order
        built: set[str] = set()
        waiting = dict(self._contributions)
        while waiting:
            done = {name for name, c in waiting.items() if c.build(spec) is not False}
            if not done:
                break  # what's left waits for a parent that isn't configured
            built |= done
            waiting = {name: c for name, c in waiting.items() if name not in done}
        skyboxes = lighting.add_skybox(spec, self.cfg.lighting)
        if skyboxes and self._skybox_warned != skyboxes:
            LOGGER.warning("lighting.skybox is ignored: %s is already in the scene", ", ".join(skyboxes))
        self._skybox_warned = skyboxes
        return spec, built

    def _rebuild(self) -> None:
        """Compile a new model; swap it in only if compilation succeeds.
        Contributions built now but not into the old model (new ones, and
        ones whose parent came back) init; the rest keep their state."""
        spec, built = self._build_spec()
        model = spec.compile()
        data = mujoco.MjData(model)
        if self.model is not None and self.data is not None:
            _copy_state(self.model, self.data, model, data)
        for name in built - self._built:
            init = self._contributions[name].init
            if init is not None:
                init(model, data)
        mujoco.mj_forward(model, data)
        self._close_renderers()
        self.model, self.data, self._built = model, data, built
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
            for name in self._built:
                init = self._contributions[name].init
                if init is not None:
                    init(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            self.slips = 0
            self._rtf_samples.clear()
            self._reanchor()

        self.run(reset)

    def scene_props(self) -> list[Any]:
        """The props of the contributions in the current model (a scene's free
        bodies). Sim thread."""
        return [p for name in sorted(self._built) for p in self._contributions[name].props]

    def _prop(self, name: str) -> Any:
        """A config prop (PropSpec) or a scene prop, by name."""
        for prop in self.cfg.props if self.cfg is not None else ():
            if prop.name == name:
                return prop
        for prop in self.scene_props():
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
            assert self.cfg is not None
            lighting.set_render_flags(renderer.scene, self.cfg.lighting)
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
