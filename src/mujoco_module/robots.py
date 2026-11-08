"""What the module needs to know about an arm, read from its MJCF
and its Viam kinematics file.

Pure functions over a parsed kinematics file and a compiled MjModel (names as
in the file, unprefixed); nothing here touches the running sim. Poses are 4x4
numpy matrices in metres.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import mujoco
import numpy as np
from viam.spatialmath import OrientationVector, Quaternion

from .meshes import ASSETS_DIR
from .shapes import GEOM_TYPES

_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
_TRN_JOINT = int(mujoco.mjtTrn.mjTRN_JOINT)
_TRN_TENDON = int(mujoco.mjtTrn.mjTRN_TENDON)
_WRAP_JOINT = int(mujoco.mjtWrap.mjWRAP_JOINT)
_BIAS_NONE = int(mujoco.mjtBias.mjBIAS_NONE)
_BIAS_AFFINE = int(mujoco.mjtBias.mjBIAS_AFFINE)
_GAIN_FIXED = int(mujoco.mjtGain.mjGAIN_FIXED)
# a position actuator with timeconst > 0 has filter dynamics
_DYN_OK = {
    int(mujoco.mjtDyn.mjDYN_NONE), int(mujoco.mjtDyn.mjDYN_FILTER), int(mujoco.mjtDyn.mjDYN_FILTEREXACT)
}


def transform(pos: Sequence[float] = (0, 0, 0), quat: Sequence[float] = (1, 0, 0, 0)) -> np.ndarray:
    """4x4 from a position (m) and a (w, x, y, z) quaternion."""
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, float) / np.linalg.norm(quat))
    result = np.eye(4)
    result[:3, :3] = matrix.reshape(3, 3)
    result[:3, 3] = pos
    return result


def quat(matrix: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of a 4x4's rotation."""
    result = np.zeros(4)
    mujoco.mju_mat2Quat(result, np.ascontiguousarray(matrix[:3, :3]).ravel())
    return result


def _axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return transform(quat=(math.cos(angle / 2), *(math.sin(angle / 2) * axis)))


def pose_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(metres, degrees) between two poses."""
    cos = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1.0, 1.0)
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), math.degrees(math.acos(cos))


def parse_pose(raw: Any, label: str) -> np.ndarray:
    """A config pose, {translation: {x, y, z} mm, orientation: {o_x, o_y, o_z,
    theta deg}}, either part optional; None is the identity."""
    if raw is None:
        return np.eye(4)
    if not isinstance(raw, Mapping) or set(raw) - {"translation", "orientation"}:
        raise ValueError(
            f"{label} must be {{translation: {{x, y, z}}, orientation: {{o_x, o_y, o_z, theta}}}}"
        )
    t = raw.get("translation") or {}
    o = raw.get("orientation") or {}
    if isinstance(o, Mapping) and not any(key in o for key in ("o_x", "o_y", "o_z")):
        o = {**o, "o_z": 1}  # Viam's default axis
    try:
        xyz = [float(t.get(key, 0)) / 1000 for key in "xyz"]
        ov = [float(o.get(key, 0)) for key in ("o_x", "o_y", "o_z", "theta")]
    except (TypeError, ValueError, AttributeError):
        raise ValueError(f"{label}: translation and orientation values must be numbers") from None
    if not all(math.isfinite(value) for value in xyz + ov) or not any(ov[:3]):
        raise ValueError(f"{label}: values must be finite, and o_x, o_y, o_z not all zero")
    q = OrientationVector(ov[0], ov[1], ov[2], math.radians(ov[3])).to_quaternion()
    return transform(xyz, (q.w, q.i, q.j, q.k))


def format_pose(matrix: np.ndarray) -> str:
    """A pose in parse_pose's form, as JSON, for error messages."""
    ov = Quaternion(*quat(matrix)).to_orientation_vector()
    t = matrix[:3, 3] * 1000

    def tidy(value: float, digits: int) -> float:
        return round(float(value), digits) + 0.0  # no -0.0

    return json.dumps({
        "translation": {"x": tidy(t[0], 1), "y": tidy(t[1], 1), "z": tidy(t[2], 1)},
        "orientation": {
            "o_x": tidy(ov.o_x, 3), "o_y": tidy(ov.o_y, 3), "o_z": tidy(ov.o_z, 3),
            "theta": tidy(math.degrees(ov.theta), 1),
        },
    })


# ----------------------------------------------------------------------------
# kinematics files
# ----------------------------------------------------------------------------


def resolve_path(source: str, label: str) -> Path:
    """An absolute path, ~, or a path relative to the module's assets/."""
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = ASSETS_DIR / path
    if not path.is_file():
        raise ValueError(f"{label}: file {path} not found")
    return path


def is_urdf(url: str) -> bool:
    """.urdf, .xml and .xacro are URDF; anything else is SVA JSON."""
    return os.path.splitext(urllib.parse.urlparse(url).path)[1].lower() in (".urdf", ".xml", ".xacro")


def load_kinematics(url: str, label: str) -> bytes:
    """A kinematics file's bytes. http(s) URLs are fetched once and cached on
    disk; file:// URLs and paths (resolve_path) are read every time, so edits
    show up when the arm is rebuilt."""
    if url.startswith("file://"):
        try:
            with urllib.request.urlopen(url) as resp:
                return resp.read()
        except OSError as e:  # URLError/HTTPError/timeouts are OSError
            raise ValueError(f"{label}: cannot read {url}: {e}") from e
    if not url.startswith(("http://", "https://")):
        return resolve_path(url, label).read_bytes()
    cache_dir = os.environ.get("VIAM_MODULE_DATA") or tempfile.gettempdir()
    ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower()
    cache = os.path.join(cache_dir, f"kinematics-{hashlib.sha1(url.encode()).hexdigest()[:12]}{ext}")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return f.read()
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
    except OSError as e:  # URLError/HTTPError/timeouts are OSError
        raise ValueError(f"{label}: cannot read {url}: {e}") from e
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = cache + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, cache)
    except OSError:
        pass  # caching is best-effort
    return data


@dataclass(frozen=True)
class Frame:
    name: str
    parent: str  # "world" for the chain's root
    offset: np.ndarray  # 4x4 from the parent, applied before the joint's motion
    axis: np.ndarray | None = None  # set for a moving joint
    prismatic: bool = False
    geometry: Mapping[str, Any] | None = None  # SVA geometry, mm


@dataclass(frozen=True)
class Kinematics:
    frames: dict[str, Frame]  # parents before children
    joints: tuple[str, ...]  # moving joints, in chain order
    end: str  # the end-effector frame

    def forward(self, q: Sequence[float]) -> dict[str, np.ndarray]:
        """Every frame's pose relative to the kinematics' world (the arm
        frame), for joint values in chain order (rad, or m for prismatic)."""
        values = dict(zip(self.joints, q, strict=True))
        poses = {"world": np.eye(4)}
        for frame in self.frames.values():
            pose = poses[frame.parent] @ frame.offset
            if frame.axis is not None:
                value = float(values[frame.name])
                if frame.prismatic:
                    pose = pose @ transform(pos=frame.axis * value)
                else:
                    pose = pose @ _axis_angle(frame.axis, value)
            poses[frame.name] = pose
        return poses


def _sva_orientation(raw: Any, where: str) -> np.ndarray:
    if not raw:
        return np.eye(4)
    kind, value = raw.get("type"), raw.get("value", {})
    if kind not in ("ov_degrees", "ov_radians"):
        raise ValueError(f"{where}: orientation type {kind!r} is not supported; use ov_degrees")
    theta = float(value.get("th", 0.0))
    if kind == "ov_degrees":
        theta = math.radians(theta)
    q = OrientationVector(
        float(value.get("x", 0.0)), float(value.get("y", 0.0)), float(value.get("z", 1.0)), theta
    ).to_quaternion()
    return transform(quat=(q.w, q.i, q.j, q.k))


def sva_pose(raw: Mapping[str, Any], where: str) -> np.ndarray:
    """An SVA link's or geometry's translation (mm) and orientation as a pose."""
    t = raw.get("translation") or {}
    result = _sva_orientation(raw.get("orientation"), where)
    result[:3, 3] = [float(t.get(key, 0.0)) / 1000 for key in "xyz"]
    return result


def _chain(frames: dict[str, Frame], label: str) -> Kinematics:
    """Order parents before children and find the one chain's end."""
    children: dict[str, list[str]] = {}
    for frame in frames.values():
        if frame.parent != "world" and frame.parent not in frames:
            raise ValueError(f"{label}: frame {frame.name!r} has unknown parent {frame.parent!r}")
        children.setdefault(frame.parent, []).append(frame.name)
    ordered: dict[str, Frame] = {}
    stack = list(reversed(children.get("world", [])))
    while stack:
        name = stack.pop()
        ordered[name] = frames[name]
        stack.extend(reversed(children.get(name, [])))
    if len(ordered) != len(frames):
        raise ValueError(
            f"{label}: frames {sorted(set(frames) - set(ordered))} are not connected to world"
        )
    leaves = [name for name in ordered if name not in children]
    if len(leaves) != 1:
        raise ValueError(f"{label}: expected one chain with one end frame, found ends {leaves}")
    path, name = [], leaves[0]
    while name != "world":
        path.append(name)
        name = frames[name].parent
    joints = tuple(name for name in reversed(path) if frames[name].axis is not None)
    return Kinematics(frames=ordered, joints=joints, end=leaves[0])


def _add(frames: dict[str, Frame], frame: Frame, label: str) -> None:
    if frame.name in frames:
        raise ValueError(f"{label}: duplicate frame name {frame.name!r}")
    frames[frame.name] = frame


def _unit(axis: Sequence[float], label: str, name: str) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError(f"{label}: joint {name!r} has a zero-length axis")
    return np.asarray(axis, float) / norm


def _parse_sva(doc: Mapping[str, Any], label: str) -> Kinematics:
    frames: dict[str, Frame] = {}
    for link in doc.get("links", []):
        _add(frames, Frame(
            name=link["id"],
            parent=link["parent"],
            offset=sva_pose(link, f"{label}: link {link['id']!r}"),
            geometry=link.get("geometry"),
        ), label)
    for joint in doc.get("joints", []):
        kind = joint.get("type", "revolute")
        if kind not in ("revolute", "prismatic"):
            raise ValueError(f"{label}: joint {joint['id']!r} has unsupported type {kind!r}")
        a = joint["axis"]
        _add(frames, Frame(
            name=joint["id"],
            parent=joint["parent"],
            offset=np.eye(4),
            axis=_unit([float(a.get(key, 0.0)) for key in "xyz"], label, joint["id"]),
            prismatic=kind == "prismatic",
        ), label)
    return _chain(frames, label)


def _numbers(text: str | None, default: str) -> list[float]:
    return [float(v) for v in (text or default).split()]


def _parse_urdf(root: ElementTree.Element, label: str) -> Kinematics:
    frames: dict[str, Frame] = {}
    joints = root.findall("joint")
    children = {joint.find("child").get("link") for joint in joints}
    for link in root.findall("link"):
        if link.get("name") not in children:
            _add(frames, Frame(name=link.get("name"), parent="world", offset=np.eye(4)), label)
    for joint in joints:
        name, kind = joint.get("name"), joint.get("type")
        origin = joint.find("origin")
        xyz = _numbers(origin.get("xyz") if origin is not None else None, "0 0 0")
        roll, pitch, yaw = _numbers(origin.get("rpy") if origin is not None else None, "0 0 0")
        offset = _axis_angle((0, 0, 1), yaw) @ _axis_angle((0, 1, 0), pitch) @ _axis_angle((1, 0, 0), roll)
        offset[:3, 3] = xyz
        axis = None
        if kind in ("revolute", "continuous", "prismatic"):
            element = joint.find("axis")
            axis = _unit(_numbers(element.get("xyz") if element is not None else None, "1 0 0"), label, name)
        elif kind != "fixed":
            raise ValueError(f"{label}: joint {name!r} has unsupported type {kind!r}")
        _add(frames, Frame(
            name=name,
            parent=joint.find("parent").get("link"),
            offset=offset,
            axis=axis,
            prismatic=kind == "prismatic",
        ), label)
        child = joint.find("child").get("link")
        _add(frames, Frame(name=child, parent=name, offset=np.eye(4)), label)
    return _chain(frames, label)


def parse_kinematics(data: bytes, urdf: bool, label: str) -> Kinematics:
    """An SVA JSON or URDF kinematics file as one chain. Only SVA links carry
    geometry."""
    try:
        if urdf:
            return _parse_urdf(ElementTree.fromstring(data), label)
        return _parse_sva(json.loads(data), label)
    except (KeyError, TypeError, AttributeError, ElementTree.ParseError, json.JSONDecodeError) as error:
        kind = "URDF" if urdf else "SVA JSON"
        raise ValueError(f"{label}: not a valid {kind} kinematics file ({error!r})") from None



# ----------------------------------------------------------------------------
# MJCF inference
# ----------------------------------------------------------------------------


def load_mjcf(path: Path | str) -> mujoco.MjSpec:
    """The MJCF as a spec, every actuator named (Menagerie leaves some
    unnamed; they take their target's name) so state carries across rebuilds.
    Every body is gravity-compensated, as a real arm controller does: without
    it, position servos sag under load (the UR5e shoulder by 0.6 degrees) and
    moves never converge."""
    spec = mujoco.MjSpec.from_file(str(path))
    for body in spec.bodies:
        body.gravcomp = 1
    for actuator in spec.actuators:
        if not actuator.name:
            actuator.name = actuator.target
    return spec


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or f"#{index}"


def _is_ancestor(model: mujoco.MjModel, ancestor: int, body: int) -> bool:
    while body != ancestor and body != 0:
        body = int(model.body_parentid[body])
    return body == ancestor


def arm_joints(
    model: mujoco.MjModel, count: int, label: str, names: Sequence[str] | None = None
) -> list[int]:
    """The arm's joint ids, in chain order: `names` if given, else the first
    `count` actuated hinge/slide joints in tree order, which must form one chain."""
    if names is not None:
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
        missing = [name for name, index in zip(names, ids) if index < 0]
        if missing:
            raise ValueError(f"{label}: joints {missing} are not in the MJCF")
        if len(ids) != count:
            raise ValueError(f"{label}: joints lists {len(ids)} joints but the kinematics file has {count}")
        return ids
    actuated = sorted({
        int(model.actuator_trnid[a, 0]) for a in range(model.nu)
        if model.actuator_trntype[a] == _TRN_JOINT
        and model.jnt_type[model.actuator_trnid[a, 0]] in (_HINGE, _SLIDE)
    })
    found = [_name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in actuated]
    if len(actuated) < count:
        raise ValueError(
            f"{label}: found {len(actuated)} actuated joints {found}, but the kinematics "
            f"file has {count}; set joints to the arm's joint names"
        )
    chosen = actuated[:count]
    for before, after in itertools.pairwise(chosen):
        if not _is_ancestor(model, int(model.jnt_bodyid[before]), int(model.jnt_bodyid[after])):
            raise ValueError(
                f"{label}: the first {count} actuated joints {found[:count]} are not one chain "
                f"({found[chosen.index(after)]} is not below {found[chosen.index(before)]}); "
                "set joints to the arm's joint names"
            )
    return chosen


def control_type(model: mujoco.MjModel, actuator: int, label: str) -> str:
    """"position" or "force"; ValueError for anything else."""
    trn = int(model.actuator_trntype[actuator])
    bias = int(model.actuator_biastype[actuator])
    if (
        trn in (_TRN_JOINT, _TRN_TENDON)
        and model.actuator_gaintype[actuator] == _GAIN_FIXED
        and model.actuator_dyntype[actuator] in _DYN_OK
    ):
        if bias == _BIAS_NONE:
            return "force"
        if bias == _BIAS_AFFINE and model.actuator_biasprm[actuator, 1] < 0:
            return "position"
    if trn == int(mujoco.mjtTrn.mjTRN_BODY):
        what = "an adhesion"
    elif bias == _BIAS_AFFINE and model.actuator_biasprm[actuator, 1] == 0:
        what = "a velocity"
    else:
        what = "an unsupported"
    raise ValueError(
        f"{label}: actuator {_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)!r} is {what} "
        "actuator; supported are position (<position>, <general biastype=\"affine\">) "
        "and force (<motor>, <general>)"
    )


def arm_actuators(model: mujoco.MjModel, joints: Sequence[int], label: str) -> list[int]:
    """The position actuator driving each arm joint, matched by transmission target."""
    result = []
    for joint in joints:
        found = [
            a for a in range(model.nu)
            if model.actuator_trntype[a] == _TRN_JOINT and model.actuator_trnid[a, 0] == joint
        ]
        if len(found) != 1:
            raise ValueError(
                f"{label}: joint {_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)!r} needs exactly "
                f"one actuator, found {len(found)}"
            )
        if control_type(model, found[0], label) != "position":
            raise ValueError(
                f"{label}: arm actuator {_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, found[0])!r} "
                "must be a position actuator; the arm writes joint positions to ctrl"
            )
        a = found[0]
        if model.actuator_gear[a, 0] != 1 or not math.isclose(model.actuator_gainprm[a, 0], -model.actuator_biasprm[a, 1]):
            raise ValueError(
                f"{label}: arm actuator {_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)!r} "
                "must have gear 1 and gain == kp so ctrl is the joint position"
            )
        result.append(a)
    return result


def home(model: mujoco.MjModel, keyframe: str, actuators: Sequence[int]) -> np.ndarray:
    """The keyframe's ctrl for the given actuators; zeros without one. A keyframe
    that sets only qpos (MuJoCo zero-fills its ctrl) gives the joints' qpos instead."""
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    if key < 0:
        return np.zeros(len(actuators))
    ctrl = model.key_ctrl[key, list(actuators)].copy()
    qpos = model.key_qpos[key, model.jnt_qposadr[model.actuator_trnid[list(actuators), 0]]]
    return qpos.copy() if not ctrl.any() and qpos.any() else ctrl


def gripper_actuator(
    model: mujoco.MjModel, label: str, exclude: Sequence[int] = (), name: str | None = None
) -> int:
    """`name` if given, else the single actuator not in `exclude` (the arm's)."""
    if name is not None:
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator < 0:
            raise ValueError(f"{label}: actuator {name!r} is not in the MJCF")
        return actuator
    arm = set(exclude)
    found = [a for a in range(model.nu) if a not in arm]
    if not found:
        raise ValueError(f"{label}: has no gripper actuator")
    if len(found) > 1:
        names = [_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in found]
        raise ValueError(f"{label}: has more than one gripper actuator {names}; set actuator")
    return found[0]


@dataclass(frozen=True)
class Jaw:
    joint: int
    open: float  # joint value the open command drives toward
    closed: float


def jaw(model: mujoco.MjModel, actuator: int, open_ctrl: float, closed_ctrl: float, label: str) -> Jaw:
    """The joint the actuator moves (a tendon's first joint) and which end of
    its range open and closed drive toward."""
    coef = 1.0
    if model.actuator_trntype[actuator] == _TRN_TENDON:
        adr = int(model.tendon_adr[model.actuator_trnid[actuator, 0]])
        if model.wrap_type[adr] != _WRAP_JOINT:
            raise ValueError(f"{label}: the gripper actuator's tendon must start with a joint")
        joint, coef = int(model.wrap_objid[adr]), float(model.wrap_prm[adr])
    else:
        joint = int(model.actuator_trnid[actuator, 0])
    if not model.jnt_limited[joint]:
        raise ValueError(
            f"{label}: the gripper's joint {_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)!r} needs a range"
        )
    low, high = (float(v) for v in model.jnt_range[joint])
    rising = model.actuator_gainprm[actuator, 0] * model.actuator_gear[actuator, 0] * coef > 0
    if (open_ctrl < closed_ctrl) == rising:
        return Jaw(joint, low, high)
    return Jaw(joint, high, low)


def subtree(model: mujoco.MjModel, root: int) -> list[int]:
    return [b for b in range(model.nbody) if _is_ancestor(model, root, b)]


def integrated_gripper_bodies(model: mujoco.MjModel, last_arm_joint: int, jaw_joint: int) -> list[int]:
    """The child of the last arm joint's body whose subtree holds the jaw
    joint, with its subtree; the jaw body's subtree when there is no such child."""
    parent = int(model.jnt_bodyid[last_arm_joint])
    jaw_body = int(model.jnt_bodyid[jaw_joint])
    for child in range(model.nbody):
        if model.body_parentid[child] == parent and child != parent and _is_ancestor(model, child, jaw_body):
            return subtree(model, child)
    return subtree(model, jaw_body)


# ----------------------------------------------------------------------------
# collision shapes from the kinematics file
# ----------------------------------------------------------------------------

MIN_SHAPE_M = 0.005  # smaller shapes are placeholders (the Lite6's 1 mm spheres)


@dataclass(frozen=True)
class Shape:
    link: str
    body: str  # MJCF body, unprefixed
    kind: str  # "box", "sphere" or "capsule"
    size: tuple[float, ...]  # MuJoCo size, m
    pos: tuple[float, ...]
    quat: tuple[float, ...]
    real: bool  # at least MIN_SHAPE_M in every dimension


def _mj_shape(geometry: Mapping[str, Any]) -> tuple[str, tuple[float, ...], float]:
    """(kind, MuJoCo size, smallest dimension), all m, from an SVA geometry (mm)."""
    if "x" in geometry:
        full = [geometry[key] / 1000 for key in "xyz"]
        return "box", tuple(v / 2 for v in full), min(full)
    radius = geometry["r"] / 1000
    length = geometry.get("l", 0) / 1000
    if length > 2 * radius:
        return "capsule", (radius, length / 2 - radius), 2 * radius
    return "sphere", (radius,), 2 * radius


def kinematics_shapes(
    model: mujoco.MjModel,
    kin: Kinematics,
    joints: Sequence[int],
    root_body: int,
    base_offset: np.ndarray,
) -> list[Shape]:
    """Each SVA link geometry as a shape on the MuJoCo body that carries it.

    RDK attaches a link's geometry in the link's parent frame, so it moves with
    the nearest joint at or above that parent; it goes on the body owning the
    matching MJCF joint (the root body when there is none). Poses are taken
    with every arm joint at zero."""
    data = mujoco.MjData(model)
    data.qpos[model.jnt_qposadr[list(joints)]] = 0.0
    mujoco.mj_kinematics(model, data)
    viam = kin.forward([0.0] * len(kin.joints))
    to_mj = np.linalg.inv(base_offset)
    shapes = []
    for frame in kin.frames.values():
        if not frame.geometry:
            continue
        rider = frame.parent
        while rider != "world" and kin.frames[rider].axis is None:
            rider = kin.frames[rider].parent
        body = root_body if rider == "world" else int(model.jnt_bodyid[joints[kin.joints.index(rider)]])
        body_pose = transform(data.xpos[body], data.xquat[body])
        geometry_pose = viam[frame.parent] @ sva_pose(frame.geometry, frame.name)
        local = np.linalg.inv(body_pose) @ to_mj @ geometry_pose
        kind, size, smallest = _mj_shape(frame.geometry)
        shapes.append(Shape(
            link=frame.name,
            body=_name(model, mujoco.mjtObj.mjOBJ_BODY, body),
            kind=kind, size=size,
            pos=tuple(local[:3, 3]), quat=tuple(quat(local)),
            real=smallest >= MIN_SHAPE_M,
        ))
    return shapes


def apply_shapes(spec: mujoco.MjSpec, shapes: Sequence[Shape], label: str) -> None:
    """Add the shapes to an unattached arm spec, colliding like the body's
    existing collision geoms (1/1 if it has none), and turn those geoms off on
    every body that got a real shape."""
    bits = {}
    for name in {shape.body for shape in shapes}:
        if spec.body(name) is None:
            raise ValueError(
                f'{label}: arm body {name} has no name; name the MJCF\'s link bodies '
                'or set collision to "mesh"'
            )
        colliding = [g for g in spec.body(name).geoms if g.contype or g.conaffinity]
        bits[name] = (colliding[0].contype, colliding[0].conaffinity) if colliding else (1, 1)
        if any(shape.real for shape in shapes if shape.body == name):
            for geom in colliding:
                geom.contype = geom.conaffinity = 0
    for shape in shapes:
        contype, conaffinity = bits[shape.body]
        spec.body(shape.body).add_geom(
            name=f"viam_{shape.link}",
            type=GEOM_TYPES[shape.kind],
            size=[*shape.size, 0, 0][:3],
            pos=list(shape.pos),
            quat=list(shape.quat),
            mass=0,
            group=3,
            contype=contype,
            conaffinity=conaffinity,
        )


def home_contacts(spec: mujoco.MjSpec, joints: Sequence[str], q: Sequence[float]) -> list[tuple[str, str]]:
    """Body pairs in contact with the joints at `q`, in a compiled copy of `spec`."""
    model = spec.compile()
    data = mujoco.MjData(model)
    for name, value in zip(joints, q):
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    pairs = set()
    for contact in data.contact[: data.ncon]:
        bodies = sorted(int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
        pairs.add(tuple(_name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies))
    return sorted(pairs)


# ----------------------------------------------------------------------------
# frame check
# ----------------------------------------------------------------------------

FRAME_TOLERANCE_M = 0.005
FRAME_TOLERANCE_DEG = 2.0


def check_frames(
    model: mujoco.MjModel,
    kin: Kinematics,
    joints: Sequence[int],
    flange_site: int,
    home_q: Sequence[float],
    base_offset: np.ndarray,
    flange_offset: np.ndarray,
    label: str,
) -> None:
    """Compare the MJCF flange (with the offsets) and Viam's end effector, in
    the arm frame, at zeros, home and a mixed pose; ValueError when they
    disagree by more than 5 mm or 2 degrees, suggesting an offset that fixes it."""
    low, high = model.jnt_range[list(joints)].T
    limited = model.jnt_limited[list(joints)].astype(bool)
    mixed = 0.5 * (-1.0) ** np.arange(len(joints))
    mixed = np.where(limited, np.clip(mixed, low, high), mixed)
    poses = {"zeros": np.zeros(len(joints)), "home": np.asarray(home_q, float), "mixed": mixed}
    data = mujoco.MjData(model)
    sim, viam = {}, {}
    for key, q in poses.items():
        data.qpos[model.jnt_qposadr[list(joints)]] = q
        mujoco.mj_kinematics(model, data)
        site = np.eye(4)
        site[:3, :3] = data.site_xmat[flange_site].reshape(3, 3)
        site[:3, 3] = data.site_xpos[flange_site]
        sim[key] = base_offset @ site @ flange_offset
        viam[key] = kin.forward(q)[kin.end]

    def worst(fix) -> tuple[str, float, float]:
        errors = {key: pose_error(fix(sim[key]), viam[key]) for key in poses}
        key = max(errors, key=lambda k: errors[k][0] / FRAME_TOLERANCE_M + errors[k][1] / FRAME_TOLERANCE_DEG)
        return key, *errors[key]

    def ok(fix) -> bool:
        _, metres, degrees = worst(fix)
        return metres <= FRAME_TOLERANCE_M and degrees <= FRAME_TOLERANCE_DEG

    if ok(lambda pose: pose):
        return
    key, metres, degrees = worst(lambda pose: pose)
    message = (
        f"{label}: the MJCF flange and the kinematics file's end effector disagree by "
        f"{metres * 1000:.1f} mm and {degrees:.1f} deg at the {key} joint position "
        f"{np.round(poses[key], 3).tolist()} (tolerance {FRAME_TOLERANCE_M * 1000:.0f} mm, "
        f"{FRAME_TOLERANCE_DEG:.0f} deg)"
    )
    flange_fix = np.linalg.inv(sim["zeros"]) @ viam["zeros"]
    base_fix = viam["zeros"] @ np.linalg.inv(sim["zeros"])
    if ok(lambda pose: pose @ flange_fix):
        message += f"; try flange_offset: {format_pose(flange_offset @ flange_fix)}"
    elif ok(lambda pose: base_fix @ pose):
        message += f"; try base_offset: {format_pose(base_fix @ base_offset)}"
    raise ValueError(message)
