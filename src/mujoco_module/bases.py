"""What the module needs to know about a mobile base, read from its MJCF or
URDF file.

Pure functions over the file's spec and its compiled model (names as in the
file, unprefixed, the chassis at the origin); nothing here touches the
running sim. Lengths are metres, angles radians.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np


_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
_TRN_JOINT = int(mujoco.mjtTrn.mjTRN_JOINT)
_ROUND = (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_SPHERE))
_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
PARALLEL = math.cos(math.radians(5))  # axles within 5 degrees count as parallel
RADIUS_TOLERANCE = 0.05
FLOOR_FRICTION = 1.0  # the world's floor geom: MuJoCo's default, priority 0
FORWARD_AXES = {"+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0), "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0)}


@dataclass(frozen=True)
class Wheel:
    joint: str
    left: bool
    sign: float  # +1 when a positive joint speed drives the base forward


@dataclass(frozen=True)
class SwerveModule:
    steer: str  # the steering hinge (vertical axis), on a body fixed to the chassis
    drive: str  # the wheel's hinge, below it
    position: tuple[float, float]  # the steer axis in the viam frame, m
    roll0: float  # the way a positive drive speed rolls the wheel at steer 0: viam frame angle, rad
    steer_sign: float  # +1 when a positive steer angle turns the wheel left (counterclockwise from above)


@dataclass(frozen=True)
class Base:
    chassis: str
    wheels: tuple[Wheel, ...]
    radius: float
    track: float
    forward: str  # the file's forward axis, a FORWARD_AXES key
    yaw: float  # turns the file's forward axis onto the viam frame's +Y
    lift: float  # chassis origin above the floor
    mass: float  # whole base
    izz: float  # whole base, about the chassis origin's vertical axis
    box: tuple[np.ndarray, np.ndarray] | None  # chassis collision (center, size), viam frame
    kinematics: str = "differential"  # or "swerve"
    modules: tuple[SwerveModule, ...] = ()  # swerve only; wheels is then empty


def take_chassis(spec: mujoco.MjSpec, name: str | None, label: str) -> str:
    """Leave only the chassis under the worldbody: its single top-level body,
    or `name`. The chassis's free joint and its pose in the file are dropped
    (the module places it and adds the drive's joints). Returns its name."""
    roots = [body.name for body in spec.worldbody.bodies]
    if name is None:
        if len(roots) != 1:
            raise ValueError(
                f"{label}: has {len(roots)} top-level bodies {roots}; set chassis to the base's body"
            )
        name = roots[0]
    elif name not in roots:
        raise ValueError(f"{label}: chassis {name!r} is not a top-level body; use one of {roots}")
    for body in list(spec.worldbody.bodies):
        if body.name != name:
            spec.delete(body)
    for geom in list(spec.worldbody.geoms):  # a floor in the file
        spec.delete(geom)
    chassis = spec.body(name)
    for joint in list(chassis.joints):
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            spec.delete(joint)
    chassis.pos = [0.0, 0.0, 0.0]
    chassis.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
    chassis.quat = [1.0, 0.0, 0.0, 0.0]
    return name


def _name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or f"#{index}"


def static_bodies(model: mujoco.MjModel) -> set[int]:
    """The chassis (body 1) and the jointless bodies fixed to it."""
    found = {1}
    for body in range(2, model.nbody):  # parents come before children
        if model.body_jntnum[body] == 0 and model.body_parentid[body] in found:
            found.add(body)
    return found


def _corners(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> np.ndarray:
    """The 8 corners of a geom's bounding box, world frame."""
    center, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
    local = center + np.array(list(itertools.product((-1, 1), repeat=3))) * half
    return data.geom_xpos[geom] + local @ data.geom_xmat[geom].reshape(3, 3).T


def _collides(model: mujoco.MjModel, geom: int) -> bool:
    return bool(model.geom_contype[geom] or model.geom_conaffinity[geom])


def _radius(model: mujoco.MjModel, data: mujoco.MjData, joint: int) -> float | None:
    """The wheel body's cylinder or sphere radius, else its mesh's furthest
    point from the axle."""
    body = model.jnt_bodyid[joint]
    geoms = range(model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body])
    for geom in geoms:
        if model.geom_type[geom] in _ROUND:
            return float(model.geom_size[geom, 0])
    for geom in geoms:
        if model.geom_type[geom] == _MESH:
            mesh = model.geom_dataid[geom]
            start = model.mesh_vertadr[mesh]
            vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
            points = data.geom_xpos[geom] + vertices @ data.geom_xmat[geom].reshape(3, 3).T
            rel = points - data.xanchor[joint]
            axis = data.xaxis[joint]
            return float(np.max(np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)))
    return None


def infer(
    model: mujoco.MjModel,
    label: str,
    wheels: Mapping[str, Sequence[str]] | None = None,
    forward_axis: str | None = None,
    wheel_radius: float | None = None,
    kinematics: str | None = None,
) -> Base:
    """Driven wheels, sides, radius, track, forward axis and placement of the
    base in `model` (from take_chassis: body 1 is the chassis, at the origin).
    `wheels` ({"left": [...], "right": [...]}), `forward_axis` and
    `wheel_radius` override the inference. A swerve base (`kinematics`
    "swerve", or inferred from two or more actuated steer hinges) has a
    SwerveModule per steer hinge instead of left and right wheels."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    hinges = [j for j in range(model.njnt) if model.jnt_type[j] == _HINGE]
    hinge_names = [_name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinges]
    steers = _steer_hinges(model, data, hinges)
    if kinematics is None:
        actuated = {int(model.actuator_trnid[a, 0]) for a in range(model.nu) if model.actuator_trntype[a] == _TRN_JOINT}
        kinematics = "swerve" if sum(s in actuated for s in steers) >= 2 else "differential"
        if kinematics == "swerve":
            steers = {s: d for s, d in steers.items() if s in actuated}
    if kinematics == "swerve":
        if len(steers) < 2:
            raise ValueError(
                f'{label}: kinematics "swerve" needs steer hinges (a vertical hinge on a body fixed to '
                f"the chassis with one wheel hinge below it), found {len(steers)} among {hinge_names}"
            )
        return _swerve(model, data, label, steers, forward_axis, wheel_radius)
    if wheels is not None:
        sides = {}
        for side in ("left", "right"):
            sides[side] = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in wheels[side]]
        unknown = [
            n for side in ("left", "right") for n, j in zip(wheels[side], sides[side]) if j not in hinges
        ]
        if unknown:
            raise ValueError(f"{label}: wheels {unknown} are not hinge joints; use names from {hinge_names}")
        driven = sides["left"] + sides["right"]
    else:
        static = static_bodies(model)
        actuated = {int(model.actuator_trnid[a, 0]) for a in range(model.nu) if model.actuator_trntype[a] == _TRN_JOINT}
        driven = [
            j for j in hinges
            if model.body_parentid[model.jnt_bodyid[j]] in static
            and abs(data.xaxis[j][2]) < 0.1
            and (not actuated or j in actuated)
        ]
    names = {j: _name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in driven}
    if driven:
        axle = data.xaxis[driven[0]]
        if any(abs(data.xaxis[j] @ axle) < PARALLEL for j in driven):
            axes = {names[j]: [round(float(v), 3) for v in data.xaxis[j]] for j in driven}
            raise ValueError(f"{label}: wheel axles are not parallel: {axes}; set wheels")
        if forward_axis is None:
            if abs(axle[1]) >= PARALLEL:
                forward_axis = "+x"  # ROS
            elif abs(axle[0]) >= PARALLEL:
                forward_axis = "+y"  # Viam
            else:
                raise ValueError(
                    f"{label}: cannot tell forward from wheel axle {[round(float(v), 3) for v in axle]}; "
                    "set forward_axis"
                )
    forward = np.array(FORWARD_AXES[forward_axis or "+y"])
    lateral = np.cross((0.0, 0.0, 1.0), forward)  # left
    if driven and abs(axle @ lateral) < PARALLEL:
        raise ValueError(
            f"{label}: forward_axis {forward_axis} is not across the wheel axles "
            f"{[round(float(v), 3) for v in axle]}; set forward_axis"
        )
    if wheels is not None:
        left = set(sides["left"])
    else:
        left = {j for j in driven if data.xanchor[j] @ lateral > 0}
    right = [j for j in driven if j not in left]
    if not left or not right:
        raise ValueError(
            f"{label}: needs driven wheels on both sides, found left "
            f"{[names[j] for j in driven if j in left]} and right {[names[j] for j in right]} "
            f"among hinge joints {hinge_names}; set wheels"
        )
    offsets = {j: float(data.xanchor[j] @ lateral) for j in driven}
    track = abs(
        float(np.mean([offsets[j] for j in left])) - float(np.mean([offsets[j] for j in right]))
    )
    if track < 1e-3:
        raise ValueError(f"{label}: left and right wheels are at the same lateral offset (track {track:.4f} m); set wheels")
    if wheel_radius is None:
        radii = {names[j]: _radius(model, data, j) for j in driven}
        missing = [n for n, r in radii.items() if r is None]
        if missing:
            raise ValueError(
                f"{label}: cannot find the radius of wheels {missing} (no cylinder, sphere or mesh "
                "geom); set wheel_radius_m"
            )
        values = list(radii.values())
        if max(values) > (1 + RADIUS_TOLERANCE) * min(values):
            shown = {n: round(r, 4) for n, r in radii.items()}
            raise ValueError(f"{label}: wheel radii differ by more than 5%: {shown}; set wheel_radius_m")
        wheel_radius = float(np.mean(values))
    return _finish(
        model, data,
        wheels=tuple(Wheel(names[j], j in left, 1.0 if data.xaxis[j] @ lateral > 0 else -1.0) for j in driven),
        radius=float(wheel_radius), track=track, forward_axis=forward_axis or "+y",
    )


def _finish(model: mujoco.MjModel, data: mujoco.MjData, **fields: Any) -> Base:
    """The Base with what every drive shares: placement, mass, yaw inertia, chassis box."""
    forward = np.array(FORWARD_AXES[fields["forward_axis"]])
    collision = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0 and _collides(model, g)]
    lift = -min((float(_corners(model, data, g)[:, 2].min()) for g in collision), default=0.0)
    yaw = math.atan2(forward[0], forward[1])
    izz = 0.0
    for body in range(1, model.nbody):
        rotation = data.ximat[body].reshape(3, 3)
        izz += float((rotation @ np.diag(model.body_inertia[body]) @ rotation.T)[2, 2])
        izz += float(model.body_mass[body] * (data.xipos[body, 0] ** 2 + data.xipos[body, 1] ** 2))
    chassis_geoms = _chassis_geoms(model)
    box = None
    if chassis_geoms:
        points = _to_viam(np.vstack([_corners(model, data, g) for g in chassis_geoms]), yaw, lift)
        low, high = points.min(axis=0), points.max(axis=0)
        box = ((low + high) / 2, high - low)
    forward_axis = fields.pop("forward_axis")
    return Base(
        chassis=model.body(1).name, forward=forward_axis, yaw=yaw, lift=lift,
        mass=float(model.body_subtreemass[1]), izz=izz, box=box, **fields,
    )


def _steer_hinges(model: mujoco.MjModel, data: mujoco.MjData, hinges: Sequence[int]) -> dict[int, int]:
    """Steer hinge -> its drive hinge: a vertical hinge on a body fixed to the
    chassis, with exactly one horizontal hinge in the bodies below it."""
    static = static_bodies(model)
    found = {}
    for j in hinges:
        body = int(model.jnt_bodyid[j])
        if model.body_parentid[body] not in static or abs(data.xaxis[j][2]) < PARALLEL:
            continue
        below = [h for h in hinges if h != j and _under(model, int(model.jnt_bodyid[h]), body)
                 and abs(data.xaxis[h][2]) < 0.1]
        if len(below) == 1:
            found[j] = below[0]
    return found


def _under(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


def _swerve(
    model: mujoco.MjModel, data: mujoco.MjData, label: str, steers: Mapping[int, int],
    forward_axis: str | None, wheel_radius: float | None,
) -> Base:
    """A swerve base: one SwerveModule per steer hinge. Forward can't come from
    axles that steer, so it is +X (ROS) unless forward_axis says otherwise."""
    forward_axis = forward_axis or "+x"
    forward = np.array(FORWARD_AXES[forward_axis])
    yaw = math.atan2(forward[0], forward[1])
    c, s = math.cos(yaw), math.sin(yaw)
    turn = np.array([[c, -s], [s, c]])
    modules = []
    for steer, drive in steers.items():
        axle = data.xaxis[drive]
        roll = np.cross(axle, (0.0, 0.0, 1.0))[:2] @ turn.T  # a positive drive speed's rolling direction
        position = turn @ data.xanchor[steer][:2]
        modules.append(SwerveModule(
            steer=_name(model, mujoco.mjtObj.mjOBJ_JOINT, steer),
            drive=_name(model, mujoco.mjtObj.mjOBJ_JOINT, drive),
            position=(float(position[0]), float(position[1])),
            roll0=float(math.atan2(roll[1], roll[0])),
            steer_sign=1.0 if data.xaxis[steer][2] > 0 else -1.0,
        ))
    if wheel_radius is None:
        radii = {m.drive: _radius(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, m.drive))
                 for m in modules}
        missing = [n for n, r in radii.items() if r is None]
        if missing:
            raise ValueError(f"{label}: cannot find the radius of wheels {missing}; set wheel_radius_m")
        values = list(radii.values())
        if max(values) > (1 + RADIUS_TOLERANCE) * min(values):
            raise ValueError(f"{label}: wheel radii differ by more than 5%: {radii}; set wheel_radius_m")
        wheel_radius = float(np.mean(values))
    xs = [m.position[0] for m in modules]
    return _finish(
        model, data, wheels=(), radius=float(wheel_radius), track=max(xs) - min(xs) or 1e-3,
        forward_axis=forward_axis, kinematics="swerve", modules=tuple(modules),
    )


SWERVE_STILL_MPS = 1e-4  # a module slower than this keeps its steer angle


def swerve_targets(
    modules: Sequence[SwerveModule], u: float, v: float, w: float, previous: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Steer angles (rad, the joints' own, unwrapped) and wheel surface speeds (m/s)
    for the body velocity (u right, v forward, m/s; w left, rad/s) in the viam
    frame. Each module turns to its own velocity, v + w x position, by whichever
    of the two wheel directions is the shorter turn from `previous` (reversing
    the wheel for the other); a module that would barely move keeps its angle."""
    angles, speeds = np.array(previous, float), np.zeros(len(modules))
    for i, m in enumerate(modules):
        vx, vy = u - w * m.position[1], v + w * m.position[0]
        speed = math.hypot(vx, vy)
        if speed < SWERVE_STILL_MPS:
            continue
        target = m.steer_sign * (math.atan2(vy, vx) - m.roll0)
        # the nearest of target + k*pi to where the joint is; odd k rolls the wheel backward
        k = round((angles[i] - target) / math.pi)
        angles[i], speeds[i] = target + k * math.pi, speed * (-1.0 if k % 2 else 1.0)
    return angles, speeds


def swerve_rolls(
    modules: Sequence[SwerveModule], angles: Sequence[float], u: float, v: float, w: float
) -> np.ndarray:
    """Each wheel's surface speed (m/s) for the body velocity (u, v, w) with the
    modules at `angles`: its module's velocity along the way the wheel rolls."""
    out = np.zeros(len(modules))
    for i, (m, angle) in enumerate(zip(modules, angles)):
        direction = m.roll0 + m.steer_sign * angle
        out[i] = (u - w * m.position[1]) * math.cos(direction) + (v + w * m.position[0]) * math.sin(direction)
    return out


def _chassis_geoms(model: mujoco.MjModel) -> list[int]:
    """The chassis's colliding geoms: on bodies fixed to it, not the wheels or casters."""
    static = static_bodies(model)
    return [
        g for g in range(model.ngeom)
        if model.geom_bodyid[g] != 0 and model.geom_bodyid[g] in static and _collides(model, g)
    ]


def _to_viam(points: np.ndarray, yaw: float, lift: float) -> np.ndarray:
    """Points in the file's frame (chassis at the origin) into the base's Viam frame:
    turned so the base's forward is +Y, and lifted onto the floor."""
    c, s = math.cos(yaw), math.sin(yaw)
    turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return points @ turn.T + (0.0, 0.0, lift)


def chassis_mesh(model: mujoco.MjModel, base: Base) -> bytes | None:
    """The geoms base.box bounds, as one PLY (metres) in the base's Viam frame: a mesh's
    convex hull (what MuJoCo collides), a primitive tessellated. None without any."""
    from .derive_kinematics import geom_piece
    from .meshes import pieces_ply

    flat = (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD))
    geoms = [g for g in _chassis_geoms(model) if int(model.geom_type[g]) not in flat]
    if not geoms:
        return None
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    pieces = []
    for g in geoms:
        vertices, faces = geom_piece(model, g, hull=True)
        world = vertices @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
        pieces.append((_to_viam(world, base.yaw, base.lift), faces))
    return pieces_ply(pieces)


def _is_velocity(model: mujoco.MjModel, actuator: int) -> bool:
    """ctrl is the joint's target speed: <velocity>, or <general> written as one."""
    kv = model.actuator_gainprm[actuator, 0]
    return bool(
        model.actuator_gaintype[actuator] == mujoco.mjtGain.mjGAIN_FIXED
        and model.actuator_biastype[actuator] == mujoco.mjtBias.mjBIAS_AFFINE
        and model.actuator_dyntype[actuator] == mujoco.mjtDyn.mjDYN_NONE
        and model.actuator_biasprm[actuator, 0] == 0
        and model.actuator_biasprm[actuator, 1] == 0
        and kv > 0
        and model.actuator_biasprm[actuator, 2] == -kv
        and model.actuator_gear[actuator, 0] == 1
    )


def wheel_actuators(model: mujoco.MjModel, base: Base, label: str) -> dict[str, str]:
    """The file's velocity actuator on each driven wheel that has one, by
    joint name. Any other actuator on a driven wheel is an error."""
    found: dict[str, str] = {}
    for wheel in base.wheels:
        joint = model.joint(wheel.joint).id
        for a in range(model.nu):
            if model.actuator_trntype[a] != _TRN_JOINT or model.actuator_trnid[a, 0] != joint:
                continue
            name = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            if not _is_velocity(model, a):
                raise ValueError(
                    f"{label}: actuator {name!r} on wheel {wheel.joint!r} is not a velocity actuator; "
                    'drive "wheels" needs <velocity> (or no actuator: the module adds one)'
                )
            found.setdefault(wheel.joint, name)
    return found


def speed_limits(model: mujoco.MjModel, base: Base) -> tuple[float, float] | None:
    """(m/s, deg/s) from the driven wheels' velocity actuators' ctrlrange;
    None when no driven wheel has a limited velocity actuator."""
    joints = {model.joint(name).id for name in (*(w.joint for w in base.wheels), *(m.drive for m in base.modules))}
    limits = [
        float(np.max(np.abs(model.actuator_ctrlrange[a])))
        for a in range(model.nu)
        if model.actuator_trntype[a] == _TRN_JOINT
        and model.actuator_trnid[a, 0] in joints
        and model.actuator_ctrllimited[a]
        and _is_velocity(model, a)
    ]
    if not limits:
        return None
    linear = min(limits) * base.radius
    if base.modules:  # spinning, the module furthest from the centre is the fastest wheel
        return linear, math.degrees(linear / max(math.hypot(*m.position) for m in base.modules))
    return linear, math.degrees(2 * linear / base.track)


def traction(model: mujoco.MjModel, joint: str) -> float:
    """Sliding friction of the wheel's first colliding geom against the
    floor: its own when it has the higher priority, else the larger of the two."""
    body = model.joint(joint).bodyid[0]
    for geom in range(model.body_geomadr[body], model.body_geomadr[body] + model.body_geomnum[body]):
        if _collides(model, geom):
            mu = float(model.geom_friction[geom, 0])
            return mu if model.geom_priority[geom] > 0 else max(mu, FLOOR_FRICTION)
    return FLOOR_FRICTION
