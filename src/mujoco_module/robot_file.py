"""A robot file with several limbs (viam-labs:mujoco-sim:robot), attached once.

The robot loads an MJCF file, checks it and keeps a standalone compile of it,
which its arms read their joints, actuators, home and chain from. Its build
attaches the file under "<robot>/" at its frame on the world or on its parent's
viam_frame (a base it rides). Arms bind to a
joint subset (ArmHandle with `robot`) and grippers to an arm; each claims the
file names it drives in sim.robot_claims[robot], and the robot's init sets
only what nobody claims, from its home keyframe.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mujoco
import numpy as np
from viam.proto.common import Geometry

from . import robots
from .handles import VIAM_FRAME_SITE, _mount
from .scenes import geom_geometry, _label
from .sim import Contribution, SimManager

# Bundled robot files: default attributes for the robot's `asset`. Paths are
# relative to assets/.
ROBOT_ASSETS: dict[str, dict[str, Any]] = {
    # mujoco_menagerie@370c439 aloha (BSD-3-Clause, Trossen Robotics)
    "aloha": {"mjcf": "trossen_aloha/aloha.xml"},
    # TIAGo++'s upper body, made by make_tiago.py from mujoco_menagerie@370c439
    # pal_tiago_dual (Apache-2.0, PAL Robotics); rides the pal_tiago_base base
    "tiago": {"mjcf": "pal_tiago_dual/tiago_upper.xml"},
}
REJECTED = ("flexes", "skins", "plugins")
_FREE = mujoco.mjtJoint.mjJNT_FREE
_QPOS_WIDTH = {int(mujoco.mjtJoint.mjJNT_BALL): 4}  # hinge and slide: 1; free: rejected


def check_robot_file(label: str, spec: mujoco.MjSpec) -> None:
    """ValueError for what a robot file can't have: flexes, skins, plugins,
    mocap bodies and free joints (a floating base isn't supported yet)."""
    for kind in REJECTED:
        found = list(getattr(spec, kind, ()))
        if found:
            first = getattr(found[0], "name", "") or "#0"
            raise ValueError(f"{label}: a robot file can't have {kind} (found {first!r})")
    for body in spec.bodies:
        if body.mocap:
            raise ValueError(f"{label}: a robot file can't have mocap bodies (found {body.name!r})")
        for joint in body.joints:
            if joint.type == _FREE:
                raise ValueError(
                    f"{label}: a robot file can't have a free joint yet (found one on {body.name!r}); "
                    "its limbs must hang from fixed bodies"
                )


class RobotHandle:
    """One robot file in the scene, under "<name>/", at `position` /
    `orientation_wxyz` on the world, or with attrs["parent"] relative to that
    component's viam_frame (it rides a base, and waits for it)."""

    def __init__(self, sim: SimManager, name: str, attrs: dict[str, Any]) -> None:
        attrs = {**ROBOT_ASSETS.get(str(attrs.get("asset")), {}), **attrs}
        label = f"robot {name}"
        self._sim = sim
        self.name = name
        self.prefix = f"{name}/"
        self.mjcf = robots.resolve_path(str(attrs["mjcf"]), f"{label}: mjcf")
        if self.mjcf.suffix.lower() != ".xml":
            raise ValueError(f"{label}: a robot file must be MJCF (.xml), got {self.mjcf.name}")
        spec = robots.load_mjcf(self.mjcf, None)
        check_robot_file(label, spec)
        self.model = spec.copy().compile()  # with its keyframes, for the arms' homes
        keyframe = attrs.get("home_keyframe")
        if keyframe is None:
            self._key = 0 if self.model.nkey else -1
        else:
            self._key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, str(keyframe))
            if self._key < 0:
                raise ValueError(f"{label}: the file has no keyframe {keyframe!r}")
        for key in list(spec.keys):
            spec.delete(key)  # its qpos length wouldn't match the world's
        defaults = mujoco.MjSpec().option
        # attach keeps the world's options; take the file's contact settings
        # (ALOHA's elliptic cones), as a gripper's are taken
        options = {
            key: getattr(spec.option, key) for key in ("cone", "impratio")
            if getattr(spec.option, key) != getattr(defaults, key)
        }
        position = [float(v) for v in attrs.get("position", (0.0, 0.0, 0.0))]
        quat = [float(v) for v in attrs.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0))]
        self.origin = robots.transform(position, quat)
        parent = attrs.get("parent")
        file_spec = spec

        def build(spec: mujoco.MjSpec) -> bool | None:
            placed = _mount(spec, parent, position, quat)
            if placed is None:
                return False  # parent not in the scene; the robot's arms say so
            body, pos, rot = placed
            child = file_spec.copy()  # the file isn't read again
            for key, value in options.items():
                setattr(spec.option, key, value)
            frame = body.add_frame(pos=list(pos), quat=list(rot))
            spec.attach(child, prefix=self.prefix, frame=frame)
            body.add_site(name=self.prefix + VIAM_FRAME_SITE, pos=list(pos), quat=list(rot))

        def init(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            self._home(model, data)

        def register() -> None:
            sim.robots[name] = self

        sim.set_contribution(name, Contribution(build=build, init=init))
        sim.run(register)

    def close(self) -> None:
        def detach() -> None:
            if self._sim.robots.get(self.name) is self:
                del self._sim.robots[self.name]

        self._sim.run(detach)
        self._sim.remove_contribution(self.name)

    def _home(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """The home keyframe for every joint and actuator nobody claims; the
        arms and grippers set their own, so init order doesn't matter."""
        if self._key < 0:
            return
        claimed = self._sim.robot_claims.get(self.name, {})
        own = self.model
        for joint in range(own.njnt):
            name = own.joint(joint).name
            if not name or f"joint:{name}" in claimed:
                continue
            width = _QPOS_WIDTH.get(int(own.jnt_type[joint]), 1)
            src = own.jnt_qposadr[joint]
            dst = model.joint(self.prefix + name).qposadr[0]
            data.qpos[dst : dst + width] = own.key_qpos[self._key, src : src + width]
        for actuator in range(own.nu):
            name = own.actuator(actuator).name
            if name and f"actuator:{name}" not in claimed:
                data.ctrl[model.actuator(self.prefix + name).id] = own.key_ctrl[self._key, actuator]

    def geometries(self) -> list[Geometry]:
        """The file's fixed parts, in the robot's frame, mm: colliding geoms on
        bodies welded to its root, other than the bodies a jointed body hangs
        from (a limb's chain base, which that arm's kinematics serve). RDK asks
        before any arm is configured, so this doesn't depend on them."""
        model = self.model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        bases = {int(model.body_parentid[model.jnt_bodyid[j]]) for j in range(model.njnt)}
        out = []
        for geom in range(model.ngeom):
            body = int(model.geom_bodyid[geom])
            if body == 0 or model.body_weldid[body] != 0 or body in bases:
                continue
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            if model.geom_type[geom] in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD):
                continue
            out.append(geom_geometry(model, data, geom, _label(model, geom)))
        return out


def check_claims(sim: SimManager, robot: str, names: Sequence[str], owner: str) -> None:
    """ValueError naming the holder of a claim ("joint:<name>" or
    "actuator:<name>", unprefixed file names) on `robot` that someone other
    than `owner` holds (sim thread). Claims live on the SimManager, so this
    works while the robot itself is out of the scene."""
    claims = sim.robot_claims.get(robot, {})
    for key in names:
        holder = claims.get(key)
        if holder is not None and holder != owner:
            kind, name = key.split(":", 1)
            raise ValueError(f"robot {robot}: {kind} {name} is already {holder}'s")


def claim(sim: SimManager, robot: str, names: Sequence[str], owner: str) -> None:
    """Claim file names on `robot` for `owner` (sim thread); ValueError as check_claims."""
    check_claims(sim, robot, names, owner)
    claims = sim.robot_claims.setdefault(robot, {})
    for name in names:
        claims[name] = owner


def release_claims(sim: SimManager, robot: str, owner: str) -> None:
    """Drop `owner`'s claims on `robot`'s file (sim thread); fine if the robot is gone."""
    claims = sim.robot_claims.get(robot)
    if claims:
        for name in [n for n, holder in claims.items() if holder == owner]:
            del claims[name]


def frame_error(expected: np.ndarray, label: str, has_offset: bool = False, parent: str | None = None) -> str:
    """How to configure an arm's frame at `expected` (in its robot's frame, or
    in its parent arm's end effector's)."""
    from viam.spatialmath import Quaternion

    w, x, y, z = robots.quat(expected)
    ov = Quaternion(w, x, y, z).to_orientation_vector()
    t = expected[:3, 3] * 1000
    return (
        f'{label}: set frame.translation to {{"x": {t[0]:.1f}, "y": {t[1]:.1f}, "z": {t[2]:.1f}}} and '
        f'frame.orientation to {{"type": "ov_degrees", "value": {{"x": {ov.o_x:.4f}, "y": {ov.o_y:.4f}, '
        f'"z": {ov.o_z:.4f}, "th": {np.degrees(ov.theta):.2f}}}}}: its chain base\'s pose '
        + (f"relative to arm {parent}'s end effector" if parent else "in the robot file")
        + (" with base_offset taken off (check base_offset first if that isn't the frame you meant)" if has_offset else "")
    )
