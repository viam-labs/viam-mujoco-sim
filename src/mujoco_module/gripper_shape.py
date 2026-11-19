"""A gripper's collision shape for Viam's motion planner.

RDK asks a gripper for GetKinematics, falls back to GetGeometries, and builds
a zero-DoF model from what comes back (rdk components/gripper/client.go), so a
gripper's served geometries are all the planner knows of its shape. These are
the gripper file's colliding geoms, settled with the jaw open and with it
closed (the closed 2F-85's pads reach 13 mm further), in the gripper's Viam
frame; meshes as convex hulls of at most HULL_VERTICES vertices, primitives
as RDK's own shapes.
"""

from types import SimpleNamespace
from typing import Sequence

import mujoco
import numpy as np
from viam.proto.common import Geometry

from .scenes import _label, geom_geometry

HULL_VERTICES = 32  # per mesh hull: the 2F-85 serves ~1,080 faces (finger links twice), not ~11,000
SETTLE_STEPS = (50, 5000)  # at least, at most
SETTLED_RAD_S = 1e-4  # every joint slower than this ...
SETTLED_FOR = 20  # ... for this many steps in a row (one slow step can be a turning point)
MOVED_M = 0.0005  # a geom that moves more than this between the jaw poses is served at both,
MOVED_RAD = 0.001  # or turns more than this


def collision_geometries(
    spec: mujoco.MjSpec,
    actuator: int,
    commands: Sequence[float],
    frame: np.ndarray,
    bodies: Sequence[int] | None = None,
    site: str | None = None,
    ctrl: Sequence[float] | None = None,
) -> list[Geometry]:
    """The colliding geoms of a gripper's own file (attached at its mount's
    flange) at each actuator command in `commands`, as Viam geometries in mm,
    moved by `frame` (4x4, metres: from the file's world to the gripper's
    Viam frame). A geom that moves between the poses is served at each, its
    label suffixed with the pose's index name (`:open`, `:closed`).

    A gripper in its arm's (or robot's) file: `bodies` are the gripper's own (ids
    in that file, compiled as the arm compiled it; a finger body may be unnamed),
    `site` is the arm's flange site (the geoms are measured from it, as the file's
    world is for an attached gripper) and `ctrl` the file's home controls, which
    hold everything but the gripper's actuator."""
    copy = spec.copy()  # the scene's hulls are the physics' contacts: reduce only this one's
    for mesh in copy.meshes:
        mesh.maxhullvert = HULL_VERTICES
    copy.option.gravity = [0, 0, 0]
    # as the scene integrates: Euler goes unstable on a light jaw under a stiff, damped servo
    copy.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model = copy.compile()
    poses = [_settled(model, actuator, command, ctrl) for command in commands]
    keep = None if bodies is None else set(bodies)
    geoms = [
        g for g in range(model.ngeom)
        if (keep is None or int(model.geom_bodyid[g]) in keep)
        and (model.geom_contype[g] or model.geom_conaffinity[g])
        and model.geom_type[g] not in (  # no Viam shape for these
            mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD, mujoco.mjtGeom.mjGEOM_SDF,
        )
    ]
    names = ["open", "closed"]  # commands: open, then closed when the gripper has a jaw
    out = []
    for geom in geoms:
        label = _label(model, geom)
        moved = len(poses) > 1 and any(
            np.linalg.norm(p.geom_xpos[geom] - poses[0].geom_xpos[geom]) > MOVED_M
            or np.abs(p.geom_xmat[geom] - poses[0].geom_xmat[geom]).max() > MOVED_RAD  # ~ radians, small turns
            for p in poses[1:]
        )
        for pose, name in zip(poses if moved else poses[:1], names):
            moved_by = frame if site is None else frame @ np.linalg.inv(_site_pose(model, pose, site))
            out.append(geom_geometry(model, _moved(pose, moved_by), geom, f"{label}:{name}" if moved else label))
    return out


def _site_pose(model: mujoco.MjModel, data: mujoco.MjData, site: str) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = data.site(site).xmat.reshape(3, 3)
    pose[:3, 3] = data.site(site).xpos
    return pose


def _settled(
    model: mujoco.MjModel, actuator: int, command: float, ctrl: Sequence[float] | None = None
) -> mujoco.MjData:
    """The gripper held at `command` (the other actuators at `ctrl`) until its
    joints stop (gravity off, base fixed): its linkage's equality constraints
    settle with it."""
    data = mujoco.MjData(model)
    if ctrl is not None:
        data.ctrl[:] = ctrl
    data.ctrl[actuator] = command
    still = 0
    for step in range(SETTLE_STEPS[1]):
        mujoco.mj_step(model, data)
        still = still + 1 if model.nv == 0 or np.abs(data.qvel).max() < SETTLED_RAD_S else 0
        if step >= SETTLE_STEPS[0] and still >= SETTLED_FOR:
            break
    mujoco.mj_forward(model, data)
    return data


def _moved(data: mujoco.MjData, frame: np.ndarray) -> SimpleNamespace:
    """The geom poses geom_geometry reads, moved by `frame`."""
    rotation, offset = frame[:3, :3], frame[:3, 3]
    xmat = data.geom_xmat.reshape(-1, 3, 3)
    return SimpleNamespace(
        geom_xpos=data.geom_xpos @ rotation.T + offset,
        geom_xmat=(rotation @ xmat).reshape(-1, 9),
    )
