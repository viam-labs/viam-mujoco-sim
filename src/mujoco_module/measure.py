"""A gripper's TCP, jaw gap and finger reach, measured on a scratch model
(never the live sim): the numbers SKILL.md's geometry table gives for the
bundled grippers, for any gripper."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

import mujoco
import numpy as np

from . import robots

SETTLE_S = 2.0  # at most; stops early once the moving bodies sit still for QUIET_S
QUIET_S, QUIET_SPEED = 0.1, 1e-3  # m/s and rad/s
TIP_BAND_M = 0.005  # the jaw gap is measured this close to the tips
MOVED_M, MOVED_RAD = 5e-4, np.radians(1.0)


def mm(value: float) -> float:
    return round(float(value) * 1000, 1) + 0.0  # a plain float, never -0.0


def _body_pose(data: mujoco.MjData, body: int) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = data.xmat[body].reshape(3, 3)
    pose[:3, 3] = data.xpos[body]
    return pose


def _flange(data: mujoco.MjData, site: int | None) -> np.ndarray:
    pose = np.eye(4)
    if site is not None:
        pose[:3, :3] = data.site_xmat[site].reshape(3, 3)
        pose[:3, 3] = data.site_xpos[site]
    return pose


def settle(
    model: mujoco.MjModel, actuator: int, ctrl: float, pin: Callable[[mujoco.MjData], None],
    bodies: Sequence[int] = (),
) -> mujoco.MjData:
    data = mujoco.MjData(model)
    pin(data)
    mujoco.mj_forward(model, data)
    quiet = 0.0
    for _ in range(int(SETTLE_S / model.opt.timestep)):
        data.ctrl[actuator] = ctrl
        pin(data)
        mujoco.mj_step(model, data)
        if len(bodies):
            still = np.abs(data.cvel[list(bodies)]).max() < QUIET_SPEED
            quiet = quiet + model.opt.timestep if still else 0.0
            if quiet >= QUIET_S:
                break
    mujoco.mj_kinematics(model, data)
    return data


def _corners(model: mujoco.MjModel, data: mujoco.MjData, geom: int, flange: np.ndarray) -> np.ndarray:
    centre, half = model.geom_aabb[geom][:3], model.geom_aabb[geom][3:]
    local = np.array([centre + half * np.array(s) for s in itertools.product((-1, 1), repeat=3)])
    world = local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    return (world - flange[:3, 3]) @ flange[:3, :3]


def _colliding(model: mujoco.MjModel, bodies: set[int]) -> list[int]:
    return [g for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in bodies and (model.geom_contype[g] or model.geom_conaffinity[g])]


def gripper_measure(
    model: mujoco.MjModel, actuator: int, open_ctrl: float, closed_ctrl: float,
    site: int | None, pin: Callable[[mujoco.MjData], None], candidates: Sequence[int],
) -> dict:
    """candidates: the bodies that may be fingers (the gripper's subtree)."""
    poses = {state: settle(model, actuator, ctrl, pin, candidates) for state, ctrl in (("open", open_ctrl), ("closed", closed_ctrl))}
    flanges = {state: _flange(data, site) for state, data in poses.items()}

    def relative(state: str, body: int) -> np.ndarray:
        return np.linalg.inv(flanges[state]) @ _body_pose(poses[state], body)

    moving = set()
    for body in candidates:
        a, b = relative("open", body), relative("closed", body)
        turn = np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1))
        if np.linalg.norm(a[:3, 3] - b[:3, 3]) > MOVED_M or turn > MOVED_RAD:
            moving.add(body)
    fingers = _colliding(model, moving)
    if not fingers:
        raise ValueError("no finger geoms moved between open and closed")
    corners = {s: {g: _corners(model, poses[s], g, flanges[s]) for g in fingers} for s in poses}
    tips = {s: max(c[:, 2].max() for c in corners[s].values()) for s in poses}
    centres = np.array([c.mean(axis=0)[:2] for c in corners["open"].values()])
    spread = centres - centres.mean(axis=0)
    jaw = np.linalg.svd(spread, full_matrices=False)[2][0]  # principal direction in the flange's xy

    def sides() -> dict[int, float]:
        mid = np.mean([c.mean(axis=0)[:2] for c in corners["open"].values()], axis=0)
        return {g: np.sign(float(np.dot(c.mean(axis=0)[:2] - mid, jaw))) for g, c in corners["open"].items()}

    side = sides()
    if len(set(side.values())) < 2:  # one jaw fixed: the gripper's other geoms are the opposite side
        for g in set(_colliding(model, set(candidates))) - set(fingers):
            for state, data in poses.items():
                corners[state][g] = _corners(model, data, g, flanges[state])
        side = sides()
        if len(set(side.values())) < 2:
            raise ValueError("measure: could not find two jaws")

    def gap(state: str) -> float:
        lo, hi = -np.inf, np.inf
        for g, points in corners[state].items():
            near = points[points[:, 2] > tips[state] - TIP_BAND_M]
            if not len(near):
                continue
            along = near[:, :2] @ jaw
            if side[g] < 0:
                lo = max(lo, float(along.max()))
            else:
                hi = min(hi, float(along.min()))
        return max(0.0, hi - lo) if np.isfinite(lo) and np.isfinite(hi) else 0.0

    tcp = tips["open"]
    return {
        "tcp_mm": mm(tcp),
        "jaw_gap_mm": {s: mm(gap(s)) for s in ("open", "closed")},
        "finger_reach_mm": {s: mm(tips[s] - tcp) for s in ("open", "closed")},
    }


def suction_measure(model: mujoco.MjModel, data: mujoco.MjData, cup: int, site: int | None) -> dict:
    mujoco.mj_kinematics(model, data)
    flange = _flange(data, site)
    geoms = _colliding(model, set(robots.subtree(model, cup)))
    if not geoms:
        raise ValueError("measure: the suction cup has no colliding geoms")
    points = np.vstack([_corners(model, data, g, flange) for g in geoms])
    return {
        "cup_diameter_mm": mm(np.ptp(points[:, :2], axis=0).max()),
        "tcp_mm": mm(points[:, 2].max()),
    }
