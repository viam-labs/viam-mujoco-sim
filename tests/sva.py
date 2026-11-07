"""Forward kinematics for a Viam SVA kinematics file, enough for tests."""

import json
from pathlib import Path

import numpy as np

from viam.spatialmath import OrientationVector


def _rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _quat_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def forward_kinematics(path: Path, joints_rad, end: str) -> np.ndarray:
    """4x4 pose of frame `end` in metres, for joint values in chain order."""
    doc = json.loads(path.read_text())
    links = {link["id"]: link for link in doc["links"]}
    joint_specs = {joint["id"]: joint for joint in doc["joints"]}
    order = [joint["id"] for joint in doc["joints"]]
    poses = {"world": np.eye(4)}

    def pose(frame):
        if frame in poses:
            return poses[frame]
        local = np.eye(4)
        if frame in links:
            link = links[frame]
            parent = link["parent"]
            t = link["translation"]
            local[:3, 3] = [t["x"] / 1000, t["y"] / 1000, t["z"] / 1000]
            orientation = link.get("orientation")
            if orientation:
                assert orientation["type"] == "ov_degrees"
                v = orientation["value"]
                q = OrientationVector(v["x"], v["y"], v["z"], np.radians(v["th"])).to_quaternion()
                local[:3, :3] = _quat_matrix((q.w, q.i, q.j, q.k))
        else:
            spec = joint_specs[frame]
            parent = spec["parent"]
            a = spec["axis"]
            local[:3, :3] = _rotation((a["x"], a["y"], a["z"]), joints_rad[order.index(frame)])
        poses[frame] = pose(parent) @ local
        return poses[frame]

    return pose(end)
