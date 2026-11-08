"""Helpers shared by the component models.

Adapted from viam-modules/viam-isaac-sim (Apache-2.0); modified for MuJoCo.
"""

import math
from typing import Any, Dict, Optional, Sequence, Tuple

from viam.proto.app.robot import ComponentConfig
from viam.spatialmath import AxisAngle, EulerAngles, OrientationVector, Quaternion
from viam.utils import struct_to_dict

from ..sim import KNOWN_ASSETS

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # (w, x, y, z), MuJoCo's order


def _wxyz(q: Quaternion) -> Quat:
    return (q.w, q.i, q.j, q.k)


def get_attrs(config: ComponentConfig) -> Dict[str, Any]:
    return struct_to_dict(config.attributes)


def frame_pose(config: ComponentConfig) -> Tuple[Optional[Vec3], Optional[Quat]]:
    """Spawn pose from the component's standard frame config: position in
    meters (frame translations are mm) and a (w,x,y,z) quaternion, or None
    for whatever isn't set. Placing components with the frame system keeps
    Viam's view of the machine (e.g. the motion service) consistent with the
    scene."""
    if not config.HasField("frame"):
        return None, None
    frame = config.frame
    t = frame.translation
    position = (t.x / 1000.0, t.y / 1000.0, t.z / 1000.0)

    quat: Optional[Quat] = None
    o = frame.orientation
    which = o.WhichOneof("type")
    if which == "quaternion":
        q = o.quaternion
        quat = (q.w, q.x, q.y, q.z)
    elif which == "vector_radians":
        v = o.vector_radians
        quat = _wxyz(OrientationVector(v.x, v.y, v.z, v.theta).to_quaternion())
    elif which == "vector_degrees":
        v = o.vector_degrees
        quat = _wxyz(OrientationVector(v.x, v.y, v.z, math.radians(v.theta)).to_quaternion())
    elif which == "euler_angles":
        e = o.euler_angles  # radians
        quat = _wxyz(EulerAngles(e.roll, e.pitch, e.yaw).to_quaternion())
    elif which == "axis_angles":
        a = o.axis_angles
        quat = _wxyz(AxisAngle(a.x, a.y, a.z, a.theta).to_quaternion())
    return position, quat


def apply_frame_to_attrs(
    config: ComponentConfig, attrs: Dict[str, Any]
) -> Dict[str, Any]:
    """Fold the frame config into the spawn attrs (frame wins over the
    position/orientation_rpy_deg fallback attributes)."""
    rpy = attrs.get("orientation_rpy_deg")
    if rpy is not None and attrs.get("orientation_wxyz") is None:
        roll, pitch, yaw = (math.radians(float(v)) for v in rpy)
        attrs["orientation_wxyz"] = list(_wxyz(EulerAngles(roll, pitch, yaw).to_quaternion()))
    position, quat = frame_pose(config)
    if position is not None:
        attrs["position"] = list(position)
    if quat is not None:
        attrs["orientation_wxyz"] = list(quat)
    return attrs


def validate_sim_component(
    config: ComponentConfig, needs_asset: bool = True, world_frame_only: bool = True
) -> Tuple[Sequence[str], Sequence[str]]:
    """Every component names its world (so viam-server starts it first) and,
    when it spawns an arm, a known asset or its own mjcf and kinematics_url. Components placed by
    their frame must be framed on the world (frame_pose ignores the parent);
    the gripper opts out because its frame parent is its arm."""
    attrs = struct_to_dict(config.attributes)
    world = attrs.get("world")
    if not world or not isinstance(world, str):
        raise ValueError(
            f'{config.name}: set the "world" attribute to the name of your '
            "viam-labs:mujoco-sim:world component"
        )
    if needs_asset:
        asset, mjcf = attrs.get("asset"), attrs.get("mjcf")
        if mjcf is None and asset not in KNOWN_ASSETS:
            raise ValueError(
                f"{config.name}: asset must be one of {sorted(KNOWN_ASSETS)}, "
                f"got {asset!r}; or set mjcf and kinematics_url for your own robot"
            )
        if mjcf is not None:
            if asset is not None:
                raise ValueError(f"{config.name}: set asset or mjcf, not both")
            if not isinstance(mjcf, str) or not mjcf:
                raise ValueError(f"{config.name}: mjcf must be a file path")
            if not attrs.get("kinematics_url") and not mjcf.lower().endswith(".urdf"):
                raise ValueError(
                    f"{config.name}: mjcf needs kinematics_url, the arm's Viam kinematics file"
                )
    if world_frame_only and config.HasField("frame") and config.frame.parent != "world":
        raise ValueError(
            f'{config.name}: frame.parent must be "world"; placing relative to '
            "other frames isn't supported yet"
        )
    return [world], []


def finite_number(name: str, attrs: Dict[str, Any], key: str, default: float) -> float:
    """attrs[key] (or default) as a finite float; ValueError otherwise."""
    raw = attrs.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
        raise ValueError(f"{name}: {key} must be a finite number")
    return float(raw)


def positive_number(name: str, attrs: Dict[str, Any], key: str, default: float) -> float:
    """attrs[key] (or default) as a finite, positive float; ValueError otherwise."""
    value = finite_number(name, attrs, key, default)
    if value <= 0:
        raise ValueError(f"{name}: {key} must be a finite, positive number")
    return value
