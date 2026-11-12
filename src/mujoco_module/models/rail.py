"""viam-labs:mujoco-sim:rail - a one-axis floor rail an arm stands on (a "7th axis").

Attributes:
  world (string, required)       - the viam-labs:mujoco-sim:world
  length_mm (number, required)   - travel, mm
  payload_kg (number)            - what the carriage carries, for the servo gains (default 25: a UR5e and gripper)
  max_speed_mm_s (number)        - top speed (default 500)
  max_accel_mm_s2 (number)       - acceleration (default 2000)
  move_timeout_sec (number)      - wall-clock time a move may take to settle after its trajectory ends (default 30)

The frame (required, parent "world") is the carriage plate's top centre at
position 0; its z is the plate's height above the floor, where the track
lies, and it may be turned about z (a yaw) only. The carriage travels along
the frame's +x. What mounts on the rail (frame parent the rail, e.g. an arm)
stands on the plate and rides the carriage. Gantry API with one axis: Home
returns true without moving, as the gantry's does.
"""

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily

from .. import FAMILY, NAMESPACE
from ..handles import ArmHandle
from ..rail_file import TRACK_MM, rail_files
from ..sim import SimManager
from .gantry import DEFAULT_MAX_ACCEL_MM_S2, DEFAULT_MAX_SPEED_MM_S, DEFAULT_MOVE_TIMEOUT_S, MujocoGantry
from .utils import apply_frame_to_attrs, frame_pose, get_attrs, positive_number, validate_sim_component

DEFAULT_PAYLOAD_KG = 25.0
MIN_HEIGHT_MM = TRACK_MM[2] + 40  # room for the carriage above the track


def _yaw_only(quat) -> bool:
    """A rotation about z only: w and z free, x and y zero."""
    return quat is None or (abs(quat[1]) < 1e-6 and abs(quat[2]) < 1e-6)


class MujocoRail(MujocoGantry):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "rail")
    AXES: ClassVar[str] = "x"

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        name = config.name
        if not config.HasField("frame"):
            raise ValueError(f'{name}: set a frame (parent "world"): the carriage plate\'s top at position 0')
        deps, _ = validate_sim_component(config, needs_asset=False, world_frame_only=True)
        attrs = get_attrs(config)
        if "length_mm" not in attrs:
            raise ValueError(f"{name}: length_mm is required: the rail's travel in mm")
        positive_number(name, attrs, "length_mm", 0.0)
        position, quat = frame_pose(config)
        if not _yaw_only(quat):
            raise ValueError(f"{name}: the rail's frame may turn only about z (a yaw); it can't tilt")
        if position is None or position[2] * 1000 < MIN_HEIGHT_MM:
            raise ValueError(f"{name}: frame z is the plate's height above the floor; at least {MIN_HEIGHT_MM:g} mm")
        positive_number(name, attrs, "payload_kg", DEFAULT_PAYLOAD_KG)
        positive_number(name, attrs, "max_speed_mm_s", DEFAULT_MAX_SPEED_MM_S)
        positive_number(name, attrs, "max_accel_mm_s2", DEFAULT_MAX_ACCEL_MM_S2)
        positive_number(name, attrs, "move_timeout_sec", DEFAULT_MOVE_TIMEOUT_S)
        return deps, []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        attrs: dict[str, Any] = apply_frame_to_attrs(config, get_attrs(config))
        self._lengths = (float(attrs["length_mm"]),)
        self._max_speed = float(attrs.get("max_speed_mm_s", DEFAULT_MAX_SPEED_MM_S))
        self._max_accel = float(attrs.get("max_accel_mm_s2", DEFAULT_MAX_ACCEL_MM_S2))
        self._move_timeout = float(attrs.get("move_timeout_sec", DEFAULT_MOVE_TIMEOUT_S))
        mjcf, kinematics = rail_files(self._lengths[0], attrs["position"][2] * 1000,
                                      float(attrs.get("payload_kg", DEFAULT_PAYLOAD_KG)))
        self._handle = ArmHandle(SimManager.get(), self.name, {
            "world": attrs["world"],
            "mjcf": str(mjcf),
            "kinematics_url": str(kinematics),
            "collision": "mesh",  # the MJCF's own geoms and contact bits
            "position": attrs["position"],
            "orientation_wxyz": list(attrs.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0))),
        })
