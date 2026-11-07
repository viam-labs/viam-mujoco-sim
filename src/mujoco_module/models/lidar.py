"""viam-labs:mujoco-sim:lidar - a 2D scanning LIDAR, served like Viam's rplidar
module: a camera that only returns point clouds.

Attributes:
  world (string, required)       - name of the viam-labs:mujoco-sim:world component
  range_min_m (float)            - nearer hits are dropped (default 0.15)
  range_max_m (float)            - farther hits are dropped (default 12)
  angular_resolution_deg (float) - ray spacing, 0.1 to 360 (default 1)
  position ([x,y,z] m)           - fallback placement when no frame is set
                                   (default (0, 0, 0.2))

Place it with the standard frame. frame.parent is "world" or another mujoco-sim
component (a base or an arm). The scan plane is the frame's XY plane; ray 0
points along +X.
"""

import asyncio
import math
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, NamedImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResponseMetadata, ResourceName
from viam.proto.component.camera import GetPropertiesResponse
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily

from .. import FAMILY, NAMESPACE
from ..handles import (
    DEFAULT_LIDAR_RANGE_MAX_M,
    DEFAULT_LIDAR_RANGE_MIN_M,
    DEFAULT_LIDAR_RESOLUTION_DEG,
    MAX_LIDAR_RAYS,
    LidarHandle,
    encode_pcd,
)
from ..sim import SimManager
from .utils import apply_frame_to_attrs, get_attrs, positive_number, validate_sim_component


class MujocoLidar(Camera, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "lidar")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[LidarHandle] = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        lidar = cls(config.name)
        lidar.reconfigure(config, dependencies)
        return lidar

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = get_attrs(config)
        raw_min = attrs.get("range_min_m", DEFAULT_LIDAR_RANGE_MIN_M)
        if (
            isinstance(raw_min, bool)
            or not isinstance(raw_min, (int, float))
            or not math.isfinite(raw_min)
            or raw_min < 0
        ):
            raise ValueError(f"{config.name}: range_min_m must be a finite number >= 0")
        range_max = positive_number(config.name, attrs, "range_max_m", DEFAULT_LIDAR_RANGE_MAX_M)
        if range_max <= raw_min:
            raise ValueError(f"{config.name}: range_max_m must be greater than range_min_m")
        resolution = positive_number(
            config.name, attrs, "angular_resolution_deg", DEFAULT_LIDAR_RESOLUTION_DEG
        )
        if not 360.0 / MAX_LIDAR_RAYS <= resolution <= 360.0:
            raise ValueError(
                f"{config.name}: angular_resolution_deg must be between "
                f"{360.0 / MAX_LIDAR_RAYS} and 360"
            )
        return validate_sim_component(config, needs_asset=False, world_frame_only=False)

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs: Dict[str, Any] = apply_frame_to_attrs(config, get_attrs(config))
        if config.HasField("frame") and config.frame.parent != "world":
            attrs["parent"] = config.frame.parent
        self._handle = LidarHandle(SimManager.get(), self.name, attrs)

    async def close(self) -> None:
        if self._handle is not None:
            await asyncio.to_thread(self._handle.close)
            self._handle = None

    def _h(self) -> LidarHandle:
        if self._handle is None:
            raise RuntimeError(f"lidar {self.name} is not attached to the sim")
        return self._handle

    async def get_images(self, **kwargs) -> Tuple[List[NamedImage], ResponseMetadata]:
        raise NotImplementedError("a lidar only serves point clouds; use GetPointCloud")

    async def get_point_cloud(self, **kwargs) -> Tuple[bytes, str]:
        points = await asyncio.to_thread(self._h().scan)
        return encode_pcd(points), CameraMimeType.PCD

    async def get_geometries(self, **kwargs):
        return []

    async def get_properties(self, **kwargs) -> GetPropertiesResponse:
        return GetPropertiesResponse(supports_pcd=True)
