"""viam-labs:mujoco-sim:camera - an RGB camera in the MuJoCo scene.

Attributes:
  world (string, required) - name of the viam-labs:mujoco-sim:world component
  target ([x,y,z] meters)  - aim point; overrides the frame's orientation
  fov_deg (float)          - vertical field of view (default 60)
  width / height (int)     - image size (default 640 x 480)
  position ([x,y,z] m)     - fallback placement when no frame is set
                             (default (1, 1, 1))
  orientation_wxyz ([w,x,y,z]) - fallback orientation when no frame is set
  orientation_rpy_deg ([r,p,y]) - same, as roll/pitch/yaw; orientation_wxyz
                             wins if both are given

Place it with the standard frame (translation in mm); frame.parent must be
"world". A frame orientation is read in Viam's camera convention: +Z forward,
+Y down. `target`, when set, takes precedence over a frame orientation.
"""

import asyncio
import math
from io import BytesIO
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image
from typing_extensions import Self
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, NamedImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResponseMetadata, ResourceName
from viam.proto.component.camera import GetPropertiesResponse, IntrinsicParameters
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily

from .. import FAMILY, NAMESPACE
from ..handles import CameraHandle
from ..sim import SimManager
from .utils import apply_frame_to_attrs, get_attrs, validate_sim_component


class MujocoCamera(Camera, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "camera")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._handle: Optional[CameraHandle] = None

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        camera = cls(config.name)
        camera.reconfigure(config, dependencies)
        return camera

    @classmethod
    def validate_config(
        cls, config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = get_attrs(config)
        for key in ("width", "height"):
            raw = attrs.get(key, 1)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(raw)
                or raw != int(raw)
                or raw <= 0
            ):
                raise ValueError(f"{config.name}: {key} must be a positive integer")
        fov = attrs.get("fov_deg", 60)
        if (
            isinstance(fov, bool)
            or not isinstance(fov, (int, float))
            or not math.isfinite(fov)
            or not 0 < fov < 180
        ):
            raise ValueError(f"{config.name}: fov_deg must be finite and between 0 and 180")
        return validate_sim_component(config, needs_asset=False)

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        attrs: Dict[str, Any] = apply_frame_to_attrs(config, get_attrs(config))
        self._handle = CameraHandle(SimManager.get(), self.name, attrs)

    async def close(self) -> None:
        if self._handle is not None:
            await asyncio.to_thread(self._handle.close)
            self._handle = None

    def _h(self) -> CameraHandle:
        if self._handle is None:
            raise RuntimeError(f"camera {self.name} is not attached to the sim")
        return self._handle

    def _jpeg(self) -> bytes:
        buffer = BytesIO()
        Image.fromarray(self._h().get_rgb(), mode="RGB").save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()

    async def get_images(self, **kwargs) -> Tuple[List[NamedImage], ResponseMetadata]:
        data = await asyncio.to_thread(self._jpeg)
        return [NamedImage(self.name, data, CameraMimeType.JPEG)], ResponseMetadata()

    async def get_point_cloud(self, **kwargs) -> Tuple[bytes, str]:
        raise NotImplementedError("point clouds are not supported")

    async def get_geometries(self, **kwargs):
        return []

    async def get_properties(self, **kwargs) -> GetPropertiesResponse:
        return GetPropertiesResponse(
            supports_pcd=False,
            intrinsic_parameters=IntrinsicParameters(**self._h().intrinsics()),
        )
