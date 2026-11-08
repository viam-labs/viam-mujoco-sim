"""viam-labs:mujoco-sim:world_state - movable props for the 3D scene viewer.

A world state store that publishes every non-fixed prop of the world as a
transform with its live pose and shape. Obstacles, arms and bases already reach
the viewer through the frame system. The world state store doesn't feed the
motion service.

Attributes:
  world (string, required) - name of the viam-labs:mujoco-sim:world component
"""

import asyncio
import math
import uuid
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import Any, ClassVar, Self

from google.protobuf.field_mask_pb2 import FieldMask
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, PoseInFrame, ResourceName, Transform
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.worldstatestore import (
    StreamTransformChangesResponse,
    TransformChangeType,
    WorldStateStore,
)
from viam.spatialmath import Quaternion
from viam.utils import dict_to_struct, struct_to_dict

from .. import FAMILY, NAMESPACE
from ..meshes import Piece, mesh_geometry
from ..shapes import viewer_geometry
from ..sim import PropSpec, SimManager

UPDATE_PERIOD_S = 0.1
MOVE_MM = 1.0
TURN_DEG = 0.5

# (prop, pose in mm and degrees, quaternion (w, x, y, z))
_Snapshot = list[
    tuple[PropSpec, Pose, tuple[float, float, float, float], list[Piece] | None]
]


_State = tuple[Pose, tuple[float, float, float, float]]


def _moved(old: _State, new: _State) -> bool:
    (a, qa), (b, qb) = old, new
    if math.dist((a.x, a.y, a.z), (b.x, b.y, b.z)) > MOVE_MM:
        return True
    dot = min(1.0, abs(sum(x * y for x, y in zip(qa, qb))))
    return math.degrees(2 * math.acos(dot)) > TURN_DEG


class MujocoWorldState(WorldStateStore, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily(NAMESPACE, FAMILY), "world_state")

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._world = ""
        self._closed = False

    @classmethod
    def new(
        cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        store = cls(config.name)
        store.reconfigure(config, dependencies)
        return store

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        world = struct_to_dict(config.attributes).get("world")
        if not world or not isinstance(world, str):
            raise ValueError(
                f'{config.name}: set the "world" attribute to the name of your '
                "viam-labs:mujoco-sim:world component"
            )
        return [world], []

    # Called once, from new(): viam-server rebuilds resources (close() then new()) on config changes.
    def reconfigure(
        self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> None:
        self._world = str(struct_to_dict(config.attributes)["world"])

    async def close(self) -> None:
        self._closed = True  # ends open streams; viam-server rebuilds us on config changes

    def _uuid(self, prop: str) -> bytes:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"viam-labs:mujoco-sim/{self._world}/{prop}")).encode()

    def _snapshot(self) -> _Snapshot:
        sim = SimManager.get()

        def read() -> _Snapshot:
            sim.require_world()
            assert sim.cfg is not None and sim.data is not None
            out = []
            for prop in sim.cfg.props:
                if prop.fixed:
                    continue
                body = sim.data.body(prop.name)
                quat = tuple(float(v) for v in body.xquat)
                x, y, z = (float(v) * 1000.0 for v in body.xpos)
                pieces = sim.prop_pieces[prop.name] if prop.mesh is not None else None
                out.append((prop, Quaternion(*quat).to_pose(x, y, z), quat, pieces))
            return out

        return sim.run(read)

    def _transform(self, prop: PropSpec, pose: Pose, pieces: list[Piece] | None) -> Transform:
        r, g, b = (round(c * 255) for c in prop.color)
        return Transform(
            reference_frame=prop.name,
            pose_in_observer_frame=PoseInFrame(reference_frame="world", pose=pose),
            physical_object=(
                mesh_geometry(pieces, prop.name)
                if pieces is not None
                else viewer_geometry(prop.shape, prop.geom_size, prop.name)
            ),
            uuid=self._uuid(prop.name),
            metadata=dict_to_struct({"color": {"r": r, "g": g, "b": b}}),
        )

    async def list_uuids(self, **kwargs: Any) -> list[bytes]:
        snapshot = await asyncio.to_thread(self._snapshot)
        return [self._uuid(prop.name) for prop, *_ in snapshot]

    async def get_transform(self, uuid: bytes, **kwargs: Any) -> Transform:
        for prop, pose, _, pieces in await asyncio.to_thread(self._snapshot):
            if self._uuid(prop.name) == uuid:
                return self._transform(prop, pose, pieces)
        raise ValueError(f"unknown transform uuid {uuid!r}")

    async def stream_transform_changes(
        self, **kwargs: Any
    ) -> AsyncGenerator[StreamTransformChangesResponse, None]:
        # ADDED for every prop first (RDK's fake expects clients to call
        # ListUUIDs/GetTransform instead; both work), then poses that moved.
        # viam-server's Go client discards a stream's first message, as RDK's own
        # server sends an empty one first; do the same or the first ADDED is lost.
        yield StreamTransformChangesResponse()
        last: dict[str, _State] = {}
        for prop, pose, quat, pieces in await asyncio.to_thread(self._snapshot):
            last[prop.name] = (pose, quat)
            yield StreamTransformChangesResponse(
                change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_ADDED,
                transform=self._transform(prop, pose, pieces),
            )
        while not self._closed:
            await asyncio.sleep(UPDATE_PERIOD_S)
            if self._closed:
                return
            for prop, pose, quat, pieces in await asyncio.to_thread(self._snapshot):
                unseen = prop.name not in last
                if not unseen and not _moved(last[prop.name], (pose, quat)):
                    continue
                last[prop.name] = (pose, quat)
                if unseen:
                    yield StreamTransformChangesResponse(
                        change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_ADDED,
                        transform=self._transform(prop, pose, pieces),
                    )
                    continue
                yield StreamTransformChangesResponse(
                    change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_UPDATED,
                    transform=Transform(
                        uuid=self._uuid(prop.name),
                        pose_in_observer_frame=PoseInFrame(reference_frame="world", pose=pose),
                    ),
                    updated_fields=FieldMask(paths=["poseInObserverFrame.pose"]),
                )
