"""viam-labs:mujoco-sim:world_state - movable props for the 3D scene viewer.

A world state store that publishes every non-fixed prop of the world, every
live spawner parcel, and the conveyors' and scales' exact shapes, as
transforms with their live pose and shape: ADDED when one appears, UPDATED
when it moves, REMOVED when a parcel returns to its pool or a conveyor or
scale closes. Obstacles, arms and bases already reach the viewer through the
frame system; RDK never asks a motor or sensor for its shape. The world state
store doesn't feed the motion service.

Attributes:
  world (string, required) - name of the viam-labs:mujoco-sim:world component
"""

import asyncio
import math
import uuid
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from typing import Any, ClassVar, Self

import mujoco
from google.protobuf.field_mask_pb2 import FieldMask
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Geometry, Pose, PoseInFrame, ResourceName, Transform
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
from ..meshes import mesh_geometry
from ..shapes import viewer_geometry
from ..sim import SimManager

UPDATE_PERIOD_S = 0.1
MOVE_MM = 1.0
TURN_DEG = 0.5

# (uuid, frame name, pose in mm and degrees, quaternion (w, x, y, z), the
# geometry and colour, built only when sent)
_Entry = tuple[
    bytes, str, Pose, tuple[float, float, float, float], Callable[[], tuple[Geometry, tuple[float, ...]]]
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

    def _snapshot(self) -> list[_Entry]:
        sim = SimManager.get()

        def read() -> list[_Entry]:
            sim.require_world()
            assert sim.cfg is not None and sim.data is not None
            out = []

            def add(
                key: str, name: str, shape: Callable[[], tuple[Geometry, tuple[float, ...]]],
                body_name: str | None = None, offset_mm: Sequence[float] = (0.0, 0.0, 0.0),
            ) -> None:
                body = sim.data.body(body_name or name)
                quat = tuple(float(v) for v in body.xquat)
                x, y, z = (body.xpos * 1000.0 + body.xmat.reshape(3, 3) @ offset_mm).tolist()
                out.append((self._uuid(key), name, Quaternion(*quat).to_pose(x, y, z), quat, shape))

            for prop in sim.cfg.props:
                if prop.fixed:
                    continue
                pieces = sim.prop_pieces[prop.name] if prop.mesh is not None else None
                add(prop.name, prop.name, lambda prop=prop, pieces=pieces: (
                    mesh_geometry(pieces, prop.name)
                    if pieces is not None
                    else viewer_geometry(prop.shape, prop.geom_size, prop.name),
                    prop.color,
                ))
            for name, dynamic in sim.dynamic_props.items():
                # a new uuid per spawn: a parcel parked and spawned again is another box
                add(f"{name}#{dynamic.serial}", name, lambda name=name, dynamic=dynamic: (
                    viewer_geometry("box", dynamic.half_size, name), dynamic.color
                ))
            for component, static in sim.static_shapes.items():
                if mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, static.body) < 0:
                    continue  # the scene was rebuilt without it
                for g in static.geometries:
                    # the transform sits at the geometry's centre (the viewer draws shapes centred on it)
                    flat = Geometry(center=Pose(o_z=1), box=g.box, label=g.label)
                    add(f"{component}/{g.label}", g.label, lambda flat=flat, static=static: (flat, static.color),
                        static.body, (g.center.x, g.center.y, g.center.z))
            return out

        return sim.run(read)

    @staticmethod
    def _transform(entry: _Entry) -> Transform:
        uid, name, pose, _, shape = entry
        geometry, color = shape()
        r, g, b = (round(c * 255) for c in color)
        return Transform(
            reference_frame=name,
            pose_in_observer_frame=PoseInFrame(reference_frame="world", pose=pose),
            physical_object=geometry,
            uuid=uid,
            metadata=dict_to_struct({"color": {"r": r, "g": g, "b": b}}),
        )

    async def list_uuids(self, **kwargs: Any) -> list[bytes]:
        snapshot = await asyncio.to_thread(self._snapshot)
        return [uid for uid, *_ in snapshot]

    async def get_transform(self, uuid: bytes, **kwargs: Any) -> Transform:
        for entry in await asyncio.to_thread(self._snapshot):
            if entry[0] == uuid:
                return self._transform(entry)
        raise ValueError(f"unknown transform uuid {uuid!r}")

    async def stream_transform_changes(
        self, **kwargs: Any
    ) -> AsyncGenerator[StreamTransformChangesResponse, None]:
        # ADDED for every transform first (RDK's fake expects clients to call
        # ListUUIDs/GetTransform instead; both work), then poses that moved,
        # new ones and gone ones. viam-server's Go client discards a stream's
        # first message, as RDK's own server sends an empty one first; do the
        # same or the first ADDED is lost.
        yield StreamTransformChangesResponse()
        last: dict[bytes, _State] = {}
        for entry in await asyncio.to_thread(self._snapshot):
            last[entry[0]] = (entry[2], entry[3])
            yield StreamTransformChangesResponse(
                change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_ADDED,
                transform=self._transform(entry),
            )
        while not self._closed:
            await asyncio.sleep(UPDATE_PERIOD_S)
            if self._closed:
                return
            snapshot = await asyncio.to_thread(self._snapshot)
            for entry in snapshot:
                uid, _, pose, quat, _ = entry
                unseen = uid not in last
                if not unseen and not _moved(last[uid], (pose, quat)):
                    continue
                last[uid] = (pose, quat)
                if unseen:
                    yield StreamTransformChangesResponse(
                        change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_ADDED,
                        transform=self._transform(entry),
                    )
                    continue
                yield StreamTransformChangesResponse(
                    change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_UPDATED,
                    transform=Transform(
                        uuid=uid,
                        pose_in_observer_frame=PoseInFrame(reference_frame="world", pose=pose),
                    ),
                    updated_fields=FieldMask(paths=["poseInObserverFrame.pose"]),
                )
            for uid in set(last) - {entry[0] for entry in snapshot}:
                del last[uid]
                yield StreamTransformChangesResponse(
                    change_type=TransformChangeType.TRANSFORM_CHANGE_TYPE_REMOVED,
                    transform=Transform(uuid=uid),
                )
