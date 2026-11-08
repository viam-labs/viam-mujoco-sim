import asyncio
import logging
import math
import uuid

import pytest
from viam.proto.app.robot import ComponentConfig
from viam.services.worldstatestore import TransformChangeType
from viam.utils import dict_to_struct, struct_to_dict

from mujoco_module.models.world import MujocoWorld
from mujoco_module.models.world_state import MujocoWorldState

PROPS = [
    {"name": "block", "size": 0.04, "position": [0.3, 0, 0.02], "color": [1, 0, 0]},
    {"name": "can", "type": "cylinder", "size": 0.06, "scale": [1, 1, 2], "position": [0.5, 0, 0.06]},
    {"name": "pad", "fixed": True, "size": 0.2, "scale": [1, 1, 0.05], "position": [0, 0.4, 0.005]},
]


@pytest.fixture
def store(running_sim):
    MujocoWorld.new(
        ComponentConfig(name="sim-world", attributes=dict_to_struct({"props": PROPS, "real_time": False})), {}
    )
    service = MujocoWorldState.new(
        ComponentConfig(name="world-state", attributes=dict_to_struct({"world": "sim-world"})), {}
    )
    return running_sim, service


def uid(prop):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"viam-labs:mujoco-sim/sim-world/{prop}")).encode()


async def handshake(stream):
    first = await anext(stream)
    assert first.change_type == TransformChangeType.TRANSFORM_CHANGE_TYPE_UNSPECIFIED
    assert not first.transform.uuid


def test_validate():
    deps = MujocoWorldState.validate_config(
        ComponentConfig(name="w", attributes=dict_to_struct({"world": "sim-world"}))
    )
    assert deps == (["sim-world"], [])
    with pytest.raises(ValueError, match="world"):
        MujocoWorldState.validate_config(ComponentConfig(name="w", attributes=dict_to_struct({})))


def test_list_and_get_transforms(store):
    _, service = store
    uuids = asyncio.run(service.list_uuids())
    assert sorted(uuids) == sorted([uid("block"), uid("can")])  # the fixed pad is left out
    block = asyncio.run(service.get_transform(uid("block")))
    assert block.reference_frame == "block"
    assert block.pose_in_observer_frame.reference_frame == "world"
    assert block.pose_in_observer_frame.pose.x == pytest.approx(300, abs=2)
    assert block.physical_object.box.dims_mm.x == pytest.approx(40)
    assert struct_to_dict(block.metadata)["color"] == {"r": 255, "g": 0, "b": 0}
    can = asyncio.run(service.get_transform(uid("can")))
    assert can.physical_object.mesh.content_type == "ply"
    with pytest.raises(ValueError, match="unknown transform"):
        asyncio.run(service.get_transform(b"nope"))


def test_stream_adds_then_updates_moved_props(store):
    sim, service = store

    async def run():
        stream = service.stream_transform_changes()
        await handshake(stream)
        added = [await anext(stream) for _ in range(2)]
        assert {a.change_type for a in added} == {TransformChangeType.TRANSFORM_CHANGE_TYPE_ADDED}
        assert {a.transform.uuid for a in added} == {uid("block"), uid("can")}
        await asyncio.to_thread(sim.set_prop_pose, "block", [0.3, 0.2, 0.02], [1, 0, 0, 0])
        update = await asyncio.wait_for(anext(stream), 2.0)
        assert update.change_type == TransformChangeType.TRANSFORM_CHANGE_TYPE_UPDATED
        assert update.transform.uuid == uid("block")
        assert update.transform.pose_in_observer_frame.pose.y == pytest.approx(200, abs=2)
        assert list(update.updated_fields.paths) == ["poseInObserverFrame.pose"]
        # a 10 degree turn about Z in place
        half = math.radians(5)
        await asyncio.to_thread(sim.set_prop_pose, "block", [0.3, 0.2, 0.02], [math.cos(half), 0, 0, math.sin(half)])
        turned = await asyncio.wait_for(anext(stream), 2.0)
        assert turned.change_type == TransformChangeType.TRANSFORM_CHANGE_TYPE_UPDATED
        assert turned.transform.uuid == uid("block")
        assert turned.transform.pose_in_observer_frame.pose.o_z == pytest.approx(1, abs=0.01)
        assert turned.transform.pose_in_observer_frame.pose.theta == pytest.approx(10, abs=0.5)
        await stream.aclose()

    asyncio.run(run())


def test_close_ends_the_stream(store):
    _, service = store

    async def run():
        stream = service.stream_transform_changes()
        await handshake(stream)
        for _ in range(2):
            await anext(stream)
        await service.close()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(stream), 1.0)

    asyncio.run(run())


def test_resting_props_send_no_updates(store):
    _, service = store

    async def run():
        stream = service.stream_transform_changes()
        await handshake(stream)
        for _ in range(2):
            await anext(stream)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(anext(stream), 0.5)

    asyncio.run(run())


def test_mesh_prop_is_one_merged_ply(running_sim, tmp_path, monkeypatch):
    import trimesh

    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path / "data"))
    trimesh.creation.box((0.05, 0.05, 0.05)).export(tmp_path / "w.stl")
    MujocoWorld.new(ComponentConfig(name="sim-world", attributes=dict_to_struct({
        "real_time": False,
        "props": [{"name": "w", "type": "mesh", "mesh": str(tmp_path / "w.stl"), "position": [0, 0, 0.1]}],
    })), {})
    service = MujocoWorldState.new(
        ComponentConfig(name="world-state", attributes=dict_to_struct({"world": "sim-world"})), {}
    )
    assert asyncio.run(service.list_uuids()) == [uid("w")]
    transform = asyncio.run(service.get_transform(uid("w")))
    assert transform.physical_object.label == "w"
    assert transform.physical_object.mesh.content_type == "ply"
    assert b"element face 12" in transform.physical_object.mesh.mesh  # a cube, one piece


def test_spawned_parcels_are_added_updated_and_removed(store):
    from mujoco_module.handles import Parcel, SpawnerHandle
    from test_spawner import CLASSES

    sim, service = store
    feed = SpawnerHandle(sim, "feed", {"conveyor": "belt", "classes": CLASSES, "pool_size": 2, "start_paused": True})
    parcel = Parcel(cls="small", half=(0.05, 0.04, 0.03), mass=0.5, color=(0, 1, 0), friction=1.0)
    from mujoco_module import handles

    serial = next(handles._PARCEL_SERIALS) + 1  # process-wide
    first = uid(f"feed/parcel_0#{serial}")

    async def run():
        stream = service.stream_transform_changes()
        await handshake(stream)
        for _ in range(2):
            await anext(stream)
        await asyncio.to_thread(feed.spawn_at, parcel, [0, -0.5, 0.0305], [1, 0, 0, 0])
        added = await asyncio.wait_for(anext(stream), 2.0)
        assert added.change_type == TransformChangeType.TRANSFORM_CHANGE_TYPE_ADDED
        assert added.transform.uuid == first and added.transform.reference_frame == "feed/parcel_0"
        assert added.transform.physical_object.box.dims_mm.x == pytest.approx(100)
        assert struct_to_dict(added.transform.metadata)["color"] == {"r": 0, "g": 255, "b": 0}
        assert first in await service.list_uuids()
        assert (await service.get_transform(first)).pose_in_observer_frame.pose.y == pytest.approx(-500, abs=2)
        joint = await asyncio.to_thread(lambda: sim.model.joint("feed/parcel_0").qposadr[0])
        await asyncio.to_thread(sim.run, lambda: sim.data.qpos.__setitem__(joint + 1, -0.3))
        moved = await asyncio.wait_for(anext(stream), 2.0)
        assert moved.change_type == TransformChangeType.TRANSFORM_CHANGE_TYPE_UPDATED and moved.transform.uuid == first
        await asyncio.to_thread(feed.clear)
        removed = await asyncio.wait_for(anext(stream), 2.0)
        assert removed.change_type == TransformChangeType.TRANSFORM_CHANGE_TYPE_REMOVED
        assert removed.transform.uuid == first
        # the same body spawned again is a new transform
        await asyncio.to_thread(feed.spawn_at, parcel, [0, -0.5, 0.0305], [1, 0, 0, 0])
        again = await asyncio.wait_for(anext(stream), 2.0)
        assert (again.change_type, again.transform.uuid) == (
            TransformChangeType.TRANSFORM_CHANGE_TYPE_ADDED, uid(f"feed/parcel_0#{serial + 1}")
        )
        await stream.aclose()

    asyncio.run(run())


def test_conveyor_and_scale_shapes_are_published_and_removed_on_close(store, caplog):
    from viam.proto.app.robot import Frame, Translation

    from mujoco_module.models.conveyor import MujocoConveyor
    from mujoco_module.models.scale import MujocoScale

    _, service = store

    def framed(name, attrs, x=0, z=0):
        cfg = ComponentConfig(name=name, attributes=dict_to_struct({"world": "sim-world", **attrs}))
        cfg.frame.CopyFrom(Frame(parent="world", translation=Translation(x=x, z=z)))
        return cfg

    with caplog.at_level(logging.WARNING):
        belt = MujocoConveyor.new(framed("belt", {"length_m": 2.0, "width_m": 0.5}, x=1000, z=750), {})
        scale = MujocoScale.new(framed("scale", {"size_m": [0.4, 0.4, 0.3], "color": [1, 0, 0]}, z=300), {})
    hints = [r.getMessage() for r in caplog.records if "never asks a motor or sensor" in r.getMessage()]
    assert len(hints) == 2
    assert hints[0].endswith(
        '"geometry": {"type": "box", "x": 500.0, "y": 2000.0, "z": 750.0, '
        '"translation": {"x": 0.0, "y": 0.0, "z": -375.0}}'
    )
    shapes = [uid(k) for k in ("belt/belt-body", "belt/belt-rail-0", "belt/belt-rail-1", "scale/scale-platform")]

    async def run():
        assert set(shapes) <= set(await service.list_uuids())
        body = await service.get_transform(shapes[0])
        assert body.reference_frame == "belt-body"
        pose = body.pose_in_observer_frame.pose
        assert (pose.x, pose.y, pose.z) == pytest.approx((1000, 0, 375), abs=0.01)  # the box's centre
        assert body.physical_object.center.z == 0 and body.physical_object.box.dims_mm.z == pytest.approx(750)
        rail = (await service.get_transform(shapes[1])).pose_in_observer_frame.pose
        assert (rail.x, rail.z) == pytest.approx((1000 - 260, 770), abs=0.01)
        platform = await service.get_transform(shapes[3])
        assert platform.pose_in_observer_frame.pose.z == pytest.approx(150, abs=0.01)
        assert struct_to_dict(platform.metadata)["color"] == {"r": 255, "g": 0, "b": 0}

        stream = service.stream_transform_changes()
        await handshake(stream)
        added = [await anext(stream) for _ in range(6)]  # the two props and the four shapes
        assert set(shapes) <= {a.transform.uuid for a in added}
        await belt.close()
        removed = [await asyncio.wait_for(anext(stream), 2.0) for _ in range(3)]
        assert {r.change_type for r in removed} == {TransformChangeType.TRANSFORM_CHANGE_TYPE_REMOVED}
        assert {r.transform.uuid for r in removed} == set(shapes[:3])
        await scale.close()
        gone = await asyncio.wait_for(anext(stream), 2.0)
        assert (gone.change_type, gone.transform.uuid) == (TransformChangeType.TRANSFORM_CHANGE_TYPE_REMOVED, shapes[3])
        await stream.aclose()

    asyncio.run(run())
