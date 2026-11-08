import asyncio
import logging
import math
import time

import mujoco
import pytest
from grpclib import GRPCError, Status
from viam.proto.app.robot import ComponentConfig
from viam.proto.robot import FrameSystemConfig
from viam.proto.common import (
    Capsule,
    Geometry,
    Mesh,
    Pose,
    PoseInFrame,
    RectangularPrism,
    ResourceName,
    Sphere,
    Transform,
    Vector3,
)
from viam.utils import dict_to_struct

from conftest import BLOCK
from mujoco_module.models import world as world_module
from mujoco_module.models.world import MujocoWorld

DECK = Geometry(center=Pose(z=72, o_z=1), box=RectangularPrism(dims_mm=Vector3(x=1200, y=800, z=144)), label="deck")
BALL = Geometry(center=Pose(x=500, o_z=1), sphere=Sphere(radius_mm=50))
POST = Geometry(center=Pose(y=300, o_z=1), capsule=Capsule(radius_mm=40, length_mm=400), label="post")
MESH = Geometry(mesh=Mesh(content_type="ply", mesh=b"ply"), label="hull")


class FakeComponent:
    def __init__(self, geometries):
        self.geometries = geometries

    async def get_geometries(self):
        if self.geometries is None:  # a pallet, a motor: viam-server's answer
            raise GRPCError(Status.UNIMPLEMENTED, "method GetGeometries not implemented")
        return self.geometries


class FakeParent:
    """viam-server as the module's parent client sees it: components appear
    after `after` refreshes."""

    def __init__(self, components, after=0, frames=()):
        self.components = components  # name: (geometries or None for UNIMPLEMENTED, world pose)
        self.frames = list(frames)  # FrameSystemConfig parts; their poses override components'
        self.after = after
        self.refreshes = 0
        self.resource_names = []

    async def refresh(self):
        self.refreshes += 1
        self.resource_names = [
            ResourceName(namespace="rdk", type="component", subtype="generic", name=name)
            for name in self.components
        ] if self.refreshes > self.after else []

    def get_component(self, name):
        return FakeComponent(self.components[name.name][0])

    async def get_frame_system_config(self):
        return self.frames

    def world_pose(self, name):
        """name's frame in world, through its parents (turns about z only)"""
        part = next((p.frame for p in self.frames if p.frame.reference_frame == name), None)
        if part is None:
            return self.components[name][1]
        local, parent = part.pose_in_observer_frame.pose, part.pose_in_observer_frame.reference_frame
        up = Pose(o_z=1) if parent == "world" else self.world_pose(parent)
        c, s = math.cos(math.radians(up.theta)), math.sin(math.radians(up.theta))
        return Pose(
            x=up.x + c * local.x - s * local.y, y=up.y + s * local.x + c * local.y, z=up.z + local.z,
            o_z=1, theta=up.theta + local.theta,
        )

    async def transform_pose(self, query, destination):
        assert destination == "world"
        return PoseInFrame(reference_frame="world", pose=self.world_pose(query.reference_frame))


class FakeModule:
    def __init__(self, parent):
        self.parent = parent


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(world_module, "IMPORT_RETRY_S", 0.01)
    monkeypatch.setattr(world_module, "IMPORT_TIMEOUT_S", 2.0)


def connect(monkeypatch, components, after=0, frames=()):
    parent = FakeParent(components, after, frames)
    monkeypatch.setattr(world_module, "MODULE", FakeModule(parent))
    return parent


def frame(name, parent, pose, geometry=None):
    """a FrameSystemConfig part, as viam-server sends it"""
    transform = Transform(reference_frame=name, pose_in_observer_frame=PoseInFrame(reference_frame=parent, pose=pose))
    if geometry is not None:
        transform.physical_object.CopyFrom(geometry)
    return FrameSystemConfig(frame=transform)


def make_world(names, props=()):
    attrs = {"real_time": False, "props": list(props), "import_geometry": names}
    return MujocoWorld.new(ComponentConfig(name="sim-world", attributes=dict_to_struct(attrs)), {})


def geom(sim, name):
    def read():
        index = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if index < 0:
            return None
        g = sim.data.geom(index)
        return int(sim.model.geom_type[index]), g.xpos.copy(), g.xmat.reshape(3, 3).copy(), sim.model.geom_size[index].copy()

    return sim.run(read)


def test_boxes_spheres_and_capsules_become_fixed_geoms_at_their_world_poses(running_sim, monkeypatch, fast):
    # a pallet 1 m out along x, turned a quarter turn
    connect(monkeypatch, {"pallet": ([DECK, BALL, POST], Pose(x=1000, o_z=1, theta=90))})

    async def run():
        world = make_world(["pallet"], [BLOCK])
        assert await world._import_task == {"imported": ["pallet"], "from_frame": [], "missing": []}
        return world

    world = asyncio.run(run())
    kind, position, rotation, size = geom(running_sim, "import/pallet/deck")
    assert kind == mujoco.mjtGeom.mjGEOM_BOX
    assert position == pytest.approx([1, 0, 0.072])
    assert size == pytest.approx([0.6, 0.4, 0.072])
    assert rotation[:, 0] == pytest.approx([0, 1, 0], abs=1e-9)  # the deck's long side along world y
    kind, position, _, size = geom(running_sim, "import/pallet/1")  # no label: its index
    assert kind == mujoco.mjtGeom.mjGEOM_SPHERE and position == pytest.approx([1, 0.5, 0])
    kind, position, _, size = geom(running_sim, "import/pallet/post")
    assert kind == mujoco.mjtGeom.mjGEOM_CAPSULE and position == pytest.approx([0.7, 0, 0])
    assert size[:2] == pytest.approx([0.04, 0.16])
    # a block dropped onto the deck rests on it
    running_sim.set_prop_pose("pick_block", [1.0, 0.0, 0.3], [1, 0, 0, 0])
    time.sleep(0.5)  # several sim seconds: real_time is off
    z = running_sim.run(lambda: float(running_sim.data.body("pick_block").xpos[2]))
    assert z == pytest.approx(0.144 + 0.0313 / 2, abs=0.002)
    asyncio.run(world.close())  # the import goes with the world
    assert geom(running_sim, "import/pallet/deck") is None


def test_empty_and_mesh_only_components_warn_and_are_skipped(running_sim, monkeypatch, fast, caplog):
    connect(monkeypatch, {"fence": ([], Pose(o_z=1)), "hull": ([MESH], Pose(o_z=1)), "pallet": ([DECK], Pose(o_z=1))})

    async def run():
        world = make_world(["fence", "hull", "pallet"])
        with caplog.at_level(logging.WARNING):
            assert await world._import_task == {"imported": ["pallet"], "from_frame": [], "missing": []}

    asyncio.run(run())
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("fence has no box, sphere or capsule" in m for m in warned)
    assert any("hull has no box, sphere or capsule" in m for m in warned)


def test_a_component_without_get_geometries_is_imported_from_its_frame(running_sim, monkeypatch, fast):
    # a workcell-components pallet: GetGeometries unimplemented, a frame geometry, 1 m out
    connect(monkeypatch, {"pallet": (None, Pose(o_z=1))}, frames=[frame("pallet", "world", Pose(x=1000, o_z=1), DECK)])

    async def run():
        world = make_world(["pallet"])
        assert await world._import_task == {"imported": ["pallet"], "from_frame": ["pallet"], "missing": []}
        kind, position, _, size = geom(running_sim, "import/pallet/deck")
        assert kind == mujoco.mjtGeom.mjGEOM_BOX and position == pytest.approx([1, 0, 0.072])
        assert size == pytest.approx([0.6, 0.4, 0.072])
        await world.close()

    asyncio.run(run())


def test_a_frame_geometry_under_another_frame_is_placed_through_it(running_sim, monkeypatch, fast):
    # a decal on a cart 1 m out along y, turned a quarter turn; the decal 0.5 m along the cart's x
    frames = [
        frame("cart", "world", Pose(y=1000, o_z=1, theta=90)),
        frame("decal", "cart", Pose(x=500, o_z=1), DECK),
    ]
    connect(monkeypatch, {"decal": (None, None)}, frames=frames)

    async def run():
        world = make_world(["decal"])
        assert await world._import_task == {"imported": ["decal"], "from_frame": ["decal"], "missing": []}
        _, position, rotation, _ = geom(running_sim, "import/decal/deck")
        assert position == pytest.approx([0, 1.5, 0.072])
        assert rotation[:, 0] == pytest.approx([0, 1, 0], abs=1e-9)
        await world.close()

    asyncio.run(run())


def test_neither_source_warns_at_once_without_retrying(running_sim, monkeypatch, fast, caplog):
    # a motor: GetGeometries unimplemented, a frame with no geometry (or none at all)
    monkeypatch.setattr(world_module, "IMPORT_TIMEOUT_S", 30.0)
    parent = connect(monkeypatch, {"motor": (None, Pose(o_z=1)), "sensor": (None, Pose(o_z=1))},
                     frames=[frame("motor", "world", Pose(o_z=1))])

    async def run():
        world = make_world(["motor", "sensor"])
        start = time.monotonic()
        with caplog.at_level(logging.WARNING):
            assert await world._import_task == {"imported": [], "from_frame": [], "missing": []}
        assert time.monotonic() - start < 1.0

    asyncio.run(run())
    assert parent.refreshes == 1
    warned = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warned) == 2 and all("Add a box, sphere or capsule geometry to its frame" in m for m in warned)
    assert "motor has no box" in warned[0] and "sensor has no box" in warned[1]


def test_missing_components_are_retried_then_logged(running_sim, monkeypatch, fast, caplog):
    parent = connect(monkeypatch, {"pallet": ([DECK], Pose(o_z=1))}, after=3)

    async def run():
        world = make_world(["pallet", "ghost"])
        with caplog.at_level(logging.ERROR):
            assert await world._import_task == {"imported": ["pallet"], "from_frame": [], "missing": ["ghost"]}

    asyncio.run(run())
    assert parent.refreshes >= 4  # the pallet showed up on the fourth
    assert geom(running_sim, "import/pallet/deck") is not None
    (error,) = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert "could not import ghost" in error and "not in the machine's resources" in error


def test_this_modules_components_are_skipped(running_sim, monkeypatch, fast):
    from mujoco_module.models.obstacle import MujocoObstacle
    from test_obstacle import BOX, config

    connect(monkeypatch, {"table": ([BOX], Pose(o_z=1))})

    async def run():
        make_world([])
        obstacle = MujocoObstacle.new(config(BOX, translation=(400, 0, 10)), {})
        world = make_world(["table"])
        assert await world._import_task == {"imported": [], "from_frame": [], "missing": []}
        await obstacle.close()

    asyncio.run(run())


def test_reimport_replaces_the_import(running_sim, monkeypatch, fast):
    parent = connect(monkeypatch, {"pallet": ([DECK, BALL], Pose(o_z=1))})

    async def run():
        world = make_world(["pallet"])
        await world._import_task
        parent.components["pallet"] = ([POST], Pose(o_z=1))
        assert await world.do_command({"command": "reimport"}) == {"imported": ["pallet"], "from_frame": [], "missing": []}
        assert geom(running_sim, "import/pallet/deck") is None
        assert geom(running_sim, "import/pallet/post") is not None
        # a second reimport supersedes one still waiting for its components
        parent.after = parent.refreshes + 1000
        first = asyncio.ensure_future(world.do_command({"command": "reimport"}))
        await asyncio.sleep(0.05)
        parent.after = 0
        assert await world.do_command({"command": "reimport"}) == {"imported": ["pallet"], "from_frame": [], "missing": []}
        assert await first == {"superseded": True}
        make_world([])  # import_geometry dropped: the import goes too
        assert geom(running_sim, "import/pallet/post") is None

    asyncio.run(run())


def test_a_reimport_outlives_its_callers_deadline_and_a_failed_refresh_is_retried(running_sim, monkeypatch, fast):
    parent = connect(monkeypatch, {"pallet": ([DECK], Pose(o_z=1))})
    refresh = parent.refresh

    async def flaky():
        if parent.refreshes == 0:
            parent.refreshes += 1
            raise ConnectionError("no route")
        await refresh()

    parent.refresh = flaky

    async def run():
        world = make_world([])
        world._import_names = ["pallet"]
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(world.do_command({"command": "reimport"}), 0.001)
        assert await world._import_task == {"imported": ["pallet"], "from_frame": [], "missing": []}
        await world.close()

    asyncio.run(run())


def test_an_import_error_is_logged_and_reaches_reimport(running_sim, monkeypatch, fast, caplog):
    connect(monkeypatch, {"pallet": ([DECK], Pose(o_z=1))})

    async def run():
        world = make_world([])
        world._import_names = ["pallet"]

        def broken(name, contribution):
            raise RuntimeError("compile failed")

        monkeypatch.setattr(running_sim, "set_contribution", broken)
        with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="compile failed"):
            await world.do_command({"command": "reimport"})

    asyncio.run(run())
    assert any("import_geometry: the import failed" in r.getMessage() for r in caplog.records)


def test_validate_import_geometry():
    def cfg(names):
        return ComponentConfig(name="w", attributes=dict_to_struct({"import_geometry": names}))

    assert MujocoWorld.validate_config(cfg(["pallet"])) == ([], [])
    with pytest.raises(ValueError, match="import_geometry must be a list of component names"):
        MujocoWorld.validate_config(cfg("pallet"))
