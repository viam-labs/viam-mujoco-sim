import asyncio
import logging
import math

import numpy as np
import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from mujoco_module.handles import (
    HIDDEN_GROUP,
    PARK_POSITION,
    CameraHandle,
    ConveyorHandle,
    Parcel,
    PhotoEyeHandle,
    SpawnerHandle,
)
from mujoco_module.models.obstacle import obstacle_contribution
from mujoco_module.models.spawner import MujocoSpawner
from mujoco_module.sim import FREE_CONTYPE
from test_conveyor import TOP
from test_obstacle import BOX, config

HEAVY = Parcel(cls="large", half=(0.2, 0.15, 0.12), mass=3.0, color=(0.8, 0.5, 0.2), friction=1.0)
LIGHT = Parcel(cls="small", half=(0.06, 0.05, 0.04), mass=0.5, color=(0.2, 0.5, 0.8), friction=0.8)
UP = [1.0, 0.0, 0.0, 0.0]
CLASSES = [
    {"name": "small", "weight": 3, "size_m": [0.1, 0.16], "mass_kg": [0.4, 0.6], "color": [0.2, 0.5, 0.8]},
    {"name": "large", "size_m": [[0.3, 0.4], [0.2, 0.3], [0.2, 0.25]], "mass_kg": [2.5, 3.0],
     "color": {"min": [0.6, 0.4, 0.2], "max": [0.8, 0.6, 0.3]}, "friction": 0.8},
]
FEED = {"conveyor": "belt", "classes": CLASSES, "pool_size": 3, "seed": 7}


@pytest.fixture
def pool(sim):
    return SpawnerHandle(sim, "feed", {**FEED, "start_paused": True})


def table(sim):
    """A 600 x 800 x 20 mm table top at z = 0.29..0.31 (test_obstacle's box)."""
    sim.set_contribution("table", obstacle_contribution(config(BOX, translation=(0, 1000, 300))))


def parked(sim, name):
    geom = sim.model.geom(name)
    return (
        geom.contype[0] == 0 and geom.conaffinity[0] == 0 and geom.group[0] == HIDDEN_GROUP
        and sim.data.body(name).xpos == pytest.approx(PARK_POSITION, abs=1e-3)  # frozen before each step
    )


def test_the_pool_starts_parked_and_stays_put(sim, pool):
    sim.advance(1.0)
    assert all(parked(sim, f"feed/parcel_{i}") for i in range(3))
    assert sim.model.body_contype[sim.model.body("feed/parcel_0").id] == 0  # compiled body masks too


def test_a_heavy_parcel_spawns_onto_a_table_and_rests(sim, pool):
    table(sim)
    name = pool.spawn_at(HEAVY, [0, 1.0, 0.31 + 0.12 + 0.0005], UP)
    assert name == "feed/parcel_0"
    body = sim.model.body(name)
    assert body.mass[0] == pytest.approx(3.0)
    assert sim.model.body_contype[body.id] == FREE_CONTYPE and sim.model.geom(name).group[0] == 0
    assert sim.model.geom(name).size == pytest.approx(HEAVY.half)
    sim.advance(2.0)
    assert np.all(np.isfinite(sim.data.qpos))
    assert not any(w.number for w in sim.data.warning)
    assert sim.data.body(name).xpos == pytest.approx([0, 1.0, 0.31 + 0.12], abs=0.002)
    # a light one on the floor beside it
    small = pool.spawn_at(LIGHT, [0.6, 0, 0.04 + 0.0005], UP)
    sim.advance(1.0)
    assert sim.data.body(small).xpos == pytest.approx([0.6, 0, 0.04], abs=0.002)


def test_a_heavy_parcel_rests_on_a_belt(sim, pool):
    ConveyorHandle(sim, "belt", {"length_m": 2.0, "width_m": 0.5, "position": [0, 0, TOP]})
    name = pool.spawn_at(HEAVY, [0, 0, TOP + 0.12 + 0.0005], UP)
    sim.advance(1.0)
    assert np.all(np.isfinite(sim.data.qpos))
    assert not any(w.number for w in sim.data.warning)
    assert sim.data.body(name).xpos == pytest.approx([0, 0, TOP + 0.12], abs=0.002)


def test_a_spawn_leaves_every_other_body_where_it_was(sim, pool):
    sim.advance(0.5)
    before = sim.data.qpos.copy()
    name = pool.spawn_at(LIGHT, [0.6, 0, 0.5], UP)
    joint = sim.model.joint(name).qposadr[0]
    after = sim.data.qpos.copy()
    after[joint : joint + 7] = before[joint : joint + 7]
    assert after == pytest.approx(before, abs=0)


def test_an_exhausted_pool_spawns_nothing_and_clear_parks_all(sim, pool):
    names = [pool.spawn_at(LIGHT, [0.3 * i, 0.5, 0.05], UP) for i in range(3)]
    assert names == ["feed/parcel_0", "feed/parcel_1", "feed/parcel_2"]
    assert pool.spawn_at(LIGHT, [0, -0.5, 0.05], UP) is None
    assert sorted(sim.dynamic_props) == names
    first = sim.dynamic_props[names[0]].serial
    assert [p.serial for p in sim.dynamic_props.values()] == [first, first + 1, first + 2]
    assert pool.clear() == 3
    assert sim.dynamic_props == {}
    sim.advance(0.1)
    assert all(parked(sim, name) for name in names)
    assert pool.spawn_at(LIGHT, [0, -0.5, 0.05], UP) == "feed/parcel_0"
    assert sim.dynamic_props["feed/parcel_0"].serial == first + 3
    # a reconfigured spawner (a new handle) never reuses a serial: world_state keys on them
    pool.close()
    again = SpawnerHandle(sim, "feed", {**FEED, "start_paused": True})
    again.spawn_at(LIGHT, [0, -0.5, 0.05], UP)
    assert sim.dynamic_props["feed/parcel_0"].serial == first + 4


def test_a_rebuild_keeps_live_parcels_and_reset_parks_them(sim, pool):
    table(sim)
    name = pool.spawn_at(HEAVY, [0, 1.0, 0.31 + 0.12 + 0.0005], UP)
    sim.advance(1.0)
    pose = sim.data.body(name).xpos.copy()
    wall = config(BOX, translation=(2000, 0, 300))
    wall.name = "wall"
    sim.set_contribution("wall", obstacle_contribution(wall))  # a rebuild
    assert sim.model.geom(name).size == pytest.approx(HEAVY.half)
    assert sim.model.body(name).mass[0] == pytest.approx(3.0)
    assert sim.model.geom(name).rgba == pytest.approx([*HEAVY.color, 1])
    assert parked(sim, "feed/parcel_1")
    sim.advance(1.0)
    assert sim.data.body(name).xpos == pytest.approx(pose, abs=0.001)
    sim.reset()
    assert parked(sim, name) and sim.dynamic_props == {}
    assert sim.model.body(name).mass[0] == pytest.approx(0.1)
    sim.advance(0.5)
    assert parked(sim, name)


def test_parked_parcels_are_invisible_to_cameras_and_rays(sim, pool):
    below = [PARK_POSITION[0] - 1.0, PARK_POSITION[1], PARK_POSITION[2]]
    eye = PhotoEyeHandle(sim, "eye", {"position": below})
    camera = CameraHandle(sim, "cam", {"position": below, "target": PARK_POSITION, "width": 64, "height": 48})
    sim.advance(0.01)
    assert not eye.readings()["blocked"]
    _, depth = camera.get_rgbd()
    assert depth[12, 16] == 0  # nothing in view
    sim.model.geom_group[sim.model.geom("feed/parcel_0").id] = 0  # unhide one
    sim.advance(0.01)
    assert eye.readings()["distance_m"] == pytest.approx(0.95)
    _, depth = camera.get_rgbd()
    assert depth[12, 16] == pytest.approx(0.95, abs=0.01)


@pytest.fixture
def line(sim):
    belt = ConveyorHandle(sim, "belt", {"length_m": 2.0, "width_m": 0.5, "position": [0, 0, TOP]})
    return belt, SpawnerHandle(sim, "feed", {**FEED, "interval_s": 1.0, "start_paused": True})


def test_classes_are_drawn_by_weight_within_their_ranges(pool):
    draws = [pool._draw() for _ in range(1000)]
    small = [parcel for parcel, _ in draws if parcel.cls == "small"]
    assert len(small) / len(draws) == pytest.approx(0.75, abs=0.04)
    for parcel, yaw in draws:
        assert abs(yaw) <= math.radians(10)
        if parcel.cls == "small":
            assert all(0.05 <= h <= 0.08 for h in parcel.half) and 0.4 <= parcel.mass <= 0.6
            assert parcel.color == (0.2, 0.5, 0.8) and parcel.friction == 1.0
        else:
            assert 0.15 <= parcel.half[0] <= 0.2 and 0.1 <= parcel.half[2] <= 0.125
            assert 2.5 <= parcel.mass <= 3.0 and 0.6 <= parcel.color[0] <= 0.8 and parcel.friction == 0.8


def test_a_parcel_spawns_at_rest_on_the_infeed_spot_and_waits_for_it_to_clear(sim, line):
    _, feed = line
    name, reason = feed.spawn_now()
    assert reason == "" and name == "feed/parcel_0"
    (parcel,) = feed.parcels()
    spot = -1.0 + 0.01 + math.hypot(0.2, 0.15)  # past the belt's end by the largest footprint
    x, y, z = parcel["pose"]["position"]
    assert (x, y, z) == pytest.approx([0, spot, TOP + parcel["size"][2] / 2 + 0.0005])
    assert sim.model.body(name).mass[0] == pytest.approx(parcel["mass_kg"])
    sim.advance(1.0)
    assert np.all(np.isfinite(sim.data.qpos)) and not any(w.number for w in sim.data.warning)
    assert feed.readings()["box_present"]
    assert feed.spawn_now() == (None, "infeed occupied")
    assert feed.spawn_now(force=True)[0] == "feed/parcel_1"  # into the first one
    assert feed.take()
    assert feed.readings()["active"] == 1
    feed.clear()
    assert not feed.readings()["box_present"] and not feed.take()


def test_a_class_wider_than_the_belt_is_refused_and_logged_once(sim, caplog):
    ConveyorHandle(sim, "belt", {"length_m": 2.0, "width_m": 0.35, "position": [0, 0, TOP]})
    feed = SpawnerHandle(sim, "feed", {**FEED, "classes": CLASSES[1:], "interval_s": 0.1})
    with caplog.at_level(logging.ERROR):
        sim.advance(1.0)  # ten tries on the timer: none spawns, the sim runs on
        assert feed.spawn_now() == (None, "class large does not fit the conveyor")
    assert feed.readings()["spawned"] == 0
    (error,) = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert "class 'large' (up to 0.4 x 0.3 m)" in error and "(0.35 m wide, 2 m long)" in error


def test_the_timer_spawns_every_interval_while_the_belt_clears_the_spot(sim, line):
    belt, feed = line
    belt.set_speed(0.5)
    feed.set_paused(False)
    sim.advance(2.5)  # spawns at 0, 1 and 2 s
    reading = feed.readings()
    assert (reading["spawned"], reading["active"], reading["pool_exhausted"]) == (3, 3, False)
    assert reading["seconds_until_next"] == pytest.approx(0.5, abs=0.003)
    assert reading["enabled"] and not reading["paused"] and reading["interval_seconds"] == 1.0
    sim.advance(1.0)  # all three live: the fourth is skipped
    reading = feed.readings()
    assert (reading["spawned"], reading["pool_exhausted"]) == (3, True)
    assert reading["seconds_until_next"] == pytest.approx(0.5, abs=0.003)
    feed.set_paused(True)
    assert not feed.readings()["enabled"]


def test_a_stopped_belt_holds_back_the_timer(sim, line):
    _, feed = line
    feed.set_paused(False)
    sim.advance(2.5)
    reading = feed.readings()
    assert (reading["spawned"], reading["seconds_until_next"]) == (1, 0.0)  # waiting for the spot


def test_the_seeded_sequence_repeats_after_reset(sim, line):
    belt, feed = line

    def run():
        belt.set_speed(0.5)
        feed.set_paused(False)
        sim.advance(2.5)
        return [(p["class"], p["size"], p["mass_kg"]) for p in feed.parcels()]

    first = run()
    sim.reset()
    assert feed.readings()["spawned"] == 0 and feed.readings()["paused"]  # back to start_paused
    assert run() == first
    assert {cls for cls, *_ in first} <= {"small", "large"}


def test_parcels_despawn_in_a_box_or_below_the_floor(sim):
    belt = ConveyorHandle(sim, "belt", {"length_m": 2.0, "width_m": 0.5, "position": [0, 0, TOP]})
    chute = {"center": [0, 1.4, 0.3], "size": [1, 0.8, 0.6]}  # under the outfeed
    feed = SpawnerHandle(sim, "feed", {**FEED, "despawn": [chute]})
    belt.set_speed(0.5)
    sim.advance(0.1)
    (parcel,) = feed.parcels()
    feed.set_paused(True)
    for _ in range(int(8 / 0.002)):
        sim.step_once()
        if not feed.parcels():
            break
    # it rode off the end and dropped into the chute
    assert sim.data.time < 7 and parcel["name"] not in sim.dynamic_props
    name, _ = feed.spawn_now()
    joint = sim.model.joint(name).qposadr[0]
    sim.data.qpos[joint + 2] = -1.0
    sim.advance(0.01)
    assert feed.parcels() == []


def test_spawner_model(running_sim):
    from mujoco_module.models.conveyor import MujocoConveyor
    from mujoco_module.models.world import MujocoWorld
    from test_conveyor import conveyor_config

    attrs = {"world": "sim-world", **FEED, "start_paused": True}
    cfg = ComponentConfig(name="feed", attributes=dict_to_struct(attrs))
    assert MujocoSpawner.validate_config(cfg) == (["sim-world", "belt"], [])
    MujocoWorld.new(ComponentConfig(name="sim-world", attributes=dict_to_struct({"real_time": False})), {})
    MujocoConveyor.new(conveyor_config(), {})
    spawner = MujocoSpawner.new(cfg, {})

    async def run():
        assert (await spawner.get_readings())["paused"]
        assert await spawner.do_command({"reset": True}) == {"spawned": "feed/parcel_0"}
        assert await spawner.do_command({"command": "spawn"}) == {"spawned": None, "reason": "infeed occupied"}
        (parcel,) = (await spawner.do_command({"command": "get_parcels"}))["parcels"]
        assert parcel["name"] == "feed/parcel_0" and parcel["class"] in ("small", "large")
        assert set(parcel["pose"]) == {"position", "orientation_wxyz"} and len(parcel["size"]) == 3
        assert await spawner.do_command({"take": True}) == {"taken": True}
        assert await spawner.do_command({"command": "resume"}) == {"paused": False}
        await asyncio.sleep(0.2)
        assert (await spawner.get_readings())["spawned"] >= 2
        assert await spawner.do_command({"command": "pause"}) == {"paused": True}
        assert (await spawner.do_command({"command": "clear"}))["cleared"] >= 1
        with pytest.raises(ValueError, match="unknown command"):
            await spawner.do_command({"command": "jam"})
        await spawner.close()

    asyncio.run(run())


@pytest.mark.parametrize("attrs, message", [
    ({"conveyor": ""}, "set conveyor"),
    ({"classes": []}, "at least one parcel class"),
    ({"classes": [{"size_m": [0.1, 0.2], "mass_kg": [1, 2]}]}, "every class needs a name"),
    ({"classes": [{"name": "a", "size_m": [0.2, 0.1], "mass_kg": [1, 2]}]}, "low <= high"),
    ({"classes": [{"name": "a", "size_m": [[0.1, 0.2]], "mass_kg": [1, 2]}]}, "list of 2 numbers"),
    ({"classes": [{"name": "a", "size_m": [0.1, 0.2]}]}, "mass_kg must be a list of 2 numbers"),
    ({"classes": [{"name": "a", "size_m": [0.1, 0.2], "mass_kg": [1, 2], "color": {"min": [0, 0, 0]}}]},
     "color max"),
    ({"classes": [{"name": "a", "size_m": [0.1, 0.2], "mass_kg": [1, 2], "weight": 0}]}, "weight must be positive"),
    ({"pool_size": 0}, "pool_size"),
    ({"seed": 1.5}, "seed"),
    ({"interval_s": 0}, "interval_s"),
    ({"start_paused": "yes"}, "start_paused"),
    ({"despawn": [{"center": [0, 0, 0], "size": [1, 1, 0]}]}, "despawn size must be positive"),
])
def test_validate_rejects(attrs, message):
    cfg = ComponentConfig(name="feed", attributes=dict_to_struct({"world": "sim-world", **FEED, **attrs}))
    with pytest.raises(ValueError, match=message):
        MujocoSpawner.validate_config(cfg)
