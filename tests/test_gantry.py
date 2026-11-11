"""viam-labs:mujoco-sim:gantry: validation, moves and what mounts on it."""

import asyncio
import math
import time

import numpy as np
import pytest
from viam.components.arm import KinematicsFileFormat
from viam.proto.app.robot import ComponentConfig, Frame
from viam.utils import dict_to_struct

import mujoco_module.models  # noqa: F401
from mujoco_module.models.camera import MujocoCamera
from mujoco_module.models.gantry import MujocoGantry, profile
from mujoco_module.models.gripper import MujocoGripper
from mujoco_module.models.world import MujocoWorld

LENGTHS = (1000.0, 800.0, 500.0)
FRAME_MM = (-500.0, -400.0, 1200.0)
BLOCK_1KG = {"name": "load", "type": "cube", "size": 0.08, "position": [3, 3, 0.04], "mass": 1.0}


def config(name, attrs, parent=None, translation=FRAME_MM, orientation=None):
    cfg = ComponentConfig(name=name, attributes=dict_to_struct(attrs))
    if parent is not None:
        cfg.frame.CopyFrom(Frame(parent=parent))
        cfg.frame.translation.x, cfg.frame.translation.y, cfg.frame.translation.z = translation
        if orientation is not None:  # an orientation vector in degrees: (x, y, z, theta)
            v = cfg.frame.orientation.vector_degrees
            v.x, v.y, v.z, v.theta = orientation
    return cfg


def gantry_config(**attrs):
    return config("gantry", {"world": "sim-world", "lengths_mm": list(LENGTHS), **attrs}, parent="world")


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def gantry(running_sim):
    """A gantry in a real-time world: moves take their real duration, so a
    test can catch one in flight."""
    MujocoWorld.new(config("sim-world", {"props": [BLOCK_1KG]}), {})
    return MujocoGantry.new(gantry_config(), {})


def sim_time(sim) -> float:
    return sim.run(lambda: sim.data.time)


# -- validation ----------------------------------------------------------------


def test_validation():
    assert MujocoGantry.validate_config(gantry_config()) == (["sim-world"], [])
    no_frame = config("gantry", {"world": "sim-world", "lengths_mm": list(LENGTHS)})
    with pytest.raises(ValueError, match="set a frame"):
        MujocoGantry.validate_config(no_frame)
    for bad in ([1000, 800], [1000, -1, 500], [1000, 800, "5"]):
        with pytest.raises(ValueError, match="lengths_mm"):
            MujocoGantry.validate_config(gantry_config(lengths_mm=bad))
    on_arm = config("gantry", {"world": "sim-world", "lengths_mm": list(LENGTHS)}, parent="arm")
    with pytest.raises(ValueError, match='frame.parent must be "world"'):
        MujocoGantry.validate_config(on_arm)
    low = config("gantry", {"world": "sim-world", "lengths_mm": list(LENGTHS)}, parent="world",
                 translation=(0, 0, 550))
    with pytest.raises(ValueError, match="frame z must be at least"):
        MujocoGantry.validate_config(low)
    turned = config("gantry", {"world": "sim-world", "lengths_mm": list(LENGTHS)}, parent="world",
                    orientation=(0, 0, 1, 30))
    with pytest.raises(ValueError, match="can't be rotated"):
        MujocoGantry.validate_config(turned)
    for key in ("max_speed_mm_s", "max_accel_mm_s2", "move_timeout_sec"):
        with pytest.raises(ValueError, match=key):
            MujocoGantry.validate_config(gantry_config(**{key: 0}))


# -- moves -----------------------------------------------------------------------


def test_profile_is_straight_and_within_limits():
    start, goal = np.array([0.0, 0.0, 0.0]), np.array([600.0, 200.0, 100.0])
    points = profile(start, goal, np.array([500.0, 500.0, 500.0]), 2000.0)
    path = np.array([p.positions for p in points])
    direction = (goal - start) / np.linalg.norm(goal - start)
    off_line = np.linalg.norm(path - np.outer(path @ direction, direction), axis=1)
    assert off_line.max() < 1e-9
    assert path[-1] == pytest.approx(goal)
    velocity = np.diff(np.vstack([start, path]), axis=0) / np.diff([0.0, *[p.time_s for p in points]])[:, None]
    assert np.abs(velocity).max() <= 500.0 + 1e-6
    # 600 mm at 500 mm/s with 0.25 s ramps: 0.25 + 0.95 + 0.25
    assert points[-1].time_s == pytest.approx(600 / 500 + 500 / 2000, abs=1e-6)


def test_short_moves_never_reach_cruise():
    points = profile(np.zeros(3), np.array([10.0, 0, 0]), np.full(3, 500.0), 2000.0)
    assert points[-1].time_s == pytest.approx(2 * math.sqrt(10 / 2000), abs=1e-6)


def test_moves_arrive_in_a_straight_line_on_time(gantry, running_sim):
    goal = np.array([600.0, 300.0, 200.0])
    seen = []

    async def watch(move):
        while not move.done():
            seen.append(await gantry.get_position())
            await asyncio.sleep(0.02)

    async def go(target, speeds):
        start = sim_time(running_sim)
        move = asyncio.create_task(gantry.move_to_position(list(target), speeds))
        await asyncio.gather(move, watch(move))
        return sim_time(running_sim) - start

    took = run(go(goal, []))
    assert run(gantry.get_position()) == pytest.approx(goal, abs=0.5)
    ideal = profile(np.zeros(3), goal, np.full(3, 500.0), 2000.0)[-1].time_s
    assert ideal <= took <= ideal * 1.15
    direction = goal / np.linalg.norm(goal)
    path = np.array(seen)
    assert np.linalg.norm(path - np.outer(path @ direction, direction), axis=1).max() < 1.0

    # given speeds: the slowest axis's share sets the pace
    took = run(go(np.array([600.0, 700.0, 200.0]), [500.0, 100.0, 500.0]))
    assert took >= 400 / 100


def test_out_of_range_fails_before_moving(gantry):
    for bad in ([1001, 0, 0], [0, -1, 0], [0, 0, 600], [0, 0]):
        with pytest.raises(ValueError):
            run(gantry.move_to_position(bad, []))
    assert run(gantry.get_position()) == pytest.approx([0, 0, 0], abs=0.5)
    assert not run(gantry.is_moving())


def test_a_move_to_where_it_is_returns_at_once(gantry, monkeypatch):
    def no_trajectory(points):
        raise AssertionError("a zero-length move started a trajectory")

    monkeypatch.setattr(gantry._handle, "start_trajectory", no_trajectory)
    run(gantry.move_to_position([0.0, 0.0, 0.2], []))


def test_a_newer_move_supersedes_the_first_which_fails(gantry):
    async def both():
        first = asyncio.create_task(gantry.move_to_position([1000.0, 0.0, 0.0], []))
        await asyncio.sleep(0.05)
        assert await gantry.is_moving()
        second = asyncio.create_task(gantry.move_to_position([0.0, 800.0, 0.0], []))
        # the first fails promptly, long before its own move would have ended
        with pytest.raises(RuntimeError, match="superseded"):
            await asyncio.wait_for(first, timeout=0.5)
        await second

    run(both())
    assert run(gantry.get_position()) == pytest.approx([0.0, 800.0, 0.0], abs=1.0)
    assert not run(gantry.is_moving())


def test_a_stop_while_a_move_settles_supersedes_it(gantry, monkeypatch):
    """The last few ms of a move wait for the axes to settle; a stop then
    ends that wait too, rather than leaving the move to time out."""
    handle = gantry._handle
    started = asyncio.Event()
    real_wait = handle.wait_for_joint_targets

    loop = None

    def never_settles(targets, tolerance):  # runs on a worker thread (asyncio.to_thread)
        future = real_wait([10.0] * 3, tolerance)  # a target it can't reach: the wait stays open
        loop.call_soon_threadsafe(started.set)
        return future

    monkeypatch.setattr(handle, "wait_for_joint_targets", never_settles)

    async def stopped():
        nonlocal loop
        loop = asyncio.get_running_loop()
        move = asyncio.create_task(gantry.move_to_position([20.0, 0.0, 0.0], []))
        await asyncio.wait_for(started.wait(), timeout=5)
        await asyncio.sleep(0.05)  # now settling
        await gantry.stop()
        with pytest.raises(RuntimeError, match="superseded"):
            await asyncio.wait_for(move, timeout=0.5)

    run(stopped())
    assert not run(gantry.is_moving())


def test_a_cancelled_call_stops_the_gantry(gantry):
    async def cancelled():
        move = asyncio.create_task(gantry.move_to_position([1000.0, 0.0, 0.0], []))
        await asyncio.sleep(0.2)
        move.cancel()
        with pytest.raises(asyncio.CancelledError):
            await move
        await asyncio.sleep(0.3)
        return await gantry.get_position()

    held = run(cancelled())
    assert 0 < held[0] < 1000
    run(asyncio.sleep(0.3))
    assert run(gantry.get_position()) == pytest.approx(held, abs=1.0)


def test_a_call_cancelled_during_setup_stops_the_gantry(gantry, monkeypatch):
    handle = gantry._handle
    real_start = handle.start_trajectory

    def slow_start(points):  # runs on a worker thread: the trajectory starts, then setup lingers
        future = real_start(points)
        time.sleep(0.2)
        return future

    monkeypatch.setattr(handle, "start_trajectory", slow_start)

    async def cancelled():
        move = asyncio.create_task(gantry.move_to_position([1000.0, 0.0, 0.0], []))
        await asyncio.sleep(0.05)
        move.cancel()
        with pytest.raises(asyncio.CancelledError):
            await move
        await asyncio.sleep(0.5)
        return await gantry.get_position()

    held = run(cancelled())
    assert held[0] < 500
    run(asyncio.sleep(0.3))
    assert run(gantry.get_position()) == pytest.approx(held, abs=1.0)


def test_a_move_that_never_settles_times_out(running_sim, monkeypatch):
    MujocoWorld.new(config("sim-world", {"props": [BLOCK_1KG]}), {})
    gantry = MujocoGantry.new(gantry_config(move_timeout_sec=0.2), {})
    handle = gantry._handle
    real_wait = handle.wait_for_joint_targets
    monkeypatch.setattr(handle, "wait_for_joint_targets", lambda targets, tolerance: real_wait([10.0] * 3, tolerance))
    with pytest.raises(RuntimeError, match="didn't settle"):
        run(gantry.move_to_position([20.0, 0.0, 0.0], []))
    assert not run(gantry.is_moving())


def test_stop_ends_a_move_and_holds(gantry, running_sim):
    async def stopped():
        move = asyncio.create_task(gantry.move_to_position([1000.0, 0.0, 0.0], []))
        await asyncio.sleep(0.1)
        await gantry.stop()
        with pytest.raises(RuntimeError, match="superseded"):
            await move

    run(stopped())
    held = run(gantry.get_position())
    assert 0 < held[0] < 1000
    run(asyncio.sleep(0.2))
    assert run(gantry.get_position()) == pytest.approx(held, abs=1.0)


def test_lengths_home_and_kinematics(gantry):
    assert run(gantry.get_lengths()) == list(LENGTHS)
    assert run(gantry.home()) is True
    fmt, data = run(gantry.get_kinematics())[:2]
    assert fmt == KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA
    assert b'"prismatic"' in data


# -- what mounts on it --------------------------------------------------------------


def test_an_epick_on_the_gantry_carries_1kg_without_sag(gantry, running_sim):
    gripper = MujocoGripper.new(
        config("cup", {"world": "sim-world", "mount": "gantry", "asset": "robotiq_epick"},
               parent="gantry", translation=(0, 0, 130)),
        {},
    )
    run(gantry.move_to_position([500.0, 400.0, 300.0], []))
    tool = running_sim.run(lambda: running_sim.data.site("gantry/viam_frame").xpos.copy())
    # the ePick sits at the column's foot, pointing down
    base = running_sim.run(lambda: (running_sim.data.body("cup/base").xpos.copy(),
                                    running_sim.data.body("cup/base").xmat.reshape(3, 3)[:, 2].copy()))
    assert base[0] == pytest.approx(tool, abs=1e-4)
    assert base[1] == pytest.approx([0, 0, -1], abs=1e-6)
    under = tool + [0, 0, -0.130 - 0.04 - 0.001]  # the block's centre, its top 1 mm under the cup
    gravity = running_sim.run(lambda: running_sim.model.opt.gravity.copy())

    def weightless(on: bool):
        def apply():
            running_sim.model.opt.gravity[:] = 0 if on else gravity

        running_sim.run(apply)

    weightless(True)  # nothing under the block while the cup seals
    running_sim.set_prop_pose("load", list(under), [1, 0, 0, 0])
    run(gripper.grab())
    run(asyncio.sleep(0.5))
    weightless(False)
    run(asyncio.sleep(0.5))
    assert run(gripper.is_holding_something()).is_holding_something
    run(gantry.move_to_position([500.0, 400.0, 100.0], []))
    run(asyncio.sleep(0.5))
    assert run(gantry.get_position())[2] == pytest.approx(100.0, abs=1.0)
    lifted = running_sim.get_prop_pose("load")["position"][2]
    assert lifted == pytest.approx(under[2] + 0.2, abs=0.01)


def test_a_camera_framed_on_the_gantry_rides_the_column(gantry, running_sim):
    MujocoCamera.new(
        config("tool-cam", {"world": "sim-world"}, parent="gantry", translation=(0, 0, 50)), {}
    )

    def camera_z():
        return running_sim.run(lambda: running_sim.data.cam("tool-cam").xpos[2])

    before = camera_z()
    run(gantry.move_to_position([0.0, 0.0, 300.0], []))
    assert camera_z() == pytest.approx(before - 0.3, abs=0.002)
