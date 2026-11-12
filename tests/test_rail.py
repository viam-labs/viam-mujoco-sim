"""viam-labs:mujoco-sim:rail: validation, moves, and an arm riding it."""

import asyncio

import pytest
from viam.components.arm import KinematicsFileFormat
from viam.proto.app.robot import ComponentConfig, Frame
from viam.utils import dict_to_struct

import mujoco_module.models  # noqa: F401
from mujoco_module.models.arm import MujocoArm
from mujoco_module.models.rail import MujocoRail
from mujoco_module.models.world import MujocoWorld

LENGTH = 1500.0
HEIGHT_MM = 400.0


def config(name, attrs, parent=None, translation=(0.0, 0.0, HEIGHT_MM), yaw_deg=None, tilt=None):
    cfg = ComponentConfig(name=name, attributes=dict_to_struct(attrs))
    if parent is not None:
        cfg.frame.CopyFrom(Frame(parent=parent))
        cfg.frame.translation.x, cfg.frame.translation.y, cfg.frame.translation.z = translation
        if yaw_deg is not None:
            v = cfg.frame.orientation.vector_degrees
            v.x, v.y, v.z, v.theta = 0, 0, 1, yaw_deg
        if tilt is not None:
            v = cfg.frame.orientation.vector_degrees
            v.x, v.y, v.z, v.theta = tilt
    return cfg


def rail_config(**kw):
    return config("rail", {"world": "sim-world", "length_mm": LENGTH}, parent="world", **kw)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def rail(running_sim):
    MujocoWorld.new(config("sim-world", {}), {})
    return MujocoRail.new(rail_config(), {})


def test_validation():
    assert MujocoRail.validate_config(rail_config()) == (["sim-world"], [])
    assert MujocoRail.validate_config(rail_config(yaw_deg=90)) == (["sim-world"], [])
    with pytest.raises(ValueError, match="length_mm"):
        MujocoRail.validate_config(config("rail", {"world": "sim-world"}, parent="world"))
    with pytest.raises(ValueError, match="length_mm"):
        MujocoRail.validate_config(config("rail", {"world": "sim-world", "length_mm": 0}, parent="world"))
    with pytest.raises(ValueError, match="set a frame"):
        MujocoRail.validate_config(config("rail", {"world": "sim-world", "length_mm": LENGTH}))
    with pytest.raises(ValueError, match="only about z"):
        MujocoRail.validate_config(rail_config(tilt=(1, 0, 1, 0)))  # orientation vector off vertical
    with pytest.raises(ValueError, match="frame z"):
        MujocoRail.validate_config(rail_config(translation=(0.0, 0.0, 50.0)))
    for key in ("max_speed_mm_s", "max_accel_mm_s2", "move_timeout_sec", "payload_kg"):
        with pytest.raises(ValueError, match=key):
            MujocoRail.validate_config(config("rail", {"world": "sim-world", "length_mm": LENGTH, key: 0},
                                              parent="world"))


def test_moves_position_lengths_home_kinematics(rail):
    run(rail.move_to_position([1000.0], []))
    assert run(rail.get_position()) == [pytest.approx(1000.0, abs=1.0)]
    assert run(rail.get_lengths()) == [LENGTH]
    assert run(rail.home()) is True
    assert run(rail.get_position()) == [pytest.approx(1000.0, abs=1.0)]  # home doesn't move
    assert not run(rail.is_moving())
    fmt, data = run(rail.get_kinematics())[:2]
    assert fmt == KinematicsFileFormat.KINEMATICS_FILE_FORMAT_SVA and b'"prismatic"' in data


def test_stop_ends_a_move_and_holds(rail):
    async def stopped():
        move = asyncio.create_task(rail.move_to_position([LENGTH], []))
        await asyncio.sleep(0.2)
        await rail.stop()
        with pytest.raises(RuntimeError, match="superseded"):
            await move

    run(stopped())
    held = run(rail.get_position())
    assert 0 < held[0] < LENGTH
    assert not run(rail.is_moving())
    run(asyncio.sleep(0.2))
    assert run(rail.get_position()) == pytest.approx(held, abs=1.0)


def test_out_of_range_and_wrong_arity_are_rejected(rail):
    with pytest.raises(ValueError, match=r"x 1600.0 mm is outside its travel \[0, 1500\]"):
        run(rail.move_to_position([1600.0], []))
    with pytest.raises(ValueError, match="expected 1 positions"):
        run(rail.move_to_position([0.0, 0.0, 0.0], []))


@pytest.mark.parametrize("yaw, along", [(None, (1, 0)), (90, (0, 1))])
def test_an_arm_on_the_rail_rides_the_carriage(running_sim, yaw, along):
    MujocoWorld.new(config("sim-world", {}), {})
    rail = MujocoRail.new(rail_config(yaw_deg=yaw), {})
    MujocoArm.new(config("arm", {"world": "sim-world", "asset": "ur5e"}, parent="rail", translation=(0, 0, 0)), {})

    def base():  # the arm's base frame, on the carriage body (arm/viam_frame is its end effector)
        return running_sim.run(lambda: running_sim.data.site("arm/arm_frame").xpos.copy())

    before = base()
    run(rail.move_to_position([800.0], []))
    run(asyncio.sleep(0.5))  # settle with the arm's mass on the carriage
    moved = base() - before
    assert moved[:2] == pytest.approx([0.8 * along[0], 0.8 * along[1]], abs=0.005)
    assert moved[2] == pytest.approx(0, abs=0.005)
