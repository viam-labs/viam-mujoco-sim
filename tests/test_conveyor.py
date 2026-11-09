import asyncio
import math

import numpy as np
import pytest
from viam.proto.app.robot import ComponentConfig, Frame, Translation
from viam.utils import dict_to_struct

from mujoco_module.handles import CONVEYOR_KV, CONVEYOR_PERIOD_M, ConveyorHandle
from mujoco_module.models.conveyor import MujocoConveyor
from mujoco_module.sim import WorldConfig
from test_base import finish

TOP = 0.75  # the belt's top surface


def parcel(name, y, mass=0.5, size=(0.16, 0.12, 0.1), x=0.0):
    return {"name": name, "size": 1, "scale": list(size), "mass": mass, "position": [x, y, TOP + size[2] / 2 + 0.0005]}


def world(sim, *props):
    sim.configure_world(WorldConfig.from_attrs({"props": list(props)}))


@pytest.fixture
def belt(sim):
    return ConveyorHandle(sim, "belt", {"length_m": 2.0, "width_m": 0.5, "position": [0, 0, TOP]})


def body(sim, name):
    joint = sim.model.joint(name)
    return sim.data.qpos[joint.qposadr[0] : joint.qposadr[0] + 7], sim.data.qvel[joint.dofadr[0] : joint.dofadr[0] + 6]


@pytest.mark.parametrize("mass, size", [(0.5, (0.16, 0.12, 0.1)), (3.0, (0.4, 0.3, 0.24))])
@pytest.mark.parametrize("speed", [0.5, -0.2])
def test_a_parcel_rides_at_belt_speed_through_the_wraps(sim, belt, mass, size, speed):
    start = -0.6 if speed > 0 else 0.6
    world(sim, parcel("p", start, mass, size))
    sim.advance(0.5)  # settle
    belt.set_speed(speed)
    slide = sim.model.joint("belt/belt").qposadr[0]
    lowest = highest = 0.0
    for _ in range(int(2.0 / 0.002)):
        sim.step_once()
        q = sim.data.qpos[slide]
        lowest, highest = min(lowest, q), max(highest, q)
        qpos, qvel = body(sim, "p")
        assert abs(qvel[2]) < 0.01 and np.abs(qvel[3:]).max() < 0.05
    step = abs(speed) * 0.002  # the shift runs before the step
    assert -step - 1e-6 <= lowest and highest <= CONVEYOR_PERIOD_M + step + 1e-6
    assert abs(belt._wraps) == pytest.approx(20 * CONVEYOR_PERIOD_M * abs(speed) / 0.5, abs=0.11)
    qpos, qvel = body(sim, "p")
    assert qvel[1] == pytest.approx(speed, rel=0.01)
    ramp = abs(speed) / 2.0  # s at 2 m/s^2
    assert qpos[1] == pytest.approx(start + speed * (2.0 - ramp / 2), abs=0.01)
    assert qpos[2] == pytest.approx(TOP + size[2] / 2, abs=0.002)
    assert qpos[3:] == pytest.approx([1, 0, 0, 0], abs=0.01)


def test_speed_ramps_and_reset_stops_the_belt(sim, belt):
    belt.set_speed(0.5)
    sim.advance(0.1)
    dof = sim.model.joint("belt/belt").dofadr[0]
    assert sim.data.qvel[dof] == pytest.approx(0.2, abs=0.01)
    sim.advance(0.5)
    assert sim.data.qvel[dof] == pytest.approx(0.5, abs=0.001)
    assert belt.is_moving() and belt.power() == pytest.approx(1.0)
    assert belt.position() > 0
    sim.reset()
    sim.advance(0.01)
    assert not belt.is_moving() and belt.power() == 0
    assert belt.position() == pytest.approx(0, abs=1e-6)


def test_go_for_go_to_position_and_zero(sim, belt):
    per_rev = math.pi * 0.05
    future = belt.go_for(120, 5)  # 0.314 m/s for 0.785 m
    finish(sim, future).result()
    assert belt.position() == pytest.approx(5, abs=0.01)
    sim.advance(0.2)
    assert not belt.is_moving()
    finish(sim, belt.go_for(-120, 2)).result()  # backwards
    assert belt.position() == pytest.approx(3, abs=0.01)
    belt.reset_zero_position(1.0)  # as RDK's encoded motors: reads -offset
    assert belt.position() == pytest.approx(-1.0, abs=1e-6)
    assert belt.go_for(60, 0).done()
    with pytest.raises(ValueError, match="rpm must be non-zero"):
        belt.go_for(0, 1)
    # a parcel travels with the belt
    world(sim, parcel("p", -0.6))
    sim.advance(0.5)
    y0 = body(sim, "p")[0][1]
    finish(sim, belt.go_for(200, 4)).result()
    assert body(sim, "p")[0][1] - y0 == pytest.approx(4 * per_rev, abs=0.01)


def test_stop_a_new_command_and_reset_fail_a_move(sim, belt):
    future = belt.go_for(120, 50)
    sim.advance(0.5)
    belt.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        future.result(timeout=0)
    future = belt.go_for(120, 50)
    belt.set_speed(0.1)
    with pytest.raises(RuntimeError, match="superseded"):
        future.result(timeout=0)
    future = belt.go_for(120, 50)
    sim.reset()
    with pytest.raises(RuntimeError, match="reset"):
        future.result(timeout=0)
    future = belt.go_for(120, 50)
    future.cancel()
    sim.advance(1.0)
    assert not belt.is_moving()


def test_parcels_pile_up_against_a_stop_at_the_outfeed(sim, belt):
    stop = {"name": "stop", "fixed": True, "size": 1, "scale": [0.5, 0.02, 0.2], "position": [0, 0.61, TOP + 0.1]}
    world(sim, stop, parcel("a", -0.2, 3.0, (0.2, 0.2, 0.2)), parcel("b", -0.7, 0.5))
    belt.set_speed(0.5)
    sim.advance(5.0)
    dof = sim.model.joint("belt/belt").dofadr[0]
    speeds = []
    for _ in range(500):  # 1 s
        sim.step_once()
        speeds.append(sim.data.qvel[dof])
    # stick-slip against the stop chatters a few mm
    assert body(sim, "a")[0][1] == pytest.approx(0.6 - 0.1, abs=0.01)
    assert body(sim, "b")[0][1] == pytest.approx(0.6 - 0.2 - 0.06, abs=0.01)
    # the belt slips under both parcels; their friction (1 x 3.5 kg x g) droops the
    # drive. Stick-slip makes the speed chatter step to step (0.34-0.50 m/s), so
    # compare the mean over a second: at least the friction droop, at most 3x it.
    droop = 3.5 * 9.81 / CONVEYOR_KV
    assert 0.5 - 3 * droop < np.mean(speeds) < 0.5 - droop / 2


def test_a_static_geom_touching_the_belt_does_not_stall_it(sim, belt):
    # a fixed box sunk into the belt across its whole width
    world(sim, {"name": "jam", "fixed": True, "size": 1, "scale": [0.6, 0.1, 0.1], "position": [0, 0, TOP]})
    belt.set_speed(0.5)
    sim.advance(1.0)
    assert sim.data.qvel[sim.model.joint("belt/belt").dofadr[0]] == pytest.approx(0.5, abs=0.001)


def test_belt_has_rails_legs_and_a_frame_site(sim, belt):
    assert sim.data.site("belt/viam_frame").xpos == pytest.approx([0, 0, TOP])
    rail = sim.data.geom("belt/rail_1")
    assert rail.xpos[0] == pytest.approx(0.26) and rail.xpos[2] + sim.model.geom("belt/rail_1").size[2] == pytest.approx(TOP + 0.04)
    leg = sim.model.geom("belt/leg_0")
    assert sim.data.geom("belt/leg_0").xpos[2] - leg.size[2] == pytest.approx(0, abs=1e-9)
    assert sim.model.body("belt/belt").parentid[0] == sim.model.body("belt/frame").id


def conveyor_config(attrs=None, z=750, frame=True):
    cfg = ComponentConfig(
        name="belt",
        attributes=dict_to_struct({"world": "sim-world", "length_m": 2.0, "width_m": 0.5, **(attrs or {})}),
    )
    if frame:
        cfg.frame.CopyFrom(Frame(parent="world", translation=Translation(z=z)))
    return cfg


@pytest.mark.parametrize("cfg, message", [
    (conveyor_config(frame=False), "set a frame"),
    (conveyor_config(z=0), "must be positive"),
    (conveyor_config({"length_m": -1}), "length_m"),
    (conveyor_config({"max_speed_mps": 0}), "max_speed_mps"),
    (conveyor_config({"rails_m": -0.1}), "rails_m must be >= 0"),
    (conveyor_config({"color": "red"}), "color"),
])
def test_validate_rejects(cfg, message):
    with pytest.raises(ValueError, match=message):
        MujocoConveyor.validate_config(cfg)


def test_motor_model(running_sim):
    from mujoco_module.models.world import MujocoWorld

    assert MujocoConveyor.validate_config(conveyor_config()) == (["sim-world"], [])
    MujocoWorld.new(ComponentConfig(name="sim-world", attributes=dict_to_struct({"real_time": False})), {})
    motor = MujocoConveyor.new(conveyor_config(), {})

    async def run():
        assert (await motor.get_properties()).position_reporting
        await motor.set_power(0.5)
        assert await motor.is_powered() == (True, 0.5)
        assert await motor.is_moving()
        await motor.stop()
        await motor.reset_zero_position(0)
        await motor.go_for(300, 3)
        assert await motor.get_position() == pytest.approx(3, abs=0.02)
        await motor.go_to(300, 1)
        assert await motor.get_position() == pytest.approx(1, abs=0.02)
        await motor.set_rpm(60)
        assert (await motor.is_powered())[1] == pytest.approx(math.pi * 0.05 / 0.5)
        body, rail, _ = await motor.get_geometries()
        assert (body.center.z, body.box.dims_mm.z, body.box.dims_mm.x) == pytest.approx((-375, 750, 500))
        assert (rail.center.x, rail.center.z, rail.box.dims_mm.z) == pytest.approx((-260, 20, 40))
        await motor.close()

    asyncio.run(run())
