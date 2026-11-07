import mujoco
import pytest

from mujoco_module.sim import MAX_SUBSTEPS, SLIP_THRESHOLD_S, WorldConfig
from conftest import BLOCK


def test_props_round_trip_and_zero_velocity(sim):
    joint = sim.model.joint("pick_block")
    sim.data.qvel[joint.dofadr[0] : joint.dofadr[0] + 6] = 1.0
    result = sim.set_prop_pose("pick_block", [0.2, 0.1, 0.3], [0, 0, 0, 2])
    assert result["prop"] == "pick_block"
    assert result["position"] == pytest.approx([0.2, 0.1, 0.3])
    assert result["orientation_wxyz"] == pytest.approx([0, 0, 0, 1])
    assert (sim.data.qvel[joint.dofadr[0] : joint.dofadr[0] + 6] == 0).all()
    assert sim.get_prop_pose("pick_block")["position"] == pytest.approx([0.2, 0.1, 0.3])


def test_park_props_restores_spawn(sim):
    sim.set_prop_pose("pick_block", [0.5, 0.5, 0.5], [1, 0, 0, 0])
    parked = sim.park_props(["pick_block"])
    assert parked["poses"]["pick_block"]["position"] == pytest.approx(BLOCK["position"])


def test_fixed_and_unknown_props_are_rejected(sim):
    pad = {"name": "pad", "fixed": True, "size": 0.2, "position": [0.3, -0.2, 0.005]}
    sim.configure_world(WorldConfig.from_attrs({"props": [BLOCK, pad]}))
    with pytest.raises(ValueError, match="fixed"):
        sim.set_prop_pose("pad", [0, 0, 0], [1, 0, 0, 0])
    with pytest.raises(ValueError, match="unknown prop"):
        sim.get_prop_pose("nope")


def test_prop_config_validation():
    with pytest.raises(ValueError, match="only type 'cube'"):
        WorldConfig.from_attrs({"props": [{"name": "s", "type": "sphere"}]})
    with pytest.raises(ValueError, match="non-negative"):
        WorldConfig.from_attrs({"props": [{"name": "b", "static_friction": -1}]})
    with pytest.raises(ValueError, match="duplicate"):
        WorldConfig.from_attrs({"props": [{"name": "b"}, {"name": "b"}]})
    WorldConfig.from_attrs({"props": [{"name": "b", "static_friction": 0}]})


def test_bad_contribution_keeps_running_scene(sim):
    from mujoco_module.sim import Contribution

    before = sim.generation

    def clash(spec):
        spec.worldbody.add_body(name="pick_block")  # duplicate body name

    with pytest.raises(ValueError):
        sim.set_contribution("bad", Contribution(build=clash))
    assert sim.generation == before
    assert "bad" not in sim._contributions


def test_real_time_pacing_and_slips(sim, clock):
    sim.tick()
    assert sim.data.time == 0.0
    clock.now = 0.05
    sim.tick()
    assert sim.data.time == pytest.approx(0.05, abs=sim.model.opt.timestep)
    clock.now = 10.0  # far behind: one tick can only do MAX_SUBSTEPS steps
    sim.tick()
    assert sim.slips == 1
    assert 10.0 - sim.data.time > SLIP_THRESHOLD_S
    assert sim.data.time == pytest.approx(0.05 + MAX_SUBSTEPS * 0.002, abs=0.002)


def test_pause_stops_stepping(sim, clock):
    sim.pause()
    clock.now = 1.0
    sim.tick()
    assert sim.data.time == 0.0
    sim.play()
    clock.now = 1.02
    sim.tick()
    assert sim.data.time == pytest.approx(0.02, abs=0.002)


def test_status_reports_rtf_and_warnings(sim, clock):
    for step in range(1, 11):
        clock.now = step * 0.01
        sim.tick()
    status = sim.status()
    assert status["playing"] is True
    assert status["real_time_factor"] == pytest.approx(1.0, abs=0.25)
    sim.data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number = 2
    assert sim.status()["warnings"] == {"mjWARN_BADQACC": 2}


def test_failing_step_callback_is_removed(sim):
    def boom(_dt):
        raise RuntimeError("boom")

    sim.step_callbacks["boom"] = boom
    sim.advance(0.01)
    assert "boom" not in sim.step_callbacks
