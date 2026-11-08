import asyncio

import mujoco
import pytest
from viam.proto.app.robot import ComponentConfig, Frame, Translation
from viam.utils import dict_to_struct

from mujoco_module.handles import HIDDEN_GROUP, ConveyorHandle, LidarHandle, PhotoEyeHandle, ScaleHandle
from mujoco_module.models.photo_eye import MujocoPhotoEye
from mujoco_module.models.scale import MujocoScale
from mujoco_module.sim import Contribution, WorldConfig
from test_conveyor import TOP, parcel, world

EYE = {"parent": "belt", "position": [-0.3, 0.0, 0.03], "range_m": 1.0}


@pytest.fixture
def belt(sim):
    return ConveyorHandle(sim, "belt", {"length_m": 2.0, "width_m": 0.5, "position": [0, 0, TOP]})


def test_an_eye_on_the_conveyor_counts_a_passing_parcel(sim, belt):
    world(sim, parcel("p", -0.5))
    eye = PhotoEyeHandle(sim, "eye", EYE)
    sim.advance(0.1)
    # the rails and belt are the parent's: the beam crosses them
    assert eye.readings() == {"blocked": False, "distance_m": 1.0, "count": 0}
    belt.set_speed(0.5)
    sim.advance(1.0)  # the parcel's 0.12 m side is across the beam
    reading = eye.readings()
    assert reading["blocked"] and reading["count"] == 1
    assert reading["distance_m"] == pytest.approx(0.3 - 0.08, abs=0.002)
    sim.advance(1.0)
    assert eye.readings() == {"blocked": False, "distance_m": 1.0, "count": 1}
    eye.reset_count()
    assert eye.readings()["count"] == 0


def test_a_world_eye_sees_the_rail_and_reset_zeroes_the_count(sim, belt):
    eye = PhotoEyeHandle(sim, "eye", {"position": [-0.3, 0.0, TOP + 0.03], "range_m": 1.0})
    sim.advance(0.01)
    assert eye.readings() == {"blocked": True, "distance_m": pytest.approx(0.03), "count": 1}
    sim.reset()
    assert eye.readings()["count"] == 0


def test_an_eye_reads_before_the_first_step(sim, belt):
    eye = PhotoEyeHandle(sim, "eye", EYE)
    assert eye.readings() == {"blocked": False, "distance_m": 1.0, "count": 0}


def test_an_eye_waits_for_its_parent(sim):
    eye = PhotoEyeHandle(sim, "eye", EYE)
    with pytest.raises(RuntimeError, match="not in the scene"):
        eye.readings()
    ConveyorHandle(sim, "belt", {"length_m": 2.0, "width_m": 0.5, "position": [0, 0, TOP]})
    sim.advance(0.01)
    assert not eye.readings()["blocked"]


def test_rays_skip_the_hidden_group(sim):
    sim.configure_world(WorldConfig.from_attrs({}))

    def build(spec):
        spec.worldbody.add_geom(
            name="ghost", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 2, 0.5], pos=[1, 0, 0.5], group=HIDDEN_GROUP
        )

    sim.set_contribution("ghost", Contribution(build=build))
    eye = PhotoEyeHandle(sim, "eye", {"position": [0, 0, 0.2]})
    sim.advance(0.01)
    assert not eye.readings()["blocked"]
    assert len(LidarHandle(sim, "lidar", {"position": [0, 0, 0.2]}).scan()) == 0
    sim.model.geom_group[sim.model.geom("ghost").id] = 0
    sim.advance(0.01)
    assert eye.readings()["distance_m"] == pytest.approx(0.95)


SCALE = {"size_m": [0.4, 0.4, 0.4], "position": [0, 0.8, 0.4]}


@pytest.mark.parametrize("mass, size", [(0.5, (0.12, 0.1, 0.08)), (3.0, (0.3, 0.24, 0.2))])
def test_scale_weighs_a_dropped_parcel(sim, mass, size):
    scale = ScaleHandle(sim, "scale", SCALE)
    sim.advance(0.5)
    assert scale.readings() == {"weight_kg": 0.0, "stable": True}
    assert scale.readings()["stable"] is True  # a bool, not numpy's (serialized as 1)
    scale.weight_kg = 1.2e-322  # what the filter decays to on a real machine
    assert str(scale.readings()["weight_kg"]) == "0.0"
    sim.configure_world(WorldConfig.from_attrs({"props": [
        {"name": "p", "size": 1, "scale": list(size), "mass": mass, "position": [0, 0.8, 0.4 + size[2] / 2 + 0.05]},
    ]}))
    sim.advance(0.15)
    assert not scale.readings()["stable"]  # landing
    sim.advance(1.5)
    reading = scale.readings()
    assert reading["stable"]
    assert reading["weight_kg"] == pytest.approx(mass, rel=0.02)
    assert scale.tare() == pytest.approx(mass, rel=0.02)
    assert scale.readings()["weight_kg"] == pytest.approx(0, abs=1e-9)
    sim.reset()  # the filter restarts; the tare stays
    assert scale.readings() == {"weight_kg": pytest.approx(-mass, rel=0.02), "stable": False}


def config(name, attrs, parent="world", y=0, z=0):
    cfg = ComponentConfig(name=name, attributes=dict_to_struct({"world": "sim-world", **attrs}))
    cfg.frame.CopyFrom(Frame(parent=parent, translation=Translation(y=y, z=z)))
    return cfg


def test_validate():
    assert MujocoPhotoEye.validate_config(config("eye", {}, parent="belt")) == (["sim-world"], [])
    with pytest.raises(ValueError, match="range_m"):
        MujocoPhotoEye.validate_config(config("eye", {"range_m": 0}))
    assert MujocoScale.validate_config(config("scale", {"size_m": [0.4, 0.4, 0.4]})) == (["sim-world"], [])
    with pytest.raises(ValueError, match="size_m"):
        MujocoScale.validate_config(config("scale", {"size_m": [0.4, 0.4]}))
    with pytest.raises(ValueError, match="frame.parent"):
        MujocoScale.validate_config(config("scale", {"size_m": [0.4, 0.4, 0.4]}, parent="belt"))
    no_frame = ComponentConfig(name="scale", attributes=dict_to_struct({"world": "sim-world", "size_m": [0.4] * 3}))
    with pytest.raises(ValueError, match="set a frame"):
        MujocoScale.validate_config(no_frame)


def test_sensor_models(running_sim):
    from mujoco_module.models.world import MujocoWorld

    MujocoWorld.new(ComponentConfig(name="sim-world", attributes=dict_to_struct({"real_time": False})), {})
    eye = MujocoPhotoEye.new(config("eye", {"range_m": 1.0}, y=1000, z=100), {})
    scale = MujocoScale.new(config("scale", {"size_m": [0.4, 0.4, 0.4]}, z=400), {})

    async def run():
        assert await eye.get_readings() == {"blocked": False, "distance_m": 1.0, "count": 0}
        assert await eye.do_command({"command": "reset"}) == {"count": 0}
        await asyncio.sleep(0.2)
        assert (await scale.get_readings())["stable"]
        assert await scale.do_command({"command": "tare"}) == {"tare_kg": 0.0}
        (box,) = await scale.get_geometries()
        assert (box.center.z, box.box.dims_mm.x) == (-200, 400)
        with pytest.raises(ValueError, match="unknown command"):
            await scale.do_command({"command": "zero"})
        await eye.close()
        await scale.close()

    asyncio.run(run())
