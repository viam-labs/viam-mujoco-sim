import asyncio
import json
import pathlib
import types

import mujoco
import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from mujoco_module.models.scene import load_scene, scene_contribution
from mujoco_module.sensors import ImuHandle, MjcfSensorHandle
from mujoco_module.sim import Contribution

# a pad the cube rests on (touch), a hinged door (jointpos), the cube's
# position (framepos), and one unnamed sensor that must be skipped
CELL = """
<mujoco model="cell">
  <worldbody>
    <body name="pad" pos="0 0 0.1">
      <geom name="pad" type="box" size="0.1 0.1 0.01"/>
      <site name="pad_top" type="box" size="0.1 0.1 0.005" pos="0 0 0.01"/>
    </body>
    <body name="cube" pos="0 0 0.14">
      <freejoint name="cube_joint"/>
      <geom type="box" size="0.02 0.02 0.02" mass="0.05"/>
    </body>
    <body name="door" pos="0.5 0.5 0.3">
      <joint name="hinge" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.01 0.2 0.3"/>
    </body>
  </worldbody>
  <sensor>
    <touch name="pad" site="pad_top"/>
    <jointpos name="door" joint="hinge"/>
    <framepos name="cube_pos" objtype="body" objname="cube"/>
    <jointvel joint="hinge"/>
  </sensor>
</mujoco>
"""

UNNAMED_ONLY = """
<mujoco model="bare">
  <worldbody>
    <body name="door" pos="0.5 0.5 0.3">
      <joint name="hinge" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.01 0.2 0.3"/>
    </body>
  </worldbody>
  <sensor><jointpos joint="hinge"/></sensor>
</mujoco>
"""


def attach(sim, tmp_path, xml=CELL, name="cell"):
    path = tmp_path / f"{name}.xml"
    path.write_text(xml)
    scene = load_scene(name, str(path), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    sim.set_contribution(name, scene_contribution(name, scene, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))


def set_hinge(sim, name, angle):
    def go():
        sim.data.qpos[sim.model.joint(name).qposadr[0]] = angle
        mujoco.mj_forward(sim.model, sim.data)

    sim.run(go)


def test_reads_every_named_sensor_of_the_file(sim, tmp_path):
    attach(sim, tmp_path)
    ImuHandle(sim, "imu", {"position": [0, 0, 0.5]})  # the world's own sensors are not the file's
    handle = MjcfSensorHandle(sim, "bench", "cell")
    sim.advance(1.0)
    r = handle.read()
    assert list(r) == ["pad", "door", "cube_pos"]  # file order; unnamed jointvel skipped
    assert r["pad"] == pytest.approx(0.05 * 9.81, abs=0.01)
    assert r["door"] == pytest.approx(0.0, abs=1e-6)
    assert r["cube_pos"] == pytest.approx([0.0, 0.0, 0.13], abs=0.003)
    assert isinstance(r["pad"], float) and isinstance(r["cube_pos"], list)


def test_jointpos_follows_the_joint(sim, tmp_path):
    attach(sim, tmp_path)
    handle = MjcfSensorHandle(sim, "bench", "cell")
    set_hinge(sim, "cell/hinge", 0.5)
    assert handle.read()["door"] == pytest.approx(0.5)


def test_a_subset_in_config_order(sim, tmp_path):
    attach(sim, tmp_path)
    handle = MjcfSensorHandle(sim, "bench", "cell", ["door", "pad"])
    assert list(handle.read()) == ["door", "pad"]


def test_a_rebuild_reresolves_the_addresses(sim, tmp_path):
    attach(sim, tmp_path)
    handle = MjcfSensorHandle(sim, "bench", "cell", ["door"])
    set_hinge(sim, "cell/hinge", 0.3)
    assert handle.read()["door"] == pytest.approx(0.3)
    before = int(sim.model.sensor("cell/door").adr[0])
    ImuHandle(sim, "imu", {"position": [0, 0, 0.5]})
    sim.remove_contribution("cell")
    attach(sim, tmp_path)  # now built after the imu: the file's sensors move
    assert int(sim.model.sensor("cell/door").adr[0]) != before
    set_hinge(sim, "cell/hinge", -0.4)
    assert handle.read()["door"] == pytest.approx(-0.4)


def test_source_not_in_the_scene(sim):
    handle = MjcfSensorHandle(sim, "bench", "nope")
    with pytest.raises(RuntimeError, match="bench: nope is not in the scene; configure a component named nope"):
        handle.read()


def test_a_file_without_named_sensors(sim, tmp_path):
    attach(sim, tmp_path, UNNAMED_ONLY, "bare")
    with pytest.raises(RuntimeError, match="bench: bare's file declares no named sensors$"):
        MjcfSensorHandle(sim, "bench", "bare").read()


def test_an_arm_inside_a_robot_points_to_the_robot(sim):
    sim.set_contribution("limb", Contribution(build=lambda spec: None))
    sim.arms["limb"] = types.SimpleNamespace(robot="bot")
    with pytest.raises(RuntimeError, match="no named sensors; its file's sensors are under the robot bot: set source to bot"):
        MjcfSensorHandle(sim, "bench", "limb").read()


def test_an_unknown_name_lists_the_file_sensors(sim, tmp_path):
    attach(sim, tmp_path)
    handle = MjcfSensorHandle(sim, "bench", "cell", ["pad", "lid"])
    with pytest.raises(RuntimeError, match="cell's file has no sensor 'lid'; it has: pad, door, cube_pos"):
        handle.read()


def test_built_names_the_contributions_in_the_model(sim, tmp_path):
    assert not sim.built("cell")
    attach(sim, tmp_path)
    assert sim.built("cell")
    sim.remove_contribution("cell")
    assert not sim.built("cell")


def sensor_config(**attrs):
    return ComponentConfig(name="bench", attributes=dict_to_struct({"world": "sim-world", **attrs}))


@pytest.mark.parametrize("attrs,message", [
    ({}, "bench: source must be the name of the component whose MJCF file declares the sensors"),
    ({"source": ""}, "bench: source must be the name"),
    ({"source": 3}, "bench: source must be the name"),
    ({"source": "cell", "sensors": []}, "bench: sensors must be a non-empty list of distinct sensor names"),
    ({"source": "cell", "sensors": "pad"}, "bench: sensors must be a non-empty list"),
    ({"source": "cell", "sensors": ["pad", ""]}, "bench: sensors must be a non-empty list"),
    ({"source": "cell", "sensors": ["pad", "pad"]}, "bench: sensors must be a non-empty list of distinct"),
])
def test_validate_rejects(attrs, message):
    from mujoco_module.models.sensor import MujocoSensor

    with pytest.raises(ValueError, match=message):
        MujocoSensor.validate_config(sensor_config(**attrs))


def test_validate_needs_the_world():
    from mujoco_module.models.sensor import MujocoSensor

    cfg = ComponentConfig(name="bench", attributes=dict_to_struct({"source": "cell"}))
    with pytest.raises(ValueError, match='set the "world" attribute'):
        MujocoSensor.validate_config(cfg)


def test_component_serves_readings(running_sim, tmp_path):
    from mujoco_module.models.scene import MujocoScene
    from mujoco_module.models.sensor import MujocoSensor
    from mujoco_module.models.world import MujocoWorld
    from test_scenes import config as scene_config

    MujocoWorld.new(ComponentConfig(name="sim-world", attributes=dict_to_struct({"real_time": False})), {})
    path = tmp_path / "cell.xml"
    path.write_text(CELL)
    scene = MujocoScene.new(scene_config(str(path)), {})  # the component is named "cell"
    cfg = sensor_config(source="cell", sensors=["pad", "door"])
    assert MujocoSensor.validate_config(cfg) == (["sim-world"], [])
    sensor = MujocoSensor.new(cfg, {})
    try:
        readings = asyncio.run(sensor.get_readings())
        assert list(readings) == ["pad", "door"]
        with pytest.raises(ValueError, match="unknown command"):
            asyncio.run(sensor.do_command({"command": "tare"}))
        assert asyncio.run(sensor.get_geometries()) == []
    finally:
        asyncio.run(sensor.close())
        asyncio.run(scene.close())
    with pytest.raises(RuntimeError, match="not attached to the sim"):
        asyncio.run(sensor.get_readings())


def test_bundled_workbench_sensors(sim):
    scene = load_scene("workbench", "scenes/workbench.xml")
    sim.set_contribution("workbench", scene_contribution("workbench", scene, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))
    handle = MjcfSensorHandle(sim, "bench-sensors", "workbench")
    sim.advance(2.0)
    r = handle.read()
    assert list(r) == ["platform", "flap"]
    assert r["platform"] == pytest.approx(2 * 0.03 * 9.81, abs=0.01)  # both blocks rest on it
    assert r["flap"] == pytest.approx(0.0, abs=1e-3)


def test_scene_pick_reads_the_workbench_sensors():
    cfg = json.loads((pathlib.Path(__file__).parent.parent / "examples" / "scene-pick.json").read_text())
    sensor = next(c for c in cfg["components"] if c["name"] == "bench-sensors")
    assert sensor["model"] == "viam-labs:mujoco-sim:sensor"
    assert sensor["api"] == "rdk:component:sensor"
    assert sensor["attributes"] == {"world": "sim-world", "source": "workbench"}
