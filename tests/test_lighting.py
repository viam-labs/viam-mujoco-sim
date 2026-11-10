import mujoco
from conftest import BLOCK
import numpy as np
import pytest
from test_scenes import CELL, attach

from mujoco_module import sim as sim_module
from mujoco_module.handles import CameraHandle
from mujoco_module.sim import Contribution, LightingConfig, WorldConfig

SKY = """
<mujoco>
  <asset><texture name="sky" type="skybox" builtin="gradient" rgb1="1 0 0" rgb2="1 0 0" width="32" height="32"/></asset>
  <worldbody><body name="post" pos="0 0 0.1"><geom type="box" size="0.01 0.01 0.1"/></body></worldbody>
</mujoco>
"""
NO_LIGHT_CELL = CELL.replace('<light name="lamp" pos="0 0 2"/>', "")


def configure(sim, lighting):
    """Reconfigure the world, keeping the fixture's block."""
    sim.configure_world(WorldConfig.from_attrs({"props": [BLOCK], "lighting": lighting}))


def lights(sim):
    return [sim.model.light(i).name for i in range(sim.model.nlight)]


# ---- parsing --------------------------------------------------------------


def test_no_lighting_is_the_defaults():
    assert WorldConfig.from_attrs({}).lighting == LightingConfig()
    assert WorldConfig.from_attrs({"lighting": None}).lighting == LightingConfig()
    assert WorldConfig.from_attrs({"lighting": {}}).lighting == LightingConfig()


@pytest.mark.parametrize("lighting,message", [
    ([], "lighting must be an object"),
    ({"lights": {}}, "lighting.lights must be a list"),
    ({"lights": [], "keep_default_light": True}, "keep_default_light only applies without lighting.lights"),
    ({"keep_default_light": 1}, "lighting.keep_default_light must be true or false"),
    ({"lights": [3]}, r"lighting.lights\[0\] must be an object"),
    ({"lights": [{"type": "area"}]}, r"lighting.lights\[0\].type must be"),
    ({"lights": [{}, {"direction": [0, 0, 0]}]}, r"lighting.lights\[1\].direction must not be all zeros"),
    ({"lights": [{"type": "point", "direction": [0, 0, 0]}]}, "direction must not be all zeros"),
    ({"lights": [{"position": [0, 0]}]}, r"lighting.lights\[0\].position must be a list of 3"),
    ({"lights": [{"color": [1, 2, 0]}]}, r"lighting.lights\[0\].color must be 3 numbers from 0 to 1"),
    ({"lights": [{"shadows": "no"}]}, r"lighting.lights\[0\].shadows must be true or false"),
    ({"lights": [{"cutoff_deg": 30}]}, r'cutoff_deg only applies to type "spot"'),
    ({"lights": [{"type": "spot", "cutoff_deg": 0}]}, r"cutoff_deg must be in \(0, 90\]"),
    ({"lights": [{"type": "spot", "cutoff_deg": 91}]}, r"cutoff_deg must be in \(0, 90\]"),
    ({"headlight": -0.1}, "lighting.headlight must be non-negative"),
    ({"skybox": [[1, 1, 1]]}, "lighting.skybox must be two colors"),
    ({"skybox": [[1, 1, 1], [0, 0, 2]]}, r"lighting.skybox\[1\] must be 3 numbers from 0 to 1"),
    ({"shadows": 0}, "lighting.shadows must be true or false"),
    ({"reflections": "yes"}, "lighting.reflections must be true or false"),
    ({"shadow_size": 0}, "lighting.shadow_size must be from 1 to 16384"),
    ({"shadow_size": 32768}, "lighting.shadow_size must be from 1 to 16384"),
    ({"shadow_size": 1024.5}, "lighting.shadow_size must be a whole number"),
    ({"antialias": 17}, "lighting.antialias must be from 0 to 16"),
    ({"antialias": True}, "lighting.antialias must be a number"),
    ({"shadow_size": float("inf")}, "lighting.shadow_size must be finite"),
    ({"antialias": float("nan")}, "lighting.antialias must be finite"),
])
def test_validation(lighting, message):
    with pytest.raises(ValueError, match=message):
        WorldConfig.from_attrs({"lighting": lighting})


def test_a_directional_light_defaults_up_its_ray_through_the_origin():
    lights = LightingConfig.parse({"lights": [
        {"direction": [0.4, 0.3, -1]},
        {"direction": [1, 0, 0]},
        {"type": "spot", "direction": [0.4, 0.3, -1]},
        {"direction": [0.4, 0.3, -1], "position": [1, 2, 3]},
        {},
        {"direction": [1, 0, -1e-308]},
    ]}).lights
    assert lights[0].position == pytest.approx((-1.2, -0.9, 3))
    assert lights[1].position == (0, 0, 3)  # not downward: the default height
    assert lights[2].position == (0, 0, 3)  # only directional lights follow their ray
    assert lights[3].position == (1, 2, 3)
    assert lights[4].position == (0, 0, 3)  # straight down: today's sun
    assert lights[5].position == (0, 0, 3)  # a near-horizontal ray would overflow


# ---- the model ------------------------------------------------------------


def test_defaults_render_as_before(sim):
    model = sim.model
    assert lights(sim) == ["sun"]
    sun = model.light("sun")
    assert sun.pos == pytest.approx([0, 0, 3])
    assert sun.dir == pytest.approx([0, 0, -1])
    assert sun.diffuse == pytest.approx([0.7, 0.7, 0.7])
    assert sun.type == mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    assert model.vis.headlight.active == 2
    assert model.vis.headlight.ambient == pytest.approx([0.1] * 3)
    assert model.vis.headlight.diffuse == pytest.approx([0.4] * 3)
    assert model.vis.headlight.specular == pytest.approx([0.5] * 3)
    assert (model.vis.quality.shadowsize, model.vis.quality.offsamples) == (4096, 4)


def test_lights_replace_the_sun(sim):
    configure(sim, {"lights": [
        {"direction": [0, 0, -2], "color": [1, 0.9, 0.8], "shadows": False},
        {"type": "spot", "position": [0.5, 0, 1.5], "cutoff_deg": 30},
        {"type": "point", "position": [0, 1, 1]},
    ]})
    assert lights(sim) == ["light_0", "light_1", "light_2"]
    first, spot, point = (sim.model.light(f"light_{i}") for i in range(3))
    assert first.dir == pytest.approx([0, 0, -1])  # MuJoCo normalizes it
    assert first.diffuse == pytest.approx([1, 0.9, 0.8])
    assert first.castshadow == 0
    assert spot.type == mujoco.mjtLightType.mjLIGHT_SPOT
    assert spot.cutoff == pytest.approx(30)
    assert spot.pos == pytest.approx([0.5, 0, 1.5])
    assert spot.castshadow == 1
    assert point.type == mujoco.mjtLightType.mjLIGHT_POINT


def test_no_lights(sim):
    configure(sim, {"lights": []})
    assert lights(sim) == []


def test_a_robot_light_keeps_the_sun(sim):
    sim.set_contribution("robot", Contribution(build=lambda spec: spec.worldbody.add_light(name="lamp")))
    assert lights(sim) == ["sun", "lamp"]


def test_headlight_and_quality(sim):
    configure(sim, {"headlight": 0.5, "shadow_size": 2048, "antialias": 0})
    vis = sim.model.vis
    assert vis.headlight.active == 2
    assert vis.headlight.ambient == pytest.approx([0.05] * 3)
    assert vis.headlight.diffuse == pytest.approx([0.2] * 3)
    assert vis.headlight.specular == pytest.approx([0.25] * 3)
    assert (vis.quality.shadowsize, vis.quality.offsamples) == (2048, 0)
    configure(sim, {"headlight": 0})
    assert sim.model.vis.headlight.active == 0


def test_a_scene_with_lights_drops_the_sun(sim, tmp_path):
    attach(sim, tmp_path)
    assert lights(sim) == ["cell/lamp"]
    sim.remove_contribution("cell")
    assert lights(sim) == ["sun"]


def test_keep_default_light(sim, tmp_path):
    configure(sim, {"keep_default_light": True})
    attach(sim, tmp_path)
    assert lights(sim) == ["sun", "cell/lamp"]


def test_a_scene_without_lights_keeps_the_sun(sim, tmp_path):
    attach(sim, tmp_path, xml=NO_LIGHT_CELL)
    assert lights(sim) == ["sun"]


def test_explicit_lights_stay_beside_scene_lights(sim, tmp_path):
    configure(sim, {"lights": [{}]})
    attach(sim, tmp_path)
    assert lights(sim) == ["light_0", "cell/lamp"]


# ---- rendering ------------------------------------------------------------


def sky_camera(sim):
    return CameraHandle(sim, "sky-cam", {"position": [0, 0, 0.5], "target": [1, 0, 3], "width": 64, "height": 48})


def test_skybox_gradient(sim):
    camera = sky_camera(sim)
    assert not any(sim.model.tex_type == mujoco.mjtTexture.mjTEXTURE_SKYBOX)
    assert camera.get_rgb().max() < 10  # no skybox: black
    configure(sim, {"skybox": [[0.2, 0.4, 0.9], [0.2, 0.4, 0.9]]})
    pixel = camera.get_rgb()[0, 32].astype(int)
    assert pixel[2] > 100 and pixel[2] > pixel[0] + 50  # blue


def test_a_scene_skybox_wins_and_warns_once(sim, tmp_path, monkeypatch):
    warned = []
    monkeypatch.setattr(sim_module.LOGGER, "warning", lambda *args: warned.append(args))
    configure(sim, {"skybox": [[0, 0, 1], [0, 0, 1]]})
    attach(sim, tmp_path, xml=SKY)
    camera = sky_camera(sim)  # a rebuild: the warning isn't repeated
    pixel = camera.get_rgb()[0, 32].astype(int)
    assert pixel[0] > 100 and pixel[0] > pixel[2] + 50  # the scene's red
    assert [a for a in warned if a[0].startswith("lighting.skybox")] == [
        ("lighting.skybox is ignored: %s is already in the scene", "cell/sky")
    ]
    assert "world/skybox" not in [sim.model.texture(i).name for i in range(sim.model.ntex)]


def test_render_flags(sim):
    camera = CameraHandle(
        sim, "cam", {"position": [0.3, -0.3, 0.3], "target": [0.3, 0, 0], "width": 64, "height": 48}
    )
    with_shadows = camera.get_rgb().astype(int)
    configure(sim, {"shadows": False, "reflections": False})  # a rebuild makes new renderers
    without = camera.get_rgb().astype(int)
    assert np.abs(with_shadows - without).max() > 20
    CameraHandle(sim, "cam2", {"width": 32, "height": 24})  # another rebuild
    camera.get_rgb()
    assert sim._renderers
    for renderer in sim._renderers.values():
        assert renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] == 0
        assert renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] == 0
