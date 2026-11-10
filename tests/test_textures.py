import io

import mujoco
import numpy as np
import pytest
from PIL import Image

from mujoco_module import meshes
from mujoco_module.meshes import add_texture, load_texture, parse_texture
from mujoco_module.sim import Contribution, WorldConfig
from test_meshes import FakeResponse, box_piece


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path / "data"))


def halves_array(size=64):
    """Left half red, right half blue."""
    img = np.zeros((size, size, 3), np.uint8)
    img[:, : size // 2] = [255, 0, 0]
    img[:, size // 2 :] = [0, 0, 255]
    return img


def halves():
    return (64, 64, halves_array().tobytes())


def quadrants():
    """2 x 2 tile: red top-left and bottom-right, blue elsewhere."""
    img = np.zeros((64, 64, 3), np.uint8)
    img[:, :] = [0, 0, 255]
    img[:32, :32] = [255, 0, 0]
    img[32:, 32:] = [255, 0, 0]
    return (64, 64, img.tobytes())


def write_image(path, mode="RGB", size=(64, 64)):
    image = Image.fromarray(halves_array()).resize(size).convert(mode)
    image.save(path)
    return path


@pytest.mark.parametrize("source", [
    "/abs/wood.PNG", "~/img/label.jpg", "textures/cardboard.png", "https://example.com/a/b.jpeg?v=2",
])
def test_parse_texture_accepts(source):
    assert parse_texture("box", source) == source


@pytest.mark.parametrize("source,message", [
    ("", "box: texture must be a file path or an https URL"),
    (3, "box: texture must be a file path or an https URL"),
    ("http://example.com/a.png", "box: texture URLs must be https, got http:"),
    ("/a/b.gif", r"box: texture must be a \.png, \.jpg or \.jpeg image, got '/a/b.gif'"),
])
def test_parse_texture_rejects(source, message):
    with pytest.raises(ValueError, match=message):
        parse_texture("box", source)


@pytest.mark.parametrize("name,mode", [
    ("a.png", "RGB"), ("b.png", "RGBA"), ("c.png", "L"), ("d.jpg", "RGB"),
])
def test_load_texture_reads_png_and_jpeg_as_rgb(tmp_path, name, mode):
    width, height, rgb = load_texture("box", str(write_image(tmp_path / name, mode)))
    assert (width, height) == (64, 64)
    assert len(rgb) == 64 * 64 * 3
    if mode != "L":
        pixels = np.frombuffer(rgb, np.uint8).reshape(64, 64, 3)
        assert pixels[0, 0, 0] > 200 and pixels[0, 0, 2] < 60  # red on the left
        assert pixels[0, 63, 2] > 200 and pixels[0, 63, 0] < 60  # blue on the right


def test_load_texture_downscales_large_images(tmp_path):
    width, height, rgb = load_texture("box", str(write_image(tmp_path / "big.png", size=(4000, 1000))))
    assert (width, height) == (2048, 512)
    assert len(rgb) == 2048 * 512 * 3


def test_load_texture_errors(tmp_path):
    with pytest.raises(ValueError, match="box: file .*nope.png not found"):
        load_texture("box", str(tmp_path / "nope.png"))
    (tmp_path / "junk.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="box: can't read image .*junk.png"):
        load_texture("box", str(tmp_path / "junk.png"))
    (tmp_path / "half.png").write_bytes(write_image(tmp_path / "ok.png").read_bytes()[:100])
    with pytest.raises(ValueError, match="box: can't read image .*half.png"):
        load_texture("box", str(tmp_path / "half.png"))


def test_load_texture_applies_exif_orientation(tmp_path):
    image = Image.fromarray(halves_array()).resize((64, 32))
    exif = Image.Exif()
    exif[0x0112] = 6
    image.save(tmp_path / "phone.jpg", exif=exif)
    assert load_texture("box", str(tmp_path / "phone.jpg"))[:2] == (32, 64)


def test_texture_url_is_downloaded_once(monkeypatch):
    buffer = io.BytesIO()
    Image.fromarray(halves_array()).save(buffer, format="PNG")
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        return FakeResponse(buffer.getvalue())

    monkeypatch.setattr(meshes.urllib.request, "urlopen", urlopen)
    url = "https://example.com/tex/label.png?v=1"
    assert load_texture("box", url)[:2] == (64, 64)
    assert load_texture("box", url)[:2] == (64, 64)
    assert calls == [url]


def test_add_texture_builds_a_material():
    spec = mujoco.MjSpec()
    name = add_texture(spec, "b/texture", halves())
    plain = spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.1, 0.1, 0.1])
    plain.material = name
    tiled = add_texture(spec, "floor/texture", quadrants(), repeat_m=0.5)
    model = spec.compile()
    assert name == "b/texture"
    assert model.material("b/texture").rgba.tolist() == [1, 1, 1, 1]
    assert model.material("b/texture").texuniform[0] == 0
    assert model.material(tiled).texrepeat.tolist() == pytest.approx([4, 4])
    assert model.material(tiled).texuniform[0] == 1
    assert model.geom(0).matid[0] == model.material("b/texture").id


def look_down(sim, x, y, height, fovy):
    """A camera named cam at (x, y, height) looking straight down; image x is
    world +x and image up is world +y."""
    def build(spec):
        spec.worldbody.add_camera(
            name="cam", pos=[x, y, height], xyaxes=[1, 0, 0, 0, 1, 0], fovy=fovy
        )
    sim.set_contribution("cam", Contribution(build=build))


def red(pixel):
    return pixel[0] > 100 and pixel[2] < 60  # a mesh renders darker than a box


def blue(pixel):
    return pixel[2] > 100 and pixel[0] < 60


def grey():
    return 64, 64, np.full((64, 64, 3), 128, np.uint8).tobytes()


def untinted(pixel):
    """Lit grey, no colour cast."""
    return int(max(pixel)) - int(min(pixel)) <= 10 and min(pixel) > 60


BOX_PROP = {"name": "b", "size": 0.1, "position": [0, 0, 0.05], "fixed": True,
            "color": [0.2, 0.8, 0.2], "texture": "/unused.png"}


def test_textured_prop_shows_its_image_upright(sim):
    sim.configure_world(WorldConfig.from_attrs({"props": [BOX_PROP]}), textures={"b": quadrants()})
    look_down(sim, 0, 0, 0.5, 10)  # sees only the top face; image up is world +y
    rgb, _ = sim.render("cam", 64, 64)
    # quadrants: red top-left and bottom-right; a flipped or mirrored image fails
    assert red(rgb[16, 16]) and blue(rgb[16, 48]) and blue(rgb[48, 16]) and red(rgb[48, 48])


def test_color_does_not_tint_a_textured_prop(sim):
    sim.configure_world(WorldConfig.from_attrs({"props": [BOX_PROP]}), textures={"b": grey()})
    look_down(sim, 0, 0, 0.5, 10)
    rgb, _ = sim.render("cam", 64, 64)
    assert untinted(rgb[32, 32])


def test_floor_texture_is_not_tinted(sim):
    sim.configure_world(WorldConfig.from_attrs({"floor_texture": "/unused.png"}), textures={"floor": grey()})
    look_down(sim, 3, 3, 2, 45)
    rgb, _ = sim.render("cam", 64, 64)
    assert untinted(rgb[32, 32])


def test_textured_mesh_prop_shows_its_image_on_every_piece(sim):
    prop = {"name": "m", "type": "mesh", "mesh": "/unused.stl", "position": [0, 0, 0.05],
            "fixed": True, "color": [0.2, 0.8, 0.2], "texture": "/unused.png"}
    pieces = {"m": [box_piece((0.05, 0.1, 0.1), (-0.025, 0, 0)),
                    box_piece((0.05, 0.1, 0.1), (0.025, 0, 0))]}
    sim.configure_world(WorldConfig.from_attrs({"props": [prop]}), pieces, {"m": halves()})
    material = sim.model.material("m/texture").id
    assert sim.model.geom("m/0").matid[0] == material
    assert sim.model.geom("m/1").matid[0] == material
    look_down(sim, 0, 0, 0.5, 10)
    rgb, _ = sim.render("cam", 64, 64)
    assert any(red(p) for p in rgb[32]) and any(blue(p) for p in rgb[32])
    assert rgb[..., 1].max() < 100  # no green: the colour doesn't show


def test_floor_tiles_are_floor_texture_size_across(sim):
    cfg = WorldConfig.from_attrs({"floor_texture": "/unused.png", "floor_texture_size": 0.5})
    sim.configure_world(cfg, textures={"floor": quadrants()})
    material = sim.model.material("floor/texture")
    assert material.texrepeat.tolist() == pytest.approx([4, 4])
    assert material.texuniform[0] == 1
    look_down(sim, 3, 3, 2, 45)
    rgb, _ = sim.render("cam", 256, 256)
    row = rgb[128, :, 0].astype(int) > 128  # red quadrants
    edges = np.flatnonzero(row[1:] != row[:-1])
    meters_per_px = 2 * 2 * np.tan(np.radians(22.5)) / 256
    # a 0.5 m tile is 2 x 2 quadrants: an edge every 0.25 m
    assert np.median(np.diff(edges)) * meters_per_px == pytest.approx(0.25, abs=0.01)


def test_texture_survives_a_rebuild_and_goes_when_dropped(sim):
    sim.configure_world(WorldConfig.from_attrs({"props": [BOX_PROP]}), textures={"b": halves()})
    look_down(sim, 0, 0, 0.5, 10)  # a contribution: rebuilds the scene
    assert sim.model.geom("b").matid[0] == sim.model.material("b/texture").id
    plain = {k: v for k, v in BOX_PROP.items() if k != "texture"}
    sim.configure_world(WorldConfig.from_attrs({"props": [plain]}))
    assert sim.model.geom("b").matid[0] == -1
    assert mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_MATERIAL, "b/texture") == -1


@pytest.mark.parametrize("attrs,message", [
    ({"floor_texture_size": 2}, "floor_texture_size only applies with floor_texture"),
    ({"floor_texture": "/f.png", "floor_texture_size": 0}, "floor_texture_size must be positive"),
    ({"floor_texture": "/f.gif"}, r"floor_texture: texture must be a \.png"),
    ({"props": [{"name": "floor"}]}, "prop floor: the name floor is taken by the floor"),
    ({"props": [{"name": "b", "texture": "http://x/a.png"}]}, "prop b: texture URLs must be https"),
])
def test_world_texture_validation(attrs, message):
    with pytest.raises(ValueError, match=message):
        WorldConfig.from_attrs(attrs)


def test_configure_world_needs_loaded_textures(sim):
    with pytest.raises(ValueError, match=r"textures not loaded for \['b'\]"):
        sim.configure_world(WorldConfig.from_attrs({"props": [BOX_PROP]}))
    with pytest.raises(ValueError, match=r"textures not loaded for \['floor'\]"):
        sim.configure_world(WorldConfig.from_attrs({"floor_texture": "/f.png"}))


def test_world_component_loads_its_textures(running_sim, tmp_path):
    from viam.proto.app.robot import ComponentConfig
    from viam.utils import dict_to_struct

    from mujoco_module.models.world import MujocoWorld

    image = str(write_image(tmp_path / "t.png"))
    attrs = {"real_time": False, "floor_texture": image,
             "props": [{"name": "b", "position": [0, 0, 0.5], "texture": image}]}
    MujocoWorld.new(ComponentConfig(name="sim-world", attributes=dict_to_struct(attrs)), {})
    assert running_sim.run(
        lambda: running_sim.model.geom("floor").matid[0]
        == running_sim.model.material("floor/texture").id
    )
    assert running_sim.run(
        lambda: running_sim.model.geom("b").matid[0]
        == running_sim.model.material("b/texture").id
    )


@pytest.mark.parametrize("name", ["textures/cardboard.png", "textures/concrete.png"])
def test_bundled_textures_load(name):
    assert load_texture("bundled", name)[:2] == (512, 512)
