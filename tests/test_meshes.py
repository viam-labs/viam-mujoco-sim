import urllib.error

import mujoco
import numpy as np
import pytest
import trimesh

from mujoco_module import meshes
from mujoco_module.meshes import MeshSource, add_pieces, load_pieces, parse_mesh, pieces_ply


def box_piece(extents, center=(0, 0, 0)):
    box = trimesh.creation.box(extents).apply_translation(center)
    return np.asarray(box.vertices, float), np.asarray(box.faces, np.int32)


@pytest.mark.parametrize("source", [
    "/abs/part.STL", "~/cad/part.obj", "meshes/bin.stl", "https://example.com/a/part.ply?v=2",
])
def test_parse_mesh_accepts(source):
    mesh = parse_mesh("bin", source, 0.001, 0.1)
    assert mesh == MeshSource(source, (0.001, 0.001, 0.001), 0.1)


def test_parse_mesh_scale_vector_and_defaults():
    assert parse_mesh("bin", "/a.stl", [1, 2, 3]).scale == (1.0, 2.0, 3.0)
    assert parse_mesh("bin", "/a.stl") == MeshSource("/a.stl", (1.0, 1.0, 1.0), 0.05)


@pytest.mark.parametrize("source,scale,threshold,message", [
    ("", 1, 0.05, "mesh must be"),
    (3, 1, 0.05, "mesh must be"),
    ("/a.step", 1, 0.05, r"\.stl, \.obj or \.ply"),
    ("http://example.com/a.stl", 1, 0.05, "https"),
    ("file:///a.stl", 1, 0.05, "https"),
    ("/a.stl", 0, 0.05, "scale"),
    ("/a.stl", [1, 1], 0.05, "scale"),
    ("/a.stl", True, 0.05, "scale"),
    ("/a.stl", float("nan"), 0.05, "scale"),
    ("/a.stl", 1, 0.005, "threshold"),
    ("/a.stl", 1, 1.5, "threshold"),
])
def test_parse_mesh_rejects(source, scale, threshold, message):
    with pytest.raises(ValueError, match=message) as error:
        parse_mesh("bin", source, scale, threshold)
    assert str(error.value).startswith("bin: ")


def test_pieces_ply_merges_pieces_in_meters():
    a, b = box_piece((0.1, 0.1, 0.1)), box_piece((0.1, 0.1, 0.1), (0.3, 0, 0))
    text = pieces_ply([a, b]).decode()
    assert "format ascii 1.0" in text
    assert "element vertex 16" in text and "element face 24" in text
    loaded = trimesh.load_mesh(__import__("io").BytesIO(text.encode()), file_type="ply")
    assert loaded.bounds[1][0] == pytest.approx(0.35)


def test_add_pieces_splits_mass_by_volume_and_compiles_thin_pieces():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="u", pos=[0, 0, 0.5])
    body.add_freejoint(name="u")
    pieces = [box_piece((0.1, 0.002, 0.1)), box_piece((0.1, 0.002, 0.1), (0, 0.05, 0)),
              box_piece((0.1, 0.1, 0.004), (0, 0.025, -0.05))]
    add_pieces(spec, body, "u", pieces, mass=0.4, rgba=[1, 0, 0, 1])
    assert [g.mass for g in body.geoms] == pytest.approx([0.1, 0.1, 0.2])
    model = spec.compile()
    assert model.body("u").mass[0] == pytest.approx(0.4)
    assert model.geom("u/0").type[0] == mujoco.mjtGeom.mjGEOM_MESH
    assert model.geom("u/0").rgba[0] == pytest.approx(1.0)


def test_add_pieces_without_mass_on_a_static_body():
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="bin/obstacle")
    add_pieces(spec, body, "bin/obstacle", [box_piece((0.1, 0.002, 0.1))])
    model = spec.compile()
    assert model.geom("bin/obstacle/0").type[0] == mujoco.mjtGeom.mjGEOM_MESH


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VIAM_MODULE_DATA", str(tmp_path / "data"))
    return tmp_path / "data" / "mesh-cache"


@pytest.mark.parametrize("ext", ["stl", "obj", "ply"])
def test_load_pieces_reads_each_format_and_scales(tmp_path, ext):
    path = tmp_path / f"cube.{ext}"
    trimesh.creation.box((100, 100, 200)).export(path)  # mm, like CAD
    pieces = load_pieces("cube", parse_mesh("cube", str(path), 0.001))
    assert len(pieces) == 1  # convex: one piece
    vertices = np.vstack([v for v, _ in pieces])
    assert vertices.max(axis=0) == pytest.approx([0.05, 0.05, 0.1], abs=0.002)


def test_load_pieces_relative_path_is_under_assets(monkeypatch, tmp_path):
    (tmp_path / "meshes").mkdir()
    trimesh.creation.box((0.1, 0.1, 0.1)).export(tmp_path / "meshes" / "cube.stl")
    monkeypatch.setattr(meshes, "ASSETS_DIR", tmp_path)
    assert len(load_pieces("cube", parse_mesh("cube", "meshes/cube.stl"))) == 1


def test_bundled_bin_decomposes_into_pieces_covering_it(u_pieces):
    assert 5 <= len(u_pieces) <= meshes.MAX_PIECES  # 13 at threshold 0.05
    vertices = np.vstack([v for v, _ in u_pieces])
    assert vertices.min(axis=0) == pytest.approx([-0.075, -0.075, 0.0], abs=0.003)
    assert vertices.max(axis=0) == pytest.approx([0.075, 0.075, 0.08], abs=0.003)


def test_missing_and_empty_files(tmp_path):
    with pytest.raises(ValueError, match="cube: mesh file .*nope.stl not found"):
        load_pieces("cube", parse_mesh("cube", str(tmp_path / "nope.stl")))
    (tmp_path / "empty.stl").write_bytes(b"")
    with pytest.raises(ValueError, match="cube: .*empty.stl"):
        load_pieces("cube", parse_mesh("cube", str(tmp_path / "empty.stl")))


def test_cache_hit_skips_coacd_and_key_covers_settings(tmp_path, monkeypatch, cache):
    path = tmp_path / "cube.stl"
    trimesh.creation.box((0.1, 0.1, 0.1)).export(path)
    first = load_pieces("cube", parse_mesh("cube", str(path)))
    assert len(list((cache / "pieces").glob("*.npz"))) == 1

    def boom(*args, **kwargs):
        raise AssertionError("CoACD ran on a cache hit")

    monkeypatch.setattr(meshes.coacd, "run_coacd", boom)
    again = load_pieces("cube", parse_mesh("cube", str(path)))
    assert np.array_equal(again[0][0], first[0][0])
    with pytest.raises(AssertionError, match="cache hit"):
        load_pieces("cube", parse_mesh("cube", str(path), 2))  # new scale: miss
    with pytest.raises(AssertionError, match="cache hit"):
        load_pieces("cube", parse_mesh("cube", str(path), 1, 0.1))  # new threshold: miss


def test_zero_pieces_is_an_error(tmp_path, monkeypatch):
    path = tmp_path / "cube.stl"
    trimesh.creation.box((0.1, 0.1, 0.1)).export(path)
    monkeypatch.setattr(meshes.coacd, "run_coacd", lambda *a, **k: [])
    with pytest.raises(ValueError, match="cube: CoACD returned no pieces"):
        load_pieces("cube", parse_mesh("cube", str(path)))


class FakeResponse:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def read(self, limit: int = -1) -> bytes:
        return self.data if limit < 0 else self.data[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def cube_stl() -> bytes:
    return trimesh.creation.box((0.1, 0.1, 0.1)).export(file_type="stl")


def test_url_is_downloaded_once(monkeypatch):
    calls = []

    def urlopen(url, timeout):
        calls.append((url, timeout))
        return FakeResponse(cube_stl())

    monkeypatch.setattr(meshes.urllib.request, "urlopen", urlopen)
    mesh = parse_mesh("cube", "https://example.com/parts/cube.stl?v=1")
    assert len(load_pieces("cube", mesh)) == 1
    assert len(load_pieces("cube", mesh)) == 1
    assert calls == [("https://example.com/parts/cube.stl?v=1", meshes.DOWNLOAD_TIMEOUT_S)]


def test_url_errors(monkeypatch):
    def not_found(url, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(meshes.urllib.request, "urlopen", not_found)
    with pytest.raises(
        ValueError, match="cube: downloading https://example.com/a.stl failed: HTTP 404"
    ):
        load_pieces("cube", parse_mesh("cube", "https://example.com/a.stl"))

    monkeypatch.setattr(meshes, "MAX_DOWNLOAD_BYTES", 10)
    monkeypatch.setattr(
        meshes.urllib.request, "urlopen", lambda url, timeout: FakeResponse(cube_stl())
    )
    with pytest.raises(ValueError, match="larger than"):
        load_pieces("cube", parse_mesh("cube", "https://example.com/b.stl"))


def test_unreadable_mesh_file_is_labelled(tmp_path):
    path = tmp_path / "x.stl"
    path.write_bytes(b"solid")
    path.chmod(0)
    with pytest.raises(ValueError, match="^cube: can't read mesh file"):
        load_pieces("cube", parse_mesh("cube", str(path)))


def test_corrupt_cache_entry_is_recomputed(tmp_path, cache):
    path = tmp_path / "cube.stl"
    trimesh.creation.box((0.1, 0.1, 0.1)).export(path)
    mesh = parse_mesh("cube", str(path))
    load_pieces("cube", mesh)
    for npz in (cache / "pieces").glob("*.npz"):
        npz.write_bytes(b"junk")
    assert len(load_pieces("cube", mesh)) == 1
    assert len(load_pieces("cube", mesh)) == 1  # rewritten, valid again
