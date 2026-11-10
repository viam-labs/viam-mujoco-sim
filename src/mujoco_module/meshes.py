"""Mesh files as convex pieces, for obstacles and props.

MuJoCo collides a mesh as its convex hull, so a concave mesh (a bin) is split
into convex pieces with CoACD. MuJoCo gets one mesh geom per piece, and Viam's
planner gets the same pieces (see models/obstacle.py), so both collide against
identical shapes. Decompositions are slow the first time and cached on disk.

Texture images load here too, sharing the path, download and cache handling.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import math
import os
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import coacd
import mujoco
import numpy as np
import trimesh
from PIL import Image, ImageOps
from viam.logging import getLogger
from viam.proto.common import Geometry, Mesh, Pose

LOGGER = getLogger("viam-mujoco-sim")
MAX_PIECES = 16
MAX_HULL_VERTICES = 256
# Bump when the fixed CoACD settings or the loader change: it's in the cache key.
CACHE_VERSION = 1

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
EXTENSIONS = (".stl", ".obj", ".ply")
TEXTURE_EXTENSIONS = (".png", ".jpg", ".jpeg")
# ponytail: a fixed cap keeps GPU uploads and rebuilds fast; raise it, or make it
# an attribute, if a close-up needs more detail
MAX_TEXTURE_PX = 2048

coacd.set_log_level("error")  # CoACD logs every step to stdout otherwise

Piece = tuple[np.ndarray, np.ndarray]  # vertices (n, 3) float meters, faces (m, 3) int
Texture = tuple[int, int, bytes]  # width, height, RGB bytes (row-major, top row first)


@dataclass(frozen=True)
class MeshSource:
    source: str  # absolute path, path relative to ASSETS_DIR, or https URL
    scale: tuple[float, float, float]
    threshold: float  # CoACD concavity


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def extension(source: str) -> str:
    path = urlparse(source).path if source.startswith("https://") else source
    return Path(path).suffix.lower()


def parse_mesh(label: str, source: Any, scale: Any = 1, threshold: Any = 0.05) -> MeshSource:
    """Check a mesh attribute's form only; nothing is read or downloaded."""
    if not isinstance(source, str) or not source:
        raise ValueError(f"{label}: mesh must be a file path or an https URL")
    scheme = urlparse(source).scheme
    if scheme and scheme != "https":
        raise ValueError(f"{label}: mesh URLs must be https, got {scheme}:")
    if extension(source) not in EXTENSIONS:
        raise ValueError(f"{label}: mesh must be a .stl, .obj or .ply file, got {source!r}")
    values = [scale] * 3 if _number(scale) else scale
    if not (
        isinstance(values, list | tuple) and len(values) == 3
        and all(_number(v) and v > 0 for v in values)
    ):
        raise ValueError(f"{label}: scale must be a positive number or [x, y, z]")
    if not (_number(threshold) and 0.01 <= threshold <= 1):
        raise ValueError(f"{label}: threshold must be between 0.01 and 1")
    return MeshSource(source, tuple(float(v) for v in values), float(threshold))


def parse_texture(label: str, source: Any) -> str:
    """Check a texture attribute's form only; nothing is read or downloaded."""
    if not isinstance(source, str) or not source:
        raise ValueError(f"{label}: texture must be a file path or an https URL")
    scheme = urlparse(source).scheme
    if scheme and scheme != "https":
        raise ValueError(f"{label}: texture URLs must be https, got {scheme}:")
    if extension(source) not in TEXTURE_EXTENSIONS:
        raise ValueError(f"{label}: texture must be a .png, .jpg or .jpeg image, got {source!r}")
    return source


def pieces_ply(pieces: Sequence[Piece]) -> bytes:
    """Pieces merged into one ASCII PLY, vertices in meters (RDK scales PLY to mm)."""
    vertices, faces, offset = [], [], 0
    for piece_vertices, piece_faces in pieces:
        vertices.extend(piece_vertices.tolist())
        faces.extend((np.asarray(piece_faces) + offset).tolist())
        offset += len(piece_vertices)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
        *(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices),
        *(f"3 {a} {b} {c}" for a, b, c in faces),
    ]
    return ("\n".join(lines) + "\n").encode()


def mesh_geometry(pieces: Sequence[Piece], label: str) -> Geometry:
    """Pieces as one Viam mesh geometry, centred on its frame."""
    return Geometry(
        center=Pose(o_z=1), mesh=Mesh(content_type="ply", mesh=pieces_ply(pieces)), label=label
    )


def add_pieces(
    spec: mujoco.MjSpec,
    body: Any,
    name: str,
    pieces: Sequence[Piece],
    mass: float | None = None,
    **geom: Any,
) -> None:
    """One convex mesh geom per piece, named f"{name}/{i}" (the mesh asset too;
    MuJoCo keeps mesh and geom names apart). With mass, it's split by volume."""
    weights = None
    if mass is not None:
        volumes = np.array([abs(trimesh.Trimesh(v, f).volume) for v, f in pieces])  # convex
        weights = volumes / volumes.sum()
    for i, (vertices, faces) in enumerate(pieces):
        piece_name = f"{name}/{i}"
        mesh = spec.add_mesh(
            name=piece_name,
            uservert=np.asarray(vertices, float).ravel().tolist(),
            userface=np.asarray(faces, int).ravel().tolist(),
        )
        # thin pieces (a bin wall) fail MuJoCo's default volume check
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
        extra = {} if weights is None else {"mass": float(mass * weights[i])}
        body.add_geom(
            name=piece_name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=piece_name, **geom, **extra
        )


# MuJoCo's default geom rgba: only at this value does a geom show its material's
# colour, so a textured geom must use it or its rgba tints the image
UNTINTED_RGBA = [0.5, 0.5, 0.5, 1.0]


def add_texture(
    spec: mujoco.MjSpec, name: str, texture: Texture, repeat_m: float | None = None
) -> str:
    """A 2D texture and a white material, both called name; returns name for a
    geom's material. With repeat_m the image tiles in world units, one tile
    repeat_m meters across (on a plane, a tile spans 2 / texrepeat meters)."""
    width, height, rgb = texture
    added = spec.add_texture(
        name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=width, height=height, nchannel=3
    )
    added.data = rgb
    material = spec.add_material(name=name, rgba=[1, 1, 1, 1])
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
    if repeat_m is not None:
        material.texuniform = True
        material.texrepeat = [2 / repeat_m, 2 / repeat_m]
    return name


def cache_dir() -> Path:
    root = os.environ.get("VIAM_MODULE_DATA")
    base = Path(root) if root else Path.home() / ".cache" / "viam-mujoco-sim"
    return base / "mesh-cache"


def _write_atomic(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    os.replace(tmp, target)


MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
DOWNLOAD_TIMEOUT_S = 30


def _download(label: str, url: str) -> Path:
    """Fetched once into the cache; change the URL (e.g. ?v=2) to refresh."""
    target = cache_dir() / "downloads" / (hashlib.sha256(url.encode()).hexdigest() + extension(url))
    if target.is_file():
        return target
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
            data = response.read(MAX_DOWNLOAD_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise ValueError(f"{label}: downloading {url} failed: HTTP {error.code}") from None
    except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
        reason = getattr(error, "reason", error)
        raise ValueError(f"{label}: downloading {url} failed: {reason}") from None
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise ValueError(f"{label}: {url} is larger than {MAX_DOWNLOAD_BYTES // 2**20} MB")
    _write_atomic(target, data)
    return target


def _resolve(label: str, source: str) -> Path:
    if source.startswith("https://"):
        return _download(label, source)
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = ASSETS_DIR / path
    if not path.is_file():
        raise ValueError(f"{label}: file {path} not found")
    return path


def load_pieces(label: str, mesh: MeshSource) -> list[Piece]:
    """The mesh's convex pieces, in meters, scaled. Runs CoACD on a cache miss."""
    path = _resolve(label, mesh.source)
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{label}: can't read mesh file {path}: {error}") from None
    settings = (CACHE_VERSION, version("coacd"), mesh.scale, mesh.threshold)
    key = hashlib.sha256(repr(settings).encode() + data).hexdigest()
    cached = cache_dir() / "pieces" / f"{key}.npz"
    if cached.is_file():
        try:
            with np.load(cached) as npz:
                return [(npz[f"v{i}"], npz[f"f{i}"]) for i in range(len(npz.files) // 2)]
        except Exception:  # corrupt or partial: recompute
            cached.unlink(missing_ok=True)

    try:
        loaded = trimesh.load_mesh(io.BytesIO(data), file_type=extension(mesh.source)[1:])
    except Exception as error:
        raise ValueError(f"{label}: can't read mesh {mesh.source}: {error}") from None
    if len(getattr(loaded, "faces", ())) == 0:
        raise ValueError(f"{label}: mesh {mesh.source} has no triangles")
    vertices = np.asarray(loaded.vertices, float) * np.asarray(mesh.scale)
    start = time.monotonic()
    result = coacd.run_coacd(
        coacd.Mesh(vertices, np.asarray(loaded.faces, int)),
        threshold=mesh.threshold,
        max_convex_hull=MAX_PIECES,
        decimate=True,  # max_ch_vertex only applies with decimate
        max_ch_vertex=MAX_HULL_VERTICES,
    )
    pieces = [(np.asarray(v, float), np.asarray(f, np.int32)) for v, f in result]
    if not pieces:
        raise ValueError(f"{label}: CoACD returned no pieces for {mesh.source}")
    LOGGER.info(
        "%s: decomposed %s into %d convex pieces in %.1f s",
        label, mesh.source, len(pieces), time.monotonic() - start,
    )
    buffer = io.BytesIO()
    arrays = {f"v{i}": v for i, (v, _) in enumerate(pieces)}
    arrays.update({f"f{i}": f for i, (_, f) in enumerate(pieces)})
    np.savez(buffer, **arrays)
    # written even if viam-server already timed the resource out: the retry hits it
    _write_atomic(cached, buffer.getvalue())
    return pieces


def load_texture(label: str, source: str) -> Texture:
    """The image as RGB, at most MAX_TEXTURE_PX on its long side."""
    path = _resolve(label, source)
    try:
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ValueError(f"{label}: can't read image {source}: {error}") from None
    rgb.thumbnail((MAX_TEXTURE_PX, MAX_TEXTURE_PX))
    return rgb.width, rgb.height, rgb.tobytes()
