"""Writes bin.stl: an open-top bin, 150 x 150 x 80 mm outside, 5 mm walls and
floor, origin at the centre of its underside, in meters.

Run: uv run python assets/meshes/make_bin.py
"""

from pathlib import Path

import trimesh

L, W, H, T = 0.15, 0.15, 0.08, 0.005
parts = [
    trimesh.creation.box((L, W, T)).apply_translation((0, 0, T / 2)),  # floor
    trimesh.creation.box((L, T, H)).apply_translation((0, (W - T) / 2, H / 2)),
    trimesh.creation.box((L, T, H)).apply_translation((0, -(W - T) / 2, H / 2)),
    trimesh.creation.box((T, W, H)).apply_translation(((L - T) / 2, 0, H / 2)),
    trimesh.creation.box((T, W, H)).apply_translation((-(L - T) / 2, 0, H / 2)),
]
trimesh.util.concatenate(parts).export(Path(__file__).with_name("bin.stl"))
