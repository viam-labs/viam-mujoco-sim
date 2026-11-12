"""The MJCF and Viam kinematics files of a one-axis floor rail, generated
from its travel (viam-labs:mujoco-sim:rail).

The rail's frame is the carriage plate's top centre with the axis at 0. A
track lies on the floor along +x; a carriage slides on it (joint x, range
[0, length]) and `attachment_site` sits on its top, facing up, so what
mounts on the rail stands upright. Nothing collides: the arm on the plate
rests on it without contact.

Both files are written to the module's data directory, named by a hash of
their content, and reused (as gantry_file.py does).
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

TRACK_MM = (None, 300.0, 60.0)  # (length from the travel, width, height)
CARRIAGE_MM = (400.0, 320.0)  # length along the track, width; its height fills track top to plate
CARRIAGE_KG = 15.0
OMEGA = 40.0  # rad/s servo bandwidth: a heavy carriage, gentler than the gantry's 100
# a moving arm can push the carriage past the 1 mm settle tolerance briefly (~3 mm at 200 N),
# which only delays a rail move's settle
FORCE_N = 20000.0


def _shapes(length_mm: float, height_mm: float) -> dict[str, list]:
    """Every box as (name, centre mm, size mm) in the rail's frame at zero."""
    floor = -height_mm
    track_len = length_mm + CARRIAGE_MM[0]
    track = ("track", (length_mm / 2, 0.0, floor + TRACK_MM[2] / 2), (track_len, TRACK_MM[1], TRACK_MM[2]))
    carriage_h = height_mm - TRACK_MM[2]
    carriage = ("carriage", (0.0, 0.0, -carriage_h / 2), (CARRIAGE_MM[0], CARRIAGE_MM[1], carriage_h))
    return {"structure": [track], "carriage": [carriage]}


def _m(values) -> str:
    return " ".join(f"{v / 1000:g}" for v in values)


def _mjcf(length_mm: float, shapes: dict[str, list], payload_kg: float) -> str:
    def geoms(body: str, rgba: str) -> str:
        return "".join(
            f'<geom name="{name}" type="box" pos="{_m(c)}" size="{_m(np.divide(s, 2))}" rgba="{rgba}" '
            f'contype="0" conaffinity="0" mass="0"/>'
            for name, c, s in shapes[body]
        )

    length = length_mm / 1000
    moved = CARRIAGE_KG + payload_kg  # kp = m w^2, kv = 2 m w: critically damped
    return f"""<mujoco model="rail">
  <worldbody>
    <body name="structure">
      {geoms("structure", "0.6 0.62 0.65 1")}
      <body name="carriage">
        <joint name="x" type="slide" axis="1 0 0" range="0 {length:g}"/>
        <inertial pos="0 0 0" mass="{CARRIAGE_KG:g}" diaginertia="1 1 1"/>
        {geoms("carriage", "0.4 0.42 0.45 1")}
        <site name="attachment_site" quat="1 0 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator><position name="x" joint="x" kp="{moved * OMEGA**2:g}" kv="{2 * moved * OMEGA:g}" forcerange="-{FORCE_N:g} {FORCE_N:g}" ctrlrange="0 {length:g}"/></actuator>
</mujoco>
"""


def _kinematics(length_mm: float, shapes: dict[str, list]) -> dict:
    """RDK SVA: the track as a static link, the prismatic joint, the carriage, the tool."""

    def box(c, s) -> dict:
        return {"x": s[0], "y": s[1], "z": s[2], "translation": {"x": c[0], "y": c[1], "z": c[2]}}

    zero = {"x": 0, "y": 0, "z": 0}
    (tname, tc, ts), = shapes["structure"]
    (cname, cc, cs), = shapes["carriage"]
    return {
        "name": "rail",
        "kinematic_param_type": "SVA",
        "links": [
            {"id": tname, "parent": "world", "translation": zero, "geometry": box(tc, ts)},
            {"id": cname, "parent": "x", "translation": zero, "geometry": box(cc, cs)},
            {"id": "tool", "parent": cname, "translation": zero,
             "orientation": {"type": "ov_degrees", "value": {"x": 0, "y": 0, "z": 1, "th": 0}}},
        ],
        "joints": [{"id": "x", "type": "prismatic", "parent": tname, "axis": {"x": 1, "y": 0, "z": 0},
                    "min": 0, "max": length_mm}],
    }


def rail_files(length_mm: float, height_mm: float, payload_kg: float) -> tuple[Path, Path]:
    """(MJCF path, kinematics path) for a rail with this travel whose frame
    (the plate's top) is height_mm above the floor."""
    length_mm, height_mm = float(length_mm), float(height_mm)
    shapes = _shapes(length_mm, height_mm)
    mjcf = _mjcf(length_mm, shapes, float(payload_kg))
    kinematics = json.dumps(_kinematics(length_mm, shapes), indent=1)
    key = hashlib.sha1((mjcf + kinematics).encode()).hexdigest()[:12]
    folder = Path(os.environ.get("VIAM_MODULE_DATA") or tempfile.gettempdir()) / "rails"
    folder.mkdir(parents=True, exist_ok=True)
    paths = folder / f"rail-{key}.xml", folder / f"rail-{key}.json"
    for path, text in zip(paths, (mjcf, kinematics)):
        if not path.exists():
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            tmp.write_text(text)
            tmp.replace(path)  # atomic: a concurrent build never reads half a file
    return paths
