"""The MJCF and Viam kinematics files of an XYZ Cartesian gantry, generated
from its travel (viam-labs:mujoco-sim:gantry).

The gantry's frame is the tool point with every axis at 0: the near corner of
the travel (min x, min y) with Z fully raised. A bridge rides two side beams
along +x, a carriage rides the bridge along +y, and a column hangs from the
carriage and moves down (-z) for a growing Z. The column's foot is the tool
point, where `attachment_site` points down (orientation vector (0, 0, -1),
theta 0). Four posts carry the side beams down to world z = 0.

Both files are written to the module's data directory, named by a hash of
their content, and reused.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

MARGIN_MM = 100  # structure clearance round the travel
BEAM_MM = 100  # side beam and bridge cross-section
POST_MM = 100
COLUMN_MM = 80
CARRIAGE_MM = (160, 160, 140)
COLUMN_EXTRA_MM = 200  # column length past the Z travel
# moving masses (kg): what each axis carries, for the servo gains
BRIDGE_KG, CARRIAGE_KG, COLUMN_KG = 10.0, 4.0, 3.0
# ponytail: gains assume PAYLOAD_KG; derive from the mounted gripper if heavier loads lag
PAYLOAD_KG = 2.0  # a gripper and its load, counted on every axis
OMEGA = 100.0  # rad/s: every axis's servo bandwidth, so all lag alike and paths stay straight
FORCE_N = 2000.0
# orientation vector (0, 0, -1) theta 0 = 180 degrees about y
DOWN_WXYZ = (0.0, 0.0, 1.0, 0.0)


def _shapes(lengths_mm: tuple[float, float, float], frame_z_mm: float) -> dict[str, list]:
    """Every box as (name, centre mm, size mm), in the gantry's frame at
    zero, grouped by the body that carries it."""
    lx, ly, lz = lengths_mm
    m = MARGIN_MM
    carriage_bottom = lz + COLUMN_EXTRA_MM  # above the tool point at Z = 0
    carriage_z = carriage_bottom + CARRIAGE_MM[2] / 2  # the bridge runs through the carriage
    beam_z = carriage_z - BEAM_MM  # the bridge's ends rest on the side beams
    beam_y = (-m - BEAM_MM / 2, ly + m + BEAM_MM / 2)
    post_x = (-m, lx + m)
    floor_z = -frame_z_mm
    post_top = beam_z - BEAM_MM / 2
    structure = [
        (f"post_{i}{j}", (x, y, (floor_z + post_top) / 2), (POST_MM, POST_MM, post_top - floor_z))
        for i, x in enumerate(post_x) for j, y in enumerate(beam_y)
    ] + [
        (f"beam_{j}", (lx / 2, y, beam_z), (lx + 2 * m + POST_MM, BEAM_MM, BEAM_MM))
        for j, y in enumerate(beam_y)
    ]
    bridge_len = beam_y[1] - beam_y[0] + BEAM_MM
    # the column slides through the carriage: at Z = 0 it stands lz above it
    column_top = carriage_bottom + CARRIAGE_MM[2] + lz
    return {
        "structure": structure,
        "bridge": [("bridge", (0, ly / 2, carriage_z), (BEAM_MM, bridge_len, BEAM_MM))],
        "carriage": [("carriage", (0, 0, carriage_z), CARRIAGE_MM)],
        "column": [("column", (0, 0, column_top / 2), (COLUMN_MM, COLUMN_MM, column_top))],
    }


def _mjcf(lengths_mm: tuple[float, float, float], shapes: dict[str, list]) -> str:
    def geoms(body: str, rgba: str, bits: int) -> str:
        return "".join(
            f'<geom name="{name}" type="box" pos="{_m(c)}" size="{_m(np.divide(s, 2))}" rgba="{rgba}" '
            f'contype="{bits}" conaffinity="{bits}" mass="0"/>'
            for name, c, s in shapes[body]
        )

    lx, ly, lz = (v / 1000 for v in lengths_mm)
    # each axis's moving mass, so kp = m w^2 and kv = 2 m w (critical damping) lag alike
    moved = {
        "x": BRIDGE_KG + CARRIAGE_KG + COLUMN_KG + PAYLOAD_KG,
        "y": CARRIAGE_KG + COLUMN_KG + PAYLOAD_KG,
        "z": COLUMN_KG + PAYLOAD_KG,
    }
    actuators = "".join(
        f'<position name="{axis}" joint="{axis}" kp="{m * OMEGA**2:g}" kv="{2 * m * OMEGA:g}" '
        f'forcerange="-{FORCE_N:g} {FORCE_N:g}" ctrlrange="0 {limit:g}"/>'
        for (axis, m), limit in zip(moved.items(), (lx, ly, lz))
    )
    grey, dark = "0.6 0.62 0.65 1", "0.4 0.42 0.45 1"
    return f"""<mujoco model="gantry">
  <worldbody>
    <body name="structure">
      {geoms("structure", grey, 0)}
      <body name="bridge">
        <joint name="x" type="slide" axis="1 0 0" range="0 {lx:g}"/>
        <inertial pos="0 {ly / 2:g} 0" mass="{BRIDGE_KG:g}" diaginertia="1 1 1"/>
        {geoms("bridge", dark, 0)}
        <body name="carriage">
          <joint name="y" type="slide" axis="0 1 0" range="0 {ly:g}"/>
          <inertial pos="0 0 0" mass="{CARRIAGE_KG:g}" diaginertia="0.1 0.1 0.1"/>
          {geoms("carriage", dark, 0)}
          <body name="column">
            <joint name="z" type="slide" axis="0 0 -1" range="0 {lz:g}"/>
            <inertial pos="0 0 0" mass="{COLUMN_KG:g}" diaginertia="0.1 0.1 0.1"/>
            {geoms("column", grey, 1)}
            <site name="attachment_site" quat="{" ".join(f"{v:g}" for v in DOWN_WXYZ)}"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>{actuators}</actuator>
</mujoco>
"""


def _kinematics(lengths_mm: tuple[float, float, float], shapes: dict[str, list]) -> dict:
    """RDK SVA: the structure as a chain of static links (this repo's parser
    takes one geometry per link and one chain), then x, y, z and the tool."""

    def box(c, s) -> dict:
        return {"x": s[0], "y": s[1], "z": s[2], "translation": {"x": c[0], "y": c[1], "z": c[2]}}

    zero = {"x": 0, "y": 0, "z": 0}
    links, parent = [], "world"
    for name, c, s in shapes["structure"]:
        links.append({"id": name, "parent": parent, "translation": zero, "geometry": box(c, s)})
        parent = name
    joints = []
    for axis, body, direction, length in (
        ("x", "bridge", {"x": 1, "y": 0, "z": 0}, lengths_mm[0]),
        ("y", "carriage", {"x": 0, "y": 1, "z": 0}, lengths_mm[1]),
        ("z", "column", {"x": 0, "y": 0, "z": -1}, lengths_mm[2]),
    ):
        joints.append({"id": axis, "type": "prismatic", "parent": parent, "axis": direction, "min": 0, "max": length})
        (name, c, s), = shapes[body]
        links.append({"id": name, "parent": axis, "translation": zero, "geometry": box(c, s)})
        parent = name
    links.append({
        "id": "tool", "parent": parent, "translation": zero,
        "orientation": {"type": "ov_degrees", "value": {"x": 0, "y": 0, "z": -1, "th": 0}},
    })
    return {"name": "gantry", "kinematic_param_type": "SVA", "links": links, "joints": joints}


def _m(values) -> str:
    return " ".join(f"{v / 1000:g}" for v in values)


def gantry_files(lengths_mm: tuple[float, float, float], frame_z_mm: float) -> tuple[Path, Path]:
    """(MJCF path, kinematics path) for a gantry with this travel whose frame
    is frame_z_mm above the world's z = 0."""
    lengths_mm = tuple(float(v) for v in lengths_mm)
    shapes = _shapes(lengths_mm, float(frame_z_mm))
    mjcf = _mjcf(lengths_mm, shapes)
    kinematics = json.dumps(_kinematics(lengths_mm, shapes), indent=1)
    key = hashlib.sha1((mjcf + kinematics).encode()).hexdigest()[:12]
    folder = Path(os.environ.get("VIAM_MODULE_DATA") or tempfile.gettempdir()) / "gantries"
    folder.mkdir(parents=True, exist_ok=True)
    paths = folder / f"gantry-{key}.xml", folder / f"gantry-{key}.json"
    for path, text in zip(paths, (mjcf, kinematics)):
        if not path.exists():
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # one per file and process
            tmp.write_text(text)
            tmp.replace(path)  # atomic: a concurrent build never reads half a file
    return paths
