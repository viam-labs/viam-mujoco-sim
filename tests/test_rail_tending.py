"""examples/rail-tending.json and rail_tending.py: the cell's layout and the script's plan."""

import importlib.util
import json
import math

import pytest
from test_parcel_sort import EXAMPLES

CELL = json.loads((EXAMPLES / "rail-tending.json").read_text())
REACH_MM = 850 + 150  # a UR5e's reach plus a margin


def comp(name):
    return next(c for c in CELL["components"] if c["name"] == name)


def box(name):
    c = comp(name)["frame"]
    t, g = c["translation"], c["geometry"]
    return (t["x"], t["y"], t["z"]), (g["x"], g["y"], g["z"])


def test_every_reference_is_configured():
    names = {c["name"] for c in CELL["components"] + CELL["services"]}
    for c in CELL["components"]:
        parent = c.get("frame", {}).get("parent")
        assert parent in (None, "world") or parent in names, c["name"]
        for key in ("world", "arm"):
            if key in c["attributes"]:
                assert c["attributes"][key] in names, (c["name"], key)


def test_the_arm_stands_on_the_rail():
    assert comp("rail")["model"] == "viam-labs:mujoco-sim:rail"
    assert comp("arm")["frame"] == {"parent": "rail"}
    assert comp("gripper")["frame"]["parent"] == "arm"
    assert comp("scene-cam")["frame"]["parent"] == "world"


def test_machine_b_is_out_of_reach_from_rail_position_zero():
    rail = comp("rail")["frame"]["translation"]
    (bx, by, _), _ = box("machine-b-base")
    assert math.hypot(bx - rail["x"], by - rail["y"]) > REACH_MM


def test_parts_sit_on_the_parts_table():
    (tx, ty, _), (sx, sy, sz) = box("parts-table")
    for p in comp("sim-world")["attributes"]["props"]:
        x, y, z = (v * 1000 for v in p["position"])
        assert abs(x - tx) < sx / 2 and abs(y - ty) < sy / 2
        assert z == pytest.approx(sz + p["scale"][2] * 500)  # resting on the top


def test_machine_walls_clear_the_fingers():
    # the part sits at the machine's centre; the 2F-85's open fingers need >= 80 mm to each wall
    for m in ("machine-a", "machine-b"):
        (mx, my, _), _ = box(f"{m}-base")
        for wall in ("left", "right", "front", "back"):
            (wx, wy, _), (sx, sy, _) = box(f"{m}-{wall}")
            gap = abs(wx - mx) - sx / 2 if wall in ("left", "right") else abs(wy - my) - sy / 2
            assert gap >= 80 + 20, (m, wall)  # 20: half the part


def load_script():
    spec = importlib.util.spec_from_file_location("rail_tending", EXAMPLES / "rail_tending.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_the_pipeline_keeps_both_machines_busy():
    rt = load_script()
    assert rt.pipeline(4) == [
        (1, "parts1", "A"), (1, "A", "B"), (2, "parts2", "A"), (1, "B", "out1"), (2, "A", "B"),
        (3, "parts3", "A"), (2, "B", "out2"), (3, "A", "B"), (4, "parts4", "A"), (3, "B", "out3"),
        (4, "A", "B"), (4, "B", "out4"),
    ]
    assert rt.pipeline(1) == [(1, "parts1", "A"), (1, "A", "B"), (1, "B", "out1")]


def test_a_transfer_goes_through_both_transits():
    rt = load_script()
    steps = rt.transfer_steps("parts1", "A")
    moves = [s for s in steps if s[0] == "move"]
    assert [s[1] for s in moves] == ["transit parts", "pre parts1", "grip parts1", "lift parts1", "transit parts",
                                     "transit A", "pre A", "place A", "up A", "transit A"]
    assert [s[0] for s in steps].count("grab") == 1 and [s[0] for s in steps].count("open") == 1
    x, y, z = moves[0][2]
    assert (x, y, z) == (rt.STATIONS["parts"], rt.TRANSIT_Y, rt.TRANSIT_Z)
    assert moves[2][2] == (*rt.SLOTS["parts1"], rt.GRIP_Z)


def test_every_slot_is_in_the_cell():
    rt = load_script()
    for slot in rt.SLOTS:
        assert rt.station_of(slot) in rt.STATIONS
    # the parts' slots match the config's props
    props = {p["name"]: p["position"] for p in comp("sim-world")["attributes"]["props"]}
    for i in range(1, 5):
        x, y, _ = props[f"part_{i}"]
        assert rt.SLOTS[f"parts{i}"] == (round(x * 1000), round(y * 1000))
    # the machines' slots are their centres
    for slot, name in (("A", "machine-a"), ("B", "machine-b")):
        (mx, my, _), _ = box(f"{name}-base")
        assert rt.SLOTS[slot] == (mx, my)


def test_on_slot():
    rt = load_script()
    assert rt.on_slot((0.961, 0.569, 0.47), "out1")
    assert not rt.on_slot((1.201, 0.747, 0.47), "out3")  # dragged off its slot
