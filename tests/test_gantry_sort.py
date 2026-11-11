"""examples/gantry-sort.json and gantry_sort.py: the config validates, the
script's idea of the cell matches it, and --direct sorts every block on the
local sim."""

import asyncio
import importlib.util
import json

import numpy as np
from test_parcel_sort import EXAMPLES, MODELS, component

from mujoco_module.models import obstacle as obstacle_module
from mujoco_module.models.gantry import MujocoGantry

CELL_MODELS = {**MODELS, "viam-labs:mujoco-sim:gantry": MujocoGantry}


def load_script():
    spec = importlib.util.spec_from_file_location("gantry_sort", EXAMPLES / "gantry_sort.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_example_validates_and_matches_the_script():
    cfg = json.loads((EXAMPLES / "gantry-sort.json").read_text())
    items = {item["name"]: item for item in cfg["components"]}
    for item in cfg["components"]:
        if item["model"] in CELL_MODELS:
            CELL_MODELS[item["model"]].validate_config(component(item))
    script = load_script()
    gantry = items[script.GANTRY]
    assert tuple(gantry["frame"]["translation"].values()) == script.GANTRY_ORIGIN_MM
    cup = items[script.GRIPPER]
    assert cup["attributes"]["mount"] == script.GANTRY and cup["frame"]["translation"]["z"] == script.TCP_MM
    props = {p["name"]: p for p in items[script.WORLD]["attributes"]["props"]}
    assert sorted(props) == sorted(script.BLOCKS)
    assert {p["size"] * 1000 for p in props.values()} == {script.BLOCK_MM}
    for colour, (x, y) in script.BINS.items():
        bin_ = items[f"{colour}-bin"]
        assert (bin_["frame"]["translation"]["x"], bin_["frame"]["translation"]["y"]) == (x, y)
        assert bin_["attributes"]["scale"] == [2, 2, 1.5]  # bin.stl is 150 x 150 x 80 mm, 5 mm walls
    # every pose the script asks for is inside the gantry's travel
    lengths = gantry["attributes"]["lengths_mm"]
    table = items["table"]["frame"]
    table_top = table["translation"]["z"] + table["geometry"]["z"] / 2
    tops = [p["position"][2] * 1000 + script.BLOCK_MM / 2 for p in props.values()]
    assert all(abs(p["position"][2] * 1000 - script.BLOCK_MM / 2 - table_top) < 1 for p in props.values())
    spots = [(p["position"][0] * 1000, p["position"][1] * 1000) for p in props.values()]
    spots += [(bx + dx, by + dy) for bx, by in script.BINS.values()
              for dx in (-script.SLOT_MM, script.SLOT_MM) for dy in (-script.SLOT_MM, script.SLOT_MM)]
    for z in (script.CARRY_MM, min(tops) + script.CUP_GAP_MM, script.RELEASE_MM):
        for x, y in spots:
            position = script.axes(x, y, z)
            assert all(0 <= v <= length for v, length in zip(position, lengths)), (x, y, z, position)


def test_direct_mode_sorts_every_block(running_sim, u_pieces, monkeypatch):
    scaled = [(np.asarray(v) * [2, 2, 1.5], f) for v, f in u_pieces]  # bin.stl at the example's scale
    monkeypatch.setattr(obstacle_module, "load_pieces", lambda label, mesh: scaled)
    cfg = json.loads((EXAMPLES / "gantry-sort.json").read_text())
    made = {}
    for item in cfg["components"]:
        if item["model"] in CELL_MODELS:
            if item["model"] == "viam-labs:mujoco-sim:world":
                item["attributes"]["real_time"] = False
            made[item["name"]] = CELL_MODELS[item["model"]].new(component(item), {})
    script = load_script()
    cell = script.Cell(made["sim-world"], made["gantry"], made["cup"], None, direct=True)

    async def run():
        # as main() does: the blocks fell while the racing sim built the table under them
        await made["sim-world"].do_command({"command": "reset"})
        await cell.sort()
        return await cell.misplaced()

    assert asyncio.run(run()) == []
