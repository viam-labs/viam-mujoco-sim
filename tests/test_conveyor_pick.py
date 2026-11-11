"""examples/conveyor-pick.json validates, and conveyor_pick.py's idea of the
cell matches it."""

import importlib.util
import json

from mujoco_module.handles import CONVEYOR_RAIL_M

from test_parcel_sort import EXAMPLES, MODELS, component


def test_example_validates_and_matches_the_script():
    cfg = json.loads((EXAMPLES / "conveyor-pick.json").read_text())
    for item in cfg["components"]:
        if item["model"] in MODELS:
            MODELS[item["model"]].validate_config(component(item))
    spec = importlib.util.spec_from_file_location("conveyor_pick", EXAMPLES / "conveyor_pick.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    items = {item["name"]: item for item in cfg["components"]}
    belt = items["belt"]
    assert (script.BELT_X, script.BELT_TOP) == (belt["frame"]["translation"]["x"], belt["frame"]["translation"]["z"])
    assert script.BELT_W == belt["attributes"]["width_m"] * 1000
    assert script.MAX_SPEED == belt["attributes"]["max_speed_mps"]
    assert abs(script.REACH_Y[0]) < belt["attributes"]["length_m"] * 500
    rail = script.RAILS[1].center.x - script.BELT_X
    assert rail == script.BELT_W / 2 + CONVEYOR_RAIL_M * 500
    arm, gripper = items[script.ARM], items[script.GRIPPER]
    assert tuple(arm["frame"]["translation"].values()) == script.ARM_BASE_MM
    assert gripper["frame"]["parent"] == script.ARM and gripper["frame"]["translation"]["z"] == script.TCP_MM
