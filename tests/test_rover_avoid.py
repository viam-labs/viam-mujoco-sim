import importlib.util
import json
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SHELL_HALF_Y_MM = 150  # assets/viam_rover/rover.xml: the shell box spans +-0.15 m in y


def load_script():
    spec = importlib.util.spec_from_file_location("rover_avoid", EXAMPLES / "rover_avoid.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def load(name):
    return json.loads((EXAMPLES / name).read_text())


def names(cfg):
    return {r["name"] for r in cfg["components"] + cfg["services"]}


@pytest.fixture(params=["rover-avoid.json", "rover-avoid-lidar.json"])
def cfg(request):
    return load(request.param)


def test_configs_differ_only_in_the_depth_camera():
    depth, lidar = load("rover-avoid.json"), load("rover-avoid-lidar.json")
    assert names(depth) - names(lidar) == {"depth-cam"}
    assert names(lidar) - names(depth) == set()
    slam = {c: next(s for s in x["services"] if s["name"] == "slam")["attributes"] for c, x in
            (("depth", depth), ("lidar", lidar))}
    assert [e["name"] for e in slam["depth"]["lidars"]] == ["lidar", "depth-cam"]
    assert [e["name"] for e in slam["lidar"]["lidars"]] == ["lidar"]
    # both share the map
    for key in ("maps_dir", "active_map"):
        assert slam["depth"][key] == slam["lidar"][key]
    # mapping: in localizing mode nav-stack's SLAM won't start without a saved map, so a fresh
    # machine could never run the map run; every goal run calls start_localizing itself
    assert slam["depth"]["mode"] == slam["lidar"]["mode"] == "mapping"
    # the default 0.2 m: the crate's no-go band (half-width 0.13 + clearance) swallows the start
    # or the goal (found on the machine)
    for x in (depth, lidar):
        assert next(s for s in x["services"] if s["name"] == "nav")["attributes"]["clearance_m"] == 0.1


def test_every_reference_is_configured(cfg):
    configured = names(cfg)
    for r in cfg["components"]:
        parent = r.get("frame", {}).get("parent")
        assert parent in (None, "world") or parent in configured, r["name"]
        for key in ("world", "base"):
            if key in r.get("attributes", {}):
                assert r["attributes"][key] in configured, (r["name"], key)
    for s in cfg["services"]:
        a = s["attributes"]
        for key in ("base", "movement_sensor", "slam_service"):
            if key in a:
                assert a[key] in configured, (s["name"], key)
        for entry in a.get("lidars", []):
            assert entry["name"] in configured


def test_the_rover_spawns_at_the_world_origin(cfg):
    # the script takes odometry (zero at the spawn, facing +Y) as world coordinates
    assert next(r for r in cfg["components"] if r["name"] == "rover")["frame"] == {"parent": "world"}


def test_the_scene_camera_watches_the_pen(cfg):
    cam = next(r for r in cfg["components"] if r["name"] == "scene-cam")
    assert cam["frame"]["parent"] == "world"
    assert cam["attributes"]["width"] == 1280


def test_the_depth_camera_sits_outside_the_shell():
    cam = next(r for r in load("rover-avoid.json")["components"] if r["name"] == "depth-cam")
    assert cam["frame"]["parent"] == "rover"
    assert cam["frame"]["translation"]["y"] > SHELL_HALF_Y_MM
    entry = next(e for s in load("rover-avoid.json")["services"] if s["name"] == "slam"
                 for e in s["attributes"]["lidars"] if e["name"] == "depth-cam")
    assert entry["obstacles_only"] and entry["cloud_frame"] == "camera_optical"
    assert entry["mount"]["y"] * 1000 == pytest.approx(0.0)  # nav-stack mount: x forward
    assert entry["mount"]["x"] * 1000 == pytest.approx(cam["frame"]["translation"]["y"])


def test_the_crate_is_on_the_route_and_the_surprise_box_is_parked(cfg):
    ra = load_script()
    props = {p["name"]: p for p in next(r for r in cfg["components"] if r["name"] == "sim-world")["attributes"]["props"]}
    crate = props["crate"]["position"]
    assert crate[0] == pytest.approx(0) and 0 < crate[1] * 1000 < ra.GOAL_MM[1]
    assert props["crate"]["scale"][2] < 0.16  # under the lidar plane
    assert abs(props["surprise"]["position"][0]) > 1.5  # outside the walls
    assert set(ra.COURSE) <= props.keys()


def test_world_to_map():
    ra = load_script()
    assert ra.world_to_map(0, 1050) == pytest.approx((1.05, 0.0))
    assert ra.world_to_map(-600, 0) == pytest.approx((0.0, 0.6))


def test_moved_props():
    ra = load_script()
    before = {"crate": (0.0, 0.40, 0.04)}
    assert ra.moved(before, {"crate": (0.0, 0.41, 0.04)}) == {}
    assert ra.moved(before, {"crate": (0.0, 0.43, 0.04)}) == {"crate": pytest.approx(30.0)}


def test_drop_point_is_ahead_of_the_rover():
    ra = load_script()
    # facing world +Y (theta 0): front 0.15 m, gap 0.25 m, half the box 0.075 m
    assert ra.drop_point(0, 500, 0) == pytest.approx((0.0, 0.5 + 0.475, 0.075))
    # facing world -X (theta 90, counterclockwise)
    x, y, _ = ra.drop_point(0, 0, 90)
    assert (x, y) == pytest.approx((-0.475, 0.0))
    assert ra.drop_ok((-0.73, 0.64, 0.075))  # where the dry run dropped it, west of the crate
    assert not ra.drop_ok((0.1, 0.6, 0.075))  # on the crate
    assert not ra.drop_ok((0.55, 0.75, 0.075))  # on box-a
    assert not ra.drop_ok((-1.4, 0.0, 0.075))  # in the wall


def test_should_stop_and_clear():
    ra = load_script()
    assert ra.should_stop([1.5, 0.29])
    assert not ra.should_stop([1.5, 0.31])
    assert ra.is_clear([1.5, 0.41]) and not ra.is_clear([1.5, 0.39])


def test_bumpers_arm_only_while_driving_away_from_the_goal():
    ra = load_script()
    assert ra.armed({"state": "active", "distance_remaining_m": 1.05})
    assert not ra.armed({"state": "active", "distance_remaining_m": 0.4})
    assert ra.armed({"state": "active"})
    for state in ("succeeded", "failed", "canceled", "idle", "suspended", None):
        assert not ra.armed({"state": state, "distance_remaining_m": 1.05})


def test_runs_for_config():
    ra = load_script()
    assert ra.runs_for(depth=True) == ["map", "depth", "surprise"]
    assert ra.runs_for(depth=False) == ["map", "blind"]
    assert ra.check_runs(["blind"], depth=False) is None
    assert "rover-avoid-lidar.json" in ra.check_runs(["blind"], depth=True)
    assert "rover-avoid.json" in ra.check_runs(["depth"], depth=False)
