"""plugin/skills/mujoco-sim-cell/scripts/twin.py: the sim-twin converter and checks."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "plugin" / "skills" / "mujoco-sim-cell" / "scripts" / "twin.py"


def load():
    spec = importlib.util.spec_from_file_location("twin", SCRIPT)
    twin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(twin)
    return twin


def test_frame_pose_orientation_types_agree():
    twin = load()
    turn_z = [
        {"type": "ov_degrees", "value": {"x": 0, "y": 0, "z": 1, "th": 90}},
        {"type": "ov_radians", "value": {"x": 0, "y": 0, "z": 1, "th": math.pi / 2}},
        {"type": "euler_angles", "value": {"roll": 0, "pitch": 0, "yaw": math.pi / 2}},
        {"type": "axis_angles", "value": {"x": 0, "y": 0, "z": 1, "th": math.pi / 2}},
        {"type": "quaternion", "value": {"w": math.cos(math.pi / 4), "x": 0, "y": 0, "z": math.sin(math.pi / 4)}},
    ]
    poses = [twin.frame_pose({"translation": {"x": 10, "y": 0, "z": 5}, "orientation": o}) for o in turn_z]
    for m in poses:
        assert np.allclose(m[:3, 3], [10, 0, 5])
        assert np.allclose(m[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-9)  # x turns to y


def test_frame_pose_defaults_and_round_trip():
    twin = load()
    assert np.allclose(twin.frame_pose({}), np.eye(4))
    m = twin.frame_pose({"translation": {"x": 1, "y": 2, "z": 3},
                         "orientation": {"type": "euler_angles", "value": {"roll": 0.3, "pitch": -0.2, "yaw": 1.1}}})
    frame = twin.pose_frame("arm", m)
    assert frame["parent"] == "arm" and frame["orientation"]["type"] == "ov_degrees"
    mm, deg = twin.pose_error(twin.frame_pose(frame), m)
    assert mm < 1e-3 and deg < 1e-3


def test_to_ancestor_composes_until_a_stop():
    twin = load()
    frames = {
        "arm": {"parent": "world"},
        "gripper": {"parent": "arm", "translation": {"x": 0, "y": 0, "z": 100}},
        "cam": {"parent": "gripper", "translation": {"x": 50, "y": 0, "z": 0},
                "orientation": {"type": "ov_degrees", "value": {"x": 0, "y": 0, "z": 1, "th": 90}}},
    }
    parent, m = twin.to_ancestor("cam", frames, stop={"arm"})
    assert parent == "arm"
    assert np.allclose(m[:3, 3], [50, 0, 100])
    parent, m = twin.to_ancestor("gripper", frames, stop=set())
    assert parent == "world" and np.allclose(m[:3, 3], [0, 0, 100])

def test_expand_merges_fragments_and_applies_mods():
    twin = load()
    part = {
        "components": [{"name": "cam", "api": "rdk:component:camera", "model": "webcam"}],
        "modules": [{"type": "registry", "name": "m1", "module_id": "a:m1"}],
        "fragments": ["F1", {"id": "F2", "version": "3"}],
        "fragment_mods": [{"fragment_id": "F1", "mods": [
            {"$set": {"components.arm.attributes.speed": 50}},
            {"$unset": {"components.arm.attributes.host": ""}},
            {"$rename": {"components.arm.attributes.x": "y"}},
        ]}],
    }
    fragments = {
        ("F1", None): {"components": [{"name": "arm", "api": "rdk:component:arm", "model": "viam:universal-robots:ur5e",
                                       "attributes": {"host": "10.0.0.2", "speed": 10}}],
                       "modules": [{"type": "registry", "name": "m1", "module_id": "a:m1"}],
                       "fragments": ["F3"]},
        ("F2", "3"): {"services": [{"name": "vision", "api": "rdk:service:vision", "model": "x:y:z"}]},
        ("F3", None): {"components": [{"name": "board", "api": "rdk:component:board", "model": "pi"}]},
    }
    calls = []

    def fetch(fid, version):
        calls.append(fid)
        return fragments[(fid, version)]

    config, unapplied = twin.expand(part, fetch)
    names = sorted(r["name"] for r in config["components"] + config["services"])
    assert names == ["arm", "board", "cam", "vision"]
    arm = next(c for c in config["components"] if c["name"] == "arm")
    assert arm["attributes"] == {"speed": 50}
    assert [m["name"] for m in config["modules"]] == ["m1"]  # deduplicated
    assert unapplied == [{"op": "$rename", "path": "components.arm.attributes.x"}]
    assert "fragments" not in config and "fragment_mods" not in config
    assert sorted(calls) == ["F1", "F2", "F3"]


def test_quat_round_trips_including_half_turns():
    twin = load()
    rng = np.random.default_rng(0)
    rots = []
    for axis in [(1, -1, 0), (0, 1, -1), (1, -1, 1), (0, 0, 1)]:
        a = np.array(axis, float) / np.linalg.norm(axis)
        k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        rots.append(np.eye(3) + 2 * k @ k)  # 180 degrees about a
    for _ in range(5):
        q = rng.normal(size=4)
        rots.append(twin._mat(q / np.linalg.norm(q)))
    for r in rots:
        assert np.allclose(twin._mat(twin._quat(r)), r)


def test_expand_unset_missing_path_and_unsupported_imports():
    twin = load()
    part = {
        "components": [{"name": "arm", "api": "rdk:component:arm", "model": "m"}],
        "fragments": ["F1", {"id": "F1", "version": "2"}, "F2@pre", {"id": "F2", "prefix": "p"}],
        "fragment_mods": [{"fragment_id": "F1", "mods": [{"$unset": {"components.arm.attributes.a.b": ""}}]}],
    }
    config, unapplied = twin.expand(part, lambda fid, version: {})
    assert config["components"] == [{"name": "arm", "api": "rdk:component:arm", "model": "m"}]
    assert [u for u in unapplied if u["op"] == "fragment_import"] == [
        {"op": "fragment_import", "path": "F1"},
        {"op": "fragment_import", "path": "F2@pre"},
        {"op": "fragment_import", "path": "F2"},
    ]


def outcome(report, name):
    return next(r for r in report["resources"] if r["name"] == name)


def comp(config, name):
    return next(c for c in config["components"] if c["name"] == name)


ARM = {"name": "arm", "api": "rdk:component:arm", "model": "viam:ufactory:lite6", "frame": {"parent": "world"},
       "attributes": {"host": "10.0.0.5"}, "service_configs": [{"type": "data_manager", "attributes": {}}]}
GRIPPER = {"name": "gripper", "api": "rdk:component:gripper", "model": "viam:ufactory:gripper_lite",
           "frame": {"parent": "arm", "translation": {"x": 0, "y": 0, "z": 80}}}


def test_convert_maps_a_bundled_arm_and_gripper():
    twin = load()
    config, report = twin.convert({"components": [ARM, GRIPPER]}, module_version="0.17.1")
    arm, gripper = comp(config, "arm"), comp(config, "gripper")
    world = report["world"]
    assert arm["model"] == "viam-labs:mujoco-sim:arm" and arm["attributes"] == {"world": world, "asset": "lite6"}
    assert arm["frame"] == {"parent": "world"} and "service_configs" not in arm
    assert gripper["model"] == "viam-labs:mujoco-sim:gripper"
    assert gripper["attributes"] == {"world": world, "mount": "arm", "asset": "lite6_gripper"}
    assert gripper["frame"]["translation"] == GRIPPER["frame"]["translation"]  # the real TCP offset is kept
    assert gripper["frame"]["geometry"]["type"] == "box"  # the Lite6 gripper serves no shape of its own
    assert outcome(report, "arm")["outcome"] == "mapped"
    assert comp(config, world)["model"] == "viam-labs:mujoco-sim:world"
    assert {"type": "registry", "name": "viam-labs_mujoco-sim", "module_id": "viam-labs:mujoco-sim",
            "version": "0.17.1"} in config["modules"]


def test_convert_fakes_an_unmapped_arm_and_everything_on_it():
    twin = load()
    arm = {**ARM, "model": "viam:ufactory:xArm6"}
    cam = {"name": "wrist", "api": "rdk:component:camera", "model": "webcam", "frame": {"parent": "arm"}}
    config, report = twin.convert({"components": [arm, GRIPPER, cam]}, module_version="0.17.1")
    for name in ("arm", "gripper", "wrist"):
        assert comp(config, name)["model"] == "rdk:builtin:fake"
        assert outcome(report, name)["outcome"] == "needs_asset"
    assert comp(config, "arm")["frame"] == {"parent": "world"}


def test_convert_reframes_an_arm_on_a_rail():
    twin = load()
    rail = {"name": "rail", "api": "rdk:component:gantry", "model": "acme:rail:one",
            "frame": {"parent": "world", "translation": {"x": 100, "y": 0, "z": 0}}}
    arm = {**ARM, "frame": {"parent": "rail", "translation": {"x": 0, "y": 0, "z": 50}}}
    config, report = twin.convert({"components": [rail, arm]}, module_version="0.17.1")
    frame = comp(config, "arm")["frame"]
    assert frame["parent"] == "world" and frame["translation"] == {"x": 100.0, "y": 0.0, "z": 50.0}
    assert outcome(report, "arm")["outcome"] == "reframed"
    assert comp(config, "rail")["model"] == "rdk:builtin:fake"


def test_convert_gripper_needs_its_arm_asset():
    twin = load()
    ur = {**ARM, "model": "viam:universal-robots:ur5e"}
    config, report = twin.convert({"components": [ur, GRIPPER]}, module_version="0.17.1")
    assert comp(config, "arm")["attributes"]["asset"] == "ur5e"
    assert comp(config, "gripper")["model"] == "rdk:builtin:fake"
    assert "lite6" in outcome(report, "gripper")["note"]


def test_convert_cameras_world_wrist_and_gripper_mounted():
    twin = load()
    overhead = {"name": "overhead", "api": "rdk:component:camera", "model": "viam:camera:realsense",
                "frame": {"parent": "world", "translation": {"x": 0, "y": 0, "z": 1500}},
                "attributes": {"intrinsic_parameters": {"width_px": 1280, "height_px": 720, "fy": 640}}}
    wrist = {"name": "wrist", "api": "rdk:component:camera", "model": "webcam",
             "frame": {"parent": "arm", "translation": {"x": 60, "y": 0, "z": 30}}}
    on_gripper = {"name": "jaw-cam", "api": "rdk:component:camera", "model": "webcam",
                  "frame": {"parent": "gripper", "translation": {"x": 20, "y": 0, "z": -10}}}
    live = {"wrist": {"size": [800, 600], "fov_deg": 70.0, "sources": ["color"]},
            "jaw-cam": {"size": [640, 480], "fov_deg": 55.0, "sources": ["rgb", "ir"], "depth": True}}
    config, report = twin.convert({"components": [ARM, GRIPPER, overhead, wrist, on_gripper]}, live, "0.17.1")

    o = comp(config, "overhead")
    assert o["model"] == "viam-labs:mujoco-sim:camera" and o["frame"] == overhead["frame"]
    assert o["attributes"]["width"] == 1280 and o["attributes"]["height"] == 720
    assert abs(o["attributes"]["fov_deg"] - math.degrees(2 * math.atan(720 / 1280))) < 1e-6
    assert outcome(report, "overhead")["outcome"] == "mapped"
    assert any(f["name"] == "overhead" and f["flag"] == "config_only" for f in report["flags"])

    w = comp(config, "wrist")
    assert w["frame"]["parent"] == "arm" and w["attributes"]["width"] == 800 and w["attributes"]["fov_deg"] == 70.0

    j = comp(config, "jaw-cam")
    assert j["frame"]["parent"] == "arm" and j["frame"]["translation"] == {"x": 20.0, "y": 0.0, "z": 70.0}
    assert j["attributes"]["depth"] is True
    assert outcome(report, "jaw-cam")["outcome"] == "reparented"
    assert any(f["name"] == "jaw-cam" and f["flag"] == "source_names_differ" for f in report["flags"])


def test_convert_camera_without_size_or_intrinsics_is_guessed():
    twin = load()
    cam = {"name": "cam", "api": "rdk:component:camera", "model": "webcam", "frame": {"parent": "world"}}
    config, report = twin.convert({"components": [cam]}, module_version="0.17.1")
    attrs = comp(config, "cam")["attributes"]
    assert (attrs["width"], attrs["height"], attrs["fov_deg"]) == (640, 480, 60.0)
    assert any(f["name"] == "cam" and f["flag"] == "guessed" for f in report["flags"])


def test_convert_rest_services_modules_and_data_capture():
    twin = load()
    real = {
        "components": [
            ARM,
            {"name": "table", "api": "rdk:component:gripper", "model": "viam:obstacle-pack:obstacle",
             "attributes": {"x": 1}, "service_configs": [{"type": "data_manager"}]},
            {"name": "board", "api": "rdk:component:board", "model": "viam:raspberry-pi:rpi5"},
            {"name": "fixture", "api": "rdk:component:motor", "model": "acme:motors:m1",
             "frame": {"parent": "world", "geometry": {"type": "box", "x": 100, "y": 100, "z": 100}}},
            {"name": "odd", "api": "acme:component:widget", "model": "acme:widgets:w"},
        ],
        "remotes": [{"name": "aux", "address": "aux.local:8080"}],
        "services": [
            {"name": "data", "api": "rdk:service:data_manager", "model": "rdk:builtin:builtin"},
            {"name": "motion", "api": "rdk:service:motion", "model": "rdk:builtin:builtin"},
            {"name": "detector", "api": "rdk:service:vision", "model": "viam:vision:mlmodel",
             "attributes": {"camera_name": "odd", "label_path": "/home/pi/labels.txt"}},
            {"name": "teleop", "api": "rdk:service:generic", "model": "acme:motors:teleop"},
        ],
        "modules": [
            {"type": "registry", "name": "ur", "module_id": "viam:ufactory", "version": "1"},
            {"type": "registry", "name": "obstacles", "module_id": "viam:obstacle-pack", "version": "2"},
            {"type": "registry", "name": "vision", "module_id": "viam:vision", "version": "3"},
            {"type": "registry", "name": "motors", "module_id": "acme:motors", "version": "4"},
            {"type": "local", "name": "mine", "executable_path": "/opt/mine"},
        ],
        "packages": [{"name": "model", "package": "org/model", "type": "ml_model", "version": "1"}],
    }
    config, report = twin.convert(real, module_version="0.17.1")
    out = {o["name"]: o["outcome"] for o in report["resources"]}
    assert out == {"arm": "mapped", "table": "copied", "board": "faked", "fixture": "faked", "odd": "unsupported",
                   "data": "removed", "motion": "copied", "detector": "copied", "teleop": "faked", "aux": "unsupported"}
    table = comp(config, "table")
    assert table["model"] == "viam:obstacle-pack:obstacle" and "service_configs" not in table
    world = comp(config, report["world"])
    assert world["attributes"]["import_geometry"] == ["table", "fixture"]
    services = {s["name"]: s for s in config["services"]}
    assert set(services) == {"motion", "detector", "teleop"} and services["teleop"]["model"] == "rdk:builtin:fake"
    modules = {m["name"] for m in config["modules"]}
    assert modules == {"viam-labs_mujoco-sim", "obstacles", "vision"}
    assert config["packages"] == real["packages"]
    flags = {(f["name"], f["flag"]) for f in report["flags"]}
    assert ("detector", "needs_review") in flags and ("mine", "needs_review") in flags
    assert any("odd" in f["note"] for f in report["flags"] if f["name"] == "detector")
    assert any(c["name"] == report["overview_camera"] for c in config["components"])
    assert not any(c["name"] == "odd" for c in config["components"])


def test_resolve_variables_defaults_and_unresolved():
    twin = load()
    config = {"components": [{"name": "cam", "frame": {
        "parent": {"$variable": {"name": "arm-name"}},
        "translation": {"x": {"$variable": {"name": "tx", "default_value": 5}}, "y": 0, "z": 0}}}]}
    out, unresolved = twin.resolve_variables(config)
    frame = out["components"][0]["frame"]
    assert frame["parent"] is None and frame["translation"]["x"] == 5
    assert unresolved == [{"name": "arm-name", "path": "components.cam.frame.parent"}]
    assert "$variable" in str(config)  # input untouched


def test_expand_fills_variables_from_import_values_then_defaults():
    twin = load()
    var = {"$variable": {"name": "arm-name"}}
    frag = {"components": [
        {"name": "gripper", "frame": {"parent": var}, "attributes": {"arm": var}},
        {"name": "cam", "frame": {"parent": {"$variable": {"name": "arm-name", "default_value": "dflt"}}}},
        {"name": "lost", "attributes": {"x": {"$variable": {"name": "nope"}}}}],
        "fragments": [{"id": "inner", "variables": {"arm-name": "inner-arm"}}]}
    inner = {"components": [{"name": "saver", "attributes": {"arm": var}}]}
    part = {"fragments": [{"id": "outer", "variables": {"arm-name": "arm"}}],
            "components": [{"name": "own", "attributes": {"a": {"$variable": {"name": "v", "default_value": 3}}}}]}
    out, unapplied = twin.expand(part, lambda fid, version: {"outer": frag, "inner": inner}[fid])
    comps = {c["name"]: c for c in out["components"]}
    assert comps["gripper"]["frame"]["parent"] == "arm" and comps["gripper"]["attributes"]["arm"] == "arm"
    assert comps["cam"]["frame"]["parent"] == "arm" and comps["saver"]["attributes"]["arm"] == "inner-arm"
    assert comps["own"]["attributes"]["a"] == 3
    assert unapplied == [{"op": "variable", "name": "nope", "path": "components.lost.attributes.x"}]


def test_convert_survives_unresolved_frame_values():
    twin = load()
    real = {"components": [
        {"name": "cam", "api": "rdk:component:camera", "model": "rdk:builtin:fake",
         "frame": {"parent": None, "translation": {"x": None, "y": 2, "z": None},
                   "orientation": {"type": "ov_degrees", "value": {"x": 0, "y": 0, "z": 1, "th": None}}}},
        {"name": "cam2", "api": "rdk:component:camera", "model": "rdk:builtin:fake",
         "frame": {"parent": "world", "orientation": {"type": "ov_degrees", "value": None}}}]}
    config, report = twin.convert(real, module_version="0.17.1")
    assert comp(config, "cam")["frame"]["parent"] == "world"
    assert any(f["name"] == "cam" and f["flag"] == "needs_review" and "parent unknown" in f["note"]
               for f in report["flags"])


def test_apply_live_frames_wins_and_keeps_geometry():
    twin = load()
    real = {"components": [
        {"name": "a", "frame": {"parent": None, "geometry": {"type": "box"}}},
        {"name": "b"}, {"name": "c", "frame": {"parent": "world"}}]}
    live = {"a": {"parent": "world", "translation": {"x": 1, "y": 2, "z": 3},
                  "orientation": {"type": "ov_degrees", "value": {"x": 0, "y": 0, "z": 1, "th": 0}}},
            "b": {"parent": "a", "translation": {"x": 0, "y": 0, "z": 0}, "orientation": None}}
    out, names = twin.apply_live_frames(real, live)
    comps = {c["name"]: c for c in out["components"]}
    assert comps["a"]["frame"]["parent"] == "world" and comps["a"]["frame"]["geometry"] == {"type": "box"}
    assert comps["a"]["frame"]["translation"]["z"] == 3
    assert comps["b"]["frame"]["parent"] == "a" and comps["c"]["frame"] == {"parent": "world"}
    assert names == ["a", "b"]


def test_convert_copies_vmodutils_components():
    twin = load()
    real = {"components": [
        {"name": "cam", "api": "rdk:component:camera", "model": "rdk:builtin:fake"},
        {"name": "crop", "api": "rdk:component:camera", "model": "erh:vmodutils:pc-crop-camera",
         "frame": {"parent": "cam"}},
        {"name": "box", "api": "rdk:component:gripper", "model": "erh:vmodutils:obstacle-open-box",
         "frame": {"parent": "world"}}],
        "modules": [{"type": "registry", "name": "vmod", "module_id": "erh:vmodutils", "version": "1"}]}
    config, report = twin.convert(real, module_version="0.17.1")
    out = {o["name"]: o["outcome"] for o in report["resources"]}
    assert out["crop"] == "copied" and out["box"] == "copied"
    assert comp(config, "crop")["model"] == "erh:vmodutils:pc-crop-camera"
    assert comp(config, report["world"])["attributes"]["import_geometry"] == ["box"]
    assert "vmod" in {m["name"] for m in config["modules"]}


def test_convert_names_never_collide():
    twin = load()
    real = {"components": [{"name": "sim-world", "api": "rdk:component:generic", "model": "acme:x:y"},
                           {"name": "twin-overview", "api": "rdk:component:sensor", "model": "acme:x:z"}]}
    config, report = twin.convert(real, module_version="0.17.1")
    names = [c["name"] for c in config["components"]]
    assert len(names) == len(set(names))
    assert report["world"] == "sim-world-twin" and report["overview_camera"] == "twin-overview-twin"


def test_convert_camera_on_unmapped_gripper_rides_the_mapped_arm():
    twin = load()
    ur = {**ARM, "model": "viam:universal-robots:ur5e"}
    grip = {**GRIPPER, "model": "acme:grippers:g"}
    cam = {"name": "cam", "api": "rdk:component:camera", "model": "webcam",
           "frame": {"parent": "gripper", "translation": {"x": 5, "y": 0, "z": 10}}}
    config, report = twin.convert({"components": [ur, grip, cam]}, module_version="0.17.1")
    assert comp(config, "gripper")["model"] == "rdk:builtin:fake"
    c = comp(config, "cam")
    assert c["model"] == "viam-labs:mujoco-sim:camera" and c["frame"]["parent"] == "arm"
    assert c["frame"]["translation"] == {"x": 5.0, "y": 0.0, "z": 90.0}
    assert outcome(report, "cam")["outcome"] == "reparented"


def test_convert_keeps_unknown_modules_flags_dead_parents_and_data_capture():
    twin = load()
    real = {
        "components": [
            {"name": "odd", "api": "acme:component:widget", "model": "acme:widgets:w", "frame": {"parent": "world"}},
            {"name": "s", "api": "rdk:component:sensor", "model": "acme:x:s",
             "frame": {"parent": "odd", "translation": {"x": 1, "y": 2, "z": 3}}},
            {"name": "cam", "api": "rdk:component:camera", "model": "webcam"},
        ],
        "services": [{"name": "ml", "api": "rdk:service:mlmodel", "model": "viam:mlmodel-tflite:tflite_cpu",
                      "depends_on": ["odd"], "service_configs": [{"type": "data_manager"}]}],
        "modules": [{"type": "registry", "name": "tf", "module_id": "viam:tflite_cpu", "version": "1"}],
    }
    config, report = twin.convert(real, module_version="0.17.1")
    assert {m["name"] for m in config["modules"]} == {"viam-labs_mujoco-sim", "tf"}
    flags = {(f["name"], f["flag"]) for f in report["flags"]}
    assert ("tf", "needs_review") in flags and ("ml", "needs_review") in flags
    assert ("s", "reframed") in flags and ("cam", "needs_review") in flags
    assert comp(config, "s")["frame"]["parent"] == "world"
    assert comp(config, "s")["frame"]["translation"] == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert ("ml", "data_capture_removed") in flags


SVA = {
    "name": "two", "links": [
        {"id": "base", "parent": "world"},
        {"id": "l1", "parent": "j1", "translation": {"x": 300, "y": 0, "z": 0}},
        {"id": "flange", "parent": "j2", "translation": {"x": 200, "y": 0, "z": 0}},
    ],
    "joints": [
        {"id": "j1", "type": "revolute", "parent": "base", "axis": {"x": 0, "y": 0, "z": 1}, "min": -170, "max": 170},
        {"id": "j2", "type": "revolute", "parent": "l1", "axis": {"x": 0, "y": 0, "z": 1}, "min": -120, "max": 120},
    ],
}
URDF = """<robot name="two">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="tool0"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-2.967" upper="2.967" effort="1" velocity="1"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/>
    <origin xyz="0.3 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-2.094" upper="2.094" effort="1" velocity="1"/></joint>
  <joint name="tool" type="fixed"><parent link="l2"/><child link="tool0"/><origin xyz="0.25 0 0"/></joint>
</robot>"""


def test_kinematics_sva_and_urdf_of_the_same_arm_match():
    twin = load()
    import json as _json
    a = twin.chain(_json.dumps(SVA).encode(), 1)
    b = twin.chain(URDF.encode(), 2)
    result = twin.compare_kinematics(a, b)
    assert result["ok"], result


def test_kinematics_flags_a_longer_link_and_a_limit():
    twin = load()
    import json as _json
    longer = _json.loads(_json.dumps(SVA))
    longer["links"][1]["translation"]["x"] = 305
    longer["joints"][1]["max"] = 110
    result = twin.compare_kinematics(twin.chain(_json.dumps(SVA).encode(), 1), twin.chain(_json.dumps(longer).encode(), 1))
    assert not result["ok"]
    assert result["limits"] == [{"joint": 1, "real": [-120.0, 120.0], "twin": [-120.0, 110.0]}]
    assert max(p["mm"] for p in result["poses"]) > 2


def test_compare_names():
    twin = load()
    report = {"resources": [
        {"name": "arm", "api": "rdk:component:arm", "outcome": "mapped"},
        {"name": "board", "api": "rdk:component:board", "outcome": "faked"},
        {"name": "data", "api": "rdk:service:data_manager", "outcome": "removed"},
    ]}
    live = {("rdk:component:arm", "arm"), ("rdk:component:generic", "sim-world")}
    assert twin.compare_names(report, live) == {"ok": False, "missing": [["rdk:component:board", "board"]]}


def test_compare_frames():
    twin = load()
    expected = {"cam": {"parent": "arm", "translation": {"x": 10, "y": 0, "z": 0}},
                "table": {"parent": "world"}}
    good = {"cam": ("arm", twin.frame_pose(expected["cam"])), "table": ("world", np.eye(4))}
    assert twin.compare_frames(expected, good)["ok"]
    moved = twin.frame_pose({"translation": {"x": 12, "y": 0, "z": 0}})
    bad = twin.compare_frames(expected, {"cam": ("gripper", moved)})
    assert not bad["ok"]
    assert [d["name"] for d in bad["deltas"]] == ["cam", "table"]


def test_edge_overlay(tmp_path):
    twin = load()
    from PIL import Image, ImageDraw
    real = Image.new("RGB", (200, 100), (128, 128, 128))
    real.save(tmp_path / "real.jpg")
    sim = Image.new("RGB", (100, 50), (0, 0, 0))
    ImageDraw.Draw(sim).rectangle([25, 10, 75, 40], fill=(255, 255, 255))
    sim.save(tmp_path / "sim.png")
    twin.edge_overlay(tmp_path / "real.jpg", tmp_path / "sim.png", tmp_path / "side.png")
    out = Image.open(tmp_path / "side.png").convert("RGB")
    assert out.size == (200, 100)
    assert out.getpixel((50, 20)) == twin.EDGE_COLOUR  # the square's edge, scaled 2x
    assert out.getpixel((100, 50)) == (128, 128, 128)  # inside the square: no edge


def test_edge_overlay_uniform_sim_paints_nothing(tmp_path):
    twin = load()
    from PIL import Image, ImageChops
    real = Image.new("RGB", (60, 40), (128, 128, 128))
    real.save(tmp_path / "real.png")
    Image.new("RGB", (30, 20), (200, 200, 200)).save(tmp_path / "sim.png")
    twin.edge_overlay(tmp_path / "real.png", tmp_path / "sim.png", tmp_path / "o.png")
    assert ImageChops.difference(real, Image.open(tmp_path / "o.png").convert("RGB")).getbbox() is None


def test_zero_ov_vector_is_z_and_nan_flagged():
    twin = load()
    m = twin.frame_pose({"orientation": {"type": "ov_degrees", "value": {"x": 0, "y": 0, "z": 0, "th": 90}}})
    ref = twin.frame_pose({"orientation": {"type": "ov_degrees", "value": {"x": 0, "y": 0, "z": 1, "th": 90}}})
    assert np.isfinite(m).all() and np.allclose(m, ref)
    nan = np.full((4, 4), np.nan)
    assert not twin.compare_frames({"a": {"parent": "world"}}, {"a": ("world", nan)})["ok"]


def test_kinematics_prismatic_limits_use_mm_and_bad_chains_raise():
    twin = load()
    import json as _json
    import pytest
    slide = {"links": [{"id": "base", "parent": "world"}], "joints": [
        {"id": "s", "type": "prismatic", "parent": "base", "axis": {"x": 0, "y": 0, "z": 1}, "min": 0, "max": 100}]}
    other = _json.loads(_json.dumps(slide))
    other["joints"][0]["max"] = 101.5
    r = twin.compare_kinematics(twin.chain(_json.dumps(slide).encode(), 1), twin.chain(_json.dumps(other).encode(), 1))
    assert not r["limits"]
    with pytest.raises(ValueError, match="no moving joints"):
        twin.chain(_json.dumps({"links": [{"id": "base", "parent": "world"}], "joints": []}).encode(), 1)
    two_roots = '<robot name="r"><link name="a"/><link name="b"/></robot>'
    with pytest.raises(ValueError, match="one root link"):
        twin.chain(two_roots.encode(), 2)


def test_main_part_id():
    twin = load()
    listing = ("Parts:\n\tID: aaa\n\tName: cell-aux\n\tLast Access: x\n"
               "\tID: 8ada56a1-603c\n\tName: franka-mug-sim-main (main)\n\tLast Access: y\n")
    assert twin.main_part_id(listing) == "8ada56a1-603c"


def test_pull_live_keeps_the_fov_when_the_images_fail(tmp_path, monkeypatch):
    import asyncio

    import pytest
    from types import SimpleNamespace

    from viam.components.camera import Camera

    twin = load()

    class FakeCamera:
        async def get_properties(self):
            return SimpleNamespace(intrinsic_parameters=SimpleNamespace(focal_y_px=480.0, height_px=480))

        async def get_images(self):
            raise RuntimeError("no frames")

    monkeypatch.setattr(Camera, "from_robot", staticmethod(lambda robot, name: FakeCamera()))
    real = {"components": [{"name": "cam", "api": "rdk:component:camera", "model": "webcam"}]}
    pose = SimpleNamespace(x=1, y=2, z=3, o_x=0, o_y=0, o_z=1, theta=90)
    # The shape of robot.v1 FrameSystemConfig: the pose sits inside `frame`.
    cfg = SimpleNamespace(frame=SimpleNamespace(
        reference_frame="cam", pose_in_observer_frame=SimpleNamespace(reference_frame="world", pose=pose)))

    class FakeRobot:
        async def get_frame_system_config(self):
            return [cfg]

    out = asyncio.run(twin.pull_live(FakeRobot(), real, tmp_path))
    assert out["cameras"]["cam"]["fov_deg"] == pytest.approx(53.13, abs=0.01)
    assert out["errors"] == {"cam": "no frames"}
    assert out["frames"]["cam"]["parent"] == "world" and out["frames"]["cam"]["orientation"]["value"]["th"] == 90


def test_kinematics_sva_keys_are_case_insensitive():
    twin = load()
    import json as _json
    upper = _json.loads(_json.dumps(SVA).replace('"x"', '"X"').replace('"y"', '"Y"').replace('"z"', '"Z"'))
    result = twin.compare_kinematics(twin.chain(_json.dumps(SVA).encode(), 1), twin.chain(_json.dumps(upper).encode(), 1))
    assert result["ok"], result


def test_convert_copies_modules_without_reload_builds():
    twin = load()
    real = {"services": [{"name": "teach", "api": "rdk:service:generic", "model": "acme:bot:teacher"}],
            "modules": [{"type": "registry", "name": "acme_bot", "module_id": "acme:bot", "version": "latest",
                         "reload_enabled": True, "reload_path": "/root/x.tar.gz", "reload_user": "a@b.c"}]}
    config, report = twin.convert(real, module_version="0.17.1")
    bot = next(m for m in config["modules"] if m["name"] == "acme_bot")
    assert bot == {"type": "registry", "name": "acme_bot", "module_id": "acme:bot", "version": "latest"}
    assert any(f["name"] == "acme_bot" and "reload" in f["note"] for f in report["flags"])


def test_convert_keeps_disabled_resources_disabled():
    twin = load()
    real = {"components": [{"name": "wall", "api": "rdk:component:gripper", "model": "erh:vmodutils:obstacle",
                            "disabled": True, "frame": {"parent": "world"}}],
            "services": [{"name": "yolo", "api": "rdk:service:vision", "model": "acme:v:y", "disabled": True}]}
    config, report = twin.convert(real, module_version="0.17.1")
    assert comp(config, "wall")["disabled"] is True
    assert next(s for s in config["services"] if s["name"] == "yolo")["disabled"] is True
    assert {r["name"]: r["outcome"] for r in report["resources"]} == {"wall": "disabled", "yolo": "disabled"}
    assert "import_geometry" not in comp(config, report["world"])["attributes"]
    assert "wall" not in report["expected_frames"]
    assert twin.compare_names(report, set())["ok"]


def test_edge_overlay_rejects_a_tiny_image(tmp_path):
    twin = load()
    import pytest as _pytest
    from PIL import Image
    Image.new("RGB", (1, 1)).save(tmp_path / "real.png")
    Image.new("RGB", (10, 10)).save(tmp_path / "sim.png")
    with _pytest.raises(ValueError, match="too small"):
        twin.edge_overlay(tmp_path / "real.png", tmp_path / "sim.png", tmp_path / "side.png")


def test_secrets_are_removed_and_flagged_by_path():
    twin = load()
    secret = "https://hooks.slack.com/services/T/B/x"
    real = {
        "components": [],
        "services": [
            {"name": "notify", "api": "rdk:service:generic", "model": "acme:mod:notify",
             "attributes": {"slack_webhook_url": "u1", "nested": {"auth": {"token": "tok-1"}}, "x": secret,
                            "camera_name": "cam", "keyframe": "home", "home_keyframe": "h", "apiKey": "k2"}},
            {"name": "sh", "api": "rdk:service:shell", "model": "rdk:builtin:shell"},
        ],
        "modules": [{"type": "registry", "name": "acme_mod", "module_id": "acme:mod", "version": "1",
                     "env": {"VIAM_API_KEY": "k3", "HF_TOKEN": "k4", "LOG_LEVEL": "debug"}},
                    {"type": "registry", "name": "acme_two", "module_id": "acme:two", "version": "1",
                     "env": {"VIAM_API_KEY_ID": "k5"}}],
    }
    config, report = twin.convert(real)
    (svc,) = config["services"]
    assert svc["attributes"] == {"camera_name": "cam", "keyframe": "home", "home_keyframe": "h", "nested": {}}
    assert any(r["name"] == "sh" and r["outcome"] == "removed" for r in report["resources"])
    mods = {m["name"]: m for m in config["modules"]}
    assert mods["acme_mod"]["env"] == {"LOG_LEVEL": "debug"}
    assert "env" not in mods["acme_two"]
    flagged = " ".join(f["note"] for f in report["flags"])
    for path in ("attributes.slack_webhook_url", "attributes.nested.auth", "attributes.x", "attributes.apiKey",
                 "env.VIAM_API_KEY", "env.HF_TOKEN", "env.VIAM_API_KEY_ID"):
        assert path in flagged
    blob = json.dumps(report)
    for value in ("u1", "tok-1", "k2", "k3", "k4", "k5"):
        assert f'"{value}"' not in blob
    assert "tok-1" not in blob and "hooks.slack.com" not in blob


def test_secret_name_matches_words_not_substrings():
    twin = load()
    for n in ("keyframe", "home_keyframe", "authority", "monkey", "hotkey", "num_keypoints", "session_id",
              "private"):
        assert not twin.secret_name(n), n
    for n in ("api_key", "apiKey", "apikey", "HF_TOKEN", "slack_webhook_url", "password", "client_secret",
              "PRIVATE_KEY", "Authorization", "passphrase", "pwd", "dsn", "connection_string", "api_secret"):
        assert twin.secret_name(n), n


def test_scrub_only_removes_stringy_values_and_known_shapes():
    twin = load()
    attrs = {"max_tokens": 512, "use_auth": True, "session_timeout": 30, "keyframe": "home", "hotkey": "x",
             "client_secret": "s", "PRIVATE_KEY": "-----BEGIN", "u": "https://a/b?api_key=1",
             "jwt": "eyJhbGciOi.eyJzdWIi.sig", "aws": "AKIA" + "A" * 16, "g": "glpat-abc", "gh": "ghs_abc", "k": "AIzaX"}
    out, gone = twin.scrub(attrs, "attributes")
    assert out == {"max_tokens": 512, "use_auth": True, "session_timeout": 30, "keyframe": "home", "hotkey": "x"}
    assert len(gone) == 8 and all("." in g for g in gone)


def test_removed_values_stay_out_of_notes_and_packages_are_scrubbed():
    twin = load()
    real = {"components": [], "services": [
        {"name": "s", "api": "rdk:service:generic", "model": "rdk:builtin:x",
         "attributes": {"path": "/home/u/x", "api_key": "/home/u/secret.pem"}}],
        "packages": [{"name": "p", "type": "ml_model", "package": "a/b", "version": "1", "api_key": "zzz"}]}
    config, report = twin.convert(real)
    blob = json.dumps([config, report])
    assert "secret.pem" not in blob and "zzz" not in blob and "/home/u/x" in blob
