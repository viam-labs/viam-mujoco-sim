"""A gripper serves its collision shape (GetGeometries) to the motion
planner: its file's colliding geoms, open and closed, in its Viam frame."""

import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pytest
from test_gripper import GRIPPER_MJCF
from test_urdf import GRIPPER as URDF_GRIPPER
from test_urdf import urdf_arm
from viam.proto.app.robot import ComponentConfig, Frame
from viam.utils import dict_to_struct

from mujoco_module import gripper_shape, robots
from mujoco_module.models.arm import MujocoArm
from mujoco_module.models.world import MujocoWorld
from mujoco_module.gantry_file import gantry_files
from mujoco_module.handles import ArmHandle, GripperHandle
from mujoco_module.models.gripper import MujocoGripper

TCP_M = 0.1558  # the 2F-85's frame on the UR5e (the README's TCP)


def ply_vertices(geometry) -> np.ndarray:
    """A served mesh's vertices, in mm (the PLY is metres)."""
    text = geometry.mesh.mesh.decode().split("end_header\n", 1)
    count = int(next(line for line in text[0].splitlines() if line.startswith("element vertex")).split()[-1])
    rows = text[1].splitlines()[:count]
    return np.array([[float(v) for v in row.split()] for row in rows]) * 1000


def epick_or_2f85(sim, asset, position=(0, 0, TCP_M)):
    ArmHandle(sim, "arm", {"asset": "ur5e"})
    return GripperHandle(sim, "g", "arm", {"asset": asset, "position": list(position)})


def by_label(gripper) -> dict:
    return {g.label: g for g in gripper.geometries}


def test_the_2f85_serves_its_links_open_and_closed(sim):
    gripper = epick_or_2f85(sim, "robotiq_2f85")
    served = by_label(gripper)
    assert len(served) == len(gripper.geometries)  # labels are unique: RDK chains them by label
    # the base doesn't move with the jaw: once; every finger link: open and closed
    assert "base:1" in served and "base:1:open" not in served
    fingers = {label.rsplit(":", 1)[0] for label in served if label.endswith((":open", ":closed"))}
    assert {"right_pad1", "left_pad1", "right_follower:1", "left_driver:1"} <= fingers
    assert all(f"{f}:open" in served and f"{f}:closed" in served for f in fingers)


def test_the_2f85_shape_is_its_real_width_not_the_old_box(sim):
    served = by_label(epick_or_2f85(sim, "robotiq_2f85"))
    pads = [served[f"{side}_pad{i}:open"] for side in ("left", "right") for i in (1, 2)]
    # the fingers open along the gripper's x (its base body is turned a quarter about z,
    # so a pad's 8 mm thickness, its box's local y, lies along x)
    xs = [p.center.x + s * p.box.dims_mm.y / 2 for p in pads for s in (-1, 1)]
    assert 95 < max(xs) - min(xs) < 115  # the box was 150 mm
    # closed, the pads reach ~13 mm further along the tool axis
    open_z = served["right_pad1:open"].center.z
    assert served["right_pad1:closed"].center.z - open_z == pytest.approx(13, abs=3)
    assert abs(served["right_pad1:closed"].center.x) < 10  # and meet in the middle


def test_shapes_sit_in_the_gripper_s_frame(sim):
    served = by_label(epick_or_2f85(sim, "robotiq_2f85"))
    # the pads hang round the tool point (the frame's origin), the mount 155.8 mm up the tool axis
    assert abs(served["right_pad1:open"].center.z) < 30
    mount = ply_vertices(served["base_mount:1"])
    assert mount[:, 2].min() == pytest.approx(-TCP_M * 1000, abs=2)


def test_hulls_are_small_and_hug_their_meshes(sim, monkeypatch):
    gripper = epick_or_2f85(sim, "robotiq_2f85")
    faces = sum(g.mesh.mesh.decode().count("\n3 ") for g in gripper.geometries if g.HasField("mesh"))
    assert faces < 1500  # ~1,100 with the finger links at both poses; MuJoCo's own hulls: ~11,000
    monkeypatch.setattr(gripper_shape, "HULL_VERTICES", -1)  # MuJoCo's full hulls
    full = by_label(GripperHandle(sim, "g2", "arm", {"asset": "robotiq_2f85", "position": [0, 0, TCP_M]}))
    for label, geometry in by_label(gripper).items():
        if not geometry.HasField("mesh"):
            continue
        reduced, exact = ply_vertices(geometry), ply_vertices(full[label])
        # a reduced hull's corners are some of the full hull's: inside it, and close to its extent
        assert (reduced.min(axis=0) >= exact.min(axis=0) - 0.01).all()
        assert (reduced.max(axis=0) <= exact.max(axis=0) + 0.01).all()
        assert np.abs(reduced.min(axis=0) - exact.min(axis=0)).max() < 3
        assert np.abs(reduced.max(axis=0) - exact.max(axis=0)).max() < 3


def test_visual_geoms_are_not_served(sim):
    labels = by_label(epick_or_2f85(sim, "robotiq_2f85"))
    # each 2F-85 body has a visual geom (index 0, contype 0) before its collision geom
    assert not any(label.split(":")[1:2] == ["0"] for label in labels)


def test_the_epick_serves_one_pose(sim):
    served = by_label(epick_or_2f85(sim, "robotiq_epick", position=(0, 0, 0.130)))
    assert served and not any(label.endswith((":open", ":closed")) for label in served)


def test_a_mimic_urdf_gripper_serves_both_fingers(sim):
    urdf_arm(sim)
    served = by_label(GripperHandle(sim, "g", "arm", {"mjcf": str(URDF_GRIPPER)}))
    moving = {label for label in served if label.endswith(":closed")}
    assert len(moving) >= 2  # the driven finger and its <mimic>


def test_a_custom_mjcf_gripper_serves_its_palm_once_and_its_jaw_twice(sim, tmp_path):
    path = tmp_path / "gripper.xml"
    path.write_text(GRIPPER_MJCF)
    ArmHandle(sim, "arm", {"asset": "ur5e"})
    served = by_label(GripperHandle(sim, "g", "arm", {"mjcf": str(path)}))
    assert set(served) == {"palm:0", "jaw:0:open", "jaw:0:closed"}
    # the jaw slides 30 mm along y between open and closed
    assert served["jaw:0:closed"].center.y - served["jaw:0:open"].center.y == pytest.approx(30, abs=1)


def test_a_gripper_with_no_frame_serves_in_the_mount_s_end_frame(sim):
    served = by_label(epick_or_2f85(sim, "robotiq_2f85", position=(0, 0, 0)))
    # no frame: the shape is relative to the mount's end effector, so the pads hang ~150 mm out
    assert served["right_pad1:open"].center.z == pytest.approx(TCP_M * 1000, abs=30)
    assert ply_vertices(served["base_mount:1"])[:, 2].min() == pytest.approx(0, abs=2)


def test_the_gripper_model_serves_the_shape_in_its_configured_frame(running_sim):
    def config(name, attrs, parent=None, z=0.0):
        cfg = ComponentConfig(name=name, attributes=dict_to_struct(attrs))
        if parent is not None:
            cfg.frame.CopyFrom(Frame(parent=parent))
            cfg.frame.translation.z = z
        return cfg

    MujocoWorld.new(config("sim-world", {"real_time": False}), {})
    MujocoArm.new(config("arm", {"world": "sim-world", "asset": "ur5e"}, parent="world"), {})
    gripper = MujocoGripper.new(
        config("g", {"world": "sim-world", "mount": "arm", "asset": "robotiq_2f85"}, parent="arm", z=155.8), {}
    )
    served = {g.label: g for g in asyncio.run(gripper.get_geometries())}
    assert abs(served["right_pad1:open"].center.z) < 30  # round the configured tool point


def test_a_gripper_in_its_arm_s_own_file_serves_nothing(sim):
    ArmHandle(sim, "arm", {"asset": "lite6"})
    assert GripperHandle(sim, "g", "arm", {}).geometries == []


def gripper_config(attrs, parent=None):
    cfg = ComponentConfig(name="g", attributes=dict_to_struct(attrs))
    if parent is not None:
        cfg.frame.CopyFrom(Frame(parent=parent))
    return cfg


def test_a_gripper_s_frame_hangs_off_its_mount():
    MujocoGripper.validate_config(gripper_config({"world": "w", "mount": "arm"}, parent="arm"))
    MujocoGripper.validate_config(gripper_config({"world": "w", "mount": "arm"}))  # no frame: fine
    with pytest.raises(ValueError, match='frame.parent must be its mount "arm"'):
        MujocoGripper.validate_config(gripper_config({"world": "w", "mount": "arm"}, parent="world"))


def served_pad_in_world(arm, gripper, frame_m, label):
    """A served pad's centre in the world, through the gripper's Viam pose: the
    mount's end pose (the module's kinematics) and the gripper's frame."""
    position, quat = arm.get_end_pose()
    end = robots.transform(position, quat)
    pad = by_label(gripper)[label]
    return (frame_m @ end @ robots.transform([0, 0, TCP_M]) @ [pad.center.x / 1000, pad.center.y / 1000, pad.center.z / 1000, 1])[:3]


def test_served_shapes_land_on_the_sim_s_pads_on_the_ur5e(sim):
    arm = ArmHandle(sim, "arm", {"asset": "ur5e"})
    gripper = GripperHandle(sim, "g", "arm", {"asset": "robotiq_2f85", "position": [0, 0, TCP_M]})
    sim.advance(1.0)  # the jaw settles open
    for name in ("right_pad1", "left_pad2"):
        world = served_pad_in_world(arm, gripper, np.eye(4), f"{name}:open")
        assert world == pytest.approx(sim.data.geom(f"g/{name}").xpos, abs=0.002)


def test_served_shapes_land_on_the_sim_s_pads_with_a_turned_frame(sim):
    turn = [math.cos(math.pi / 8), math.sin(math.pi / 8), 0, 0]  # 45 degrees about x
    offset = [0.01, 0.02, 0.15]
    arm = ArmHandle(sim, "arm", {"asset": "ur5e"})
    gripper = GripperHandle(sim, "g", "arm", {"asset": "robotiq_2f85", "position": offset, "orientation_wxyz": turn})
    sim.advance(1.0)
    position, quat = arm.get_end_pose()
    gripper_frame = robots.transform(position, quat) @ robots.transform(offset, turn)
    for name in ("right_pad1", "left_pad2"):
        pad = by_label(gripper)[f"{name}:open"]
        world = (gripper_frame @ [pad.center.x / 1000, pad.center.y / 1000, pad.center.z / 1000, 1])[:3]
        assert world == pytest.approx(sim.data.geom(f"g/{name}").xpos, abs=0.002)


def test_served_shapes_land_on_the_sim_s_pads_on_the_gantry(sim):
    frame = np.eye(4)
    frame[:3, 3] = [-0.5, -0.4, 1.2]
    mjcf, kinematics = gantry_files((1000.0, 800.0, 500.0), 1200.0)
    gantry = ArmHandle(sim, "gantry", {
        "mjcf": str(mjcf), "kinematics_url": str(kinematics), "collision": "mesh", "position": [-0.5, -0.4, 1.2],
    })
    gripper = GripperHandle(sim, "g", "gantry", {"asset": "robotiq_2f85", "position": [0, 0, TCP_M]})
    sim.advance(1.0)
    world = served_pad_in_world(gantry, gripper, frame, "right_pad1:open")
    assert world == pytest.approx(sim.data.geom("g/right_pad1").xpos, abs=0.002)


@pytest.mark.parametrize("example", sorted(p.name for p in (Path(__file__).parents[1] / "examples").glob("*.json")))
def test_example_grippers_carry_no_frame_box(example):
    """A gripper serves its own shape; a frame box would be checked as well (RDK keeps both)."""
    cfg = json.loads((Path(__file__).parents[1] / "examples" / example).read_text())
    for item in cfg.get("components", []):
        if item["model"] == "viam-labs:mujoco-sim:gripper" and item["attributes"].get("asset") != "lite6_gripper":
            assert "geometry" not in item.get("frame", {}), f"{example}: {item['name']} has a frame geometry"
