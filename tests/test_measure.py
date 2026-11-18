"""The gripper's measure DoCommand and the world's prop measure."""

import pytest

from mujoco_module.handles import ArmHandle, GripperHandle
from test_urdf import GRIPPER, urdf_arm


def lite6(sim):
    ArmHandle(sim, "arm", {"asset": "lite6"})
    return GripperHandle(sim, "grip", "arm", {})


def ur5e_2f85(sim):
    ArmHandle(sim, "arm", {"asset": "ur5e"})
    return GripperHandle(sim, "grip", "arm", {"asset": "robotiq_2f85"})


def test_lite6_gripper_tips_are_its_tcp_and_open_to_39_mm(sim):
    m = lite6(sim).measure()
    assert m["tcp_mm"] == pytest.approx(81.1, abs=3)  # the bundled frame's translation.z
    assert m["jaw_gap_mm"]["open"] == pytest.approx(39.4, abs=3)
    assert m["jaw_gap_mm"]["closed"] == pytest.approx(21.2, abs=3)


def test_2f85_opens_to_85_mm_and_its_pads_swing_down_as_they_close(sim):
    m = ur5e_2f85(sim).measure()
    assert m["jaw_gap_mm"]["open"] == pytest.approx(85.2, abs=3)
    assert m["jaw_gap_mm"]["closed"] < 5
    assert m["finger_reach_mm"]["open"] == 0.0
    assert m["finger_reach_mm"]["closed"] > 8  # measured 13 mm


def test_epick_reports_its_cup(sim):
    ArmHandle(sim, "arm", {"asset": "ur5e"})
    m = GripperHandle(sim, "cup", "arm", {"asset": "robotiq_epick"}).measure()
    assert m["cup_diameter_mm"] == pytest.approx(45, abs=8)


def test_measure_leaves_the_live_sim_alone(sim):
    gripper = lite6(sim)
    before = sim.data.qpos.copy(), float(sim.data.time), sim.data.ctrl.copy()
    gripper.measure()
    assert gripper._measure_model is not sim.model
    assert (sim.data.qpos == before[0]).all() and sim.data.time == before[1]
    assert (sim.data.ctrl == before[2]).all()


def test_world_measures_a_props_size_and_top(sim):
    # the conftest block: a 31.3 mm cube resting at z 15.65 mm
    m = sim.prop_extent("pick_block")
    assert m["size_m"] == pytest.approx([0.0313] * 3, abs=1e-4)
    assert m["top_z_m"] == pytest.approx(0.0313, abs=1e-3)


def test_urdf_gripper_measures_two_positive_gaps(sim):
    urdf_arm(sim)
    m = GripperHandle(sim, "g", "arm", {"mjcf": str(GRIPPER)}).measure()
    assert 0 < m["jaw_gap_mm"]["closed"] < m["jaw_gap_mm"]["open"]


def test_world_measure_of_an_unknown_prop_is_the_usual_error(sim):
    with pytest.raises(ValueError, match="unknown prop 'nope'"):
        sim.prop_extent("nope")


def test_world_measure_of_a_mesh_prop_is_its_true_size_and_centre(sim):
    # a mesh cylinder 80 mm across, 100 mm tall, its file origin 20 mm off its axis, turned 45 degrees:
    # bounding-box corners of its convex piece would read up to 113 mm across
    import trimesh

    from mujoco_module.sim import WorldConfig

    cylinder = trimesh.creation.cylinder(radius=0.04, height=0.1, sections=64).apply_translation((0.02, 0, 0.05))
    cfg = WorldConfig.from_attrs({"props": [{
        "name": "cup", "type": "mesh", "mesh": "/cup.stl", "mass": 0.2,
        "position": [0.3, 0, 0.2], "orientation_wxyz": [0.92388, 0, 0, 0.38268],
    }]})
    sim.configure_world(cfg, {"cup": [(cylinder.vertices.copy(), cylinder.faces.copy())]})
    m = sim.prop_extent("cup")
    assert m["size_m"] == pytest.approx([0.08, 0.08, 0.1], abs=0.001)
    origin = sim.get_prop_pose("cup")["position"]
    # the centre is 20 mm from the origin along the prop's own x, turned 45 degrees
    assert m["center_m"][:2] == pytest.approx([origin[0] + 0.01414, origin[1] + 0.01414], abs=0.001)
