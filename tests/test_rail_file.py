import json

import mujoco
import pytest

from mujoco_module.rail_file import CARRIAGE_MM, rail_files


@pytest.mark.parametrize("length", [1000.0, 2600.0])
def test_mjcf_slides_the_carriage_along_x_with_the_plate_on_top(length):
    mjcf, _ = rail_files(length, 400.0, 25.0)
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    joint = model.joint("x")
    assert joint.type == mujoco.mjtJoint.mjJNT_SLIDE
    assert joint.axis == pytest.approx([1, 0, 0])
    assert joint.range == pytest.approx([0, length / 1000])
    assert model.actuator("x").ctrlrange == pytest.approx([0, length / 1000])
    mujoco.mj_forward(model, data)
    site = model.site("attachment_site").id
    assert data.site_xpos[site] == pytest.approx([0, 0, 0], abs=1e-9)  # the frame: the plate's top at 0
    assert data.site_xmat[site].reshape(3, 3)[:, 2] == pytest.approx([0, 0, 1])  # facing up
    # the track lies on the floor (frame z is 400 mm above it) and spans the travel
    track = model.geom("track")
    assert data.geom_xpos[track.id][2] - track.size[2] == pytest.approx(-0.4, abs=1e-9)
    assert 2 * track.size[0] >= length / 1000 + CARRIAGE_MM[0] / 1000 - 1e-9
    assert all(model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0 for g in range(model.ngeom))
    data.qpos[model.joint("x").qposadr[0]] = length / 1000
    mujoco.mj_forward(model, data)
    assert data.site_xpos[site] == pytest.approx([length / 1000, 0, 0], abs=1e-9)


def test_kinematics_one_prismatic_joint_and_an_upright_tool():
    _, kinematics = rail_files(2600.0, 400.0, 25.0)
    k = json.loads(kinematics.read_text())
    assert k["kinematic_param_type"] == "SVA"
    (joint,) = k["joints"]
    assert (joint["type"], joint["axis"], joint["min"], joint["max"]) == ("prismatic", {"x": 1, "y": 0, "z": 0}, 0, 2600.0)
    tool = k["links"][-1]
    assert tool["id"] == "tool" and tool["parent"] == "carriage"
    assert tool["orientation"]["value"] == {"x": 0, "y": 0, "z": 1, "th": 0}


def test_files_are_cached_by_content():
    first = rail_files(1000.0, 400.0, 25.0)
    written = first[0].stat().st_mtime_ns
    assert rail_files(1000.0, 400.0, 25.0) == first
    assert first[0].stat().st_mtime_ns == written  # reused, not rewritten
    assert rail_files(1000.0, 400.0, 25.0) != rail_files(1000.0, 400.0, 30.0)
