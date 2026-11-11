"""The gantry's generated MJCF and kinematics files (gantry_file.py)."""

import mujoco
import numpy as np
import pytest

from mujoco_module import robots
from mujoco_module.gantry_file import gantry_files

LENGTHS = (1000.0, 800.0, 500.0)
FRAME_Z_MM = 1200.0


def test_kinematics_agree_with_the_mjcf():
    mjcf, kinematics = gantry_files(LENGTHS, FRAME_Z_MM)
    model = robots.load_mjcf(mjcf, None).compile()
    data = mujoco.MjData(model)
    kin = robots.parse_kinematics(kinematics.read_bytes(), False, str(kinematics))
    assert list(kin.joints) == ["x", "y", "z"]
    assert kin.forward(np.array([0, 0, 0.5]))[kin.end][:3, 3] == pytest.approx([0, 0, -0.5])  # Z lowers the tool; the tool point is the origin at zero
    site = model.site("attachment_site").id
    rng = np.random.default_rng(0)
    for q in [np.zeros(3), *rng.uniform(0, np.array(LENGTHS) / 1000, (20, 3))]:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        pose = kin.forward(q)[kin.end]
        assert pose[:3, 3] == pytest.approx(data.site_xpos[site], abs=1e-4)
        assert pose[:3, :3] == pytest.approx(data.site_xmat[site].reshape(3, 3), abs=1e-6)
    # the tool points down: its +z along the frame's -z
    assert data.site_xmat[site].reshape(3, 3)[:, 2] == pytest.approx([0, 0, -1])


def test_the_structure_boxes_match_its_geoms():
    mjcf, kinematics = gantry_files(LENGTHS, FRAME_Z_MM)
    model = robots.load_mjcf(mjcf, None).compile()
    kin = robots.parse_kinematics(kinematics.read_bytes(), False, str(kinematics))
    statics = [frame for name, frame in kin.frames.items() if name.startswith(("post", "beam"))]
    assert len(statics) == 6
    for frame in statics:
        geom = model.geom(frame.name)
        g = frame.geometry
        centre = [g["translation"][axis] / 1000 for axis in "xyz"]
        assert geom.pos == pytest.approx(centre)
        assert geom.size * 2 == pytest.approx([g[axis] / 1000 for axis in "xyz"])
    # posts stand on the floor: their feet at the frame's -z
    post = model.geom("post_00")
    assert post.pos[2] - post.size[2] == pytest.approx(-FRAME_Z_MM / 1000)


def test_only_the_column_collides():
    mjcf, _ = gantry_files(LENGTHS, FRAME_Z_MM)
    model = robots.load_mjcf(mjcf, None).compile()
    data = mujoco.MjData(model)
    assert [model.geom(n).contype.item() for n in ("post_00", "beam_0", "bridge", "carriage", "column")] == [0, 0, 0, 0, 1]
    for corner in np.array(np.meshgrid(*[[0, v / 1000] for v in LENGTHS])).T.reshape(-1, 3):
        data.qpos[:] = corner
        mujoco.mj_forward(model, data)
        assert data.ncon == 0, f"contacts at {corner}"


def test_the_same_inputs_reuse_the_files():
    first = gantry_files(LENGTHS, FRAME_Z_MM)
    mtimes = [p.stat().st_mtime_ns for p in first]
    assert gantry_files(LENGTHS, FRAME_Z_MM) == first
    assert [p.stat().st_mtime_ns for p in first] == mtimes
    assert gantry_files((900.0, 800.0, 500.0), FRAME_Z_MM) != first
