import pytest

import numpy as np
from viam.media.video import CameraMimeType, ViamImage

from mujoco_module.handles import ArmHandle, CameraHandle, encode_depth, encode_pcd


def test_camera_renders_configured_size(sim):
    ArmHandle(sim, "pick-arm", {"asset": "lite6"})
    camera = CameraHandle(
        sim,
        "scene-cam",
        {"position": [1.2, 1.2, 1.0], "target": [0.3, 0, 0.15], "width": 1280, "height": 720},
    )
    image = camera.get_rgb()
    assert image.shape == (720, 1280, 3)
    assert image.std() > 5  # not a blank frame


def test_intrinsics_follow_fovy(sim):
    camera = CameraHandle(sim, "cam", {"width": 640, "height": 480, "fov_deg": 90})
    intrinsics = camera.intrinsics()
    assert intrinsics["focal_y_px"] == pytest.approx(240.0)
    assert intrinsics["center_x_px"] == 320.0
    assert intrinsics["focal_x_px"] == pytest.approx(240.0)
    assert intrinsics["center_y_px"] == 240.0


def test_adding_camera_keeps_arm_state(sim):
    arm = ArmHandle(sim, "pick-arm", {"asset": "lite6"})
    arm.set_joint_targets([0.3, 0, 1.57, 0, 1.57, 0])
    sim.advance(1.0)
    before = arm.get_joint_positions()
    CameraHandle(sim, "cam", {})
    assert arm.get_joint_positions() == pytest.approx(before)


def test_viam_frame_orientation_maps_to_mujoco_camera(sim):
    # Viam cameras look along their +Z; a half turn about X points that down world -Z.
    CameraHandle(sim, "down", {"position": [0, 0, 2], "orientation_wxyz": [0, 1, 0, 0]})
    forward = -sim.data.cam_xmat[sim.model.camera("down").id].reshape(3, 3)[:, 2]
    assert forward == pytest.approx([0, 0, -1], abs=1e-6)


def _camera_pose(sim, name):
    camera = sim.model.camera(name).id
    return sim.data.cam_xpos[camera].copy(), sim.data.cam_xmat[camera].reshape(3, 3).copy()


def test_wrist_camera_rides_the_end_effector(sim):
    arm = ArmHandle(sim, "pick-arm", {"asset": "lite6"})
    CameraHandle(sim, "wrist", {"parent": "pick-arm", "position": [0, 0, 0.05]})
    end = sim.model.site("pick-arm/viam_end").id
    position, rotation = _camera_pose(sim, "wrist")
    end_z = sim.data.site_xmat[end].reshape(3, 3)[:, 2]
    # 5 cm out along the end effector's +Z, looking along it (MuJoCo looks down -Z)
    assert position == pytest.approx(sim.data.site_xpos[end] + 0.05 * end_z, abs=1e-6)
    assert -rotation[:, 2] == pytest.approx(end_z, abs=1e-6)

    arm.set_joint_targets([0.6, 0.2, 1.2, 0, 1.2, 0])
    sim.advance(1.0)
    moved, _ = _camera_pose(sim, "wrist")
    assert np.linalg.norm(moved - position) > 0.05


def test_wrist_camera_waits_for_its_arm(sim):
    camera = CameraHandle(sim, "wrist", {"parent": "pick-arm"})
    with pytest.raises(RuntimeError, match="not in the scene"):
        camera.get_rgb()
    arm = ArmHandle(sim, "pick-arm", {"asset": "lite6"})
    assert camera.get_rgb().shape == (480, 640, 3)
    arm.close()  # the camera's build skips instead of failing the rebuild
    ArmHandle(sim, "pick-arm", {"asset": "lite6"})  # re-added after the camera
    assert camera.get_rgb().shape == (480, 640, 3)


def test_depth_and_point_cloud_measure_the_floor(sim):
    camera = CameraHandle(
        sim, "down", {"position": [-1, 0, 1.0], "orientation_wxyz": [0, 1, 0, 0], "width": 64, "height": 48}
    )
    rgb, depth = camera.get_rgbd()
    assert rgb.shape == (48, 64, 3)
    assert depth.shape == (24, 32)  # half the color size by default
    assert depth[12, 16] == pytest.approx(1.0, abs=1e-3)
    points, colors = camera.point_cloud()
    assert len(points) == 32 * 24 == len(colors)
    assert np.abs(points[:, 2] - 1.0).max() < 1e-3  # a flat floor, square to the camera
    assert points[:, 0].min() < 0 < points[:, 0].max()


def test_depth_is_zero_where_nothing_is_in_view(sim):
    camera = CameraHandle(
        sim, "up", {"position": [0, 0, 1.0], "orientation_wxyz": [1, 0, 0, 0], "width": 16, "height": 12}
    )
    _, depth = camera.get_rgbd()
    assert not depth.any()
    assert len(camera.point_cloud()[0]) == 0


def test_encode_depth_round_trips_through_the_sdk():
    depth = np.array([[0.0, 0.5, 1.25], [2.0, 0.001, 70.0]])
    image = ViamImage(encode_depth(depth), CameraMimeType.VIAM_RAW_DEPTH)
    assert image.bytes_to_depth_array() == [[0, 500, 1250], [2000, 1, 65535]]


def test_encode_pcd_packs_points_and_colors():
    data = encode_pcd(np.array([[0.1, -0.2, 1.0]], np.float32), np.array([[255, 128, 1]], np.uint8))
    header, body = data.split(b"DATA binary\n")
    assert b"POINTS 1" in header
    record = np.frombuffer(body, dtype=[("xyz", "<f4", 3), ("rgb", "<u4")])
    assert record["xyz"][0] == pytest.approx([0.1, -0.2, 1.0])
    assert record["rgb"][0] == (255 << 16) | (128 << 8) | 1


def test_depth_size_is_configurable(sim):
    camera = CameraHandle(
        sim,
        "down",
        {"position": [-1, 0, 1.0], "orientation_wxyz": [0, 1, 0, 0], "width": 64, "height": 48,
         "depth_width": 16, "depth_height": 12},
    )
    _, depth = camera.get_rgbd()
    assert depth.shape == (12, 16)
    assert camera.intrinsics(16, 12)["focal_y_px"] == pytest.approx(camera.intrinsics()["focal_y_px"] / 4)


def test_point_cloud_colors_come_from_the_matching_color_pixel(sim):
    # straight down on the pale grey block from 30 cm: the center point is the
    # block and sits higher and brighter than the darker blue-grey floor at the edge
    camera = CameraHandle(
        sim, "down", {"position": [0.3, 0, 0.3], "orientation_wxyz": [0, 1, 0, 0], "width": 64, "height": 48}
    )
    points, colors = camera.point_cloud()
    center = np.argmin(np.hypot(points[:, 0], points[:, 1]))
    edge = np.argmax(np.hypot(points[:, 0], points[:, 1]))
    assert points[center, 2] < points[edge, 2] - 0.02
    assert int(colors[center].sum()) > int(colors[edge].sum()) + 60


def test_renderers_are_warm_before_the_first_frame(sim):
    CameraHandle(sim, "cam", {"width": 64, "height": 48})
    assert {(64, 48), (32, 24)} <= set(sim._renderers)
    ArmHandle(sim, "pick-arm", {"asset": "lite6"})  # a rebuild re-warms them
    assert {(64, 48), (32, 24)} <= set(sim._renderers)
