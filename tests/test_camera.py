import pytest

from mujoco_module.handles import ArmHandle, CameraHandle


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


def test_adding_camera_keeps_arm_state(sim):
    arm = ArmHandle(sim, "pick-arm", {"asset": "lite6"})
    arm.set_joint_targets([0.3, 0, 1.57, 0, 1.57, 0])
    sim.advance(1.0)
    before = arm.get_joint_positions()
    CameraHandle(sim, "cam", {})
    assert arm.get_joint_positions() == pytest.approx(before)
