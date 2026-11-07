import math
import time

import pytest

from mujoco_module.handles import BaseHandle, CameraHandle
from mujoco_module.sim import WorldConfig


def pose(sim, name="rover"):
    """(x, y, yaw) of the rover's joints: spawn-frame meters and radians."""
    def q(joint):
        return float(sim.data.qpos[sim.model.joint(f"{name}/{joint}").qposadr[0]])
    return q("x"), q("y"), q("yaw")


@pytest.fixture
def rover(sim):
    return BaseHandle(sim, "rover", {"asset": "viam_rover"})


def test_rover_spawns_at_rest_with_its_frame_site(sim, rover):
    assert pose(sim) == (0.0, 0.0, 0.0)
    assert sim.model.site("rover/viam_frame").id >= 0
    assert not rover.is_moving()


def test_set_velocity_drives_an_arc(sim, rover):
    rover.set_velocity(0.2, math.radians(45))
    sim.advance(2.0)
    x, y, yaw = pose(sim)
    radius = 0.2 / math.radians(45)
    # starts heading +Y and turns left (counterclockwise) a quarter turn
    assert yaw == pytest.approx(math.radians(90), abs=math.radians(2))
    assert (x, y) == pytest.approx((-radius, radius), abs=0.01)


def test_velocity_is_clamped_to_the_maxima(sim, rover):
    rover.set_velocity(10.0, 0.0)
    sim.advance(1.0)
    assert pose(sim)[1] == pytest.approx(0.762, abs=0.02)


def test_stop_and_is_moving(sim, rover):
    rover.set_velocity(0.2, 0.0)
    sim.advance(0.2)
    assert rover.is_moving()
    rover.stop()
    sim.advance(0.2)
    assert not rover.is_moving()


def test_reset_returns_the_rover_to_spawn(sim, rover):
    rover.set_velocity(0.3, 0.5)
    sim.advance(1.0)
    sim.reset()
    assert pose(sim) == (0.0, 0.0, 0.0)
    sim.advance(0.5)
    assert pose(sim) == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)  # the command was cleared


def test_spawn_pose_comes_from_attrs(sim):
    BaseHandle(
        sim, "r2", {"asset": "viam_rover", "position": [1.0, 0.0, 0.0],
                    "orientation_wxyz": [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]},
    )
    site = sim.model.site("r2/viam_frame").id
    assert sim.data.site_xpos[site] == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    forward = sim.data.site_xmat[site].reshape(3, 3)[:, 1]  # chassis +Y
    assert forward == pytest.approx([-1.0, 0.0, 0.0], abs=1e-9)  # yawed 90° left


def test_camera_rides_the_rover(sim, rover):
    CameraHandle(sim, "cam", {"parent": "rover", "position": [0, 0.1, 0.2]})
    rover.set_velocity(0.25, 0.0)
    sim.advance(2.0)
    rover.stop()
    sim.advance(0.1)
    y = pose(sim)[1]
    assert sim.data.cam_xpos[sim.model.camera("cam").id] == pytest.approx([0, y + 0.1, 0.2], abs=1e-6)


def wall_world(sim, y=1.0):
    """A fixed wall across the rover's path, its near face at y - 0.02."""
    sim.configure_world(WorldConfig.from_attrs({"props": [
        {"name": "wall", "fixed": True, "size": 1.0, "scale": [2, 0.04, 0.3],
         "position": [0, y, 0.15]},
    ]}))


def finish(sim, future, limit_s=10.0):
    end = sim.data.time + limit_s
    while not future.done() and sim.data.time < end:
        sim.advance(0.01)
    assert future.done(), "move did not finish"
    return future


def settle(sim):
    sim.advance(0.3)


def test_move_straight_forward_and_back(sim, rover):
    finish(sim, rover.move_straight(0.5, 0.2)).result()
    settle(sim)
    x, y, _ = pose(sim)
    assert y == pytest.approx(0.5, abs=0.01) and x == pytest.approx(0.0, abs=0.005)
    finish(sim, rover.move_straight(-0.3, 0.2)).result()  # negative distance reverses
    settle(sim)
    assert pose(sim)[1] == pytest.approx(0.2, abs=0.01)
    finish(sim, rover.move_straight(0.1, -0.2)).result()  # so does negative speed
    settle(sim)
    assert pose(sim)[1] == pytest.approx(0.1, abs=0.01)


def test_move_straight_follows_the_starting_heading(sim, rover):
    finish(sim, rover.spin(math.radians(90), math.radians(90))).result()
    settle(sim)
    finish(sim, rover.move_straight(0.4, 0.2)).result()
    settle(sim)
    x, y, _ = pose(sim)
    assert (x, y) == pytest.approx((-0.4, 0.0), abs=0.01)  # facing -X after a left turn


def test_spin_left_right_and_past_a_half_turn(sim, rover):
    finish(sim, rover.spin(math.radians(90), math.radians(90))).result()
    settle(sim)
    assert pose(sim)[2] == pytest.approx(math.radians(90), abs=math.radians(1))
    finish(sim, rover.spin(math.radians(-90), math.radians(90))).result()
    settle(sim)
    assert pose(sim)[2] == pytest.approx(0.0, abs=math.radians(1))
    # 90 °/s keeps the braking coast under 0.5° (yaw accel is capped at 200 rad/s^2)
    finish(sim, rover.spin(math.radians(270), math.radians(90))).result()
    settle(sim)
    assert pose(sim)[2] == pytest.approx(math.radians(270), abs=math.radians(1))


def test_zero_velocity_is_rejected_and_zero_distance_is_done(sim, rover):
    with pytest.raises(ValueError, match="non-zero"):
        rover.move_straight(0.5, 0.0)
    with pytest.raises(ValueError, match="non-zero"):
        rover.spin(1.0, 0.0)
    assert rover.move_straight(0.0, 0.2).done()
    assert rover.spin(0.0, 1.0).done()


def test_a_wall_stalls_the_move(sim):
    wall_world(sim)
    rover = BaseHandle(sim, "rover", {"asset": "viam_rover", "move_timeout_margin_sec": 0.5})
    future = finish(sim, rover.move_straight(2.0, 0.4))
    with pytest.raises(RuntimeError, match="stalled"):
        future.result()
    # chassis front (half-length 0.15) stops at the wall face (0.98)
    assert pose(sim)[1] == pytest.approx(0.83, abs=0.01)


def test_stop_new_command_and_close_fail_a_move(sim, rover):
    future = rover.move_straight(1.0, 0.2)
    sim.advance(0.5)
    rover.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        future.result()
    future = rover.spin(math.radians(90), math.radians(30))
    sim.advance(0.5)
    rover.set_velocity(0.1, 0.0)
    with pytest.raises(RuntimeError, match="superseded"):
        future.result()
    future = rover.move_straight(1.0, 0.2)
    rover.close()
    with pytest.raises(RuntimeError, match="closed"):
        future.result()
    assert "rover/x" not in [sim.model.joint(i).name for i in range(sim.model.njnt)]


def test_close_zeroes_the_rover_actuators(sim, rover, monkeypatch):
    rover.set_velocity(0.2, 0.0)
    sim.advance(0.1)
    actuators = [sim.model.actuator(f"rover/{j}").id for j in ("x", "y", "yaw")]
    assert any(sim.data.ctrl[actuators])
    seen = []
    remove = sim.remove_contribution
    monkeypatch.setattr(
        sim, "remove_contribution", lambda name: (seen.append(list(sim.data.ctrl[actuators])), remove(name))
    )
    rover.close()
    assert seen == [[0.0, 0.0, 0.0]]


def test_a_paused_sim_does_not_time_out_a_move(sim):
    rover = BaseHandle(sim, "rover", {"asset": "viam_rover", "move_timeout_margin_sec": 0.1})
    future = rover.move_straight(0.02, 0.2)  # sim-time deadline: 0.1 s + 0.1 s margin
    # the deadline is sim time: wall time passes but the sim doesn't step, so no timeout
    time.sleep(0.3)
    assert not future.done()
    finish(sim, future).result()


def test_cancelling_the_future_stops_the_rover(sim, rover):
    future = rover.move_straight(1.0, 0.2)
    sim.advance(0.2)
    assert future.cancel()
    sim.advance(0.3)
    assert not rover.is_moving()


def test_a_goal_error_halts_the_rover_and_keeps_the_callback(sim, rover, monkeypatch):
    future = rover.move_straight(1.0, 0.2)
    sim.advance(0.1)
    monkeypatch.setattr(BaseHandle, "_progress", staticmethod(lambda *a: 1 / 0))
    sim.advance(0.05)
    with pytest.raises(ZeroDivisionError):
        future.result()
    assert "base:rover" in sim.step_callbacks
    sim.advance(0.3)
    assert not rover.is_moving()


@pytest.mark.parametrize("args", [(float("nan"), 0), (0, float("inf"))])
def test_set_velocity_rejects_non_finite(rover, args):
    with pytest.raises(ValueError, match="finite"):
        rover.set_velocity(*args)
