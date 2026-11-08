import math
import time
from pathlib import Path

import mujoco
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


def test_the_rovers_wheels_turn_by_distance_over_radius(sim, rover):
    radius = 0.381 / (2 * math.pi)

    def wheel(side):
        return float(sim.data.qpos[sim.model.joint(f"rover/{side}_wheel").qposadr[0]])

    finish(sim, rover.move_straight(0.5, 0.2)).result()
    settle(sim)
    travelled = pose(sim)[1]
    # both axles point +X, the rover's right: rolling forward is a negative turn about them
    assert wheel("right") == pytest.approx(-travelled / radius, rel=0.01)
    assert wheel("left") == pytest.approx(-travelled / radius, rel=0.01)
    finish(sim, rover.spin(math.radians(90), math.radians(90))).result()
    settle(sim)
    arc = pose(sim)[2] * 0.356 / 2  # a left turn: the right wheel rolls forward, the left back
    assert wheel("right") == pytest.approx(-(travelled + arc) / radius, rel=0.01)
    assert wheel("left") == pytest.approx(-(travelled - arc) / radius, rel=0.01)


def test_in_ideal_mode_only_the_chassis_collides(sim, rover):
    model = sim.model
    chassis = model.body("rover/chassis").id
    for geom in range(model.ngeom):
        if model.body_rootid[model.geom_bodyid[geom]] == chassis:
            touches = bool(model.geom_contype[geom] or model.geom_conaffinity[geom])
            assert touches == (model.geom_bodyid[geom] == chassis), model.geom(geom).name
    rover.set_velocity(0.3, 0.5)
    sim.advance(1.0)
    rover_geoms = {g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == chassis}
    contacts = sim.data.contact.geom[: sim.data.ncon]
    assert not rover_geoms & set(contacts.ravel())  # the shell is 3 cm up; nothing else touches


ROS_BASE = """<mujoco>
  <worldbody>
    <body name="base_link">
      <geom type="box" size="0.3 0.2 0.05" pos="0 0 0.1" mass="10"/>
      <body name="fl" pos="0.2 0.25 0"><joint name="fl" axis="0 1 0"/>
        <geom type="cylinder" size="0.1 0.03" euler="90 0 0" mass="1"/></body>
      <body name="fr" pos="0.2 -0.25 0"><joint name="fr" axis="0 1 0"/>
        <geom type="cylinder" size="0.1 0.03" euler="90 0 0" mass="1"/></body>
      <body name="rl" pos="-0.2 0.25 0"><joint name="rl" axis="0 1 0"/>
        <geom type="cylinder" size="0.1 0.03" euler="90 0 0" mass="1"/></body>
      <body name="rr" pos="-0.2 -0.25 0"><joint name="rr" axis="0 1 0"/>
        <geom type="cylinder" size="0.1 0.03" euler="90 0 0" mass="1"/></body>
    </body>
  </worldbody>
</mujoco>"""


def test_a_ros_axis_base_drives_along_its_configured_frame(sim, tmp_path):
    path = tmp_path / "skid.xml"
    path.write_text(ROS_BASE)
    with pytest.raises(ValueError, match="set max_linear_mps and max_angular_dps"):
        BaseHandle(sim, "skid", {"mjcf": str(path)})
    quarter = [math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]  # yawed 90° left
    skid = BaseHandle(sim, "skid", {
        "mjcf": str(path), "max_linear_mps": 1.0, "max_angular_dps": 90,
        "position": [1.0, 0.0, 0.0], "orientation_wxyz": quarter,
    })
    site = sim.model.site("skid/viam_frame").id
    assert sim.data.site_xpos[site] == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    frame = sim.data.site_xmat[site].reshape(3, 3)
    assert frame[:, 1] == pytest.approx([-1.0, 0.0, 0.0], abs=1e-9)  # +Y is the configured +Y
    assert frame[:, 2] == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    # base_link sits at axle height, wheel radius 0.1 m above the frame; its +X (ROS forward) is +Y
    chassis = sim.model.body("skid/base_link").id
    assert sim.data.xpos[chassis] == pytest.approx([1.0, 0.0, 0.1], abs=1e-9)
    assert sim.data.xmat[chassis].reshape(3, 3)[:, 0] == pytest.approx(frame[:, 1], abs=1e-9)
    finish(sim, skid.move_straight(0.3, 0.2)).result()
    settle(sim)
    assert sim.data.site_xpos[site] == pytest.approx([0.7, 0.0, 0.0], abs=0.01)


def frame_pose(sim, name="rover"):
    """(x, y, yaw) of the base's viam_frame site in the world, yaw from its +Y."""
    site = sim.model.site(f"{name}/viam_frame").id
    forward = sim.data.site_xmat[site].reshape(3, 3)[:, 1]
    x, y, _ = sim.data.site_xpos[site]
    return float(x), float(y), math.atan2(-forward[0], forward[1])


@pytest.fixture
def wheeled(sim):
    return BaseHandle(sim, "rover", {"asset": "viam_rover", "drive": "wheels"})


def test_a_wheel_driven_rover_rests_on_its_wheels(sim, wheeled):
    sim.advance(1.0)
    site = sim.model.site("rover/viam_frame").id
    assert sim.data.site_xpos[site] == pytest.approx([0, 0, 0], abs=0.001)  # no gravcomp: it sits
    assert not wheeled.is_moving()
    assert sim.model.joint("rover/free").type == mujoco.mjtJoint.mjJNT_FREE
    # the file has no actuators: the module adds one per wheel
    assert {sim.model.actuator(a).name for a in range(sim.model.nu)} == {
        "rover/left_wheel_drive", "rover/right_wheel_drive"
    }
    chassis = sim.model.body("rover/chassis").id
    assert not any(sim.model.body_gravcomp[b] for b in range(sim.model.nbody) if sim.model.body_rootid[b] == chassis)


def turned_by(before, after, target):
    """The angle turned from `before` to `after` (wrapped yaws), the turn nearest `target`."""
    return target + math.remainder(after - before - target, 2 * math.pi)


def test_wheels_move_straight_and_spin_on_the_measured_pose(sim, wheeled):
    finish(sim, wheeled.move_straight(1.0, 0.3)).result()
    settle(sim)
    x, y, yaw = frame_pose(sim)
    assert (x, y) == pytest.approx((0.0, 1.0), abs=0.02)
    assert yaw == pytest.approx(0.0, abs=math.radians(2))
    for target in (math.radians(90), math.radians(-90), math.radians(270)):
        before = frame_pose(sim)[2]
        finish(sim, wheeled.spin(target, math.radians(90))).result()
        settle(sim)
        assert turned_by(before, frame_pose(sim)[2], target) == pytest.approx(target, abs=math.radians(2))


def test_wheels_set_velocity_is_open_loop(sim, wheeled):
    wheeled.set_velocity(0.3, 0.0)
    sim.advance(1.0)
    assert wheeled.is_moving()
    assert all(sim.data.ctrl[sim.model.actuator(f"rover/{s}_wheel_drive").id] != 0 for s in ("left", "right"))
    assert frame_pose(sim)[1] == pytest.approx(0.3 * 1.0, abs=0.05)
    wheeled.stop()
    sim.advance(0.5)
    assert not wheeled.is_moving()


def test_a_wall_stalls_a_wheel_driven_move(sim):
    wall_world(sim)
    rover = BaseHandle(sim, "rover", {"asset": "viam_rover", "drive": "wheels", "move_timeout_margin_sec": 0.5})
    future = finish(sim, rover.move_straight(2.0, 0.4))
    with pytest.raises(RuntimeError, match="stalled"):
        future.result()
    assert frame_pose(sim)[1] == pytest.approx(0.83, abs=0.02)


def test_a_urdf_base_drives_and_warns_about_density_masses(sim, caplog):
    urdf = Path(__file__).parent / "data" / "skid.urdf"
    skid = BaseHandle(sim, "skid", {
        "mjcf": str(urdf), "mesh_dir": "../../assets/meshes", "max_linear_mps": 1.0, "max_angular_dps": 90,
    })
    assert "base skid: links ['base_link'] have no <inertial>; their mass comes from geom density" in caplog.text
    finish(sim, skid.move_straight(0.3, 0.2)).result()
    settle(sim)
    assert frame_pose(sim, "skid")[1] == pytest.approx(0.3, abs=0.01)


def test_a_rebuild_does_not_read_the_file_again(sim, tmp_path):
    path = tmp_path / "skid.xml"
    path.write_text(ROS_BASE)
    BaseHandle(sim, "skid", {"mjcf": str(path), "max_linear_mps": 1.0, "max_angular_dps": 90})
    path.unlink()
    BaseHandle(sim, "rover", {"asset": "viam_rover"})  # rebuilds the scene
    assert sim.model.body("skid/base_link").id >= 0


def test_a_negative_integral_never_reverses_a_spin(sim, wheeled):
    wheeled.spin(math.radians(90), math.radians(45))
    wheeled._integral = -10.0
    v, w = wheeled._wheel_command(1.0, 0.0, 0.0, 0.01)
    assert (v, w) == (0.0, 0.0)


def test_a_missing_mesh_names_the_file(sim, tmp_path):
    path = tmp_path / "missing.xml"
    path.write_text('<mujoco><asset><mesh file="nope.stl"/></asset><worldbody><body name="b"/></worldbody></mujoco>')
    with pytest.raises(ValueError, match=r"cannot load mjcf .*missing.xml"):
        BaseHandle(sim, "b", {"mjcf": str(path)})
