import asyncio
import math

import mujoco
import numpy as np
import pytest
from viam.proto.app.robot import ComponentConfig, Frame, Orientation, Translation
from viam.utils import dict_to_struct

from mujoco_module.handles import ArmHandle, BaseHandle
from mujoco_module.models.imu import MujocoImu
from mujoco_module.sensors import ImuHandle

QUARTER_X = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]


def test_at_rest_it_reads_gravity_up(sim):
    imu = ImuHandle(sim, "imu", {"position": [0, 0, 0.5]})
    sim.advance(0.1)
    r = imu.read()
    assert r.orientation_wxyz == pytest.approx([1, 0, 0, 0])
    assert r.angular_velocity_dps == pytest.approx([0, 0, 0], abs=1e-9)
    assert r.linear_acceleration_mps2 == pytest.approx([0, 0, 9.81], abs=1e-6)


def test_a_tilted_mount_rotates_gravity(sim):
    imu = ImuHandle(sim, "imu", {"position": [0, 0, 0.5], "orientation_wxyz": QUARTER_X})
    sim.advance(0.1)
    r = imu.read()
    assert r.orientation_wxyz == pytest.approx(QUARTER_X)
    assert r.linear_acceleration_mps2 == pytest.approx([0, 9.81, 0], abs=1e-6)  # sensor +Y is world up


def test_a_turning_base_reads_its_yaw_rate(sim):
    rover = BaseHandle(sim, "rover", {"asset": "viam_rover"})
    imu = ImuHandle(sim, "imu", {"parent": "rover", "position": [0, 0, 0.1]})
    rover.set_velocity(0.0, 0.5)
    sim.advance(1.0)
    r = imu.read()
    assert r.angular_velocity_dps[2] == pytest.approx(math.degrees(0.5), rel=0.02)
    assert r.angular_velocity_dps[:2] == pytest.approx([0, 0], abs=0.5)
    # the orientation turned about world Z by the yaw so far
    yaw = 2 * math.atan2(r.orientation_wxyz[3], r.orientation_wxyz[0])
    site = sim.model.site("rover/viam_frame").id
    xmat = sim.data.site_xmat[site].reshape(3, 3)
    assert yaw == pytest.approx(math.atan2(xmat[1, 0], xmat[0, 0]), abs=1e-6)


def test_an_arm_mounted_imu_follows_the_end_effector(sim):
    arm = ArmHandle(sim, "arm", {"asset": "lite6"})
    imu = ImuHandle(sim, "imu", {"parent": "arm"})
    arm.set_joint_targets([0.5, 0.2, 1.2, 0, 0.6, 0])
    sim.advance(2.0)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, sim.data.site_xmat[sim.model.site("arm/viam_frame").id])
    assert abs(float(np.dot(imu.read().orientation_wxyz, quat))) == pytest.approx(1.0, abs=1e-9)


def test_an_imu_waits_for_its_parent_and_survives_rebuilds(sim):
    imu = ImuHandle(sim, "imu", {"parent": "rover"})
    with pytest.raises(RuntimeError, match="not in the scene"):
        imu.read()
    BaseHandle(sim, "rover", {"asset": "viam_rover"})
    sim.advance(0.1)
    assert imu.read().linear_acceleration_mps2[2] == pytest.approx(9.81, abs=0.05)
    ImuHandle(sim, "other", {})  # a rebuild moves the sensors' addresses
    sim.advance(0.1)
    assert imu.read().linear_acceleration_mps2[2] == pytest.approx(9.81, abs=0.05)


def test_imu_noise_settings_apply(sim):
    quiet = ImuHandle(sim, "quiet", {"position": [0, 0, 0.5]})
    noisy = ImuHandle(sim, "noisy", {"position": [0, 0, 0.5], "gyro_noise_dps": 0.5, "gyro_bias_dps": [1, 0, 0],
                                     "accel_bias_mps2": [0, 0, 0.2], "seed": 3})
    sim.advance(0.1)
    assert quiet.read().angular_velocity_dps == pytest.approx([0, 0, 0], abs=1e-9)
    gyro = np.array([noisy.read().angular_velocity_dps for _ in range(500)])
    assert gyro.mean(axis=0) == pytest.approx([1, 0, 0], abs=0.1)
    assert noisy.read().linear_acceleration_mps2[2] == pytest.approx(10.01, abs=1e-6)


def config(attrs, parent="world", orientation=None):
    cfg = ComponentConfig(name="imu", attributes=dict_to_struct({"world": "sim-world", **attrs}))
    frame = Frame(parent=parent, translation=Translation(z=500))
    if orientation is not None:
        frame.orientation.CopyFrom(orientation)
    cfg.frame.CopyFrom(frame)
    return cfg


@pytest.mark.parametrize("attrs,message", [
    ({"gyro_noise_dps": -1}, "gyro_noise_dps must be a finite number >= 0"),
    ({"accel_noise_mps2": "a"}, "accel_noise_mps2 must be a finite number"),
    ({"orientation_noise_deg": float("inf")}, "orientation_noise_deg must be a finite number"),
    ({"gyro_bias_dps": [1, 2]}, "gyro_bias_dps must be a list of 3 finite numbers"),
    ({"accel_bias_mps2": [1, 2, "x"]}, "accel_bias_mps2 must be a list of 3 finite numbers"),
    ({"seed": 1.5}, "seed must be a whole number"),
    ({"seed": -1}, "seed must be 0 or more"),
])
def test_validate_rejects(attrs, message):
    with pytest.raises(ValueError, match=message):
        MujocoImu.validate_config(config(attrs))


def test_imu_model(running_sim):
    from mujoco_module.models.world import MujocoWorld

    MujocoWorld.new(ComponentConfig(name="sim-world", attributes=dict_to_struct({})), {})
    assert MujocoImu.validate_config(config({}, parent="rover")) == (["sim-world"], [])
    imu = MujocoImu.new(config({}, orientation=Orientation(vector_degrees=Orientation.OrientationVectorDegrees(z=1, theta=90))), {})

    async def run():
        await asyncio.sleep(0.2)
        accel = await imu.get_linear_acceleration()
        assert (accel.x, accel.y, accel.z) == pytest.approx((0, 0, 9.81), abs=1e-3)
        orientation = await imu.get_orientation()
        assert (orientation.o_z, orientation.theta) == pytest.approx((1, 90), abs=1e-6)
        readings = await imu.get_readings()
        assert set(readings) == {"orientation", "angular_velocity", "linear_acceleration"}
        properties = await imu.get_properties()
        assert properties.linear_acceleration_supported and properties.orientation_supported
        assert not properties.position_supported
        with pytest.raises(NotImplementedError):
            await imu.get_position()
        await imu.close()

    asyncio.run(run())
