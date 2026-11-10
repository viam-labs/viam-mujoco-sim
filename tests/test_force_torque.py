import asyncio

import numpy as np
import pytest
from viam.proto.app.robot import ComponentConfig, Frame, Orientation, Translation
from viam.utils import dict_to_struct

from mujoco_module.handles import ArmHandle, BaseHandle, GripperHandle
from mujoco_module.models.force_torque import MujocoForceTorque
from mujoco_module.sensors import ForceTorqueHandle
from mujoco_module.sim import WorldConfig

G = 9.81


def tool_mass(sim, arm):
    """The mass hanging on `arm`'s flange: the subtrees of the flange body's
    children that hold no arm joint."""
    ft = ForceTorqueHandle(sim, "probe", arm, {})
    sim.run(ft._resolve)
    mass = sum(float(sim.model.body_subtreemass[b]) for b in ft._ids.roots)
    ft.close()
    return mass


def expected_force(sim, arm, mass):
    """A hanging load's weight, in the arm's viam_frame axes."""
    rotation = sim.data.site_xmat[sim.model.site(f"{arm}/viam_frame").id].reshape(3, 3)
    return rotation.T @ np.array([0.0, 0.0, -mass * G])


def vec(reading):
    return np.array([reading[k] for k in "xyz"])


@pytest.mark.parametrize("arm_attrs,gripper_attrs,kg", [
    ({"asset": "lite6"}, {}, 0.308),  # its gripper is part of the arm's MJCF
    ({"asset": "ur5e"}, {"asset": "robotiq_2f85"}, 1.053),  # attached at the flange site
])
def test_at_rest_it_reads_the_tools_weight(sim, arm_attrs, gripper_attrs, kg):
    ArmHandle(sim, "arm", arm_attrs)
    GripperHandle(sim, "grip", "arm", gripper_attrs)
    ft = ForceTorqueHandle(sim, "ft", "arm", {})
    sim.advance(1.0)
    mass = tool_mass(sim, "arm")
    assert mass == pytest.approx(kg, abs=1e-3)
    r = ft.readings()
    assert vec(r["force"]) == pytest.approx([0, 0, mass * G], abs=2e-3)  # hanging, flange down: +Z
    assert vec(r["torque"]) == pytest.approx([0, 0, 0], abs=2e-3)
    assert r["tared"] is False


@pytest.mark.parametrize("asset,gripper", [("lite6", {}), ("ur5e", {"asset": "robotiq_2f85"})])
def test_a_tilted_wrist_turns_the_weight_in_the_sensor_frame(sim, asset, gripper):
    arm = ArmHandle(sim, "arm", {"asset": asset})
    GripperHandle(sim, "grip", "arm", gripper)
    ft = ForceTorqueHandle(sim, "ft", "arm", {"filter_s": 0})
    home = arm.get_joint_positions()
    arm.set_joint_targets([home[0], home[1], home[2], home[3] + 0.7, home[4] + 0.9, home[5] + 0.5])
    sim.advance(2.0)
    mass = tool_mass(sim, "arm")
    force = vec(ft.readings()["force"])
    assert np.linalg.norm(force) == pytest.approx(mass * G, rel=0.01)
    # in the arm's viam_frame (the Lite6's turns half a turn from its flange site)
    assert force == pytest.approx(expected_force(sim, "arm", mass), abs=0.05)
    # the weight's torque about the frame's origin: its lever is the tool's centre of mass
    site = sim.model.site("arm/viam_frame").id
    rotation = sim.data.site_xmat[site].reshape(3, 3)
    ft._resolve()
    centre = sum(sim.data.subtree_com[b] * sim.model.body_subtreemass[b] for b in ft._ids.roots) / mass
    weight = np.array([0.0, 0.0, -mass * G])
    torque = rotation.T @ np.cross(centre - sim.data.site_xpos[site], weight)
    assert vec(ft.readings()["torque"]) == pytest.approx(torque, abs=2e-3)
    assert np.linalg.norm(torque) > 0.02  # tilted: a real lever, not ~0


def test_a_suction_cup_adds_what_it_holds(sim):
    sim.configure_world(WorldConfig.from_attrs({"props": [{"name": "box", "size": 0.05, "mass": 0.2, "position": [2, 0, 0.03]}]}))
    ArmHandle(sim, "arm", {"asset": "ur5e"})
    gripper = GripperHandle(sim, "grip", "arm", {"asset": "robotiq_epick"})
    ft = ForceTorqueHandle(sim, "ft", "arm", {})
    sim.advance(1.0)
    empty = vec(ft.readings()["force"])
    tip = min((sim.data.xpos[b] for b in range(sim.model.nbody) if sim.model.body(b).name.startswith("grip/")),
              key=lambda p: p[2])
    sim.set_prop_pose("box", list(tip + [0, 0, -0.03]), [1, 0, 0, 0])
    gripper.close_jaws()
    sim.advance(1.0)
    assert sim.data.body("box").xpos[2] > 0.2  # held up off the floor
    held = vec(ft.readings()["force"])
    # the adhesion pull is an actuator force MuJoCo's internal forces leave out
    assert held - empty == pytest.approx([0, 0, 0.2 * G], abs=0.03)


def test_tare_filter_and_noise(sim):
    ArmHandle(sim, "arm", {"asset": "lite6"})
    GripperHandle(sim, "grip", "arm", {})
    ft = ForceTorqueHandle(sim, "ft", "arm", {})
    ft.tare()  # before any step: the filter starts at the first raw value, not 0
    r = ft.readings()
    assert vec(r["force"]) == pytest.approx([0, 0, 0], abs=1e-3) and r["tared"]
    sim.advance(0.5)
    assert vec(ft.readings()["force"]) == pytest.approx([0, 0, 0], abs=5e-3)
    ft.untare()
    assert vec(ft.readings()["force"])[2] == pytest.approx(0.308 * G, abs=5e-3)
    noisy = ForceTorqueHandle(sim, "noisy", "arm", {"force_noise_n": 0.5, "seed": 2})
    sim.advance(0.2)
    samples = np.array([vec(noisy.readings()["force"]) for _ in range(1000)])
    assert samples.std(axis=0) == pytest.approx([0.5] * 3, rel=0.1)
    assert samples.mean(axis=0)[2] == pytest.approx(0.308 * G, abs=0.06)


def test_filter_zero_reads_each_step_raw(sim):
    arm = ArmHandle(sim, "arm", {"asset": "lite6"})
    GripperHandle(sim, "grip", "arm", {})
    mass = tool_mass(sim, "arm")  # before the sensors: adding the probe rebuilds, which restarts filters
    raw = ForceTorqueHandle(sim, "raw", "arm", {"filter_s": 0})
    slow = ForceTorqueHandle(sim, "slow", "arm", {"filter_s": 5.0})
    sim.advance(0.5)
    home = arm.get_joint_positions()
    arm.set_joint_targets([*home[:4], home[4] + 0.9, home[5]])  # the wrist swings: the weight turns in the frame
    sim.advance(2.0)
    # once the wrist holds still, the raw reading is the weight; a 5 s filter still lags behind
    assert vec(raw.readings()["force"]) == pytest.approx(expected_force(sim, "arm", mass), abs=0.02)
    assert np.linalg.norm(vec(slow.readings()["force"]) - vec(raw.readings()["force"])) > 0.5


def test_a_reset_restarts_the_filter(sim):
    arm = ArmHandle(sim, "arm", {"asset": "lite6"})
    GripperHandle(sim, "grip", "arm", {})
    ft = ForceTorqueHandle(sim, "ft", "arm", {"filter_s": 5.0})  # slow: a stale value would linger
    arm.set_joint_targets([0, 0.5, 1.0, 0, 1.2, 0])
    sim.advance(2.0)
    sim.reset()
    assert vec(ft.readings()["force"])[2] == pytest.approx(0.308 * G, abs=0.02)


def test_the_arm_can_go_and_come_back(sim):
    arm = ArmHandle(sim, "arm", {"asset": "lite6"})
    GripperHandle(sim, "grip", "arm", {})
    ft = ForceTorqueHandle(sim, "ft", "arm", {})
    sim.advance(0.2)
    arm.close()
    sim.advance(0.1)
    with pytest.raises(RuntimeError, match="not a mujoco-sim arm"):
        ft.readings()
    ArmHandle(sim, "arm", {"asset": "lite6"})
    sim.advance(0.5)
    assert vec(ft.readings()["force"])[2] == pytest.approx(0.308 * G, abs=5e-3)


def test_a_parent_that_is_not_an_arm_raises(sim):
    BaseHandle(sim, "rover", {"asset": "viam_rover"})
    ft = ForceTorqueHandle(sim, "ft", "rover", {})
    sim.advance(0.1)
    with pytest.raises(RuntimeError, match="'rover' is not a mujoco-sim arm"):
        ft.readings()
    ArmHandle(sim, "rover2", {"asset": "lite6"})  # unrelated arm: still not its parent
    with pytest.raises(RuntimeError, match="not a mujoco-sim arm"):
        ft.readings()


def config(attrs, parent="pick-arm", translation=None, orientation=None):
    cfg = ComponentConfig(name="ft", attributes=dict_to_struct({"world": "sim-world", **attrs}))
    frame = Frame(parent=parent)
    if translation is not None:
        frame.translation.CopyFrom(translation)
    if orientation is not None:
        frame.orientation.CopyFrom(orientation)
    cfg.frame.CopyFrom(frame)
    return cfg


def test_validate():
    assert MujocoForceTorque.validate_config(config({})) == (["sim-world", "pick-arm"], [])
    identity = Orientation(vector_degrees=Orientation.OrientationVectorDegrees(z=1, theta=0))
    assert MujocoForceTorque.validate_config(config({}, orientation=identity))[0] == ["sim-world", "pick-arm"]


@pytest.mark.parametrize("cfg,message", [
    (config({}, parent="world"), "set frame.parent to the mujoco-sim arm"),
    (config({}, parent=""), "set frame.parent to the mujoco-sim arm"),
    (config({}, translation=Translation(z=10)), "remove the frame's translation and orientation"),
    (config({}, orientation=Orientation(vector_degrees=Orientation.OrientationVectorDegrees(z=1, theta=90))),
     "remove the frame's translation and orientation"),
    (config({"filter_s": -0.1}), "filter_s must be a finite number >= 0"),
    (config({"force_noise_n": -1}), "force_noise_n must be a finite number >= 0"),
    (config({"torque_noise_nm": "x"}), "torque_noise_nm must be a finite number"),
    (config({"seed": True}), "seed must be a whole number"),
    (config({"seed": -1}), "seed must be 0 or more"),
])
def test_validate_rejects(cfg, message):
    with pytest.raises(ValueError, match=message):
        MujocoForceTorque.validate_config(cfg)


def test_force_torque_model(running_sim):
    from mujoco_module.models.arm import MujocoArm
    from mujoco_module.models.gripper import MujocoGripper
    from mujoco_module.models.world import MujocoWorld

    MujocoWorld.new(ComponentConfig(name="sim-world", attributes=dict_to_struct({})), {})
    arm_cfg = ComponentConfig(name="pick-arm", attributes=dict_to_struct({"world": "sim-world", "asset": "lite6"}))
    arm_cfg.frame.CopyFrom(Frame(parent="world"))
    MujocoArm.new(arm_cfg, {})
    grip_cfg = ComponentConfig(name="grip", attributes=dict_to_struct({"world": "sim-world", "arm": "pick-arm"}))
    grip_cfg.frame.CopyFrom(Frame(parent="pick-arm", translation=Translation(z=81.1)))
    MujocoGripper.new(grip_cfg, {})
    ft = MujocoForceTorque.new(config({}), {})

    async def run():
        await asyncio.sleep(0.3)
        r = await ft.get_readings()
        assert r["force"]["z"] == pytest.approx(0.308 * G, abs=0.05)
        assert await ft.do_command({"command": "tare"}) == {"tared": True}
        assert (await ft.get_readings())["tared"] is True
        assert await ft.do_command({"command": "untare"}) == {"tared": False}
        with pytest.raises(ValueError, match="unknown command"):
            await ft.do_command({"command": "zero"})
        await ft.close()

    asyncio.run(run())
