"""A free-floating robot file (a legged robot's trunk on a free joint) in viam-labs:mujoco-sim:robot,
its BAM actuators, read-only limb arms and the io DoCommand. A synthetic two-legged robot stands in
for the MicroDuck, whose meshes are CC BY-NC-SA and stay out of this repo."""

import asyncio
import time

import mujoco
import numpy as np
import pytest
from viam.proto.app.robot import ComponentConfig
from bam.mujoco import MujocoController
from viam.utils import dict_to_struct, struct_to_dict

from mujoco_module.bam import BamConfig, rewrite_spec
from mujoco_module.handles import ArmHandle, JointTrajectoryPoint
from mujoco_module.models.robot import MujocoRobot
from mujoco_module.models.world import MujocoWorld
from mujoco_module.robot_file import RobotHandle
from mujoco_module.sim import Contribution, WorldConfig

# A 0.3 kg trunk on a free joint, and two 2-hinge legs about y with wide flat box feet, so it
# stands statically. Actuator order interleaves the legs the way the duck's file does (left, then
# right), so io's joint order is the file's actuator order. Geometry at STAND (hip 0.15,
# knee -0.3):
# - each 0.1 m segment drops 0.1*cos(0.15) = 0.0989 m;
# - the shin's world pitch is -0.15, so the foot box is pitched +0.15 on the shin to lie flat;
# - trunk centre to sole is 0.03 + 2*0.0989 + 0.005 = 0.2328, so the trunk starts at 0.234,
#   about 1 mm above the floor.
# STAND must differ from qpos0 (all zeros): several tests use that to catch a qpos wipe.
FLOATER = """
<mujoco>
  <compiler angle="radian"/>
  <default><joint damping="0.05" frictionloss="0.01" armature="0.001"/><position kp="5"/></default>
  <worldbody>
    <body name="trunk" pos="0 0 0.234">
      <freejoint name="root"/>
      <geom type="box" size="0.05 0.04 0.03" mass="0.3"/>
      <body name="l_thigh" pos="0 0.03 -0.03">
        <joint name="l_hip" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" mass="0.05"/>
        <body name="l_shin" pos="0 0 -0.1">
          <joint name="l_knee" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" mass="0.05"/>
          <geom type="box" size="0.04 0.02 0.005" pos="0 0 -0.105" euler="0 0.15 0" mass="0.02"/>
          <site name="l_foot" pos="0 0 -0.1"/>
        </body>
      </body>
      <body name="r_thigh" pos="0 -0.03 -0.03">
        <joint name="r_hip" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" mass="0.05"/>
        <body name="r_shin" pos="0 0 -0.1">
          <joint name="r_knee" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" mass="0.05"/>
          <geom type="box" size="0.04 0.02 0.005" pos="0 0 -0.105" euler="0 0.15 0" mass="0.02"/>
          <site name="r_foot" pos="0 0 -0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="l_hip" joint="l_hip"/>
    <position name="l_knee" joint="l_knee"/>
    <position name="r_hip" joint="r_hip"/>
    <position name="r_knee" joint="r_knee"/>
  </actuator>
  <keyframe>
    <key name="stand" qpos="0 0 0.234 1 0 0 0  0.15 -0.3 0.15 -0.3" ctrl="0.15 -0.3 0.15 -0.3"/>
  </keyframe>
</mujoco>
"""
STAND = [0.15, -0.3, 0.15, -0.3]
TRUNK_Z = 0.234


@pytest.fixture
def floater(tmp_path):
    path = tmp_path / "floater.xml"
    path.write_text(FLOATER)
    return path


def robot(sim, path, **extra):
    return RobotHandle(sim, "duck", {"mjcf": str(path), **extra})


def free_qpos(sim, name="duck/root"):
    adr = sim.model.joint(name).qposadr[0]
    return sim.data.qpos[adr : adr + 7].copy()


def test_a_free_root_loads_and_homes_from_its_keyframe(sim, floater):
    robot(sim, floater)
    assert sim.model.joint("duck/root").type == 0  # mjJNT_FREE
    assert free_qpos(sim) == pytest.approx([0, 0, TRUNK_Z, 1, 0, 0, 0])


def test_a_floating_robot_has_weight(sim, floater):
    # load_mjcf gravity-compensates every body (for arms); a floating robot must not hover
    robot(sim, floater)
    trunk = sim.model.body("duck/trunk").id
    assert sim.model.body_gravcomp[trunk] == 0.0
    assert sim.model.body_gravcomp[sim.model.body("duck/l_shin").id] == 0.0


def test_a_free_root_homes_relative_to_the_robots_frame(sim, floater):
    # the keyframe's trunk pose is in the file's world; the robot's frame moves it
    robot(sim, floater, position=[1.0, 0.5, 0.0], orientation_wxyz=[0.7071068, 0, 0, 0.7071068])
    q = free_qpos(sim)
    assert q[:3] == pytest.approx([1.0, 0.5, TRUNK_Z], abs=1e-6)
    assert q[3:] == pytest.approx([0.7071068, 0, 0, 0.7071068], abs=1e-6)


@pytest.mark.parametrize(
    "edit, message",
    [
        # a free joint below the root
        (('<body name="l_thigh" pos="0 0.03 -0.03">', '<body name="l_thigh" pos="0 0.03 -0.03"><freejoint name="loose"/>'),
         "free joint on 'l_thigh', which isn't a root body"),
        # an unnamed free joint, and an unnamed trunk
        (('<freejoint name="root"/>', "<freejoint/>"), "free joint and its body need names"),
        (('<body name="trunk" pos', '<body pos'), "free joint and its body need names"),
        # a second floating root
        (("</worldbody>", '<body name="other"><freejoint/><geom size="0.01"/></body></worldbody>'),
         "more than one floating root"),
    ],
)
def test_free_joint_errors(sim, floater, edit, message):
    floater.write_text(FLOATER.replace(*edit, 1))
    with pytest.raises(ValueError, match=message):
        robot(sim, floater)


def test_a_floating_robot_cannot_ride_a_base(sim, floater):
    with pytest.raises(ValueError, match="floating robot .* must be framed on the world"):
        robot(sim, floater, parent="rover")


def leg(side, **extra):
    # chain base is the trunk itself, so the arm's frame relative to the robot is the identity
    return {"robot": "duck", "parent": "duck", "joints": [f"{side}_hip", f"{side}_knee"],
            "flange_site": f"{side}_foot", "home_joints": STAND[:2], **extra}


def test_limbs_bind_to_a_floating_trunk(sim, floater):
    robot(sim, floater)
    left = ArmHandle(sim, "left-leg", leg("l"))
    ArmHandle(sim, "right-leg", leg("r"))
    assert left.joint_names() == ("l_hip", "l_knee")
    assert left.get_joint_positions() == pytest.approx(STAND[:2], abs=1e-9)
    # the arm frame site sits on the trunk and moves with it
    assert sim.data.site("left-leg/arm_frame").xpos == pytest.approx(sim.data.body("duck/trunk").xpos)


def test_a_limb_frame_off_the_trunk_says_what_to_paste(sim, floater):
    robot(sim, floater)
    # -?: inv(parent_end) @ base can leave -1e-17, printed as -0.0
    with pytest.raises(ValueError, match=r'set frame.translation to \{"x": -?0.0, "y": -?0.0, "z": -?0.0\}'):
        ArmHandle(sim, "left-leg", leg("l", position=[0, 0, 0.1]))


POLLEN_SOLREF = (-5.0e4, -2.0e2)
POLLEN_SOLIMP = (0.99, 0.9999, 0.001, 0.5, 2.0)


def pollen_rewrite(spec, bam_model):
    """microduck_rl@0daab16 scripts/infer_policy.py load_mujoco_with_bam, the spec part."""
    force_limit = bam_model.actuator.vin * bam_model.kt.value / bam_model.R.value
    for act in spec.actuators:
        if act.target.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        joint = spec.joint(act.target)
        joint.damping = np.zeros(np.shape(joint.damping))
        joint.frictionloss = 0.0
        joint.solref_friction = POLLEN_SOLREF
        joint.solimp_friction = POLLEN_SOLIMP


def test_bam_config_defaults_and_errors():
    cfg = BamConfig.from_attrs({"model": "bam"}, "robot duck")
    assert (cfg.motor, cfg.bam_model, cfg.kp_fw, cfg.vin, cfg.vin_drop_gain, cfg.vin_min) == (
        "xl330", "m6", 200.0, 7.4, 0.1, 6.0)
    with pytest.raises(ValueError, match='actuators.model must be "bam"'):
        BamConfig.from_attrs({"model": "pd"}, "robot duck")
    with pytest.raises(ValueError, match="unknown actuators keys"):
        BamConfig.from_attrs({"model": "bam", "kp": 3}, "robot duck")
    with pytest.raises(ValueError, match="BAM has no motor"):
        BamConfig.from_attrs({"model": "bam", "motor": "nope"}, "robot duck")
    for bad in ({"vin": 0}, {"vin": float("nan")}, {"kp_fw": -1}, {"vin_drop_gain": -0.1}, {"vin_min": -1}):
        with pytest.raises(ValueError, match=next(iter(bad))):
            BamConfig.from_attrs({"model": "bam", **bad}, "robot duck")


def test_rewrite_matches_pollens_load_mujoco_with_bam():
    cfg = BamConfig.from_attrs({"model": "bam"}, "t")
    # one actuator with a ctrlrange and gear, so the gear and ctrllimited comparisons bite
    xml = FLOATER.replace('<position name="l_hip" joint="l_hip"/>',
                          '<position name="l_hip" joint="l_hip" ctrlrange="-1 1" gear="2"/>')
    assert xml != FLOATER
    ours, theirs = mujoco.MjSpec.from_string(xml), mujoco.MjSpec.from_string(xml)
    names = rewrite_spec(ours, cfg, cfg.load())
    pollen_rewrite(theirs, cfg.load())
    assert names == ["l_hip", "l_knee", "r_hip", "r_knee"]
    a, b = ours.compile(), theirs.compile()
    for field in ("actuator_gainprm", "actuator_biasprm", "actuator_gear", "actuator_forcerange",
                  "actuator_forcelimited", "actuator_ctrllimited", "actuator_dyntype",
                  "dof_damping", "dof_frictionloss", "dof_solref", "dof_solimp"):
        assert getattr(a, field) == pytest.approx(getattr(b, field)), field


BAM = {"model": "bam"}


@pytest.fixture
def sim5(sim):
    """The sim fixture's world at BAM's 5 ms step."""
    sim.configure_world(WorldConfig.from_attrs({"physics_dt": 0.005}))
    return sim


def test_bam_drives_the_robot_and_holds_a_stand(sim5, floater):
    duck = robot(sim5, floater, actuators=BAM)
    assert sim5.model.actuator("duck/l_hip").gainprm[0] == 1.0  # a motor now
    sim5.advance(0.005)
    # binding must not move anything (MujocoController's mj_setConst resets the data it's given)
    assert duck.drive.targets(sim5) == pytest.approx(STAND, abs=1e-9)
    assert free_qpos(sim5)[2] == pytest.approx(TRUNK_Z, abs=0.002)
    sim5.advance(1.0)
    adr = [sim5.model.joint(f"duck/{n}").qposadr[0] for n in ("l_hip", "l_knee", "r_hip", "r_knee")]
    # held near the keyframe; the knees sag ~0.03 rad under the trunk's weight (BAM's P-only law)
    assert sim5.data.qpos[adr] == pytest.approx(STAND, abs=0.06)
    assert duck.io(None, None)["gravity"][2] < -0.99  # still upright
    assert np.any(sim5.data.ctrl != 0.0)  # BAM is writing torques


def test_bam_rollout_equals_the_controller_run_directly(sim5, floater):
    """Same model, same start: mujoco-sim's BAM callback and a bare MujocoController agree."""
    duck = robot(sim5, floater, actuators=BAM)
    sim5.advance(0.005)  # bind
    ref_data = mujoco.MjData(sim5.model)
    # construct first (its mj_setConst resets ref_data), then copy the live state in exactly
    ref = MujocoController(duck.drive.cfg.load(), list(duck.drive.names), sim5.model, ref_data,
                           vin_drop_gain=0.1, vin_min=6.0)
    mujoco.mj_copyData(ref_data, sim5.model, sim5.data)
    ref.q_target[:] = duck.drive.targets(sim5)
    ref._prev_motor_torque[:] = duck.drive._ctrl._prev_motor_torque
    ref.last_ts = duck.drive._ctrl.last_ts
    steps = 200
    sim5.advance(steps * 0.005)
    for _ in range(steps):
        ref.update()
        mujoco.mj_step(sim5.model, ref_data)
    assert sim5.data.qpos == pytest.approx(ref_data.qpos, abs=1e-9)


def test_adding_a_limb_rebinds_bam_to_the_new_model(sim5, floater):
    duck = robot(sim5, floater, actuators=BAM)
    sim5.advance(0.01)
    first = duck.drive._ctrl
    # a rebuild; a limb arm on a BAM robot is Task 6's, so its init pose isn't checked here
    sim5.set_contribution("extra", Contribution(build=lambda spec: None))
    sim5.advance(0.005)
    assert duck.drive._ctrl is not first and duck.drive._generation == sim5.generation
    dof = sim5.model.joint("duck/l_hip").dofadr[0]
    assert sim5.model.dof_armature[dof] == pytest.approx(duck.drive.cfg.load().actuator.get_extra_inertia())
    # q_target restarts at the current pose (the knees have sagged ~2e-4 rad in 10 ms)
    assert duck.drive.targets(sim5) == pytest.approx(STAND, abs=1e-3)


def test_bam_warns_off_its_fitted_step(sim, floater, caplog):
    robot(sim, floater, actuators=BAM)  # the fixture's world runs at 2 ms
    assert "fitted at 0.005" in caplog.text


def test_world_reset_rebinds_bam_to_the_reset_pose(sim5, floater):
    duck = robot(sim5, floater, actuators=BAM)
    sim5.advance(0.01)
    duck.drive.set_targets(sim5, np.array([0.5, -0.9, 0.5, -0.9]))
    sim5.advance(0.05)
    sim5.reset()
    sim5.advance(0.005)
    assert duck.drive.targets(sim5) == pytest.approx(STAND, abs=1e-6)
    assert duck.drive._generation == sim5.generation


def test_limbs_of_a_bam_robot_are_read_only(sim5, floater):
    robot(sim5, floater, actuators=BAM)
    left = ArmHandle(sim5, "left-leg", leg("l"))
    assert "arm:left-leg" not in sim5.step_callbacks  # its position hold would fight BAM for ctrl
    for call in (lambda: left.set_joint_targets([0, 0]),
                 lambda: left.start_trajectory([JointTrajectoryPoint(0.1, (0.0, 0.0))]),
                 lambda: left.extend_trajectory([JointTrajectoryPoint(0.1, (0.0, 0.0))]),
                 lambda: left.extend_trajectory([]),
                 left.stop_trajectory):
        with pytest.raises(RuntimeError, match="read-only: robot duck's io drives it"):
            call()
    left.stop()  # Stop (session end, e-stop) must not fail
    sim5.advance(0.1)
    assert left.get_commanded_positions() == pytest.approx(STAND[:2], abs=1e-6)  # BAM's q_target
    assert left.get_joint_positions() == pytest.approx(STAND[:2], abs=0.05)


def test_a_limb_outside_bams_joints_is_an_error(sim5, floater):
    robot(sim5, floater, actuators={"model": "bam", "joints": "^r_"})
    with pytest.raises(ValueError, match="isn't driven by robot duck's BAM actuators"):
        ArmHandle(sim5, "left-leg", leg("l"))


def test_arm_init_pose_rebinds_bam(sim5, floater):
    duck = robot(sim5, floater, actuators=BAM)
    sim5.advance(0.01)
    ArmHandle(sim5, "left-leg", leg("l", home_joints=[0.2, -0.5]))  # rebuild; the arm's init sets qpos
    sim5.advance(0.005)
    assert duck.drive._generation == sim5.generation
    assert duck.drive.targets(sim5)[:2] == pytest.approx([0.2, -0.5], abs=1e-6)


def test_a_robot_on_a_running_sim_never_steps_unbound(running_sim, floater):
    # the contribution and the BAM callback must land in one sim-thread task: a step in
    # between would feed the keyframe's joint angles to the motors as torques
    running_sim.configure_world(WorldConfig.from_attrs({"physics_dt": 0.005, "real_time": False}))
    duck = robot(running_sim, floater, actuators=BAM)
    time.sleep(0.5)
    z, targets = running_sim.run(lambda: (free_qpos(running_sim)[2], duck.drive.targets(running_sim)))
    assert z == pytest.approx(TRUNK_Z, abs=0.005)
    assert targets == pytest.approx(STAND, abs=1e-6)


def test_io_reads_one_instant_and_applies_targets(sim5, floater):
    duck = robot(sim5, floater, actuators=BAM)
    snap = duck.io(None, None)  # binds on the spot: no step has run yet
    assert snap["joints"] == ["l_hip", "l_knee", "r_hip", "r_knee"]
    assert snap["q"] == pytest.approx(STAND, abs=1e-9)
    assert snap["gravity"] == pytest.approx([0, 0, -1], abs=1e-9)
    assert snap["trunk_pos"] == pytest.approx([0, 0, TRUNK_Z]) and snap["generation"] == sim5.generation
    t0 = snap["t"]
    crouch = [0.25, -0.5, 0.25, -0.5]
    duck.io(crouch, True)
    assert duck.drive.targets(sim5) == pytest.approx(crouch, abs=1e-12)  # applied exactly
    sim5.advance(1.0)
    snap = duck.io(None, None)
    assert snap["t"] == pytest.approx(t0 + 1.0, abs=0.006)
    # BAM's P-only firmware law plus stiction leaves ~0.1 rad of steady-state error under load
    # (probe: [0.223, -0.616] at 0.5, 1 and 2 s), so assert the move, not the exact angle
    assert snap["q"][1] < STAND[1] - 0.1 and snap["q"][3] < STAND[3] - 0.1  # knees went deeper
    assert snap["gravity"][2] < -0.95  # still upright (probe: -0.970)


def test_io_after_a_rebuild_acts_on_the_new_model(sim5, floater):
    duck = robot(sim5, floater, actuators=BAM)
    sim5.advance(0.01)
    ArmHandle(sim5, "left-leg", leg("l", home_joints=[0.2, -0.5]))  # rebuild; no step after it
    snap = duck.io([0.2, -0.5, 0.15, -0.3], None)  # binds to the new model itself
    assert duck.drive._generation == sim5.generation == snap["generation"]
    assert snap["q"][:2] == pytest.approx([0.2, -0.5], abs=1e-6)


def test_io_gravity_and_gyro_are_in_the_trunk_frame(sim5, floater):
    duck = robot(sim5, floater, actuators=BAM)
    adr = sim5.model.joint("duck/root").qposadr[0]
    dof = sim5.model.joint("duck/root").dofadr[0]
    sim5.data.qpos[adr + 3 : adr + 7] = [0.7071068, 0.7071068, 0, 0]  # rolled +90 deg about x
    sim5.data.qvel[dof + 3 : dof + 6] = [0.1, 0.2, 0.3]
    mujoco.mj_forward(sim5.model, sim5.data)
    snap = duck.io(None, None)
    assert snap["gravity"] == pytest.approx([0, -1, 0], abs=1e-6)  # world -z seen from the rolled trunk
    assert snap["gyro"] == pytest.approx([0.1, 0.2, 0.3])  # free-joint angular qvel: body frame


def test_torque_off_collapses_slowly_and_torque_on_resets_targets(sim5, floater):
    duck = robot(sim5, floater, actuators=BAM)
    sim5.advance(0.2)
    standing = duck.io(None, None)["trunk_pos"][2]
    duck.io(None, False)
    sim5.advance(0.05)
    assert np.all(sim5.data.ctrl[[sim5.model.actuator(f"duck/{n}").id for n in ("l_hip", "l_knee")]] == 0.0)
    sim5.advance(0.45)
    sagged = duck.io(None, None)
    assert sagged["trunk_pos"][2] < standing - 0.005  # it collapsed
    duck.io(None, True)
    assert duck.drive.targets(sim5) == pytest.approx(sagged["q"], abs=1e-9)  # not flung at STAND


def test_io_validates(sim5, floater):
    duck = robot(sim5, floater, actuators=BAM)
    with pytest.raises(ValueError, match="expected 4 targets"):
        duck.io([0.0], None)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="must be finite"):
            duck.io([bad, 0.0, 0.0, 0.0], None)
    plain = RobotHandle(sim5, "pd", {"mjcf": str(floater), "position": [2, 0, 0]})
    with pytest.raises(ValueError, match="io needs actuators"):
        plain.io(None, None)


def _component(name, attrs):
    return ComponentConfig(name=name, attributes=dict_to_struct(attrs))


def _world_and_duck(floater):
    world = MujocoWorld.new(_component("sim-world", {"physics_dt": 0.005}), {})
    duck = MujocoRobot.new(_component("duck", {"world": "sim-world", "mjcf": str(floater), "actuators": BAM}), {})
    return world, duck


def test_io_do_command(running_sim, floater):
    async def run():
        world, duck = _world_and_duck(floater)
        first = await duck.do_command({"io": {"targets": None}})
        second = await duck.do_command({"io": {"targets": STAND, "torque": True}})
        third = await duck.do_command({"io": {"targets": STAND}})  # no torque: unchanged
        with pytest.raises(ValueError, match="unknown command"):
            await duck.do_command({"nope": 1})
        await duck.close()
        await world.close()
        return first, second, third

    first, second, third = asyncio.run(run())
    assert first["joints"] == ["l_hip", "l_knee", "r_hip", "r_knee"]
    assert second["t"] >= first["t"] and len(second["gravity"]) == 3
    assert second["torque"] is True and third["torque"] is True


def test_validate_rejects_bad_actuators(floater):
    with pytest.raises(ValueError, match="unknown actuators keys"):
        MujocoRobot.validate_config(
            _component("duck", {"world": "sim-world", "mjcf": str(floater), "actuators": {"model": "bam", "kp": 1}})
        )


def test_io_do_command_over_the_wire(running_sim, floater):
    # viam-sdk decodes a protobuf null as int 0; requests go through that same round trip
    def wire(command):
        return struct_to_dict(dict_to_struct(command))

    async def run():
        world, duck = _world_and_duck(floater)
        off = await duck.do_command(wire({"io": {"targets": STAND, "torque": False}}))
        nulled = await duck.do_command(wire({"io": {"torque": None, "targets": None}}))
        omitted = await duck.do_command(wire({"io": {"targets": STAND}}))
        whole = await duck.do_command(wire({"io": None}))
        await duck.do_command(wire({"io": {"torque": True}}))
        still_on = await duck.do_command(wire({"io": {"torque": None}}))
        errors = []
        for bad in ({"io": [1, 2]}, {"io": {"targets": 1.0}}, {"io": {"torque": "false"}}):
            with pytest.raises(ValueError) as err:
                await duck.do_command(wire(bad))
            errors.append(str(err.value))
        await duck.close()
        await world.close()
        return off, nulled, omitted, whole, still_on, errors

    off, nulled, omitted, whole, still_on, errors = asyncio.run(run())
    assert [r["torque"] for r in (off, nulled, omitted, whole)] == [False] * 4
    assert still_on["torque"] is True  # a wire null (int 0) must not turn torque off
    assert errors == [
        "robot duck: io must be an object",
        "robot duck: io.targets must be a list of numbers or null",
        "robot duck: io.torque must be true, false or null",
    ]


def test_a_fixed_robot_with_actuators_has_no_gravity_compensation(sim5, tmp_path):
    fixed = FLOATER.replace('<freejoint name="root"/>', "").replace("0 0 0.234 1 0 0 0  ", "")
    path = tmp_path / "fixed.xml"
    path.write_text(fixed)
    robot(sim5, path, actuators=BAM)
    assert not sim5.model.body_gravcomp.any()


def test_a_limb_reads_the_live_robot_after_a_rebuild(sim5, floater):
    old = robot(sim5, floater, actuators=BAM)
    left = ArmHandle(sim5, "left-leg", leg("l"))
    old.close()
    new = robot(sim5, floater, actuators=BAM)
    assert new.drive is not old.drive
    new.io([0.2, -0.5, 0.15, -0.3], None)
    assert left.get_commanded_positions() == pytest.approx([0.2, -0.5], abs=1e-9)
    new.close()
    with pytest.raises(RuntimeError, match="not in the scene"):
        left.get_commanded_positions()


def test_a_failing_bam_update_keeps_the_callback(sim5, floater, monkeypatch):
    duck = robot(sim5, floater, actuators=BAM)
    sim5.advance(0.005)
    calls = []
    real = duck.drive._ctrl.update

    def update():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        real()

    monkeypatch.setattr(duck.drive._ctrl, "update", update)
    sim5.advance(0.005)
    assert "robot-bam:duck" in sim5.step_callbacks
    sim5.advance(0.01)
    assert len(calls) == 3  # it kept stepping after the failure


def test_two_bam_robots_have_independent_targets(sim5, floater):
    one = RobotHandle(sim5, "one", {"mjcf": str(floater), "actuators": BAM, "position": [0, 0, 0]})
    two = RobotHandle(sim5, "two", {"mjcf": str(floater), "actuators": BAM, "position": [2, 0, 0]})
    one.io([0.25, -0.5, 0.25, -0.5], None)
    assert one.drive.targets(sim5) == pytest.approx([0.25, -0.5, 0.25, -0.5])
    assert two.drive.targets(sim5) == pytest.approx(STAND)
    sim5.advance(0.1)
    assert two.io(None, None)["q"] == pytest.approx(STAND, abs=0.05)


def test_a_robot_closed_and_recreated_under_one_name_stands(sim5, floater):
    robot(sim5, floater, actuators=BAM).close()
    assert "robot-bam:duck" not in sim5.step_callbacks
    duck = robot(sim5, floater, actuators=BAM)
    assert "robot-bam:duck" in sim5.step_callbacks
    sim5.advance(0.2)
    assert free_qpos(sim5)[2] == pytest.approx(TRUNK_Z, abs=0.005)
    assert duck.drive.targets(sim5) == pytest.approx(STAND, abs=1e-6)
