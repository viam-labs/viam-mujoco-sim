import json
import math
from pathlib import Path

import numpy as np
import pytest

from mujoco_module import robots
from mujoco_module.handles import ArmHandle, JointTrajectoryPoint
from mujoco_module.sim import WorldConfig
from viam.spatialmath import AxisAngle
from conftest import BLOCK

KINEMATICS = Path(__file__).parents[1] / "assets" / "ufactory_lite6" / "lite6_kinematics.json"
HOME = [0.0, 0.0, 1.57, 0.0, 1.57, 0.0]


def make_arm(sim, **attrs):
    return ArmHandle(sim, "pick-arm", {"asset": "lite6", **attrs})


def test_arm_starts_at_home(sim):
    arm = make_arm(sim)
    assert arm.get_joint_positions() == pytest.approx(HOME)


@pytest.mark.parametrize("seed", range(4))
def test_flange_matches_viam_kinematics(sim, seed):
    """If this fails, the motion service plans against the wrong arm."""
    q = AxisAngle(0, 0, 1, math.radians(30)).to_quaternion()
    yaw = (q.w, q.i, q.j, q.k)
    arm = make_arm(sim, position=[0.1, 0.2, 0.05], orientation_wxyz=list(yaw))
    rng = np.random.default_rng(seed)
    joints = list(np.asarray(HOME) + rng.uniform(-0.8, 0.8, 6))
    ids = arm._now()[1]
    sim.data.qpos[ids.qpos] = joints
    import mujoco

    mujoco.mj_kinematics(sim.model, sim.data)
    (x, y, z), quat = arm.get_end_pose()
    kinematics = robots.parse_kinematics(KINEMATICS.read_bytes(), urdf=False, label="lite6")
    expected = kinematics.forward(joints)["gripper_mount"]
    assert np.allclose([x, y, z], expected[:3, 3], atol=1.5e-3)
    assert np.allclose(robots.transform(quat=quat)[:3, :3], expected[:3, :3], atol=1e-2)
    # and the base really sits at the configured frame in the world
    base = sim.data.body("pick-arm/link_base")
    assert np.allclose(base.xpos, [0.1, 0.2, 0.05])
    assert np.allclose(base.xquat, yaw)


def test_move_settles_within_velocity_limit(sim):
    arm = make_arm(sim, max_joint_deg_s=60)
    target = [0.5, -0.3, 1.2, 0.2, 1.0, -0.4]
    arm.set_joint_targets(target)
    done = arm.wait_for_joint_targets(target, math.radians(0.5))
    previous = np.asarray(arm.get_joint_positions())
    peak = 0.0
    while not done.done() and sim.data.time < 10.0:
        sim.advance(0.01)
        now = np.asarray(arm.get_joint_positions())
        peak = max(peak, float(np.max(np.abs(now - previous)) / 0.01))
        previous = now
    assert done.done()
    assert arm.get_joint_positions() == pytest.approx(target, abs=math.radians(0.5))
    assert peak <= math.radians(60) * 1.1


def test_trajectory_hits_waypoints_on_time(sim):
    arm = make_arm(sim)
    mid = [0.3, 0.0, 1.57, 0.0, 1.57, 0.0]
    end = [0.6, 0.0, 1.57, 0.0, 1.57, 0.0]
    points = [
        JointTrajectoryPoint(0.0, tuple(HOME)),
        JointTrajectoryPoint(1.0, tuple(mid)),
        JointTrajectoryPoint(2.0, tuple(end)),
    ]
    done = arm.start_trajectory(points)
    sim.advance(1.0)
    # the kp=2000/kv=200 servo trails a moving target by about 0.1 s
    assert arm.get_joint_positions()[0] == pytest.approx(0.3, abs=0.04)
    sim.advance(1.05)
    assert done.done()
    sim.advance(0.5)
    assert arm.get_joint_positions() == pytest.approx(end, abs=0.01)


def test_stop_trajectory_holds_measured(sim):
    arm = make_arm(sim)
    done = arm.start_trajectory(
        [JointTrajectoryPoint(0.0, tuple(HOME)), JointTrajectoryPoint(2.0, (1.0, 0, 1.57, 0, 1.57, 0))]
    )
    sim.advance(0.5)
    arm.stop_trajectory()
    assert done.cancelled()
    held = arm.get_joint_positions()
    sim.advance(0.5)
    assert arm.get_joint_positions() == pytest.approx(held, abs=0.01)


def test_dynamics_metadata(sim):
    meta = make_arm(sim, max_joint_deg_s=30, max_joint_deg_s2=60).dynamics_metadata()
    assert meta["max_velocity_rad_s"] == pytest.approx([math.radians(30)] * 6)
    assert meta["max_acceleration_rad_s2"] == pytest.approx([math.radians(60)] * 6)


def test_rebuild_keeps_arm_state_and_time(sim):
    arm = make_arm(sim)
    target = [0.4, -0.2, 1.3, 0.1, 1.2, 0.2]
    arm.set_joint_targets(target)
    sim.advance(2.0)
    before_q = arm.get_joint_positions()
    before_t = sim.data.time
    distractor = {"name": "distractor", "size": 0.03, "position": [0.2, 0.2, 0.015]}
    sim.configure_world(WorldConfig.from_attrs({"props": [BLOCK, distractor]}))
    assert sim.data.time == before_t
    assert arm.get_joint_positions() == pytest.approx(before_q)
    sim.advance(0.5)
    assert arm.get_joint_positions() == pytest.approx(target, abs=0.01)


def test_bad_trajectory_is_rejected_up_front(sim):
    arm = make_arm(sim)
    with pytest.raises(ValueError):
        arm.start_trajectory([JointTrajectoryPoint(0.0, (0.0,) * 5)])
    with pytest.raises(ValueError):
        arm.start_trajectory([JointTrajectoryPoint(1.0, tuple(HOME)), JointTrajectoryPoint(0.5, tuple(HOME))])
    assert "arm:pick-arm" in sim.step_callbacks


def test_batch_starting_after_zero_interpolates_from_command(sim):
    arm = make_arm(sim)
    sim.advance(0.5)
    done = arm.start_trajectory([JointTrajectoryPoint(0.5, (0.5, 0, 1.57, 0, 1.57, 0))])
    sim.step_once()
    ids = arm._now()[1]
    assert sim.data.ctrl[ids.ctrl][0] == pytest.approx(0.0, abs=0.01)  # no backwards jump
    sim.advance(0.6)
    assert done.done()


def test_close_fails_inflight_moves(sim):
    arm = make_arm(sim)
    done = arm.start_trajectory([JointTrajectoryPoint(0.0, tuple(HOME)), JointTrajectoryPoint(5.0, tuple(HOME))])
    wait = arm.wait_for_joint_targets([1.0] * 6, 1e-6)
    arm.close()
    for future in (done, wait):
        assert future.done() and not future.cancelled()
        assert isinstance(future.exception(), RuntimeError)
        assert "closed" in str(future.exception())
    assert "arm:pick-arm" not in sim.step_callbacks


def test_arm_reaches_pose_where_link_hulls_overlap(sim):
    """Regression: Menagerie's link4/link6 convex hulls overlap here, which jammed joint 5."""
    arm = make_arm(sim, max_joint_deg_s=30)
    target = [-0.32019518688087617, -0.13123530856204854, 1.595436485882882,
              -0.001766502550737528, 1.7313311471953172, -0.3204752787998066]
    arm.set_joint_targets(target)
    sim.advance(8.0)
    assert arm.get_joint_positions() == pytest.approx(target, abs=math.radians(0.5))


def test_gripper_still_collides_with_arm_base(sim):
    """The self-collision exclusion must stay narrow: Viam's planner can't see the gripper."""
    import mujoco

    arm = make_arm(sim)
    ids = arm._now()[1]
    # found by random sweep; within both the MJCF joint ranges and Viam's joint limits, so
    # the planner can command it. gripper_body is ~98 mm into link_base
    sim.data.qpos[ids.qpos] = [-6.082, 1.28, 0.422, -3.222, -0.427, -2.86]
    mujoco.mj_forward(sim.model, sim.data)
    gripper = sim.model.body("pick-arm/gripper_body").id
    base = sim.model.body("pick-arm/link_base").id
    pairs = [{int(sim.model.geom_bodyid[c.geom1]), int(sim.model.geom_bodyid[c.geom2])}
             for c in sim.data.contact[: sim.data.ncon] if c.dist < -0.005]
    assert {gripper, base} in pairs

THREE_JOINT_MJCF = """<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.3">
      <geom size="0.03" pos="0 0 -0.06"/>
      <body name="l1"><joint name="a" axis="0 0 1"/><geom size="0.02"/>
        <body name="l2" pos="0 0 0.1"><joint name="b" axis="0 1 0"/><geom size="0.02"/>
          <body name="l3" pos="0.1 0 0"><joint name="c" axis="0 1 0"/><geom size="0.02"/>
            <site name="attachment_site"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="act_c" joint="c" kp="50" kv="5"/>
    <position name="act_a" joint="a" kp="50" kv="5"/>
    <position name="act_b" joint="b" kp="50" kv="5"/>
  </actuator>
  <keyframe><key name="home" qpos="0.1 0.2 0.3" ctrl="0.3 0.1 0.2"/></keyframe>
</mujoco>"""

THREE_JOINT_SVA = {
    "name": "three",
    "links": [
        {"id": "base", "parent": "world", "translation": {"x": 0, "y": 0, "z": 300}},
        {"id": "upper", "parent": "a", "translation": {"x": 0, "y": 0, "z": 100}},
        {"id": "fore", "parent": "b", "translation": {"x": 100, "y": 0, "z": 0}},
        {"id": "tool", "parent": "c", "translation": {"x": 0, "y": 0, "z": 0}},
    ],
    "joints": [
        {"id": "a", "type": "revolute", "parent": "base", "axis": {"x": 0, "y": 0, "z": 1}},
        {"id": "b", "type": "revolute", "parent": "upper", "axis": {"x": 0, "y": 1, "z": 0}},
        {"id": "c", "type": "revolute", "parent": "fore", "axis": {"x": 0, "y": 1, "z": 0}},
    ],
}


@pytest.fixture
def three_joint_files(tmp_path):
    mjcf, kinematics = tmp_path / "three.xml", tmp_path / "three.json"
    mjcf.write_text(THREE_JOINT_MJCF)
    kinematics.write_text(json.dumps(THREE_JOINT_SVA))
    return {"mjcf": str(mjcf), "kinematics_url": str(kinematics)}


def test_custom_arm_from_mjcf_and_kinematics(sim, three_joint_files):
    arm = ArmHandle(sim, "mine", three_joint_files)
    assert arm.joint_names() == ("a", "b", "c")
    # home is the keyframe's ctrl, mapped to joints by actuator target, not by name
    assert arm.get_joint_positions() == pytest.approx([0.1, 0.2, 0.3])
    target = [0.5, -0.2, 0.4]
    arm.set_joint_targets(target)
    sim.advance(3.0)
    assert arm.get_joint_positions() == pytest.approx(target, abs=0.01)


def test_end_pose_is_in_the_arm_frame_with_base_offset(sim, three_joint_files, tmp_path):
    # this arm's kinematics put its base 100 mm higher and turned 90 degrees;
    # base_offset says where the MJCF world sits in the arm frame to match
    sva = json.loads(json.dumps(THREE_JOINT_SVA))
    sva["links"][0]["translation"]["z"] = 400
    sva["links"][0]["orientation"] = {"type": "ov_degrees", "value": {"x": 0, "y": 0, "z": 1, "th": 90}}
    kinematics_file = tmp_path / "turned.json"
    kinematics_file.write_text(json.dumps(sva))
    offset = {"translation": {"x": 0, "y": 0, "z": 100}, "orientation": {"o_z": 1, "theta": 90}}
    arm = ArmHandle(sim, "mine", {
        **three_joint_files, "kinematics_url": str(kinematics_file),
        "base_offset": offset, "position": [1.0, 0, 0],
    })
    kinematics = robots.parse_kinematics(json.dumps(sva).encode(), urdf=False, label="turned")
    expected = kinematics.forward([0.1, 0.2, 0.3])["tool"]
    position, quat = arm.get_end_pose()
    assert position == pytest.approx(expected[:3, 3], abs=1e-6)
    assert robots.transform(quat=quat)[:3, :3] == pytest.approx(expected[:3, :3], abs=1e-6)
    # the MJCF world sits at the arm's frame composed with base_offset
    assert sim.data.body("mine/base").xpos == pytest.approx([1.0, 0, 0.4])


def test_flange_site_euler_orientation_is_honoured(sim, three_joint_files):
    # MjSpec keeps euler in site.alt, so its quat reads identity; the compiled site has it (90 deg; this MJCF is in radians)
    Path(three_joint_files["mjcf"]).write_text(
        THREE_JOINT_MJCF.replace('<site name="attachment_site"/>', '<site name="attachment_site" euler="0 0 1.5707963267948966"/>')
    )
    # the frame check needs the -90 deg flange_offset to undo the site's turn, so the end pose is the tool's
    arm = ArmHandle(sim, "mine", {**three_joint_files, "flange_offset": {"orientation": {"theta": -90}}})
    kinematics = robots.parse_kinematics(
        Path(three_joint_files["kinematics_url"]).read_bytes(), urdf=False, label="three"
    )
    expected = kinematics.forward([0.1, 0.2, 0.3])["tool"]
    position, quat = arm.get_end_pose()
    assert position == pytest.approx(expected[:3, 3], abs=1e-6)
    assert robots.transform(quat=quat)[:3, :3] == pytest.approx(expected[:3, :3], abs=1e-6)


def test_missing_flange_site_names_the_attribute(sim, three_joint_files):
    with pytest.raises(ValueError, match="has no site 'tcp'; set flange_site"):
        ArmHandle(sim, "mine", {**three_joint_files, "flange_site": "tcp"})


def suggestion(message, key):
    """The pose an error message suggests for `key`, parsed."""
    return robots.parse_pose(json.loads(message.split(f"try {key}: ")[1]), key)


def test_frame_check_suggests_the_lite6_flange_offset(sim):
    with pytest.raises(ValueError, match="disagree by .* 180.0 deg") as error:
        make_arm(sim, flange_offset={})  # identity, not the bundled half turn
    fix = suggestion(str(error.value), "flange_offset")
    assert fix[:3, :3] == pytest.approx(np.diag([-1.0, -1.0, 1.0]), abs=1e-2)
    assert np.linalg.norm(fix[:3, 3]) < 0.002


def test_frame_check_suggests_a_base_offset(sim, three_joint_files):
    with pytest.raises(ValueError, match="try base_offset") as error:
        ArmHandle(sim, "mine", {**three_joint_files, "base_offset": {"translation": {"x": 50}}})
    assert suggestion(str(error.value), "base_offset") == pytest.approx(np.eye(4), abs=1e-6)


def test_frame_check_without_a_rigid_fix_has_no_suggestion(sim, three_joint_files, tmp_path):
    sva = json.loads(json.dumps(THREE_JOINT_SVA))
    sva["links"][2]["translation"]["x"] = 200  # the forearm is twice as long
    kinematics_file = tmp_path / "long.json"
    kinematics_file.write_text(json.dumps(sva))
    with pytest.raises(ValueError, match="disagree by") as error:
        ArmHandle(sim, "mine", {**three_joint_files, "kinematics_url": str(kinematics_file)})
    assert "try" not in str(error.value)
    assert "arm:mine" not in sim.step_callbacks


def test_kinematics_shapes_sit_where_viam_puts_them(sim):
    import mujoco

    make_arm(sim)
    kinematics = robots.parse_kinematics(KINEMATICS.read_bytes(), urdf=False, label="lite6")
    viam = kinematics.forward(HOME)
    checked = 0
    for frame in kinematics.frames.values():
        if not frame.geometry:
            continue
        expected = viam[frame.parent] @ robots.sva_pose(frame.geometry, frame.name)
        geom = sim.data.geom(f"pick-arm/viam_{frame.name}")
        assert np.linalg.norm(geom.xpos - expected[:3, 3]) < 0.005, frame.name
        if sim.model.geom_type[geom.id] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            assert abs(geom.xmat.reshape(3, 3)[:, 2] @ expected[:3, 2]) > 0.999, frame.name
        checked += 1
    assert checked == 8


def test_kinematics_shapes_replace_link_meshes_but_not_the_base_or_gripper(sim):
    make_arm(sim)
    model = sim.model
    for link in ("link1", "link2", "link3", "link4", "link5"):
        assert model.geom(f"pick-arm/{link}_c").contype == 0, link
    # link_base and link6 only have Viam's 1 mm placeholder spheres
    for geom in ("link_base_c", "link6_c", "gripper_lite_body_c", "gripper_left_finger"):
        assert model.geom(f"pick-arm/{geom}").contype == 1, geom
    assert model.geom("pick-arm/viam_base_top").contype == 1


def test_arm_is_not_in_self_contact_at_home(sim):
    import mujoco

    make_arm(sim)
    mujoco.mj_forward(sim.model, sim.data)
    root = sim.model.body("pick-arm/link_base").id
    arm_pairs = [
        c for c in sim.data.contact[: sim.data.ncon]
        if sim.model.body_rootid[sim.model.geom_bodyid[c.geom1]] == root
        and sim.model.body_rootid[sim.model.geom_bodyid[c.geom2]] == root
    ]
    assert arm_pairs == []


def test_mesh_collision_leaves_the_mjcf_as_is(sim):
    make_arm(sim, collision="mesh")
    names = [sim.model.geom(i).name for i in range(sim.model.ngeom)]
    assert not [name for name in names if "viam_" in name]
    assert sim.model.geom("pick-arm/link1_c").contype == 1


def test_kinematics_collision_needs_named_bodies(sim, tmp_path, three_joint_files):
    mjcf, kinematics = tmp_path / "unnamed.xml", tmp_path / "unnamed.json"
    mjcf.write_text(THREE_JOINT_MJCF.replace('<body name="l1">', "<body>"))
    sva = {**THREE_JOINT_SVA, "links": [dict(link) for link in THREE_JOINT_SVA["links"]]}
    sva["links"][1]["geometry"] = {"r": 20, "l": 100}
    kinematics.write_text(json.dumps(sva))
    with pytest.raises(ValueError, match=r"arm body #\d+ has no name"):
        ArmHandle(sim, "other", {"mjcf": str(mjcf), "kinematics_url": str(kinematics), "collision": "kinematics"})


def test_kinematics_collision_needs_geometries(sim, three_joint_files):
    ArmHandle(sim, "mine", three_joint_files)  # no geometries: mesh by default
    with pytest.raises(ValueError, match='collision "kinematics" needs an SVA kinematics file'):
        ArmHandle(sim, "other", {**three_joint_files, "collision": "kinematics"})


def test_mesh_dir_is_only_for_a_urdf_arm(sim):
    with pytest.raises(ValueError, match="mesh_dir is for a .urdf mjcf"):
        make_arm(sim, mesh_dir="meshes")
