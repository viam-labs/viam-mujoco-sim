import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_module import robots
from mujoco_module.handles import ArmHandle, JointTrajectoryPoint
from mujoco_module.sim import KNOWN_ASSETS, WorldConfig
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


def test_an_mjcf_arm_without_kinematics_url_derives_them(sim, tmp_path):
    mjcf = tmp_path / "three.xml"
    mjcf.write_text(THREE_JOINT_MJCF)
    arm = ArmHandle(sim, "mine", {"mjcf": str(mjcf)})
    assert arm.joint_names() == ("a", "b", "c")
    assert not arm.kinematics_urdf
    sva = json.loads(arm.kinematics_data)
    assert [j["id"] for j in sva["joints"]] == ["a", "b", "c"]
    target = [0.5, -0.2, 0.4]
    arm.set_joint_targets(target)
    sim.advance(3.0)
    assert arm.get_joint_positions() == pytest.approx(target, abs=0.01)


@pytest.mark.parametrize("target", [[0.5, -0.3, 1.2, 0.2, 1.0, -0.4], [1.5, 0.8, 2.0, 1.0, 1.5, 0.5]])
def test_the_bundled_lite6_mjcf_reaches_its_targets_without_its_kinematics_file(sim, target):
    # fitted boxes are conservative; MuJoCo collides the arm's own meshes, so they can't jam it
    attrs = {k: v for k, v in KNOWN_ASSETS["lite6"].items() if k != "kinematics_url"}
    arm = ArmHandle(sim, "derived", attrs)
    assert len(arm.joint_names()) == 6
    arm.set_joint_targets(target)
    sim.advance(6.0)
    assert arm.get_joint_positions() == pytest.approx(target, abs=0.01)


def viam_geoms(sim, arm):
    return [g.name for g in map(sim.model.geom, range(sim.model.ngeom)) if g.name.startswith(f"{arm}/viam_")]


def test_a_derived_arm_collides_as_its_mesh_but_a_kinematics_url_adds_fitted_shapes(sim, tmp_path):
    attrs = {k: v for k, v in KNOWN_ASSETS["lite6"].items() if k != "kinematics_url"}
    ArmHandle(sim, "derived", attrs)
    assert viam_geoms(sim, "derived") == []
    ArmHandle(sim, "vendor", dict(KNOWN_ASSETS["lite6"]))
    assert viam_geoms(sim, "vendor")


def test_an_arm_files_contact_options_reach_the_scene(sim, tmp_path, recwarn):
    mjcf = tmp_path / "opt.xml"
    mjcf.write_text(THREE_JOINT_MJCF.replace("<mujoco>", '<mujoco><option impratio="10" cone="elliptic"/>', 1))
    ArmHandle(sim, "opt", {"mjcf": str(mjcf)})
    assert sim.model.opt.impratio == 10
    assert sim.model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert not [w for w in recwarn if "ttach" in str(w.message)]


def test_an_empty_kinematics_url_is_unset(sim, tmp_path):
    mjcf = tmp_path / "three.xml"
    mjcf.write_text(THREE_JOINT_MJCF)
    assert ArmHandle(sim, "mine", {"mjcf": str(mjcf), "kinematics_url": ""}).derived


def derived_lite6(**extra):
    return {**{k: v for k, v in KNOWN_ASSETS["lite6"].items() if k != "kinematics_url"}, **extra}


def test_a_derived_arm_serves_collision_hulls_and_visual_models_when_asked(sim):
    shapes = ArmHandle(sim, "shapes", derived_lite6())
    assert all("mesh_data" not in link.get("geometry", {}) for link in json.loads(shapes.kinematics_data)["links"])
    assert shapes.visual_models == {}
    meshes = ArmHandle(sim, "meshes", derived_lite6(planning_geometry="mesh", visual_models=True))
    links = json.loads(meshes.kinematics_data)["links"]
    shaped = [link["id"] for link in links if link.get("geometry")]
    assert shaped and all(link["geometry"]["type"] == "mesh" for link in links if link.get("geometry"))
    assert set(meshes.visual_models) == set(shaped)
    # each GLB (visual meshes) sits on its PLY (collision hulls): the Lite6's differ by under 1 cm
    import base64
    import io

    import trimesh

    for link in links:
        if link.get("geometry"):
            lines = base64.b64decode(link["geometry"]["mesh_data"]).decode().splitlines()
            count = int(next(line for line in lines if line.startswith("element vertex")).split()[-1])
            first = lines.index("end_header") + 1
            hull = np.array([[float(v) for v in line.split()] for line in lines[first:first + count]])
            glb = trimesh.load(io.BytesIO(meshes.visual_models[link["id"]]), file_type="glb", force="scene").bounds
            np.testing.assert_allclose(glb, [hull.min(axis=0), hull.max(axis=0)], atol=0.01, err_msg=link["id"])


@pytest.mark.parametrize("extra", [{"planning_geometry": "mesh"}, {"visual_models": True}])
def test_meshes_and_models_are_only_for_derived_kinematics(sim, extra):
    with pytest.raises(ValueError, match="only for an arm whose kinematics come from its own MJCF or URDF"):
        ArmHandle(sim, "vendor", {**KNOWN_ASSETS["lite6"], **extra})


def test_mesh_planning_geometry_cannot_be_kinematics_collision(sim):
    with pytest.raises(ValueError, match='planning_geometry "mesh" needs collision "mesh"'):
        ArmHandle(sim, "both", derived_lite6(planning_geometry="mesh", collision="kinematics"))


NO_GEOMETRY = {
    "type": "", "x": 0, "y": 0, "z": 0, "r": 0, "l": 0,
    "translation": {"x": 0, "y": 0, "z": 0}, "orientation": {"type": ""}, "label": "",
}


def real_driver_sva():
    """THREE_JOINT_SVA as a real Viam driver serves it (the UFactory xArm's): Go-style
    uppercase translation and axis keys, null orientations, an empty placeholder
    geometry on every link, and extra joint fields."""
    return {
        "name": "three",
        "links": [
            {
                "id": link["id"], "parent": link["parent"], "orientation": None,
                "translation": {k.upper(): v for k, v in link["translation"].items()},
                "geometry": dict(NO_GEOMETRY),
            }
            for link in THREE_JOINT_SVA["links"]
        ],
        "joints": [
            {
                **joint, "axis": {k.upper(): v for k, v in joint["axis"].items()},
                "min": -3.0, "max": 3.0, "max_velocity": 3.14, "max_acceleration": 10,
            }
            for joint in THREE_JOINT_SVA["joints"]
        ],
    }


def test_an_arm_accepts_the_kinematics_a_real_driver_serves(sim, three_joint_files, tmp_path):
    import asyncio

    from mujoco_module.models.arm import MujocoArm

    path = tmp_path / "real.json"
    path.write_text(json.dumps(real_driver_sva()))
    arm = ArmHandle(sim, "mine", {**three_joint_files, "kinematics_url": str(path)})
    assert arm.joint_names() == ("a", "b", "c")
    # served as the file is, for RDK to read with its own case-insensitive keys
    served = object.__new__(MujocoArm)
    served._handle = arm
    _, data, *_ = asyncio.run(served.get_kinematics())
    assert json.loads(data) == real_driver_sva()
    q = [0.1, 0.2, 0.3]
    real = robots.parse_kinematics(path.read_bytes(), urdf=False, label="real")
    plain = robots.parse_kinematics(json.dumps(THREE_JOINT_SVA).encode(), urdf=False, label="plain")
    assert real.forward(q)["tool"] == pytest.approx(plain.forward(q)["tool"])
    assert real.forward(q)["tool"][:3, 3] != pytest.approx([0, 0, 0])
    assert not any(frame.geometry for frame in real.frames.values())
    assert arm.kinematics_url == str(path)


@pytest.mark.parametrize("orientation, expected", [
    ({"type": "axis_angles", "value": {"x": 0, "y": 0, "z": 1, "th": math.pi / 2}}, [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    ({"type": "quaternion", "value": {"w": 0.7071067811865476, "x": 0, "y": 0, "z": 0.7071067811865476}}, [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    ({"type": "euler_angles", "value": {"roll": 0, "pitch": 0, "yaw": math.pi / 2}}, [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    ({"type": "ov_radians", "value": {"x": 0, "y": 0, "z": 1, "th": math.pi / 2}}, [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    (None, np.eye(3)),
    ({"type": ""}, np.eye(3)),
])
def test_sva_orientation_types(orientation, expected):
    assert robots.sva_pose({"orientation": orientation}, "x")[:3, :3] == pytest.approx(np.array(expected), abs=1e-9)


def test_unknown_sva_orientation_type_names_the_supported_ones():
    with pytest.raises(ValueError, match="'spin' is not supported; use ov_degrees"):
        robots.sva_pose({"orientation": {"type": "spin"}}, "x")


def test_every_kind_of_arm_leaves_get_geometries_to_viam_server(sim, three_joint_files):
    import asyncio

    from viam.errors import MethodNotImplementedError

    from mujoco_module.models.arm import MujocoArm

    for attrs in (dict(KNOWN_ASSETS["lite6"]), three_joint_files):  # kinematics_url: vendor SVA, custom SVA
        arm = object.__new__(MujocoArm)
        arm._handle = ArmHandle(sim, f"arm{len(attrs)}", attrs)
        assert asyncio.run(arm.get_kinematics())[1]  # served, so viam-server has geometry to pose
        with pytest.raises(MethodNotImplementedError):
            asyncio.run(arm.get_geometries())


def test_an_untyped_geometry_is_inferred_like_rdk_does(sim):
    """The xArm driver serves type "" with real sizes: x/y/z set is a box, else l a
    capsule, else r a sphere; nothing set is none."""
    geometry = robots._geometry
    assert geometry({**NO_GEOMETRY, "r": 0, "l": 0}) is None
    assert geometry({"type": "", "x": 0, "y": 0, "z": 0, "r": 1, "l": 0})["type"] == "sphere"
    capsule = geometry({"type": "", "x": 0, "y": 0, "z": 0, "r": 50, "l": 320})
    box = geometry({"type": "", "x": 80, "y": 180, "z": 250, "r": 0, "l": 0})
    assert (capsule["type"], box["type"]) == ("capsule", "box")
    assert robots._mj_shape(capsule)[:2] == ("capsule", pytest.approx((0.05, 0.11)))
    assert robots._mj_shape(box)[:2] == ("box", pytest.approx((0.04, 0.09, 0.125)))
    assert robots._mj_shape({"x": 20, "y": 20, "z": 20, "type": "box"})[0] == "box"  # our own files omit type
    assert robots._mj_shape(geometry({"x": 20, "y": 20, "z": 20}))[0] == "box"


def test_a_real_drivers_untyped_geometry_becomes_shapes(sim, three_joint_files, tmp_path):
    sva = real_driver_sva()
    sva["links"][1]["geometry"] = {**NO_GEOMETRY, "r": 50, "l": 320}
    sva["links"][2]["geometry"] = {**NO_GEOMETRY, "x": 80, "y": 180, "z": 250}
    path = tmp_path / "real.json"
    path.write_text(json.dumps(sva))
    ArmHandle(sim, "mine", {**three_joint_files, "kinematics_url": str(path), "collision": "kinematics"})
    assert viam_geoms(sim, "mine") == ["mine/viam_upper", "mine/viam_fore"]
    upper, fore = sim.model.geom("mine/viam_upper"), sim.model.geom("mine/viam_fore")
    assert (upper.type[0], fore.type[0]) == (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX)
    assert upper.size[:2] == pytest.approx([0.05, 0.11])
    assert fore.size == pytest.approx([0.04, 0.09, 0.125])
