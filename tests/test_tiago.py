"""TIAGo++ (pal_tiago_dual): its upper body a robot file riding the
pal_tiago_base base, the torso an arm on the robot, the arms and the head arms
on the torso."""

import importlib.util
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mujoco_module import bases, robots
from mujoco_module.handles import ArmHandle, BaseHandle, CameraHandle, GripperHandle
from mujoco_module.robot_file import RobotHandle
from mujoco_module.shapes import GEOM_TYPES
from mujoco_module.sim import ROBOT_LIMB_ASSETS, WorldConfig
from test_base import finish, frame_pose, settle

EXAMPLES = Path(__file__).parents[1] / "examples"
LIMBS = {"torso": "tiago_torso", "left-arm": "tiago_arm_left", "right-arm": "tiago_arm_right", "head": "tiago_head"}
ARM_HOME = [-0.12, 1.42, 1.42, 2.21, -0.11, -0.95, 1.86]
HOME = {"torso": [0.175], "left-arm": ARM_HOME, "right-arm": ARM_HOME, "head": [0.0, 0.0]}
BLOCK = {"name": "block", "type": "cube", "size": 0.04, "position": [2.0, 2.0, 0.02], "mass": 0.05,
         "static_friction": 1.5, "dynamic_friction": 1.5}


def tiago(sim, props=()):
    sim.configure_world(WorldConfig.from_attrs({"props": list(props)}))
    base = BaseHandle(sim, "tiago-base", {"asset": "pal_tiago_base"})
    robot = RobotHandle(sim, "tiago", {"asset": "tiago", "parent": "tiago-base"})  # on the base: z 0
    arms = {
        name: ArmHandle(sim, name, {"robot": "tiago", "asset": asset, "parent": "tiago" if name == "torso" else "torso"})
        for name, asset in LIMBS.items()
    }
    grippers = {side: GripperHandle(sim, f"{side}-gripper", f"{side}-arm", {"asset": "tiago"}) for side in ("left", "right")}
    return base, robot, arms, grippers


@pytest.fixture
def rig(sim):
    return tiago(sim)


def test_tiago_loads_at_home_and_holds_it(sim, rig):
    _, robot, arms, _ = rig
    assert robot.geometries() == []  # its root is the torso's chain base: the torso serves the column
    assert [len(arm.kinematics.joints) for arm in arms.values()] == [1, 7, 7, 2]
    sim.advance(1.0)
    for name, arm in arms.items():
        assert arm.get_joint_positions() == pytest.approx(HOME[name], abs=1e-3), name  # no drift
    assert sim.data.ncon == 0  # nothing touches: not the arms, the torso, nor the robot and its base


def test_no_contacts_with_the_torso_down_and_the_arms_tucked(sim, rig):
    _, _, arms, _ = rig
    arms["torso"].set_joint_targets([0.0])
    sim.advance(2.5)
    assert arms["torso"].get_joint_positions() == pytest.approx([0.0], abs=1e-3)
    assert sim.data.ncon == 0


def test_the_wrist_bends_to_its_limits_without_its_hulls_holding_it_off():
    # found on the machine: at 77 deg the forearm's and wrist's convex hulls
    # met across the wrist joint and held the arm 2.5 deg short of a plan
    model = mujoco.MjModel.from_xml_path(str(Path(robots.resolve_path("pal_tiago_dual/tiago_upper.xml", "tiago"))))
    data = mujoco.MjData(model)
    for side in ("left", "right"):
        wrist = model.joint(f"arm_{side}_6_joint")
        for angle in wrist.range:
            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
            data.qpos[wrist.qposadr[0]] = angle
            mujoco.mj_forward(model, data)
            bodies = {model.body(model.geom_bodyid[g]).name for c in data.contact[: data.ncon] for g in (c.geom1, c.geom2)}
            assert not {f"arm_{side}_5_link", f"arm_{side}_7_link"} <= bodies, (side, angle)


def test_the_loaded_base_drives_within_its_move_timeouts(sim, rig):
    base, _, _, _ = rig
    finish(sim, base.move_straight(1.0, 0.3), limit_s=1.0 / 0.3 + base.move_margin_s).result()
    finish(sim, base.spin(math.radians(90), math.radians(45)), limit_s=2.0 + base.move_margin_s).result()
    settle(sim)
    x, y, yaw = frame_pose(sim, "tiago-base")
    assert (x, y) == pytest.approx((0.0, 1.0), abs=0.01)
    assert yaw == pytest.approx(math.radians(90), abs=math.radians(1))
    # the robot rides it at its frame (a pure z of 0)
    assert sim.data.site("tiago/viam_frame").xpos == pytest.approx(sim.data.site("tiago-base/viam_frame").xpos, abs=1e-9)


def test_the_coupled_fingers_open_and_close_together(sim, rig):
    _, _, _, grippers = rig

    def fingers(side):
        return [sim.data.joint(f"tiago/gripper_{side}_{finger}_finger_joint").qpos[0] for finger in ("left", "right")]

    sim.advance(1.0)
    for side, gripper in grippers.items():
        assert fingers(side) == pytest.approx([0.045, 0.045], abs=1e-3)  # open
        gripper.close_jaws()
    sim.advance(1.0)
    for side in grippers:
        assert fingers(side) == pytest.approx([0.0, 0.0], abs=1e-3)


def test_the_gripper_holds_a_block(sim):
    _, _, arms, grippers = tiago(sim, [BLOCK])
    sim.advance(0.5)
    site = sim.data.site("tiago/gripper_left_grasp")
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, site.xmat)
    gravity = sim.model.opt.gravity.copy()
    sim.model.opt.gravity[:] = 0  # the block floats between the open fingers while they close
    sim.set_prop_pose("block", site.xpos.tolist(), quat.tolist())
    grippers["left"].close_jaws()
    sim.advance(0.5)
    sim.model.opt.gravity[:] = gravity
    sim.advance(1.0)
    assert grippers["left"].firm_grasp()
    assert sim.data.body("block").xpos == pytest.approx(site.xpos, abs=0.01)
    arms["torso"].set_joint_targets([0.3])  # the torso lifts it
    sim.advance(2.0)
    assert grippers["left"].firm_grasp()
    assert sim.data.body("block").xpos[2] == pytest.approx(site.xpos[2], abs=0.01)


def test_the_head_camera_renders(sim, rig):
    CameraHandle(sim, "head-cam", {"parent": "head"})  # on the head's end effector: its camera site
    assert sim.data.cam("head-cam").xpos == pytest.approx(sim.data.site("tiago/head_camera").xpos, abs=1e-9)
    rgb = CameraHandle(sim, "head-cam-2", {"parent": "head", "width": 64, "height": 48}).get_rgb()
    assert rgb.shape == (48, 64, 3) and rgb.std() > 0


@pytest.mark.parametrize("name", ["torso", "head"])
def test_the_torso_and_head_reach_their_kinematics_limits(sim, rig, name):
    _, _, arms, _ = rig
    arm = arms[name]
    joints = json.loads(robots.load_kinematics(ROBOT_LIMB_ASSETS[LIMBS[name]]["kinematics_url"], name))["joints"]
    scale = np.array([1000.0 if joint["type"] == "prismatic" else math.degrees(1) for joint in joints])
    tolerance = np.where(arm.slides, 0.001, math.radians(0.5))
    for end in ("min", "max"):
        target = np.array([joint[end] for joint in joints]) / scale
        arm.set_joint_targets(target)
        finish(sim, arm.wait_for_joint_targets(target, tolerance))


# ---- the kinematics files ---------------------------------------------------


def _model():
    return robots.load_mjcf(robots.resolve_path("pal_tiago_dual/tiago_upper.xml", "tiago")).compile()


def _limb(model, asset):
    """(kinematics, joint ids, chain base body)."""
    preset = ROBOT_LIMB_ASSETS[asset]
    kin = robots.parse_kinematics(robots.load_kinematics(preset["kinematics_url"], asset), False, asset)
    joints = [model.joint(name).id for name in preset["joints"]]
    return kin, joints, int(model.body_parentid[model.jnt_bodyid[joints[0]]])


def _points(model, data, body):
    """The colliding geoms' points (mesh vertices, else box corners) on `body`
    and the bodies below it that no limb joint moves (a gripper's fingers), in the world."""
    limb_joints = {j for asset in LIMBS.values() for j in _limb(model, asset)[1]}
    moved = {int(model.jnt_bodyid[j]) for j in limb_joints}
    bodies, stack = [body], [body]
    while stack:
        parent = stack.pop()
        for child in range(model.nbody):
            if model.body_parentid[child] == parent and child != parent and child not in moved:
                bodies.append(child)
                stack.append(child)
    found = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] not in bodies or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            local = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
        else:
            centre, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
            local = centre + half * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        found.append(local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
    return np.vstack(found)


@pytest.mark.parametrize("asset", LIMBS.values())
def test_the_kinematics_shapes_sit_on_their_bodies(asset):
    """Each SVA link geometry, placed as RDK places it (the link's parent
    frame times the geometry's pose, from the arm frame: its chain base), is
    on the body that carries it. At zeros (the fingers open) a capsule or
    sphere covers its body's collision points and a box lies within them (the
    column and the torso shell are cut down to clear the base and the arms);
    at a mixed pose with the torso raised each shape's pose is unchanged
    relative to its body."""
    model = _model()
    data = mujoco.MjData(model)
    for joint in range(model.njnt):
        if "finger" in model.joint(joint).name:
            data.qpos[model.jnt_qposadr[joint]] = model.jnt_range[joint, 1]
    kin, joints, base = _limb(model, asset)
    torso = model.joint("torso_lift_joint").qposadr[0]

    def place(q, lift):
        data.qpos[torso] = lift
        data.qpos[model.jnt_qposadr[joints]] = q
        mujoco.mj_kinematics(model, data)
        arm_frame = robots.transform(data.xpos[base], data.xquat[base])
        poses = kin.forward(q)
        out = {}
        for frame in kin.frames.values():
            if frame.geometry:
                rider = frame.parent
                while rider != "world" and kin.frames[rider].axis is None:
                    rider = kin.frames[rider].parent
                body = base if rider == "world" else int(model.jnt_bodyid[joints[kin.joints.index(rider)]])
                pose = arm_frame @ poses[frame.parent] @ robots.sva_pose(frame.geometry, frame.name)
                out[frame.name] = (body, pose, np.linalg.inv(robots.transform(data.xpos[body], data.xquat[body])) @ pose)
        return out

    at_zero = place(np.zeros(len(joints)), 0.0)
    assert len(at_zero) == {"tiago_torso": 2, "tiago_head": 1}.get(asset, 7)  # every arm link has one
    for name, (body, pose, _) in at_zero.items():
        geometry = kin.frames[name].geometry
        world = _points(model, data, body)
        local = (np.c_[world, np.ones(len(world))] @ np.linalg.inv(pose).T)[:, :3]  # in the shape's frame
        if "x" in geometry:
            half = np.array([geometry[k] for k in "xyz"]) / 2000
            assert np.all(-half >= local.min(axis=0) - 1e-3) and np.all(half <= local.max(axis=0) + 1e-3), name
        else:
            radius = geometry["r"] / 1000
            reach = max(geometry.get("l", 0) / 2000 - radius, 0)
            along = np.clip(local[:, 2], -reach, reach)
            gaps = np.linalg.norm(local - np.c_[np.zeros((len(local), 2)), along], axis=1)
            assert gaps.max() <= radius + 1e-6, name
    lows, highs = model.jnt_range[joints].T
    mixed = np.clip(0.5 * (-1.0) ** np.arange(len(joints)), lows, highs)
    for name, (_, _, in_body) in place(mixed, 0.2 if asset != "tiago_torso" else mixed[0]).items():
        metres, degrees = robots.pose_error(in_body, at_zero[name][2])
        assert metres < 1e-6 and degrees < 1e-4, name


def test_the_kinematics_shapes_of_different_limbs_and_the_base_dont_overlap_at_home():
    """RDK exempts shapes touching at the start from collision checks for the
    whole plan: no shape of one limb (or the base's box) touches another's at home."""
    spec = robots.load_mjcf(robots.resolve_path("pal_tiago_dual/tiago_upper.xml", "tiago"))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    for geom in spec.geoms:
        geom.contype = geom.conaffinity = 0
    for exclude in list(spec.excludes):
        spec.delete(exclude)
    for asset in LIMBS.values():
        kin, joints, base = _limb(model, asset)
        arm_frame = robots.transform(data.xpos[base], data.xquat[base])
        for shape in robots.kinematics_shapes(model, kin, joints, base, np.linalg.inv(arm_frame)):
            spec.body(shape.body).add_geom(
                name=f"{asset}:{shape.link}", type=GEOM_TYPES[shape.kind], size=[*shape.size, 0, 0][:3],
                pos=list(shape.pos), quat=list(shape.quat),
            )
    base_spec = robots.load_file(robots.resolve_path("pal_tiago_dual/tiago_base.xml", "base"), "base")
    bases.take_chassis(base_spec, None, "base")
    # the box the base's GetGeometries serves, in its frame: the robot's
    centre, size = bases.infer(base_spec.compile(), "base").box
    spec.worldbody.add_geom(name="base:box", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size / 2), pos=list(centre))
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_FILTERPARENT  # the torso's shapes and the arms' are parent and child
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    pairs = {tuple(sorted(model.geom(g).name for g in (c.geom1, c.geom2))) for c in data.contact[: data.ncon]}
    assert pairs  # neighbouring shapes of one limb do touch
    assert not [pair for pair in pairs if pair[0].split(":")[0] != pair[1].split(":")[0]]


# ---- the example ------------------------------------------------------------


def _script():
    spec = importlib.util.spec_from_file_location("tiago_pick", EXAMPLES / "tiago_pick.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def test_the_example_validates_and_matches_its_script():
    from viam.proto.app.robot import Orientation

    from mujoco_module.models.arm import MujocoArm
    from mujoco_module.models.base import MujocoBase
    from mujoco_module.models.gripper import MujocoGripper
    from mujoco_module.models.robot import MujocoRobot
    from test_robot_models import config

    cfg = json.loads((EXAMPLES / "tiago-pick.json").read_text())
    components = {c["name"]: c for c in cfg["components"]}
    models = {"robot": MujocoRobot, "arm": MujocoArm, "gripper": MujocoGripper, "base": MujocoBase}
    deps = {}
    for item in cfg["components"]:
        cls = models.get(item["model"].rsplit(":", 1)[1])
        if cls is None:
            continue
        frame = item.get("frame", {"parent": "world"})
        orientation = None
        if "orientation" in frame:
            v = frame["orientation"]["value"]
            orientation = Orientation(vector_degrees=Orientation.OrientationVectorDegrees(x=v["x"], y=v["y"], z=v["z"], theta=v["th"]))
        deps[item["name"]] = cls.validate_config(config(
            item["name"], item["attributes"], parent=frame["parent"], translation=frame.get("translation"), orientation=orientation,
        ))[0]
    assert deps == {
        "tiago-base": ["sim-world"],
        "tiago": ["sim-world", "tiago-base"],  # rides the base
        "torso": ["sim-world", "tiago"],
        "left-arm": ["sim-world", "torso"],  # on the torso
        "right-arm": ["sim-world", "torso"],
        "head": ["sim-world", "torso"],
        "left-gripper": ["sim-world", "left-arm"],
        "right-gripper": ["sim-world", "right-arm"],
    }

    script = _script()
    for name, (center, size) in script.TABLES.items():
        frame = components[name]["frame"]
        t, g = frame["translation"], frame["geometry"]
        assert ((t["x"], t["y"], t["z"]), (g["x"], g["y"], g["z"])) == (center, size)
        assert center[2] * 2 == size[2]  # standing on the floor
        nearest = [max(abs(c) - s / 2, 0) for c, s in zip(center[:2], size[:2])]
        assert math.hypot(*nearest) > 2000  # out of the arms' reach from TIAGo's configured pose
    assert script.TABLES["table"][1][2] == script.TABLE_TOP_MM and script.TABLES["shelf"][1][2] == script.SHELF_TOP_MM
    (block,) = components["sim-world"]["attributes"]["props"]
    bx, by, bz = (v * 1000 for v in block["position"])
    assert script.on_top("table", bx, by, bz, script.TABLE_TOP_MM)
    assert block["size"] * 1000 == pytest.approx(script.BLOCK_MM)
    # the drives stop the base's front a few cm short of each table, the block in the left arm's reach
    (_, ty, _), (_, t_depth, _) = script.TABLES["table"]
    assert 0 < (ty - t_depth / 2) - (script.TO_TABLE_MM + script.BASE_FRONT_MM) < 60
    assert script.to_base(bx, by, (0, script.TO_TABLE_MM, 0)) == pytest.approx(script.REACH_MM)
    stop = (-script.TO_SHELF_MM, script.TO_TABLE_MM - script.BACK_UP_MM, math.pi / 2)
    (sx, _, _), (s_depth, _, _) = script.TABLES["shelf"]
    assert 0 < (stop[0] - script.BASE_FRONT_MM) - (sx + s_depth / 2) < 60
    assert script.to_base(*script.PLACE_MM, stop) == pytest.approx(script.REACH_MM)
    assert script.on_top("shelf", *script.PLACE_MM, script.SHELF_TOP_MM + script.BLOCK_MM / 2, script.SHELF_TOP_MM)


def test_at_the_shelf_stop_only_the_left_arm_comes_near_the_shelf():
    """The right arm stays at home all run, its gripper ~400 mm ahead at the
    shelf's height: at the shelf stop no kinematics shape or collision geom
    but the left arm's (carrying the block) is near the shelf, at any torso
    height (RDK rejects a plan whose start state touches an obstacle)."""
    script = _script()
    stop = (-script.TO_SHELF_MM, script.TO_TABLE_MM - script.BACK_UP_MM, math.pi / 2)
    (cx, cy, cz), (sx, sy, sz) = script.TABLES["shelf"]
    x, y = script.to_base(cx, cy, stop)
    spec = robots.load_mjcf(robots.resolve_path("pal_tiago_dual/tiago_upper.xml", "tiago"))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    for asset in ("tiago_torso", "tiago_arm_right", "tiago_head"):
        kin, joints, base = _limb(model, asset)
        arm_frame = robots.transform(data.xpos[base], data.xquat[base])
        for shape in robots.kinematics_shapes(model, kin, joints, base, np.linalg.inv(arm_frame)):
            spec.body(shape.body).add_geom(type=GEOM_TYPES[shape.kind], size=[*shape.size, 0, 0][:3],
                                           pos=list(shape.pos), quat=list(shape.quat))
    # in the base frame (tiago_upper's world), turned a quarter turn: the shelf's x and y swap
    spec.worldbody.add_geom(name="shelf", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sy / 2000, sx / 2000, sz / 2000],
                            pos=[x / 1000, y / 1000, cz / 1000])
    model = spec.compile()
    data = mujoco.MjData(model)
    shelf = model.geom("shelf").id
    near = [g for g in range(model.ngeom) if g != shelf and (model.geom_contype[g] or model.geom_conaffinity[g])
            and not model.body(model.geom_bodyid[g]).name.startswith(("arm_left", "gripper_left"))]
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    for lift in (0.0, 0.175, 0.35):
        data.qpos[model.joint("torso_lift_joint").qposadr[0]] = lift
        mujoco.mj_forward(model, data)
        assert min(mujoco.mj_geomDistance(model, data, g, shelf, 1.0, None) for g in near) > 0.05, lift
