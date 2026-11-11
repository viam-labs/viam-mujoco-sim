import itertools
import warnings

import mujoco
import numpy as np
import pytest
import test_robot_file
from test_robot_file import limb, robot

from mujoco_module import robots
from mujoco_module.handles import ArmHandle, GripperHandle
from mujoco_module.robot_file import RobotHandle
from mujoco_module.sensors import ForceTorqueHandle

torso = test_robot_file.torso  # the fixture


def test_a_gripper_finds_only_its_arms_actuator(sim, torso):
    robot(sim, torso)
    ArmHandle(sim, "left-arm", limb(torso, "left"))
    ArmHandle(sim, "right-arm", limb(torso, "right"))
    gripper = GripperHandle(sim, "left-gripper", "left-arm", {})  # the right arm's actuators don't count
    gripper.close_jaws()
    sim.advance(1.0)
    assert gripper.jaw() == pytest.approx(0.04, abs=2e-3)
    with pytest.raises(ValueError, match="has no gripper actuator"):
        GripperHandle(sim, "right-gripper", "right-arm", {})
    assert sim.robot_claims["bot"]["actuator:left/grip"] == "left-gripper"
    with pytest.raises(ValueError, match="'right/elbow' doesn't drive a body below the arm's last joint"):
        GripperHandle(sim, "stray", "left-arm", {"actuator": "right/elbow"})


def test_a_gripper_while_its_robot_is_away(sim, torso):
    bot = robot(sim, torso)
    ArmHandle(sim, "left-arm", limb(torso, "left"))
    bot.close()
    gripper = GripperHandle(sim, "left-gripper", "left-arm", {})  # no KeyError: claims don't need the robot
    with pytest.raises(RuntimeError, match="robot 'bot' is not in the scene"):
        gripper.jaw()
    robot(sim, torso)
    assert gripper.jaw() == pytest.approx(0.02, abs=1e-3)  # back, its jaw at the robot's keyframe


def test_wrist_force_torque_on_a_bound_arm(sim, torso):
    robot(sim, torso)
    ArmHandle(sim, "left-arm", limb(torso, "left"))
    GripperHandle(sim, "left-gripper", "left-arm", {})
    ft = ForceTorqueHandle(sim, "ft", "left-arm", {})
    sim.advance(0.5)
    rotation = sim.data.site("left-arm/viam_frame").xmat.reshape(3, 3)
    assert np.array([ft.readings()["force"][k] for k in "xyz"]) == pytest.approx(
        rotation.T @ [0, 0, -0.1 * 9.81], abs=2e-3
    )  # the finger's 0.1 kg


# ---- ALOHA ------------------------------------------------------------------

ALOHA_LEFT = {"robot": "aloha", "asset": "aloha_left", "parent": "aloha", "position": [-0.469, -0.019, 0.02]}
ALOHA_RIGHT = {"robot": "aloha", "asset": "aloha_right", "parent": "aloha", "position": [0.469, -0.019, 0.02],
               "orientation_wxyz": [0, 0, 0, 1]}


def test_aloha(sim):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no attach warning about its contact options
        aloha = RobotHandle(sim, "aloha", {"asset": "aloha"})
        left = ArmHandle(sim, "left-arm", ALOHA_LEFT)
        right = ArmHandle(sim, "right-arm", ALOHA_RIGHT)
        grippers = [GripperHandle(sim, f"{side}-gripper", f"{side}-arm", {"asset": "aloha"}) for side in ("left", "right")]
    assert sim.model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    sim.advance(0.5)
    neutral = [0, -0.96, 1.16, 0, -0.3, 0]
    assert left.get_joint_positions() == pytest.approx(neutral, abs=1e-3)
    assert right.get_joint_positions() == pytest.approx(neutral, abs=1e-3)
    for gripper in grippers:
        assert gripper.jaw() == pytest.approx(0.037, abs=1e-3)  # open
        gripper.close_jaws()
    sim.advance(1.0)
    assert all(gripper.jaw() < 0.01 for gripper in grippers)
    sim.reset()  # the jaw keeps the keyframe with its coupled finger: no snap
    for side in ("left", "right"):
        fingers = [sim.data.joint(f"aloha/{side}/{f}_finger").qpos[0] for f in ("left", "right")]
        assert fingers[0] == pytest.approx(fingers[1], abs=1e-6)
    assert aloha.geometries() == []  # its only fixed geoms are the arms' base_links
    assert list(left.kinematics.joints) == ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]


# SVA link -> the ALOHA bodies its geometry stands for: the body carrying the
# link's parent joint (RDK places a link's geometry in its parent's frame),
# and for the gripper everything from gripper_link down
VX300S_BODIES = {
    "base": ["base_link"], "shoulder_link": ["shoulder_link"], "upper_arm_link": ["upper_arm_link"],
    "upper_forearm_link": ["upper_forearm_link"], "lower_forearm_link": ["lower_forearm_link"],
    "wrist_link": ["wrist_link"], "gripper": "gripper_link",
}


def _bodies(model, side, spec):
    if isinstance(spec, str):
        return robots.subtree(model, model.body(f"{side}/{spec}").id)
    return [model.body(f"{side}/{name}").id for name in spec]


def _aabb(model, data, bodies, to_arm):
    """The bodies' geoms' bounding box (centre, size), in the arm frame (to_arm: world -> arm)."""
    points = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] in bodies:
            centre, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
            rotation, position = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g]
            for signs in itertools.product((-1, 1), repeat=3):
                world = rotation @ (centre + half * np.array(signs)) + position
                points.append((to_arm @ np.append(world, 1.0))[:3])
    points = np.array(points)
    return (points.min(0) + points.max(0)) / 2, points.max(0) - points.min(0)


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_vx300s_kinematics_shapes_sit_on_the_arm(side):
    """Each SVA link geometry, placed as RDK places it (the link's parent
    frame times the geometry's pose), sits on the MuJoCo bodies it stands
    for. At zeros the fixed base's and the gripper's boxes (any rotation
    included) span their bounding boxes, and each moving link's capsule runs
    from its parent joint to its own, along the link and on its body (joint to
    joint, as the Lite6's do, so neighbouring capsules overlap at the start
    and RDK ignores those pairs instead of finding boxes' corners colliding
    mid-move). At a mixed pose each shape's whole pose is unchanged relative
    to its body (it moves with the right link). Every link has one."""
    spec = robots.load_mjcf(robots.resolve_path("trossen_aloha/aloha.xml", "aloha"), None)
    model = spec.compile()
    data = mujoco.MjData(model)
    kin = robots.parse_kinematics(robots.load_kinematics("trossen_aloha/vx300s_kinematics.json", "k"), False, "k")
    assert {f.name for f in kin.frames.values() if f.geometry} == set(VX300S_BODIES)  # a dropped shape would vanish
    base = model.body(f"{side}/base_link").id
    joints = [model.joint(f"{side}/{name}").id for name in kin.joints]

    def place(q):
        """(the shapes' poses in their bodies' frames, and their poses in the arm frame), at q."""
        data.qpos[model.jnt_qposadr[joints]] = q
        mujoco.mj_kinematics(model, data)
        to_arm = np.linalg.inv(robots.transform(data.xpos[base], data.xquat[base]))
        viam = kin.forward(q)
        in_body, in_arm = {}, {}
        for frame in kin.frames.values():
            if frame.geometry:
                pose = viam[frame.parent] @ robots.sva_pose(frame.geometry, frame.name)
                body = _bodies(model, side, VX300S_BODIES[frame.name])[0]
                to_body = np.linalg.inv(to_arm @ robots.transform(data.xpos[body], data.xquat[body]))
                in_body[frame.name], in_arm[frame.name] = to_body @ pose, pose
        return to_arm, in_body, in_arm

    to_arm, at_zero, in_arm = place(np.zeros(6))
    anchors = {"world": np.zeros(3)}
    for name in kin.joints:
        joint = model.joint(f"{side}/{name}").id
        anchors[name] = (to_arm @ np.append(data.xanchor[joint], 1.0))[:3]
    ends_at = {f.parent: f.name for f in kin.frames.values() if f.axis is not None}  # link -> the joint it carries
    for frame in kin.frames.values():
        if not frame.geometry:
            continue
        centre, size = _aabb(model, data, _bodies(model, side, VX300S_BODIES[frame.name]), to_arm)
        pose = in_arm[frame.name]
        if "x" in frame.geometry:  # the fixed base and the gripper: their bounding box
            dims = np.array([frame.geometry[k] for k in "xyz"]) / 1000
            assert pose[:3, 3] == pytest.approx(centre, abs=2e-3), frame.name
            assert np.abs(pose[:3, :3]) @ dims == pytest.approx(size, abs=2e-3), frame.name  # rotated, its span
        else:  # a moving link: a capsule from its parent joint to its own joint, along the link
            start, end = anchors[frame.parent], anchors[ends_at[frame.name]]
            assert pose[:3, 3] == pytest.approx((start + end) / 2, abs=1e-3), frame.name
            axis = (end - start) / np.linalg.norm(end - start)
            assert abs(float(np.dot(pose[:3, 2], axis))) == pytest.approx(1.0, abs=1e-6), frame.name
            assert frame.geometry["l"] / 1000 == pytest.approx(np.linalg.norm(end - start) + 2 * frame.geometry["r"] / 1000, abs=1e-3)
            assert np.all(np.abs(pose[:3, 3] - centre) <= size / 2), frame.name  # on its body
            # it covers its body: every collision-mesh vertex lies within the capsule
            body = _bodies(model, side, VX300S_BODIES[frame.name])[0]
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == body and model.geom_contype[g] and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh = model.geom_dataid[g]
                    verts = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
                    world = verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                    points = (np.c_[world, np.ones(len(world))] @ to_arm.T)[:, :3]
                    along = np.clip((points - start) @ (end - start) / np.dot(end - start, end - start), 0, 1)
                    gaps = np.linalg.norm(points - (start + along[:, None] * (end - start)), axis=1)
                    assert gaps.max() <= frame.geometry["r"] / 1000, frame.name
    _, at_mixed, _ = place(np.array([0.4, -0.5, 0.6, 0.5, 0.4, -0.6]))
    for name, pose in at_mixed.items():
        metres, degrees = robots.pose_error(pose, at_zero[name])
        assert metres < 2e-3 and degrees < 0.5, name
