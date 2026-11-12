"""examples/vision-pick.json and vision_pick.py: the config matches the
script, the script's point and grasp maths, and an end to end run on the
local sim with stand-ins for the motion and vision services."""

import asyncio
import importlib.util
import json
import math

import mujoco
import numpy as np
import pytest
from test_parcel_sort import EXAMPLES, MODELS, component
from viam.components.arm import JointPositions
from viam.proto.common import Geometry, PointCloud, Pose, PoseInFrame, RectangularPrism, Transform, Vector3
from viam.proto.service.vision import Detection3D
from viam.spatialmath import OrientationVector, Quaternion

from mujoco_module.handles import JointTrajectoryPoint, encode_pcd
from mujoco_module.models.camera import MujocoCamera

CELL_MODELS = {**MODELS, "viam-labs:mujoco-sim:camera": MujocoCamera}


def load_script():
    spec = importlib.util.spec_from_file_location("vision_pick", EXAMPLES / "vision_pick.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


script = load_script()


def block_cloud(cx, cy, yaw_deg, length, width, step=2.0, top=40.0, base=script.TABLE_TOP_MM):
    """A block's top face and sides as world points (mm), as a camera from
    above sees it, with some table round it."""
    a = math.radians(yaw_deg)
    u, v = np.meshgrid(np.arange(-length / 2, length / 2, step), np.arange(-width / 2, width / 2, step))
    u, v = u.ravel(), v.ravel()
    x, y = cx + u * math.cos(a) - v * math.sin(a), cy + u * math.sin(a) + v * math.cos(a)
    block = np.column_stack([x, y, np.full_like(x, base + top)])
    tx, ty = np.meshgrid(np.arange(cx - 60, cx + 60, step), np.arange(cy - 60, cy + 60, step))
    table = np.column_stack([tx.ravel(), ty.ravel(), np.full(tx.size, float(base))])
    return np.vstack([block, table])


# -- config ---------------------------------------------------------------------


def test_example_validates_and_matches_the_script():
    cfg = json.loads((EXAMPLES / "vision-pick.json").read_text())
    items = {item["name"]: item for item in cfg["components"]}
    for item in cfg["components"]:
        if item["model"] in CELL_MODELS:
            CELL_MODELS[item["model"]].validate_config(component(item))
    props = {p["name"]: p for p in items[script.WORLD]["attributes"]["props"]}
    assert sorted(props) == sorted(script.BLOCKS)
    assert {round(p["scale"][2] * 1000) for p in props.values()} == {script.BLOCK_H_MM}
    assert {n for n, p in props.items() if p["scale"][0] == p["scale"][1]} == script.SQUARES
    assert all(min(p["scale"][:2]) * 1000 <= script.MAX_GRASP_MM for p in props.values())
    for name, (x, y) in script.BINS.items():
        frame = items[f"{name}-bin"]["frame"]["translation"]
        assert (frame["x"], frame["y"]) == (x, y)
        assert items[f"{name}-bin"]["attributes"]["scale"][:2] == [1.5, 1.5]
    assert script.BIN_HALF_MM == 150 * 1.5 / 2
    # the bins stay out of the survey's view (~690 x 520 mm at 450 mm, 60 deg vertical fov, 4:3)
    half_y = script.SURVEY_MM[2] * math.tan(math.radians(30))
    assert all(abs(y) - script.BIN_HALF_MM > script.SURVEY_MM[1] + half_y for _, y in script.BINS.values())
    table = items["table"]["frame"]
    assert table["translation"]["z"] + table["geometry"]["z"] / 2 == script.TABLE_TOP_MM
    assert items[script.GRIPPER]["attributes"]["mount"] == script.ARM
    services = {s["name"]: s for s in cfg["services"]}
    assert set(script.SERVICES.values()) <= set(services)
    assert services["shape-segments"]["attributes"]["detector_name"] == "shapes"


# -- points and grasps ------------------------------------------------------------


def test_parse_pcd_reads_the_module_s_own_point_clouds():
    points = np.array([[0.1, -0.2, 0.5], [0.0, 0.0, 1.25]], dtype=np.float32)
    for colors in (None, np.array([[255, 0, 0], [0, 0, 255]])):
        assert script.parse_pcd(encode_pcd(points, colors)) == pytest.approx(points)


@pytest.mark.parametrize("heading", [0, 30, 90, -45, 180])
def test_down_points_the_tool_down_with_x_at_the_heading(heading):
    rotation = rotation_of(script.down(500, 0, 450, heading).pose)
    assert rotation[:, 2] == pytest.approx([0, 0, -1], abs=1e-9)  # the tool points down
    x_heading = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    assert (x_heading - heading + 180) % 360 - 180 == pytest.approx(0, abs=1e-6)


def test_block_points_drop_the_table_and_a_neighbours_corner():
    cloud = block_cloud(0, 0, 30, 80, 40)
    neighbour = block_cloud(75, 0, 0, 50, 50)
    corner = neighbour[(neighbour[:, 2] > script.TABLE_TOP_MM) & (neighbour[:, 0] < 55)]  # its near edge, inside the box
    kept = script.block_points(np.vstack([cloud, corner]))
    assert (kept[:, 2] > script.BAND_MM[0]).all()
    assert kept[:, 0].max() < 45  # none of the neighbour's points
    assert len(kept) == (cloud[:, 2] > script.TABLE_TOP_MM).sum()


@pytest.mark.parametrize("yaw", [0, 30, -60, 89])
def test_grasp_finds_a_rectangle_s_centre_yaw_and_width(yaw):
    item = script.grasp(script.block_points(block_cloud(510, -40, yaw, 80, 40)))
    assert (item["x"], item["y"]) == pytest.approx((510, -40), abs=3)
    assert script.yaw_error(item["yaw"], yaw, square=False) < 3
    assert item["width"] == pytest.approx(40, abs=4) and item["length"] == pytest.approx(80, abs=4)
    assert item["top"] == pytest.approx(script.TABLE_TOP_MM + 40)


def test_grasp_on_a_noisy_square():
    rng = np.random.default_rng(1)
    cloud = block_cloud(380, 130, 20, 50, 50)
    cloud += rng.normal(0, 1.0, cloud.shape)
    item = script.grasp(script.block_points(cloud))
    assert (item["x"], item["y"]) == pytest.approx((380, 130), abs=3)
    assert item["width"] == pytest.approx(50, abs=5) and item["length"] == pytest.approx(50, abs=5)


def test_score_matches_detections_to_blocks():
    truths = {
        "red_1": {"x": 0, "y": 0, "yaw": 10, "square": True},
        "blue_1": {"x": 200, "y": 0, "yaw": 170, "square": False},
        "red_2": {"x": 400, "y": 0, "yaw": 0, "square": False},
    }
    found = [
        {"x": 3, "y": 4, "yaw": 100, "label": "square-red"},  # a square turned 90: no error
        {"x": 200, "y": 6, "yaw": -8, "label": "rectangle-red"},  # 2 deg off across +-90; wrong colour
        {"x": 900, "y": 0, "yaw": 0, "label": ""},  # nothing there
    ]
    report = script.score(found, truths)
    assert report["seen"] == 2 and report["missed"] == ["red_2"] and report["spurious"] == 1
    first, second = report["errors"]
    assert (first["block"], first["mm"], first["yaw_deg"], first["colour_ok"]) == ("red_1", 5.0, 0.0, True)
    assert (second["block"], second["yaw_deg"], second["colour_ok"]) == ("blue_1", 2.0, False)


class ScriptedCell(script.Cell):
    """A cell whose survey and picks are scripted: each object a dict at a
    position, each pick's outcome taken from `outcomes` (True held, False
    slipped, an exception for a plan that fails)."""

    def __init__(self, objects, outcomes) -> None:
        super().__init__(None, None, None, None, None, "pointcloud")
        self.objects, self.outcomes, self.attempts = objects, outcomes, []

    async def truths(self):
        return {}

    async def survey(self):
        return [dict(o, top=script.TABLE_TOP_MM + 40.0, yaw=0.0, length=50.0, width=50.0, label="", obstacle=None) for o in self.objects]

    async def pick(self, item, others):
        name = item["name"]
        self.attempts.append(name)
        outcome = self.outcomes[name].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if outcome:
            self.objects = [o for o in self.objects if o["name"] != name]
        return outcome


def run_scripted(objects, outcomes):
    cell = ScriptedCell(objects, outcomes)
    result = asyncio.run(cell.run())
    return cell.attempts, result["picked"]


NEAR, FAR = {"name": "near", "x": 300, "y": 0}, {"name": "far", "x": 600, "y": 0}


def test_a_plan_that_fails_moves_on_to_the_next_object():
    attempts, picked = run_scripted([NEAR, FAR], {"near": [RuntimeError("no plan"), True], "far": [True]})
    assert attempts == ["near", "far", "near"] and picked == 2


def test_a_slip_surveys_again_and_two_failures_leave_an_object():
    attempts, picked = run_scripted([NEAR, FAR], {"near": [False, False], "far": [True]})
    assert attempts == ["near", "near", "far"] and picked == 1


def test_three_failures_in_a_row_stop_the_run():
    third = {"name": "third", "x": 700, "y": 0}
    attempts, picked = run_scripted(
        [NEAR, FAR, third], {"near": [RuntimeError("a")], "far": [RuntimeError("b")], "third": [RuntimeError("c")]}
    )
    assert attempts == ["near", "far", "third"] and picked == 0


def test_a_vision_service_error_names_the_service():
    class Broken:
        async def get_detections_3d(self, camera_name, **kwargs):
            raise ConnectionError("down")

    class Still:
        async def move(self, **kwargs):
            pass

        async def get_pose(self, **kwargs):
            return PoseInFrame(reference_frame="world", pose=Pose(o_z=-1))

    cell = script.Cell(None, None, None, Still(), Broken(), "shapes")
    with pytest.raises(RuntimeError, match="vision service shape-segments: down"):
        asyncio.run(cell.survey())


# -- end to end ------------------------------------------------------------------------


OFFSETS_M = {script.GRIPPER: np.array([0, 0, 0.1558]), script.CAMERA: np.array([0, 0.1, 0.03])}  # in the flange frame


def rotation_of(pose: Pose) -> np.ndarray:
    q = OrientationVector(pose.o_x, pose.o_y, pose.o_z, math.radians(pose.theta)).to_quaternion()
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.array([q.w, q.i, q.j, q.k]))
    return matrix.reshape(3, 3)


class FakeMotion:
    """The motion service, for the harness: IK on the sim's own model for the
    gripper or the wrist camera (each a fixed offset from the flange), then a
    straight joint path. No planning: the script's moves stay clear."""

    def __init__(self, sim, arm) -> None:
        self.sim, self.arm = sim, arm
        self.world_states = []  # each move's, for the obstacle checks

    def _flange(self):
        _, ids = self.arm._h()._now()
        data = self.sim.data
        return data.site_xpos[ids.end_site].copy(), data.site_xmat[ids.end_site].reshape(3, 3).copy()

    def _solve(self, position, rotation, offset):
        model = self.sim.model
        _, ids = self.arm._h()._now()
        data = mujoco.MjData(model)
        data.qpos[:] = self.sim.data.qpos
        start = data.qpos[ids.qpos].copy()
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        for _ in range(500):
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            frame = data.site_xmat[ids.end_site].reshape(3, 3)
            point = data.site_xpos[ids.end_site] + frame @ offset
            error = np.concatenate([position - point, 0.5 * sum(np.cross(frame[:, i], rotation[:, i]) for i in range(3))])
            if np.linalg.norm(error) < 1e-5:
                joints = start + np.remainder(data.qpos[ids.qpos] - start + np.pi, 2 * np.pi) - np.pi
                low, high = model.jnt_range[[model.dof_jntid[d] for d in ids.dof]].T
                return joints - 2 * np.pi * (joints > high) + 2 * np.pi * (joints < low)
            mujoco.mj_jac(model, data, jacp, jacr, point, model.site_bodyid[ids.end_site])
            jac = np.vstack([jacp[:, ids.dof], jacr[:, ids.dof]])
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
            data.qpos[ids.qpos] += step * min(1.0, 0.2 / np.abs(step).max())
        raise RuntimeError(f"no IK solution for {position}")

    async def move(self, component_name, destination, world_state=None, **kwargs):
        assert destination.reference_frame == "world"
        self.world_states.append(world_state)
        pose = destination.pose
        position = np.array([pose.x, pose.y, pose.z]) / 1000
        joints = await asyncio.to_thread(
            self.sim.run, lambda: self._solve(position, rotation_of(pose), OFFSETS_M[component_name])
        )
        handle = self.arm._h()
        start = await asyncio.to_thread(handle.get_joint_positions)
        speed = min(handle.dynamics_metadata()["max_velocity_rad_s"])
        duration = float(np.abs(joints - start).max()) / speed
        await asyncio.wrap_future(handle.start_trajectory([JointTrajectoryPoint(duration, tuple(joints))]))
        await self.arm.move_to_joint_positions(JointPositions(values=list(np.degrees(joints))))

    async def get_pose(self, component_name, destination_frame, **kwargs):
        assert destination_frame == "world"
        position, frame = await asyncio.to_thread(self.sim.run, self._flange)
        point = (position + frame @ OFFSETS_M[component_name]) * 1000
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, frame.ravel())
        ov = Quaternion(*quat).to_orientation_vector()
        pose = Pose(x=point[0], y=point[1], z=point[2], o_x=ov.o_x, o_y=ov.o_y, o_z=ov.o_z, theta=math.degrees(ov.theta))
        return PoseInFrame(reference_frame="world", pose=pose)


def camera_to_world(points_m: np.ndarray, camera: Pose) -> np.ndarray:
    """Camera-frame points (m) in the world frame (mm), as the frame system moves them."""
    return points_m * 1000 @ rotation_of(camera).T + [camera.x, camera.y, camera.z]


class FakeRobot:
    """The robot client's transform_pose, for the harness: a pose on the wrist
    camera moved to the world by the camera's pose from the stand-in motion
    service (the sim's camera)."""

    def __init__(self, motion) -> None:
        self.motion = motion

    async def transform_pose(self, query: PoseInFrame, destination: str, **kwargs) -> PoseInFrame:
        assert query.reference_frame == script.CAMERA and destination == "world"
        camera = (await self.motion.get_pose(script.CAMERA, "world")).pose
        local = query.pose
        x, y, z = camera_to_world(np.array([[local.x, local.y, local.z]]) / 1000, camera)[0]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (rotation_of(camera) @ rotation_of(local)).ravel())
        return PoseInFrame(reference_frame="world", pose=Quaternion(*quat).to_pose(x, y, z))


class FakeVision:
    """A 3D segmenter that knows where the blocks are, its detections shaped
    as RDK's detection3d.FromSegmenter makes them: for each block on the
    table, the wrist camera's points inside an axis-aligned box round it (as
    a 2D detection box would take them, table and neighbours' corners
    included), as a root transform on the camera at the points' bounding box
    (labelled with the block's colour) and a "<root>/points" child with the
    points in the root's frame."""

    def __init__(self, sim, camera, motion, labelled: bool) -> None:
        self.sim, self.camera, self.motion, self.labelled = sim, camera, motion, labelled

    async def get_detections_3d(self, camera_name, **kwargs):
        data, _ = await self.camera.get_point_cloud()
        cam_points = script.parse_pcd(data)
        pose = (await self.motion.get_pose(camera_name, "world")).pose
        world = camera_to_world(cam_points, pose)
        detections = []
        for name in script.BLOCKS:
            x, y, z = (v * 1000 for v in self.sim.get_prop_pose(name)["position"])
            if z > script.TABLE_TOP_MM + script.BLOCK_H_MM or script.in_bin(x, y):
                continue
            points = cam_points[(np.abs(world[:, 0] - x) < 55) & (np.abs(world[:, 1] - y) < 55)]
            low, high = points.min(axis=0), points.max(axis=0)
            centre = (low + high) / 2
            root = f"fake/object-{len(detections)}"
            label = f"square-{name.split('_')[0]}" if self.labelled else ""
            dims = Vector3(x=(high - low)[0] * 1000, y=(high - low)[1] * 1000, z=(high - low)[2] * 1000)
            detections.append(Detection3D(transforms=[
                Transform(
                    reference_frame=root,
                    pose_in_observer_frame=PoseInFrame(reference_frame=camera_name, pose=Pose(
                        x=centre[0] * 1000, y=centre[1] * 1000, z=centre[2] * 1000, o_z=1)),
                    physical_object=Geometry(center=Pose(o_z=1), box=RectangularPrism(dims_mm=dims), label=label),
                ),
                Transform(
                    reference_frame=f"{root}/points",
                    pose_in_observer_frame=PoseInFrame(reference_frame=root, pose=Pose(o_z=1)),
                    physical_object=Geometry(pointcloud=PointCloud(
                        point_cloud=encode_pcd((points - centre).astype(np.float32)))),
                ),
            ]))
        return detections


def make_cell(running_sim, mode):
    cfg = json.loads((EXAMPLES / "vision-pick.json").read_text())
    made = {}
    for item in cfg["components"]:
        if item["model"] in CELL_MODELS:
            if item["model"] == "viam-labs:mujoco-sim:world":
                item["attributes"]["real_time"] = False
            made[item["name"]] = CELL_MODELS[item["model"]].new(component(item), {})
    motion = FakeMotion(running_sim, made[script.ARM])
    vision = FakeVision(running_sim, made[script.CAMERA], motion, labelled=mode == "shapes")
    cell = script.Cell(FakeRobot(motion), made[script.WORLD], made[script.GRIPPER], motion, vision, mode)
    return cell, made


def test_the_survey_sees_the_blocks_and_nothing_else(running_sim):
    cell, made = make_cell(running_sim, "shapes")

    async def look():
        await made[script.WORLD].do_command({"command": "reset"})
        await cell.motion.move(
            component_name=script.CAMERA, destination=script.down(*script.SURVEY_MM, script.SURVEY_HEADING_DEG)
        )
        pose = (await cell.motion.get_pose(script.CAMERA, "world")).pose
        data, _ = await made[script.CAMERA].get_point_cloud()
        return pose, camera_to_world(script.parse_pcd(data), pose)

    pose, world = asyncio.run(look())
    # the stand-in's camera pose is the sim's camera
    cam = running_sim.run(lambda: running_sim.data.cam(script.CAMERA).xpos.copy() * 1000)
    assert (pose.x, pose.y, pose.z) == pytest.approx(tuple(cam), abs=1)
    # nothing above the blocks (no fingers), nothing below the table (no floor), no bins
    assert world[:, 2].max() < script.TABLE_TOP_MM + script.BLOCK_H_MM + 2 and world[:, 2].min() > script.TABLE_TOP_MM - 2
    assert not any(script.in_bin(x, y) for x, y, _ in world[::50])


@pytest.mark.parametrize("mode", ["shapes", "pointcloud"])
def test_the_script_sorts_every_block(running_sim, mode):
    cell, made = make_cell(running_sim, mode)

    async def run():
        await made[script.WORLD].do_command({"command": "reset"})
        result = await cell.run()
        return result, await cell.landed()

    result, where = asyncio.run(run())
    want = {name: name.split("_")[0] if mode == "shapes" else "clear" for name in script.BLOCKS}
    assert where == want
    assert result["picked"] == len(script.BLOCKS)
    first = result["reports"][0]
    assert first["seen"] == len(script.BLOCKS) and first["spurious"] == 0
    # the first pick's moves avoid the other seven detections, each pinned to the world
    # at its block, with its own box, under a unique label
    state = next(s for s in cell.motion.world_states if s is not None)
    assert len(state.transforms) == len(script.BLOCKS) - 1
    assert len({t.physical_object.label for t in state.transforms}) == len(state.transforms)
    world = next(c for c in json.loads((EXAMPLES / "vision-pick.json").read_text())["components"] if c["name"] == script.WORLD)
    blocks = [[v * 1000 for v in prop["position"]] for prop in world["attributes"]["props"]]  # where they started
    for t in state.transforms:
        assert t.pose_in_observer_frame.reference_frame == "world" and t.physical_object.HasField("box")
        at = t.pose_in_observer_frame.pose
        assert min(math.dist((at.x, at.y), b[:2]) for b in blocks) < 15
    assert max(e["mm"] for e in first["errors"]) < 5 and max(e["yaw_deg"] for e in first["errors"]) < 5
