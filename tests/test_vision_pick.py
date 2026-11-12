"""examples/vision-pick.json and vision_pick.py: the config matches the
script, the script's pose maths and scoring, and an end to end run on the
local sim with stand-ins for the motion service and the grasp tracker."""

import asyncio
import importlib.util
import json
import math

import mujoco
import numpy as np
import pytest
from test_parcel_sort import EXAMPLES, MODELS, component
from viam.components.arm import JointPositions
from viam.proto.common import Pose, PoseInFrame
from viam.spatialmath import EulerAngles, OrientationVector, Quaternion

from mujoco_module.handles import JointTrajectoryPoint
from mujoco_module.models.camera import MujocoCamera

CELL_MODELS = {**MODELS, "viam-labs:mujoco-sim:camera": MujocoCamera}


def load_script():
    spec = importlib.util.spec_from_file_location("vision_pick", EXAMPLES / "vision_pick.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


script = load_script()


def down(x: float, y: float, z: float, heading_deg: float) -> PoseInFrame:
    """A world pose pointing straight down (+z along world -z) with its x
    axis turned heading_deg from world +x, counterclockwise seen from above:
    a half turn about x (roll 180), then the turn about the vertical (yaw)."""
    ov = EulerAngles(roll=math.pi, pitch=0, yaw=math.radians(heading_deg)).to_quaternion().to_orientation_vector()
    pose = Pose(x=x, y=y, z=z, o_x=ov.o_x, o_y=ov.o_y, o_z=ov.o_z, theta=math.degrees(ov.theta))
    return PoseInFrame(reference_frame="world", pose=pose)


def parse_pcd(data: bytes) -> np.ndarray:
    """The x, y, z (metres) of a binary PCD, as Viam's vision services return it."""
    header, rest = {}, data
    while True:
        line, rest = rest.split(b"\n", 1)
        key, _, value = line.decode().partition(" ")
        header[key] = value.split()
        if key == "DATA":
            break
    if header["DATA"] != ["binary"]:
        raise ValueError(f"expected a binary PCD, got DATA {header['DATA']}")
    kinds = {("F", "4"): "<f4", ("F", "8"): "<f8", ("U", "4"): "<u4", ("I", "4"): "<i4", ("U", "1"): "u1"}
    dtype = np.dtype([(name, kinds[(kind, size)]) for name, kind, size in zip(header["FIELDS"], header["TYPE"], header["SIZE"])])
    rows = np.frombuffer(rest, dtype=dtype, count=int(header["POINTS"][0]))
    return np.column_stack([rows["x"], rows["y"], rows["z"]]).astype(float)


# -- config ---------------------------------------------------------------------


@pytest.mark.parametrize("name", ["vision-pick.json", "vision-pick-suction.json"])
def test_example_validates_and_matches_the_script(name):
    cfg = json.loads((EXAMPLES / name).read_text())
    items = {item["name"]: item for item in cfg["components"]}
    for item in cfg["components"]:
        if item["model"] in CELL_MODELS:
            CELL_MODELS[item["model"]].validate_config(component(item))
    props = {p["name"]: p for p in items[script.WORLD]["attributes"]["props"]}
    assert sorted(props) == sorted(script.BLOCKS)
    assert {round(p["scale"][2] * 1000) for p in props.values()} == {script.BLOCK_H_MM}
    assert {n for n, p in props.items() if p["scale"][0] == p["scale"][1]} == script.SQUARES
    for colour, (x, y) in script.BINS.items():
        frame = items[f"{colour}-bin"]["frame"]["translation"]
        assert (frame["x"], frame["y"]) == (x, y)
        assert items[f"{colour}-bin"]["attributes"]["scale"][:2] == [1.5, 1.5]
    assert script.BIN_HALF_MM == 150 * 1.5 / 2
    # the bins stay out of the survey's view (~690 x 520 mm at 450 mm, 60 deg vertical fov, 4:3)
    half_y = script.SURVEY_MM[2] * math.tan(math.radians(30))
    assert all(abs(y) - script.BIN_HALF_MM > script.SURVEY_MM[1] + half_y for _, y in script.BINS.values())
    table = items["table"]["frame"]
    assert table["translation"]["z"] + table["geometry"]["z"] / 2 == script.TABLE_TOP_MM
    assert items[script.GRIPPER]["attributes"]["mount"] == script.ARM
    services = {s["name"]: s for s in cfg["services"]}
    trackers = {c["name"]: c for c in cfg["components"] if c["model"] == "viam-labs:grasp-pose:top-down"}
    assert set(trackers) == set(script.TRACKERS.values())
    for mode, tracker in script.TRACKERS.items():
        attrs = trackers[tracker]["attributes"]
        assert attrs["vision"] == script.SERVICES[mode] and attrs["vision"] in services
        assert attrs["camera"] == script.CAMERA and attrs["support_z_mm"] == script.TABLE_TOP_MM
        suction = items[script.GRIPPER]["attributes"]["asset"] == "robotiq_epick"
        assert attrs["gripper"]["type"] == ("suction" if suction else "parallel")
    assert services["shape-segments"]["attributes"]["detector_name"] == "shapes"


# -- poses and scoring ------------------------------------------------------------


@pytest.mark.parametrize("heading", [0, 30, 90, -45, 180])
def test_down_points_the_tool_down_with_x_at_the_heading(heading):
    rotation = rotation_of(down(500, 0, 450, heading).pose)
    assert rotation[:, 2] == pytest.approx([0, 0, -1], abs=1e-9)  # the tool points down
    x_heading = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    assert (x_heading - heading + 180) % 360 - 180 == pytest.approx(0, abs=1e-6)


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


class Hand:
    """A gripper stand-in that counts its opens and says whether it holds something."""

    def __init__(self, holding=False, log=None) -> None:
        self.holding, self.opens, self.log = holding, 0, log if log is not None else []

    async def open(self):
        self.opens += 1
        self.log.append("open")

    async def grab(self):
        pass

    async def is_holding_something(self):
        return type("Held", (), {"is_holding_something": self.holding})()


class ScriptedCell(script.Cell):
    """A cell whose survey and picks are scripted: each object a dict at a
    position, each pick's outcome taken from `outcomes` (True held, False
    slipped, an exception for a plan that fails)."""

    def __init__(self, objects, outcomes) -> None:
        super().__init__(None, None, Hand(), None, None, "pointcloud")
        self.objects, self.outcomes, self.attempts = objects, outcomes, []

    async def truths(self):
        return {}

    async def survey(self):
        return [dict(o, yaw=0.0, label="", box=None, grasps=[grasp_at(o["x"])]) for o in self.objects]

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


def test_a_failed_pick_opens_a_gripper_that_holds_something():
    cell = ScriptedCell([NEAR], {"near": [RuntimeError("no plan after the grab")] * 2})
    cell.gripper = Hand(holding=True)
    asyncio.run(cell.run())
    assert cell.gripper.opens == 2  # once per failed pick


def test_a_failed_pick_lets_go_before_it_lifts():
    log = []

    class Lift:
        async def move(self, **kwargs):
            log.append("lift")

    cell = ScriptedCell([NEAR], {"near": [RuntimeError("no plan after the grab")] * 2})
    cell.gripper, cell.motion = Hand(holding=True, log=log), Lift()
    asyncio.run(cell.run())
    assert log == ["open", "lift"] * 2


def grasp_at(x):
    return {"pose": {"x": x, "y": 0, "z": 45, "o_x": 0, "o_y": 0, "o_z": -1, "theta": 0}}


class Recorder:
    """A motion stand-in: records the x of each destination, failing the ones in `fail`."""

    def __init__(self, fail) -> None:
        self.fail, self.xs = fail, []

    async def move(self, component_name, destination, world_state=None, **kwargs):
        if destination.pose.x in self.fail:
            raise RuntimeError(f"no plan to {destination.pose.x}")
        self.xs.append(destination.pose.x)


def test_pick_falls_back_to_the_next_grasp_when_a_plan_fails():
    motion = Recorder(fail={111})
    cell = script.Cell(None, None, Hand(holding=True), motion, None, "pointcloud")
    assert asyncio.run(cell.pick({"label": "", "grasps": [grasp_at(111), grasp_at(222)]}, [])) is True
    assert 222 in motion.xs and 111 not in motion.xs


def test_pick_raises_when_every_grasp_fails():
    cell = script.Cell(None, None, Hand(holding=True), Recorder(fail={111, 222}), None, "pointcloud")
    with pytest.raises(RuntimeError, match="no plan to 222"):
        asyncio.run(cell.pick({"label": "", "grasps": [grasp_at(111), grasp_at(222)]}, []))


def test_every_survey_moves_the_arm_to_the_fixed_joints():
    class Motion:
        async def move(self, **kwargs):
            raise AssertionError("a survey does not plan")

    class Arm:
        def __init__(self) -> None:
            self.went = []

        async def move_to_joint_positions(self, joints):
            self.went.append(list(joints.values))

    class Empty:
        async def do_command(self, command, **kwargs):
            return {"grasps": [], "objects": []}

    arm = Arm()
    cell = script.Cell(None, arm, None, Motion(), Empty(), "pointcloud")

    async def look():
        await cell.survey()
        await cell.survey()

    asyncio.run(look())
    assert arm.went == [script.SURVEY_JOINTS_DEG] * 2


def test_a_failed_pick_lifts_clear_before_the_next_survey():
    class Lift:
        def __init__(self) -> None:
            self.moves = []

        async def move(self, component_name, destination, world_state=None, **kwargs):
            self.moves.append((component_name, destination.pose.x, destination.pose.z))

    cell = ScriptedCell([NEAR], {"near": [RuntimeError("no plan after the grab")] * 2})
    cell.motion = Lift()
    asyncio.run(cell.run())
    assert cell.motion.moves[0] == (script.GRIPPER, 300, script.CARRY_MM)


def test_a_tracker_error_names_the_tracker():
    class Broken:
        async def do_command(self, command, **kwargs):
            raise ConnectionError("down")

    class Still:
        async def move(self, **kwargs):
            pass

        async def move_to_joint_positions(self, joints):
            pass

    cell = script.Cell(None, Still(), None, Still(), Broken(), "shapes")
    with pytest.raises(RuntimeError, match="grasp tracker shape-grasps: down"):
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

    def __init__(self, sim, arm, gripper_m) -> None:
        self.sim, self.arm = sim, arm
        self.offsets = {**OFFSETS_M, script.GRIPPER: np.array([0, 0, gripper_m])}
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
            self.sim.run, lambda: self._solve(position, rotation_of(pose), self.offsets[component_name])
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
        point = (position + frame @ self.offsets[component_name]) * 1000
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, frame.ravel())
        ov = Quaternion(*quat).to_orientation_vector()
        pose = Pose(x=point[0], y=point[1], z=point[2], o_x=ov.o_x, o_y=ov.o_y, o_z=ov.o_z, theta=math.degrees(ov.theta))
        return PoseInFrame(reference_frame="world", pose=pose)


def camera_to_world(points_m: np.ndarray, camera: Pose) -> np.ndarray:
    """Camera-frame points (m) in the world frame (mm), as the frame system moves them."""
    return points_m * 1000 @ rotation_of(camera).T + [camera.x, camera.y, camera.z]


class FakeTracker:
    """The viam-labs:grasp-pose tracker, for the harness: each block still on
    the table as an object (its true centre and yaw, its size from the
    config), nearest the arm first, with the grasps the module gives a lone
    block (labelled square-<colour> whatever its shape: only the colour
    matters to the script): parallel across each side that fits the 2F-85 (+10 mm under
    85 mm), the tool point 15 mm under the top; suction at the top's centre,
    1 mm over it."""

    def __init__(self, sim, cfg: dict, labelled: bool) -> None:
        self.sim, self.labelled = sim, labelled
        world = next(c for c in cfg["components"] if c["name"] == script.WORLD)
        self.sizes = {p["name"]: (p["scale"][0] * 1000, p["scale"][1] * 1000) for p in world["attributes"]["props"]}
        self.suction = next(c for c in cfg["components"] if c["name"] == script.GRIPPER)["attributes"]["asset"] == "robotiq_epick"

    async def do_command(self, command, **kwargs):
        assert command == {"grasps": {}}
        blocks = []
        for name in script.BLOCKS:
            pose = self.sim.get_prop_pose(name)
            x, y, z = (v * 1000 for v in pose["position"])
            if z > script.TABLE_TOP_MM + script.BLOCK_H_MM or script.in_bin(x, y):
                continue
            w, _, _, k = pose["orientation_wxyz"]
            sx, sy = self.sizes[name]
            yaw = math.degrees(2 * math.atan2(k, w)) + (90 if sx < sy else 0)  # the long side's yaw
            blocks.append((math.hypot(x, y), name, x, y, (yaw + 90) % 180 - 90, max(sx, sy), min(sx, sy)))
        top = script.TABLE_TOP_MM + script.BLOCK_H_MM
        objects, grasps = [], []
        for i, (_, name, x, y, yaw, length, width) in enumerate(sorted(blocks)):
            obj = f"object-{i}"
            label = f"square-{name.split('_')[0]}" if self.labelled else ""
            objects.append({
                "name": obj, "label": label, "yaw_deg": yaw, "dims_mm": {"x": length, "y": width, "z": script.BLOCK_H_MM},
                "center": {"x": x, "y": y, "z": script.TABLE_TOP_MM + script.BLOCK_H_MM / 2, "o_x": 0, "o_y": 0, "o_z": 1, "theta": yaw},
            })
            if self.suction:
                spans = [(yaw, width, top + 1)]
            else:
                spans = [(h, w, top - 15) for h, w in ((yaw + 90, width), (yaw, length)) if w + 10 <= 85]
            for j, (h, span, tool_z) in enumerate(spans):
                p = down(x, y, tool_z, (h + 90) % 180 - 90).pose
                grasps.append({
                    "name": f"{obj}/grasp-{j}", "object": obj, "label": label, "width_mm": span, "score": 100.0,
                    "pose": {"x": p.x, "y": p.y, "z": p.z, "o_x": p.o_x, "o_y": p.o_y, "o_z": p.o_z, "theta": p.theta},
                })
        return {"grasps": grasps, "objects": objects}


def make_cell(running_sim, mode, name="vision-pick.json"):
    cfg = json.loads((EXAMPLES / name).read_text())
    items = {item["name"]: item for item in cfg["components"]}
    made = {}
    for item in cfg["components"]:
        if item["model"] in CELL_MODELS:
            if item["model"] == "viam-labs:mujoco-sim:world":
                item["attributes"]["real_time"] = False
            made[item["name"]] = CELL_MODELS[item["model"]].new(component(item), {})
    motion = FakeMotion(running_sim, made[script.ARM], items[script.GRIPPER]["frame"]["translation"]["z"] / 1000)
    cell = script.Cell(made[script.WORLD], made[script.ARM], made[script.GRIPPER], motion, FakeTracker(running_sim, cfg, mode == "shapes"), mode)
    return cell, made


def test_the_survey_joints_put_the_camera_over_the_blocks(running_sim):
    cell, made = make_cell(running_sim, "shapes")

    async def look():
        await made[script.ARM].move_to_joint_positions(JointPositions(values=script.SURVEY_JOINTS_DEG))
        return (await cell.motion.get_pose(script.CAMERA, "world")).pose

    pose = asyncio.run(look())
    # whole degrees put it ~6 mm off in y: nothing, in a view ~690 x 520 mm
    assert (pose.x, pose.y, pose.z) == pytest.approx(script.SURVEY_MM, abs=10)
    rotation = rotation_of(pose)
    assert rotation[:, 2] == pytest.approx([0, 0, -1], abs=0.02)  # pointing down
    heading = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    assert (heading - script.SURVEY_HEADING_DEG + 180) % 360 - 180 == pytest.approx(0, abs=2)


def test_the_survey_sees_the_blocks_and_nothing_else(running_sim):
    cell, made = make_cell(running_sim, "shapes")

    async def look():
        await made[script.WORLD].do_command({"command": "reset"})
        await made[script.ARM].move_to_joint_positions(JointPositions(values=script.SURVEY_JOINTS_DEG))
        pose = (await cell.motion.get_pose(script.CAMERA, "world")).pose
        data, _ = await made[script.CAMERA].get_point_cloud()
        return pose, camera_to_world(parse_pcd(data), pose)

    pose, world = asyncio.run(look())
    # the stand-in's camera pose is the sim's camera
    cam = running_sim.run(lambda: running_sim.data.cam(script.CAMERA).xpos.copy() * 1000)
    assert (pose.x, pose.y, pose.z) == pytest.approx(tuple(cam), abs=1)
    # nothing above the blocks (no fingers), nothing below the table (no floor), no bins
    assert world[:, 2].max() < script.TABLE_TOP_MM + script.BLOCK_H_MM + 2 and world[:, 2].min() > script.TABLE_TOP_MM - 2
    assert not any(script.in_bin(x, y) for x, y, _ in world[::50])


@pytest.mark.parametrize(
    "name, mode",
    [("vision-pick.json", "shapes"), ("vision-pick.json", "pointcloud"), ("vision-pick-suction.json", "pointcloud")],
)
def test_the_script_sorts_every_block(running_sim, name, mode):
    cell, made = make_cell(running_sim, mode, name)

    async def run():
        await made[script.WORLD].do_command({"command": "reset"})
        result = await cell.run()
        return result, await cell.landed()

    result, where = asyncio.run(run())
    want = {block: block.split("_")[0] if mode == "shapes" else "clear" for block in script.BLOCKS}
    assert where == want
    assert result["picked"] == len(script.BLOCKS)
    first = result["reports"][0]
    assert first["seen"] == len(script.BLOCKS) and first["spurious"] == 0
    # the first pick's moves avoid the other seven objects' boxes, each under a unique label
    state = next(s for s in cell.motion.world_states if s is not None)
    geometries = state.obstacles[0].geometries
    assert len(geometries) == len(script.BLOCKS) - 1
    assert all(g.HasField("box") and g.label.startswith("object-") for g in geometries)
    assert len({g.label for g in geometries}) == len(geometries)
    world = next(c for c in json.loads((EXAMPLES / name).read_text())["components"] if c["name"] == script.WORLD)
    blocks = [[v * 1000 for v in prop["position"]] for prop in world["attributes"]["props"]]  # where they started
    for g in geometries:
        assert min(math.dist((g.center.x, g.center.y), b[:2]) for b in blocks) < 15
    assert max(e["mm"] for e in first["errors"]) < 5 and max(e["yaw_deg"] for e in first["errors"]) < 5
