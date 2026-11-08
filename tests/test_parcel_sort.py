"""examples/parcel-sort.json and parcel_sort.py, run end to end on the local
sim (stepping as fast as it can) with a stand-in for the motion service."""

import asyncio
import importlib.util
import json
import logging
import math
from pathlib import Path

import mujoco
import numpy as np
from google.protobuf.json_format import ParseDict
from viam.components.arm import JointPositions
from viam.proto.app.robot import ComponentConfig, Frame
from viam.spatialmath import OrientationVector
from viam.utils import dict_to_struct

from mujoco_module.models import obstacle as obstacle_module
from mujoco_module.models.arm import MujocoArm
from mujoco_module.models.conveyor import MujocoConveyor
from mujoco_module.models.gripper import MujocoGripper
from mujoco_module.models.obstacle import MujocoObstacle
from mujoco_module.models.photo_eye import MujocoPhotoEye
from mujoco_module.models.scale import MujocoScale
from mujoco_module.models.spawner import MujocoSpawner
from mujoco_module.models.world import MujocoWorld
from mujoco_module.handles import CONVEYOR_RAIL_M, ArmHandle, ConveyorHandle, GripperHandle, JointTrajectoryPoint
from mujoco_module.models.arm import _densify
from mujoco_module.sim import WorldConfig

EXAMPLES = Path(__file__).parents[1] / "examples"
MODELS = {
    "viam-labs:mujoco-sim:world": MujocoWorld,
    "viam-labs:mujoco-sim:obstacle": MujocoObstacle,
    "viam-labs:mujoco-sim:conveyor": MujocoConveyor,
    "viam-labs:mujoco-sim:spawner": MujocoSpawner,
    "viam-labs:mujoco-sim:photo_eye": MujocoPhotoEye,
    "viam-labs:mujoco-sim:scale": MujocoScale,
    "viam-labs:mujoco-sim:arm": MujocoArm,
    "viam-labs:mujoco-sim:gripper": MujocoGripper,
}
TOOL_M = 0.13  # the gripper frame past the arm's end frame


def load_script():
    spec = importlib.util.spec_from_file_location("parcel_sort", EXAMPLES / "parcel_sort.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


def component(item: dict) -> ComponentConfig:
    cfg = ComponentConfig(name=item["name"], attributes=dict_to_struct(item["attributes"]))
    if "frame" in item:
        frame = item["frame"]
        parsed = {"parent": frame["parent"], "translation": frame.get("translation", {})}
        if "geometry" in frame:
            g = frame["geometry"]
            parsed["geometry"] = {"box": {"dims_mm": {"x": g["x"], "y": g["y"], "z": g["z"]}}}
            if "translation" in g:
                parsed["geometry"]["center"] = {**g["translation"], "o_z": 1}
        cfg.frame.CopyFrom(ParseDict(parsed, Frame()))
    return cfg


def test_example_validates_and_matches_the_script():
    cfg = json.loads((EXAMPLES / "parcel-sort.json").read_text())
    for item in cfg["components"]:
        if item["model"] in MODELS:
            MODELS[item["model"]].validate_config(component(item))
    script = load_script()
    frames = {item["name"]: item.get("frame", {}).get("translation") for item in cfg["components"]}
    attrs = {item["name"]: item["attributes"] for item in cfg["components"]}
    assert script.BELT == (frames["belt"]["x"], frames["belt"]["z"], attrs["belt"]["width_m"] * 1000)
    assert script.BELT_LENGTH_MM == attrs["belt"]["length_m"] * 1000 and frames["belt"]["y"] == 0
    assert script.RAIL_MM == (CONVEYOR_RAIL_M * 1000, attrs["belt"].get("rails_m", 0.04) * 1000)
    assert script.SCALE == (frames["scale"]["x"], frames["scale"]["y"], frames["scale"]["z"])
    for name, (x, y) in script.BINS.items():
        assert (frames[f"{name}-bin"]["x"], frames[f"{name}-bin"]["y"]) == (x, y)
        assert attrs[f"{name}-bin"]["scale"] == [4, 4, 2.5]  # bin.stl is 150 x 150 x 80 mm
    assert script.BIN_HALF_MM == 150 * 4 / 2 and script.BIN_RIM_MM == 80 * 2.5
    classes = {c["name"]: c["mass_kg"] for c in attrs["feed"]["classes"]}
    assert classes["small"][1] < script.LIGHT_KG < classes["medium"][0]


def test_the_frame_boxes_match_the_shapes_the_module_builds(running_sim, caplog):
    """RDK never asks a motor or sensor for its geometries, so the planner sees
    the belt and the scale only through these frame boxes."""
    from mujoco_module.models.utils import frame_box

    cfg = json.loads((EXAMPLES / "parcel-sort.json").read_text())
    items = {item["name"]: item for item in cfg["components"]}
    world = items["sim-world"]
    MujocoWorld.new(component({**world, "attributes": {**world["attributes"], "real_time": False}}), {})
    for name in ("belt", "scale"):
        with caplog.at_level(logging.WARNING):
            model = MODELS[items[name]["model"]].new(component(items[name]), {})
        assert frame_box(model._geometries()[0]) == items[name]["frame"]["geometry"]
        asyncio.run(model.close())
    assert "never asks a motor or sensor" not in caplog.text  # the frames have their geometry


class FakeMotion:
    """The motion service, for the harness: inverse kinematics on the sim's
    own model, then a straight joint path. No planning; the script's
    waypoints keep the straight joint paths clear."""

    def __init__(self, sim, arm: MujocoArm) -> None:
        self.sim, self.arm = sim, arm

    def _solve(self, position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        sim = self.sim
        model = sim.model
        _, ids = self.arm._h()._now()
        data = mujoco.MjData(model)
        data.qpos[:] = sim.data.qpos
        start = data.qpos[ids.qpos].copy()
        # start from the arm swung round to face the target: the shoulder pan
        # turns the tool point about the arm's base, which is on the world's z axis
        tcp = sim.data.site_xpos[ids.end_site] + sim.data.site_xmat[ids.end_site].reshape(3, 3) @ [0, 0, TOOL_M]
        data.qpos[ids.qpos[0]] += math.atan2(position[1], position[0]) - math.atan2(tcp[1], tcp[0])
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        for _ in range(500):
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            frame = data.site_xmat[ids.end_site].reshape(3, 3)
            tcp = data.site_xpos[ids.end_site] + frame @ [0, 0, TOOL_M]
            error = np.concatenate([
                position - tcp, 0.5 * sum(np.cross(frame[:, i], rotation[:, i]) for i in range(3))
            ])
            if np.linalg.norm(error) < 1e-5:
                # the same pose, each joint the nearest way round from where it is, within its range
                joints = start + np.remainder(data.qpos[ids.qpos] - start + np.pi, 2 * np.pi) - np.pi
                low, high = model.jnt_range[[model.dof_jntid[d] for d in ids.dof]].T
                return joints - 2 * np.pi * (joints > high) + 2 * np.pi * (joints < low)
            mujoco.mj_jac(model, data, jacp, jacr, tcp, model.site_bodyid[ids.end_site])
            jac = np.vstack([jacp[:, ids.dof], jacr[:, ids.dof]])
            step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
            data.qpos[ids.qpos] += step * min(1.0, 0.2 / np.abs(step).max())  # small steps stay on one branch
        raise RuntimeError(f"no IK solution for {position}")

    async def move(self, component_name, destination, world_state=None, **kwargs):
        assert component_name == "pick-gripper" and destination.reference_frame == "world"
        pose = destination.pose
        q = OrientationVector(pose.o_x, pose.o_y, pose.o_z, math.radians(pose.theta)).to_quaternion()
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, np.array([q.w, q.i, q.j, q.k]))
        position = np.array([pose.x, pose.y, pose.z]) / 1000
        joints = await asyncio.to_thread(self.sim.run, lambda: self._solve(position, rotation.reshape(3, 3)))
        # one straight joint path, every joint arriving together, as the motion
        # service's plan runs (MoveToJointPositions slews each joint on its own,
        # which tips the tool mid-move); timed in sim time, so it keeps pace
        handle = self.arm._h()
        start = await asyncio.to_thread(handle.get_joint_positions)
        speed = min(handle.dynamics_metadata()["max_velocity_rad_s"])
        duration = float(np.abs(joints - start).max()) / speed
        await asyncio.wrap_future(handle.start_trajectory([JointTrajectoryPoint(duration, tuple(joints))]))
        await self.arm.move_to_joint_positions(JointPositions(values=list(np.degrees(joints))))  # settle


def test_the_script_sorts_parcels_end_to_end(running_sim, u_pieces, monkeypatch):
    # bin.stl at the example's scale, without decomposing it again
    scaled = [(np.asarray(v) * [4, 4, 2.5], f) for v, f in u_pieces]
    monkeypatch.setattr(obstacle_module, "load_pieces", lambda label, mesh: scaled)
    cfg = json.loads((EXAMPLES / "parcel-sort.json").read_text())
    made = {}
    for item in cfg["components"]:
        if item["model"] in MODELS:
            if item["model"] == "viam-labs:mujoco-sim:world":
                item["attributes"]["real_time"] = False
            made[item["name"]] = MODELS[item["model"]].new(component(item), {})
    script = load_script()
    cell = script.Cell(
        made["belt"], made["eye"], made["feed"], made["scale"], made["pick-gripper"],
        FakeMotion(running_sim, made["pick-arm"]),
    )

    async def run():
        await cell.feed.do_command({"command": "clear"})  # as main() does
        sorted_ = await cell.sort(3)
        return sorted_, await cell.check(sorted_)

    sorted_, wrong = asyncio.run(run())
    assert len(sorted_) == 3 and wrong == []
    assert {bin_name for _, bin_name in sorted_} == {"light", "heavy"}  # seed 42: both kinds


# Joint positions (deg) from a real-time run: the cup on a parcel stopped at
# y -0.5 on the belt, carry height above it, then over the scale. With the
# real pacing the belt stops there; the harness's faster sim lets the parcel
# run on to where the lift doesn't swing it, so this case is checked alone.
BELT_PICK = [-229.5, -49.6, 53.0, -93.4, -90.0, 40.5]
BELT_CARRY = [-229.5, -39.1, 10.3, -59.8, -90.0, 40.5]
OVER_SCALE = [-104.1, -83.7, 77.9, -83.2, -90.0, 165.8]


def test_the_epick_carries_the_heaviest_parcel_level(sim):
    """A large parcel at the class's top size and mass stays flat on the cup
    through the lift and the swing to the scale. Peeled onto one rim contact,
    it swung 30-40 degrees, landed corner first on the scale and stalled the arm."""
    cfg = json.loads((EXAMPLES / "parcel-sort.json").read_text())
    attrs = {item["name"]: item["attributes"] for item in cfg["components"]}
    arm_attrs = attrs["pick-arm"]
    (large,) = [c for c in attrs["feed"]["classes"] if c["name"] == "large"]
    size = [high for _, high in large["size_m"]]
    parcel = {"name": "parcel", "type": "cube", "size": 1, "scale": size, "mass": large["mass_kg"][1],
              "position": [2, 2, 1]}
    sim.configure_world(WorldConfig.from_attrs({"props": [parcel]}))
    arm = ArmHandle(sim, "arm", {k: arm_attrs[k] for k in ("asset", "max_joint_deg_s", "max_joint_deg_s2")})
    gripper = GripperHandle(sim, "grip", "arm", {"asset": attrs["pick-gripper"]["asset"]})
    worst = 0.0

    def move(target):
        nonlocal worst
        target = list(np.radians(target))
        for step in _densify(arm.get_joint_positions(), [target], math.radians(30) * 0.02) + [target] * 25:
            arm.set_joint_targets(step)
            sim.advance(0.02)
            worst = max(worst, math.degrees(math.acos(min(1.0, abs(sim.data.body("parcel").xmat[8])))))

    move(BELT_PICK)
    pinch = sim.data.site("grip/pinch").xpos
    sim.set_prop_pose("parcel", list(pinch - [0, 0, size[2] / 2 + 0.001]), [1, 0, 0, 0])  # CUP_GAP_MM
    gripper.close_jaws()
    sim.advance(0.5)
    move(BELT_CARRY)
    move(OVER_SCALE)
    assert gripper.firm_grasp()
    assert worst < 8  # the tool itself pitches a few degrees mid-move


# The motion service's plan (radians) from a real-time run: from carry height
# over a small parcel at the photo-eye down to 100 mm above it.
OVER_PARCEL = (
    [-3.9478487041342154, -0.3637085709478122, -0.5734237943260614,
     -0.6334796841346785, -1.5708337227397111, 0.7620779278866183],
    [-3.949664382817493, 0.08443467412724313, -1.1029287689842528,
     -0.5523031950762975, -1.5707962580620116, 0.7627281905014102],
)


def test_the_planner_sees_the_rail_the_upper_arm_stalled_on(sim):
    """That plan lowered the upper arm onto the belt's near rail, which the
    belt's frame box leaves out (it stops at the belt top so it doesn't
    enclose the parcels): the shoulder stalled 4 degrees short at its force
    limit. Every point where the arm touches the cell must be inside
    something the planner was given."""
    script = load_script()
    cfg = json.loads((EXAMPLES / "parcel-sort.json").read_text())
    items = {item["name"]: item for item in cfg["components"]}
    belt = items["belt"]
    ConveyorHandle(sim, "belt", {**belt["attributes"], "position": [v / 1000 for v in belt["frame"]["translation"].values()]})
    arm_attrs = items["pick-arm"]["attributes"]
    arm = ArmHandle(sim, "pick-arm", {
        **{k: arm_attrs[k] for k in ("asset", "max_joint_deg_s", "max_joint_deg_s2")},
        "position": [v / 1000 for v in items["pick-arm"]["frame"]["translation"].values()],
    })
    GripperHandle(sim, "pick-gripper", "pick-arm", {"asset": items["pick-gripper"]["attributes"]["asset"]})
    start, end = OVER_PARCEL
    data, ids = arm._now()
    data.qpos[ids.qpos] = data.ctrl[ids.ctrl] = start
    arm.set_joint_targets(start)
    touches = []
    for step in _densify(start, [end], math.radians(arm_attrs["max_joint_deg_s"]) * 0.02) + [end] * 50:
        arm.set_joint_targets(step)
        sim.advance(0.02)
        model = sim.model
        for contact in data.contact[: data.ncon]:
            names = [model.body(model.geom_bodyid[g]).name for g in contact.geom]
            if any(name.startswith(("pick-arm/", "pick-gripper/")) for name in names):
                touches.append((names, contact.pos * 1000))
    assert touches, "the plan clears the rails now; find another that doesn't"

    def frame_box(item):
        g, t = item["frame"]["geometry"], item["frame"]["translation"]
        c = {k: t[k] + g.get("translation", {}).get(k, 0) for k in "xyz"}
        return [c["x"], c["y"], c["z"]], [g["x"], g["y"], g["z"]]

    def obstacle(g):
        return [g.center.x, g.center.y, g.center.z], [g.box.dims_mm.x, g.box.dims_mm.y, g.box.dims_mm.z]

    seen = [frame_box(item) for item in cfg["components"]
            if item.get("frame", {}).get("parent") == "world" and "geometry" in item["frame"]]
    seen += [obstacle(g) for g in getattr(script, "RAILS", [])]
    for names, point in touches:
        assert any(np.all(np.abs(point - c) <= np.asarray(s) / 2 + 1) for c, s in seen), (
            f"{names} touch at {np.round(point)} mm, outside every box the planner sees"
        )
