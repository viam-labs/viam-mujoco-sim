"""Writes the Viam SVA kinematics of tiago_upper.xml's four limbs:
tiago_torso_kinematics.json, tiago_arm_left_kinematics.json,
tiago_arm_right_kinematics.json and tiago_head_kinematics.json.

Each is generated from the file's joint anchors and axes at zeros, in its
chain base's frame (torso_fixed for the torso, torso_lift_link for the
others), joint limits equal to the file's ranges (prismatic in mm). Every
link frame keeps the chain base's axes; the end link turns to the flange
site. Shapes are placed in each link's parent frame, as RDK places them:

- torso: the column (a box on the base link, from just above the base's
  shapes up) and the torso shell (a box on the lift link, narrowed to clear
  the arms' first capsules);
- arms: each moving link a capsule from its joint to the next (a sphere
  where they meet) covering its collision-mesh vertices, and the last link a
  box bounding the wrist and the open gripper; no base shape (it's the
  torso's);
- head: a box bounding the tilting head.

Run (after make_tiago.py): uv run python assets/pal_tiago_dual/make_kinematics.py
"""

import json
import math
from pathlib import Path

import mujoco
import numpy as np
from viam.spatialmath import Quaternion

HERE = Path(__file__).parent
CLEARANCE = 0.005  # between shapes of different limbs at home


def ov(rotation: np.ndarray) -> dict:
    """A rotation matrix as an SVA orientation (ov_degrees)."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    vector = Quaternion(*quat).to_orientation_vector()
    return {"type": "ov_degrees", "value": {
        "x": round(vector.o_x, 6), "y": round(vector.o_y, 6), "z": round(vector.o_z, 6),
        "th": round(math.degrees(vector.theta), 4) + 0.0,
    }}


def mm(vector) -> dict:
    return {key: round(float(v) * 1000, 2) + 0.0 for key, v in zip("xyz", vector)}


class Chain:
    """The file at zeros (fingers open), in a chain base's frame."""

    def __init__(self, model: mujoco.MjModel, base: str) -> None:
        self.model, self.data = model, mujoco.MjData(model)
        for joint in range(model.njnt):
            if "finger" in model.joint(joint).name:
                self.data.qpos[model.jnt_qposadr[joint]] = model.jnt_range[joint, 1]  # open
        mujoco.mj_kinematics(model, self.data)
        body = model.body(base).id
        self.rotation = self.data.xmat[body].reshape(3, 3)
        self.origin = self.data.xpos[body].copy()

    def point(self, world: np.ndarray) -> np.ndarray:
        return self.rotation.T @ (world - self.origin)

    def anchor(self, joint: str) -> np.ndarray:
        return self.point(self.data.xanchor[self.model.joint(joint).id])

    def axis(self, joint: str) -> np.ndarray:
        return self.rotation.T @ self.data.xaxis[self.model.joint(joint).id]

    def site(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        site = self.model.site(name).id
        return self.point(self.data.site_xpos[site]), self.rotation.T @ self.data.site_xmat[site].reshape(3, 3)

    def points(self, bodies: list[str], kind: int | None = None) -> np.ndarray:
        """The colliding geoms' points on the bodies (only geoms of type `kind`
        when given): mesh vertices, else bounding-box corners."""
        model, data = self.model, self.data
        ids = {model.body(name).id for name in bodies}
        found = []
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] not in ids or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            if kind is not None and model.geom_type[geom] != kind:
                continue
            if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[geom]
                local = model.mesh_vert[model.mesh_vertadr[mesh] : model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            else:
                centre, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
                local = centre + half * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            world = local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            found.append(np.array([self.point(p) for p in world]))
        return np.vstack(found)


def box(points: np.ndarray, frame: np.ndarray, low=None, high=None) -> dict:
    """A box bounding the points, clipped to [low, high], placed from `frame` (a point)."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    if low is not None:
        lo = np.maximum(lo, low)
    if high is not None:
        hi = np.minimum(hi, high)
    return {**mm(hi - lo), "translation": mm((lo + hi) / 2 - frame)}


def capsule(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> dict:
    """From start to end (a sphere if they meet), its radius covering the points, mm up."""
    segment = end - start
    length = float(np.linalg.norm(segment))
    along = np.clip((points - start) @ segment / length**2, 0, 1) if length else np.zeros(len(points))
    radius = math.ceil(np.linalg.norm(points - (start + along[:, None] * segment), axis=1).max() * 1000)
    if not length:
        return {"r": radius, "translation": mm((0, 0, 0))}
    return {
        "r": radius, "l": round(length * 1000 + 2 * radius, 1),
        "translation": mm(segment / 2),
        "orientation": ov(_z_along(segment / length)),
    }


def _z_along(direction: np.ndarray) -> np.ndarray:
    """A rotation whose z is `direction`."""
    helper = np.array([1.0, 0, 0]) if abs(direction[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(helper, direction)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(direction, x), direction])


def sva(name: str, chain: Chain, joints: list[str], links: list[str], end_site: str,
        shapes: dict[str, dict], base_shape: dict | None = None) -> dict:
    """links[i] is the body joints[i] moves; its link runs to the next joint
    (the end link to end_site)."""
    model = chain.model
    anchors = [chain.anchor(joint) for joint in joints]
    site, rotation = chain.site(end_site)
    out_links = [{"id": "base", "parent": "world", "translation": mm(anchors[0])}]
    if base_shape:
        out_links[0]["geometry"] = base_shape
    out_joints = []
    for i, (joint, link) in enumerate(zip(joints, links)):
        index = model.joint(joint).id
        prismatic = model.jnt_type[index] == mujoco.mjtJoint.mjJNT_SLIDE
        low, high = model.jnt_range[index]
        scale = 1000 if prismatic else 180 / math.pi
        out_joints.append({
            "id": joint, "type": "prismatic" if prismatic else "revolute",
            "parent": out_links[-1]["id"],
            "axis": {key: round(float(v), 6) + 0.0 for key, v in zip("xyz", chain.axis(joint))},
            "min": round(low * scale, 4), "max": round(high * scale, 4),
        })
        last = i == len(joints) - 1
        entry = {"id": link, "parent": joint,
                 "translation": mm((site if last else anchors[i + 1]) - anchors[i])}
        if last:
            entry["orientation"] = ov(rotation)
        if link in shapes:
            entry["geometry"] = shapes[link]
        out_links.append(entry)
    return {"name": name, "kinematic_param_type": "SVA", "links": out_links, "joints": out_joints}


def arm(model: mujoco.MjModel, side: str) -> dict:
    chain = Chain(model, "torso_lift_link")
    joints = [f"arm_{side}_{i}_joint" for i in range(1, 8)]
    links = [f"arm_{side}_{i}_link" for i in range(1, 8)]
    anchors = [chain.anchor(joint) for joint in joints]
    shapes = {
        links[i]: capsule(chain.points([links[i]]), anchors[i], anchors[i + 1]) for i in range(6)
    }
    gripper = [model.body(b).name for b in range(model.nbody)
               if model.body(b).name == links[6] or model.body(b).name.startswith(f"gripper_{side}_")]
    shapes[links[6]] = box(chain.points(gripper), anchors[6])
    return sva(f"tiago_arm_{side}", chain, joints, links, f"gripper_{side}_grasp", shapes)


def torso(model: mujoco.MjModel, arms: list[dict], base_top: float) -> dict:
    chain = Chain(model, "torso_fixed")
    anchor = chain.anchor("torso_lift_joint")
    # the column box (the torso_fix mesh is all below the base's top)
    column = box(chain.points(["torso_fixed"], mujoco.mjtGeom.mjGEOM_BOX), np.zeros(3), low=(-1, -1, base_top + CLEARANCE))
    # narrowed to clear each arm's first capsule (on the shell's sides, in its x here)
    first = [a["links"][1]["geometry"] for a in arms]
    shoulder = min(abs(a["links"][0]["translation"]["y"]) - g["r"] for a, g in zip(arms, first)) / 1000
    half = shoulder - CLEARANCE
    shell = box(chain.points(["torso_lift_link"]), anchor, low=(-half, -1, -1), high=(half, 1, 1))
    return sva("tiago_torso", chain, ["torso_lift_joint"], ["torso_lift_link"], "torso_frame",
               {"torso_lift_link": shell}, base_shape=column)


def head(model: mujoco.MjModel) -> dict:
    chain = Chain(model, "torso_lift_link")
    joints, links = ["head_1_joint", "head_2_joint"], ["head_1_link", "head_2_link"]
    shapes = {"head_2_link": box(chain.points(["head_2_link"]), chain.anchor("head_2_joint"))}
    return sva("tiago_head", chain, joints, links, "head_camera", shapes)


def base_top() -> float:
    """The top of pal_tiago_base's shape (its static colliding geoms; base_link
    is on the floor, where the robot's frame is)."""
    model = mujoco.MjModel.from_xml_path(str(HERE / "tiago_base.xml"))
    return float(Chain(model, "base_link").points(["base_link"])[:, 2].max())


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(HERE / "tiago_upper.xml"))
    arms = [arm(model, side) for side in ("left", "right")]
    files = {"torso": torso(model, arms, base_top()), "arm_left": arms[0], "arm_right": arms[1], "head": head(model)}
    for name, doc in files.items():
        path = HERE / f"tiago_{name}_kinematics.json"
        path.write_text(json.dumps(doc, indent=2) + "\n")
        print("wrote", path)
