"""Writes tiago_base.xml and tiago_upper.xml, TIAGo++ split into a mujoco-sim
base and a robot file riding it, from mujoco_menagerie's pal_tiago_dual
tiago_dual_position.xml (at 370c439).

tiago_base.xml: base_link without its free joint, with the base, lidar, dock
and antenna meshes and the four wheels with their velocity actuators.

tiago_upper.xml: a welded root torso_fixed (no inertial: base_link's mass stays
on the base) with the torso_fix mesh and the column, turned a quarter turn
about z so +Y is TIAGo's forward, its origin on the base's frame axis (the
robot's frame on the base is a pure z); the torso on its slide with the head
and both arms. Torso and head ctrlranges are widened to their joint ranges.
Each gripper's two fingers hang from a new welded gripper_<side>_link, coupled
by an equality and driven by one position actuator (twice one finger's
stiffness, damping and force: the pair has twice one finger's inertia, so it
keeps menagerie's damping ratio); each finger collides with a box at its pad (its mesh is visual
only). Each arm's forearm (arm_<side>_5_link) and wrist (arm_<side>_7_link)
don't collide: their hulls meet across the wrist joint inside its range.
Sites: torso_frame, gripper_<side>_grasp (between the fingertips,
+Z out of the fingers), head_camera (+Z along the view, +Y down). Keyframe
home: torso mid-height, arms tucked in front, head level.

Run: uv run python assets/pal_tiago_dual/make_tiago.py <menagerie>/pal_tiago_dual
"""

import math
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

HERE = Path(__file__).parent
SIDES = ("left", "right")
TURN = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))  # a quarter turn about z
GRASP_Z = -0.2  # in the finger frame: the pads reach the tips 19 mm further
# Each finger's pad (from its mesh: inner face on x = 0 at zero, closed):
# half sizes and centre z in the finger frame. MuJoCo collides a mesh's
# convex hull, and the L-shaped finger's slants from the pad out to its root,
# squeezing a grasped block out of the fingers; the pads collide instead.
PAD_HALF = (0.0041, 0.025, 0.0441)
PAD_Z = -0.1747
HEAD_CAMERA = ("0.112 0.09 0", "0 0.707107 0 0.707107")  # on head_2's face; +Z forward, +Y down
# home: torso mid-height, upper arms down, forearms and grippers forward,
# head level. The arms are mirrored in their body frames, so equal joint
# values are mirror poses. Joints stay 8 deg or more inside their ranges
# (the right arm's joint 3 tops out at 90 deg).
TORSO_HOME = 0.175
ARM_HOME = (-0.12, 1.42, 1.42, 2.21, -0.11, -0.95, 1.86)  # arm_*_1 ... 7
OPEN = 0.045


def numbers(element: ET.Element, key: str, default: str) -> np.ndarray:
    return np.array([float(v) for v in element.get(key, default).split()])


def text(values) -> str:
    return " ".join(f"{round(float(v), 9) + 0.0:.6g}" for v in values)  # no -0 or 1e-18 noise


def turn(element: ET.Element) -> None:
    """Rotate an element's pose by TURN about its parent's origin."""
    pos, quat = numbers(element, "pos", "0 0 0"), numbers(element, "quat", "1 0 0 0")
    rotated, product = np.zeros(3), np.zeros(4)
    mujoco.mju_rotVecQuat(rotated, pos, np.array(TURN))
    mujoco.mju_mulQuat(product, np.array(TURN), quat)
    element.set("pos", text(rotated))
    element.set("quat", text(product))


def find(root: ET.Element, tag: str, name: str) -> ET.Element:
    found = root.find(f".//{tag}[@name='{name}']")
    assert found is not None, name
    return found


def prune_meshes(root: ET.Element) -> None:
    used = {g.get("mesh") for g in root.iter("geom")}
    asset = root.find("asset")
    for mesh in asset.findall("mesh"):
        if mesh.get("name") not in used:
            asset.remove(mesh)


def base(root: ET.Element) -> ET.Element:
    root.set("model", "tiago_base")
    body = find(root, "body", "base_link")
    body.remove(body.find("joint"))
    for geom in body.findall("geom"):
        if geom.get("mesh") == "torso_fix" or geom.get("type") == "box":
            body.remove(geom)
    body.remove(find(body, "body", "torso_lift_link"))
    actuators = root.find("actuator")
    for actuator in list(actuators):
        if not actuator.get("name").startswith("wheel_"):
            actuators.remove(actuator)
    root.remove(root.find("contact"))
    prune_meshes(root)
    return root


def upper(root: ET.Element) -> ET.Element:
    root.set("model", "tiago_upper")
    body = find(root, "body", "base_link")
    body.set("name", "torso_fixed")
    for child in list(body):
        keep = child.tag == "geom" and (child.get("mesh") == "torso_fix" or child.get("type") == "box")
        if child.tag == "body" and child.get("name") == "torso_lift_link":
            keep = True
        if not keep:
            body.remove(child)
        else:
            turn(child)
    torso = find(body, "body", "torso_lift_link")
    ET.SubElement(torso, "site", name="torso_frame")
    head = find(torso, "body", "head_2_link")
    ET.SubElement(head, "site", name="head_camera", pos=HEAD_CAMERA[0], quat=HEAD_CAMERA[1])
    actuators = root.find("actuator")
    menagerie = {actuator.get("name"): actuator for actuator in actuators}
    for actuator in list(actuators):
        name = actuator.get("name")
        if name.startswith(("wheel_", "gripper_")):
            actuators.remove(actuator)
        elif name.startswith(("torso_", "head_")):  # the file's are narrower: SVA limits unreachable
            actuator.set("ctrlrange", find(root, "joint", actuator.get("joint")).get("range"))
    equality = ET.Element("equality")
    root.insert(list(root).index(actuators), equality)
    for side in SIDES:
        last = find(root, "body", f"arm_{side}_7_link")
        fingers = [b for b in last.findall("body") if b.get("name").startswith(f"gripper_{side}_")]
        gripper = ET.SubElement(last, "body", name=f"gripper_{side}_link", pos=fingers[0].get("pos"),
                                quat=fingers[0].get("quat"))
        for finger in fingers:
            last.remove(finger)
            del finger.attrib["pos"], finger.attrib["quat"]
            gripper.append(finger)
            mesh = finger.find("geom")
            mesh.set("contype", "0")
            mesh.set("conaffinity", "0")
            side_x = -PAD_HALF[0] if "_left_finger" in finger.get("name") else PAD_HALF[0]
            ET.SubElement(finger, "geom", type="box", size=text(PAD_HALF), pos=text((side_x, 0, PAD_Z)),
                          group="3", mass="0")
        ET.SubElement(gripper, "site", name=f"gripper_{side}_grasp", pos=f"0 0 {GRASP_Z}", quat="0 1 0 0")
        driven, follower = (f"gripper_{side}_{finger}_finger_joint" for finger in ("left", "right"))
        ET.SubElement(equality, "joint", joint1=follower, joint2=driven, polycoef="0 1 0 0 0")
        # one actuator does both fingers' work: twice the gains and the force (twice the
        # inertia too, so the damping ratio stays menagerie's)
        finger = menagerie[f"{driven}_position"]
        gain = 2 * float(finger.get("gainprm"))
        bias = 2 * numbers(finger, "biasprm", "0 0 0")
        force = 2 * numbers(finger, "forcerange", "0 0")
        ET.SubElement(actuators, "general", name=f"gripper_{side}_position", joint=driven,
                      ctrlrange=f"0 {OPEN}", forcerange=text(force), gainprm=f"{gain:g}",
                      biasprm=text(bias), biastype="affine", **{"class": "position"})
        # the forearm's and the wrist's convex hulls meet across the wrist joint well inside
        # its range (from about 70 deg), holding the arm off a planned pose
        ET.SubElement(root.find("contact"), "exclude", body1=f"arm_{side}_5_link", body2=f"arm_{side}_7_link")
    prune_meshes(root)
    root.append(keyframe(root))
    return root


def keyframe(root: ET.Element) -> ET.Element:
    """home, in the order of the file's joints (each one qpos) and actuators."""
    values = {"torso_lift_joint": TORSO_HOME}
    for side in SIDES:
        for i, value in enumerate(ARM_HOME, start=1):
            values[f"arm_{side}_{i}_joint"] = value
    qpos = [values.get(joint.get("name"), 0.0) for joint in root.find("worldbody").iter("joint")]
    ctrl = [values.get(actuator.get("joint"), 0.0) for actuator in root.find("actuator")]
    keys = ET.Element("keyframe")
    ET.SubElement(keys, "key", name="home", qpos=text(qpos), ctrl=text(ctrl))
    return keys


def write(root: ET.Element, name: str) -> None:
    ET.indent(root)
    path = HERE / name
    path.write_text(ET.tostring(root, encoding="unicode") + "\n")
    mujoco.MjModel.from_xml_path(str(path))  # it loads
    print("wrote", path)


if __name__ == "__main__":
    source = Path(sys.argv[1]) / "tiago_dual_position.xml"
    flat = mujoco.MjSpec.from_file(str(source)).to_xml()  # the includes inlined
    write(base(ET.fromstring(flat)), "tiago_base.xml")
    write(upper(ET.fromstring(flat)), "tiago_upper.xml")
