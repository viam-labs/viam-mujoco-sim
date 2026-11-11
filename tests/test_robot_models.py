import asyncio
import json
import math

import pytest
from viam.proto.app.robot import ComponentConfig, Frame, Orientation, Translation
from viam.utils import dict_to_struct

from mujoco_module.models.arm import MujocoArm
from mujoco_module.models.robot import MujocoRobot


# ---- validation -------------------------------------------------------------


def config(name, attrs, parent="world", translation=None, orientation=None):
    cfg = ComponentConfig(name=name, attributes=dict_to_struct({"world": "sim-world", **attrs}))
    frame = Frame(parent=parent)
    if translation is not None:
        frame.translation.CopyFrom(Translation(**translation))
    if orientation is not None:
        frame.orientation.CopyFrom(orientation)
    cfg.frame.CopyFrom(frame)
    return cfg


def test_validate():
    assert MujocoRobot.validate_config(config("aloha", {"asset": "aloha"})) == (["sim-world"], [])
    arm = config("left-arm", {"robot": "aloha", "asset": "aloha_left"}, parent="aloha")
    assert MujocoArm.validate_config(arm)[0] == ["sim-world", "aloha"]


@pytest.mark.parametrize("cls,cfg,message", [
    (MujocoRobot, config("r", {}), "set asset"),
    (MujocoRobot, config("r", {"asset": "aloha", "mjcf": "x.xml"}), "set asset"),
    (MujocoRobot, config("r", {"asset": "panda"}), "asset must be one of"),
    (MujocoRobot, config("r", {"mjcf": "robot.urdf"}), "URDF robot files aren't supported yet"),
    (MujocoRobot, config("r", {"asset": "aloha"}, parent="husky"), "on a base is not supported yet"),
    (MujocoArm, config("a", {"robot": "aloha", "mjcf": "x.xml"}, parent="aloha"), "set robot, or mjcf/asset, not both"),
    (MujocoArm, config("a", {"robot": "aloha", "asset": "ur5e"}, parent="aloha"), "set robot, or mjcf/asset, not both"),
    (MujocoArm, config("a", {"robot": "aloha"}, parent="aloha"), "needs joints"),
    (MujocoArm, config("a", {"robot": "aloha", "joints": ["j"]}, parent="aloha"), "needs kinematics_url"),
    (MujocoArm, config("a", {"robot": "aloha", "asset": "aloha_left", "collision": "kinematics"}, parent="aloha"),
     'collision "kinematics" is not supported'),
    (MujocoArm, config("a", {"robot": "aloha", "asset": "aloha_left"}), 'needs frame.parent "aloha"'),
    (MujocoArm, config("a", {"asset": "aloha_left"}), "is a limb of a robot file; set robot"),
])
def test_validate_rejects(cls, cfg, message):
    with pytest.raises(ValueError, match=message):
        cls.validate_config(cfg)


def test_robot_model(running_sim):
    from mujoco_module.models.world import MujocoWorld

    MujocoWorld.new(ComponentConfig(name="sim-world", attributes=dict_to_struct({})), {})
    robot_model = MujocoRobot.new(config("aloha", {"asset": "aloha"}), {})
    left = MujocoArm.new(config("left-arm", {"robot": "aloha", "asset": "aloha_left"}, parent="aloha",
                                translation={"x": -469, "y": -19, "z": 20}), {})
    right = MujocoArm.new(config(
        "right-arm", {"robot": "aloha", "asset": "aloha_right"}, parent="aloha",
        translation={"x": 469, "y": -19, "z": 20},
        orientation=Orientation(vector_degrees=Orientation.OrientationVectorDegrees(z=1, theta=180)),
    ), {})

    async def run():
        assert await robot_model.get_geometries() == []
        await asyncio.sleep(0.3)
        positions = await left.get_joint_positions()
        assert [math.radians(v) for v in positions.values] == pytest.approx([0, -0.96, 1.16, 0, -0.3, 0], abs=1e-2)
        for component in (left, right, robot_model):
            await component.close()

    asyncio.run(run())


def test_the_example_validates():
    import pathlib

    from mujoco_module.models.gripper import MujocoGripper

    cfg = json.loads((pathlib.Path(__file__).parent.parent / "examples" / "aloha-handover.json").read_text())
    models = {"robot": MujocoRobot, "arm": MujocoArm, "gripper": MujocoGripper}
    checked = []
    for item in cfg["components"]:
        cls = models.get(item["model"].rsplit(":", 1)[1])
        if cls is None:
            continue  # the obstacle, world and camera have their own example tests
        frame = item["frame"]
        orientation = None
        if "orientation" in frame:
            v = frame["orientation"]["value"]
            orientation = Orientation(vector_degrees=Orientation.OrientationVectorDegrees(x=v["x"], y=v["y"], z=v["z"], theta=v["th"]))
        cls.validate_config(config(item["name"], item["attributes"], parent=frame["parent"],
                                   translation=frame.get("translation"), orientation=orientation))
        checked.append(item["name"])
    assert checked == ["aloha", "left-arm", "right-arm", "left-gripper", "right-gripper"]
