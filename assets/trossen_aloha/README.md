# ALOHA

From [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/aloha)
at `370c439`: `aloha.xml`, `joint_position_actuators.xml`, `keyframe_ctrl.xml`
and the 12 meshes `aloha.xml` uses (menagerie's table and frame are left out).
BSD-3-Clause, Copyright (c) 2023 Trossen Robotics; see LICENSE.

`vx300s_kinematics.json` is written here: Viam SVA kinematics for one ViperX
300 arm, `base_link` to the `gripper` site turned so +Z points out of the
fingers, generated from `aloha.xml`'s joint anchors and axes, joint limits
equal to its ranges. Shapes are placed in each link's parent frame, as RDK
places them: the fixed base and the gripper are boxes bounding their geoms at
zeros (the gripper's reaches 19 mm past the site, to the fingertips); each
moving link is a capsule from its parent joint to its own, as the Lite6's are,
with a radius covering every collision-mesh vertex, so neighbouring capsules
overlap at the start and RDK doesn't flag them mid-move
(`tests/test_robot_grippers.py` checks they sit on and cover the arm). Both
arms use it. Known limit: past about ±100° of wrist_angle the forearm and wrist
capsules touch, so the planner treats such bends as self-colliding though the
meshes clear; top-down grasps stay well inside it.
