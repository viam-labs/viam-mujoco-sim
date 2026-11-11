# TIAGo++

From [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/pal_tiago_dual)
at `370c439`: the 17 meshes under `assets/` that the derived files use
(menagerie's dock, cover and spare base meshes are left out). Apache-2.0,
PAL Robotics; see LICENSE.

`tiago_base.xml` (the `pal_tiago_base` base) and `tiago_upper.xml` (the
`tiago` robot) are written by `make_tiago.py` from menagerie's
`tiago_dual_position.xml`; regenerate them rather than editing them:

```sh
uv run python assets/pal_tiago_dual/make_tiago.py <mujoco_menagerie>/pal_tiago_dual
uv run python assets/pal_tiago_dual/make_kinematics.py
```

- The base is `base_link` without its free joint: the base, lidar, dock and
  antenna meshes and the four wheels with their velocity actuators. Its omni
  wheels drive as a diff-drive.
- The upper body hangs from a welded `torso_fixed` (the torso_fix mesh and
  the column, no mass: the base keeps its 59.5 kg), turned so +Y is forward
  with its origin on the base's frame axis, so the robot's frame on the base
  is a pure z of 0. Torso and head ctrlranges are widened to the joints'
  ranges. Each gripper's fingers hang from a welded `gripper_<side>_link`,
  coupled by an equality and driven by one position actuator (twice one
  finger's stiffness, damping and force, so the pair, with twice the inertia,
  keeps menagerie's damping ratio; ctrl 0 closed to 0.045 open); each finger
  collides with a box at its pad (the finger mesh's convex hull would squeeze
  a block out). Sites: `torso_frame`, `gripper_<side>_grasp` (between the
  fingertips, 19 mm short of them, +Z out of the fingers), `head_camera` (+Z
  along the view, +Y down). Keyframe `home`.

`tiago_torso_kinematics.json`, `tiago_arm_left_kinematics.json`,
`tiago_arm_right_kinematics.json` and `tiago_head_kinematics.json` are written
by `make_kinematics.py`: Viam SVA kinematics from `tiago_upper.xml`'s joint
anchors and axes at zeros, in each limb's chain base frame, joint limits equal
to its ranges (the torso's prismatic joint in mm). Shapes are placed in each
link's parent frame, as RDK places them: each moving arm link a capsule from
its joint to the next (a sphere where they meet) covering its collision-mesh
vertices, the last link a box over the wrist and the open gripper, the head a
box, and the torso's column (above the base's box) and shell (narrowed to
clear the arms' first capsules) boxes, so no two limbs' shapes touch at home
(`tests/test_tiago.py` checks they sit on and cover the robot).
