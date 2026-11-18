# Custom assets

Read this when the prompt names an arm, gripper, object or environment that isn't in SKILL.md's bundled lists. Source it, place it, and check it before you compose the task. Custom assets need `MODULE_VERSION` 0.17.0 or later.

## Trusted sources

Use these without asking:
- `google-deepmind/mujoco_menagerie` (arms, grippers, scenes; check each model's `LICENSE`);
- vendor ROS description repos: `UniversalRobots/Universal_Robots_ROS2_Description`, `frankaemika/franka_description`, `Kinovarobotics/ros2_kortex` (`kortex_description`), `xArm-Developer/xarm_ros2` (`xarm_description`), `ros-industrial/robotiq`;
- the YCB object set (`https://www.ycbbenchmarks.com`, meshes for research use);
- any URL the user gives.

Anything else: propose it with its license and wait for the user. Record every source, commit and license in the run report.

## Where files go

Put them in `<scratch>/assets/<name>/`. Clone with `git clone --depth 1 https://github.com/<owner>/<repo>.git <dir>` (public repos need no login) and record `git rev-parse HEAD`. Reference files by absolute path. Never commit or redistribute them.

## Recipes

### Arm from Menagerie

Set `"mjcf": "<abs>/<model>/<arm>_nohand.xml"` (the model's file without a hand) and leave out `kinematics_url`: the module derives it from the MJCF. The file needs a flange site, `attachment_site` by default; set `flange_site` if it has another name. The Franka Panda needs nothing else: `franka_emika_panda/panda_nohand.xml` (checked at Menagerie commit 370c439) configures as is, takes the bundled `robotiq_2f85` and reached two test targets exactly. Record your commit. If it differs and configuring fails, check out 370c439. Menagerie's `panda.xml`, the version with its own hand, has no flange site, so it doesn't configure as is. Use `panda_nohand.xml` with the 2F-85, and record the gripper as a guess.

Derived kinematics has rules: the joints from the base to the flange are the chain, one joint per body, and a hinge's `pos` must be zero. A configured `joints` list must be exactly that chain, base first. The README's "Derived kinematics" lists them. The arm collides as its own MJCF geoms (`collision` is `"mesh"`); only the planner uses the fitted boxes and capsules. Two options for a derived arm:
- `"planning_geometry": "mesh"` gives the planner the arm's collision hulls instead. Use it when the boxes are too fat for a tight cell, or the user wants the viewer to show what MuJoCo collides.
- `"visual_models": true` serves the arm's own visual meshes to the 3D viewer. Turn it on when the user wants to see the arm.

For a custom gripper, run `cell.py measure --gripper <name>` and set the gripper frame's `translation.z` to its `tcp_mm`. A bundled gripper keeps its own frame values on any arm. Copy the gripper component from the example that bundles it: `ur5e-pick.json` for the 2F-85, with its frame `translation.z` 155.8 and its orientation.

### Arm from a URDF or xacro

Clone the vendor repo, then run `viam xacro convert --input-file <xacro> --output-file <name>.urdf [--args k:=v]`. Never pass `--collapse-fixed-joints`. Set `"mjcf": "<abs>.urdf"`, plus `mesh_dir` and `flange_link` as the README's "Bring your own robot" shows for the UR5e. If the flange check errors, it prints the `flange_offset` to set. Planning can be slow with dense meshes.

### Gripper from a URDF or MJCF

Set `"mjcf"` on the gripper. A URDF gripper needs exactly one driven (non-`mimic`) joint, with `<limit effort>`. Anything more complex needs an MJCF. Run `cell.py measure --gripper <name>` and set the frame's `translation.z` to its `tcp_mm`.

### Object from a mesh

Add a prop with `"type": "mesh"`, `"mesh"` set to a URL or absolute path, a `scale` that gives meters (YCB is already in meters), a realistic `mass` (a mug is about 0.3 kg), and `texture` if there is one. Place it in this order:

1. Configure the prop with its `position` z a little above its support, so it exists to measure.
2. Run `cell.py measure --prop <name>`.
3. Check `center_m` against its `position`. A mesh prop's origin is its file's, often at its base or off its axis (the YCB mug's is 17 mm off). `cell.py pick` grasps at the origin. When the two differ in x or y by more than about 5 mm, re-centre a copy of the mesh, and point `mesh` at the copy: `uv run --with trimesh python -c "import sys, trimesh; m = trimesh.load(sys.argv[1], force='mesh'); m.apply_translation(-m.bounds.mean(axis=0)); m.export(sys.argv[2])" in.obj centred.obj`.
4. Set its z to `support_top + (z - (top_z_m - size_m[2]))`, so its lowest point rests on the support. Write the config and `reset`.

Fit check: compare the width the jaws close across with the gripper's open `jaw_gap_mm`, leaving at least 2 mm each side. Measure that width at the height the jaws will close, not just the body's. A handle doesn't count, but a flared rim does. The YCB mug's body is about 82 mm, but its rim flares to about 90 mm, so a top-down grasp around it stalls on the 2F-85's 85 mm jaws. What worked was a pinch of the rim wall nearest the arm, one pad inside the cup, in a script of your own. Turn the prop (`orientation_wxyz`) so a handle points along the jaws' closing direction, not across their path, and confirm in a `GetImages` look. `cell.py pick --size` only shapes what the planner avoids on the approach and carries while held, so pass the body's largest dimension and leave out the handle.

### Scene from an MJCF

Add a `viam-labs:mujoco-sim:scene` component with `"mjcf": "<abs>.xml"`, placed by its frame. Its free bodies become props named `<scene>/<body>`.

## What `measure` returns

`cell.py measure --gripper G` prints `{"tcp_mm", "jaw_gap_mm": {"open", "closed"}, "finger_reach_mm": {"open", "closed"}}`, or for suction `{"cup_diameter_mm", "tcp_mm"}`. `cell.py measure --prop NAME` prints `{prop, size_m: [x, y, z], center_m: [x, y, z], top_z_m}`. It measures a scratch copy, so the live sim isn't disturbed.

`tcp_mm` is the fingertips. The bundled 2F-85 frame uses the pad centre (155.8), about 4.5 mm short of its tips (160.3), so the two differ on purpose. Measured: Lite6's own gripper tcp 81.1, gap 39.4 open and 22.2 closed; 2F-85 gap 85.2 and 0.0, closed reach 13.2; ePick cup 45.0, tcp 130.0.

## Checks, before you compose the task

- **Arm:** `GetMachineStatus` is ready, and a short script (`cell.connect`, then `MotionClient.from_robot(robot, "builtin").move(...)` of the gripper) reaches a pose 30 cm in front of the arm's base, pointing down.
  - A config error naming a body or joint means a chain rule (README, "Derived kinematics"). Fix the file's structure or set `kinematics_url`.
  - A frame check error from derived kinematics means a module bug: set `kinematics_url` and tell the user.
- **Gripper:** `measure` returns sane numbers (a gap of 20 to 150 mm), and a test block lifts with `cell.py pick`.
- **Object:** `cell.py check` reports it at rest at its spawn pose, and `measure` shows it fits the gripper.
- **Scene:** it loads, and every one of its props is at rest.
