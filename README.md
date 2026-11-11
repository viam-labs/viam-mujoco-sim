# viam-mujoco-sim

A [Viam](https://www.viam.com) module that simulates robots in
[MuJoCo](https://mujoco.org): arms and grippers (finger and suction), mobile
bases, arms mounted on bases, RGB-D cameras, 2D and 3D LIDAR, odometry, IMUs,
wrist force-torque sensors, obstacles, and a workcell kit (conveyors, parcel
spawners, photo-eyes, scales). Bundled
robots cover common hardware, and your own MJCF or URDF models load the same
way. Viam's motion planner, 3D scene viewer and SDKs work with it, and
components are placed with the standard Viam frame.

Physics runs on the CPU and the module is a few tens of MB, so it runs on a Mac
as well as on Linux. Cameras render offscreen with OpenGL: CGL on macOS, EGL
(the GPU) on Linux.

## Models

| Model | API | What it does |
|---|---|---|
| [`viam-labs:mujoco-sim:world`](#model-viam-labsmujoco-simworld) | `generic` | Owns the scene, props and step loop. Configure exactly one. |
| [`viam-labs:mujoco-sim:arm`](#model-viam-labsmujoco-simarm) | `arm` | A UFactory Lite6, a UR5e, or your own MJCF or URDF arm, placed by its frame on the world or a base, with joint control. |
| [`viam-labs:mujoco-sim:gripper`](#model-viam-labsmujoco-simgripper) | `gripper` | The Lite6's gripper, a Robotiq 2F-85, or your own MJCF or URDF gripper. |
| [`viam-labs:mujoco-sim:camera`](#model-viam-labsmujoco-simcamera) | `camera` | An RGB-D camera placed by its frame, on the world or an arm's end effector. |
| [`viam-labs:mujoco-sim:base`](#model-viam-labsmujoco-simbase) | `base` | A Viam Rover, a Clearpath Husky, TIAGo++'s base, or your own MJCF or URDF base; ideal motion or driven through its wheels. |
| [`viam-labs:mujoco-sim:lidar`](#model-viam-labsmujoco-simlidar) | `camera` | A 2D or 3D LIDAR on the world, a base or an arm; point clouds only. |
| [`viam-labs:mujoco-sim:movement_sensor`](#model-viam-labsmujoco-simmovement_sensor) | `movement_sensor` | Wheel odometry for a mujoco-sim base, with optional drift. |
| [`viam-labs:mujoco-sim:imu`](#model-viam-labsmujoco-simimu) | `movement_sensor` | An IMU on the world, a base or an arm: orientation, angular velocity and acceleration, with optional noise. |
| [`viam-labs:mujoco-sim:force_torque`](#model-viam-labsmujoco-simforce_torque) | `sensor` | A wrist force-torque sensor at an arm's flange, with optional noise. |
| [`viam-labs:mujoco-sim:obstacle`](#model-viam-labsmujoco-simobstacle) | `generic` | A fixed box, sphere, capsule or mesh that Viam's motion planner also sees. |
| [`viam-labs:mujoco-sim:robot`](#model-viam-labsmujoco-simrobot) | `generic` | A robot file with several limbs, such as the bundled ALOHA bimanual cell or TIAGo++'s torso, arms and head on its base; each limb is a mujoco-sim arm. |
| [`viam-labs:mujoco-sim:scene`](#model-viam-labsmujoco-simscene) | `generic` | An MJCF scene file placed by its frame: simulated, planned against, and its free bodies movable as props. |
| [`viam-labs:mujoco-sim:world_state`](#model-viam-labsmujoco-simworld_state) | `world_state_store` | Streams movable props, parcels, and the conveyor and scale shapes to the Viam app's 3D scene viewer. |
| [`viam-labs:mujoco-sim:conveyor`](#model-viam-labsmujoco-simconveyor) | `motor` | A belt conveyor that carries parcels by friction. |
| [`viam-labs:mujoco-sim:spawner`](#model-viam-labsmujoco-simspawner) | `sensor` | Feeds box parcels of random classes onto a conveyor; `box-detect` compatible. |
| [`viam-labs:mujoco-sim:photo_eye`](#model-viam-labsmujoco-simphoto_eye) | `sensor` | A through-beam photo-eye that counts what passes. |
| [`viam-labs:mujoco-sim:scale`](#model-viam-labsmujoco-simscale) | `sensor` | A platform scale. |

## Getting started

Add the components in the Viam app; it adds the module to your machine's config
for you. In JSON, the module entry is:

```json
{
  "type": "registry",
  "name": "viam-labs_mujoco-sim",
  "module_id": "viam-labs:mujoco-sim",
  "version": "latest"
}
```

Configure one world first, then the arm, gripper and cameras, each naming the
world. [`examples/lite6-pick.json`](examples/lite6-pick.json) is a complete
pick cell, and [`examples/ur5e-pick.json`](examples/ur5e-pick.json) is the same
cell with a UR5e and a Robotiq 2F-85. Its `pick_block` starts parked out of reach at x = 2.5 m on purpose;
move it into reach with the world's `set_prop_pose` DoCommand (see the example
under the world model). [`examples/rover-lidar.json`](examples/rover-lidar.json) is a
rover cell: a rover with a LIDAR inside a walled pen, and
[`examples/rover-nav.json`](examples/rover-nav.json) adds odometry and
nav-stack for SLAM and navigation in it (see
[Mobile base, SLAM and navigation](#mobile-base-slam-and-navigation)).
[`examples/depalletize.json`](examples/depalletize.json) is a suction
de-palletizing cell (see [De-palletizing](#de-palletizing)), and
[`examples/parcel-sort.json`](examples/parcel-sort.json) a parcel-sorting cell
with a conveyor, photo-eye and scale (see [Parcel sorting](#parcel-sorting)), and
[`examples/conveyor-pick.json`](examples/conveyor-pick.json) a UR5e picking
parcels off the running belt (see [Conveyor pick](#conveyor-pick)), and
[`examples/husky-ur5e.json`](examples/husky-ur5e.json) a UR5e riding a Husky
between two tables (see [Mobile manipulation](#mobile-manipulation)), and
[`examples/aloha-handover.json`](examples/aloha-handover.json) the ALOHA, two
arms from one robot file passing a block across a table (see
[Model viam-labs:mujoco-sim:robot](#model-viam-labsmujoco-simrobot)), and
[`examples/tiago-pick.json`](examples/tiago-pick.json) a TIAGo++ carrying a
block from a low table to a shelf (see [Semi-humanoid: TIAGo++](#semi-humanoid-tiago)). With an
arm and gripper configured, the builtin motion service plans for the arm as it
would for a real one.

## Model viam-labs:mujoco-sim:world

Owns the MuJoCo scene: the floor, lighting, props and the step loop. Configure
exactly one; every other component names it in its `world` attribute, so
viam-server starts it first. Sim time is paced to wall time by default.

### Configuration

```json
{
  "name": "sim-world",
  "api": "rdk:component:generic",
  "model": "viam-labs:mujoco-sim:world",
  "attributes": {
    "props": [
      {
        "name": "pick_block",
        "size": 0.0313,
        "position": [0.3, 0, 0.02],
        "mass": 0.03,
        "color": [1, 0, 0],
        "static_friction": 2.0
      }
    ]
  }
}
```

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `physics_dt` | float | No | `0.002` | Physics step in seconds. |
| `real_time` | bool | No | `true` | Pace sim time to wall time. `false` steps as fast as possible. |
| `props` | list | No | `[]` | Movable objects in the scene; see below. |
| `import_geometry` | list | No | `[]` | Names of components, of any module, whose geometry becomes fixed and physical; see [Importing geometry from other modules](#importing-geometry-from-other-modules). |
| `floor_texture` | string | No | | A `.png` or `.jpg` image tiled across the floor: a path, an assets-relative path (the module bundles `textures/concrete.png` and `textures/cardboard.png`) or an https URL. |
| `floor_texture_size` | float | No | `1` | One floor tile's edge in meters. Only with `floor_texture`. |
| `lighting` | object | No | | Lights, headlight, skybox and render quality for cameras; see [Lighting](#lighting). |

Each prop:

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | Yes | | Unique prop name, used by the prop commands. `floor` is reserved. |
| `type` | string | No | `"cube"` | `"cube"`, `"sphere"`, `"cylinder"`, `"capsule"` or `"mesh"`. Each fills a box of `size × scale`: a sphere needs a uniform scale; a cylinder's or capsule's diameter is `size × scale[0]` (equal to `scale[1]`) and its height `size × scale[2]`, tip to tip for a capsule. A capsule's height must exceed its diameter (`scale[2] > scale[0]`). |
| `size` | float | No | `0.05` | Width in meters; see `type` for each shape. |
| `mesh`, `threshold` | string, float | No | | `type` `mesh` only: `mesh` is a path, assets-relative path or https URL; `scale` (number or [x, y, z]) scales the vertices and `size` is ignored; mass is split across the convex pieces by volume. `threshold` is the CoACD concavity threshold, 0.01 to 1 (default `0.05`). |
| `scale` | [x, y, z] | No | `[1, 1, 1]` | Per-axis scale of `size`. A `mesh` prop also takes a single number, and scales its vertices instead. |
| `position` | [x, y, z] | No | `[0, 0, 0]` | Spawn position in meters. |
| `orientation_wxyz` | [w, x, y, z] | No | `[1, 0, 0, 0]` | Spawn orientation. |
| `mass` | float | No | `0.05` | Mass in kg. |
| `color` | [r, g, b] | No | `[0.7, 0.7, 0.7]` | RGB, 0 to 1. |
| `texture` | string | No | | A `.png` or `.jpg` image on each face (projected onto a mesh): a path, an assets-relative path or an https URL. Cameras see the image untinted; `color` is still the 3D viewer's colour. |
| `fixed` | bool | No | `false` | A fixed prop never moves. |
| `static_friction` | float | No | `1.0` | Sliding friction, non-negative. |
| `dynamic_friction` | float | No | | Accepted for compatibility; MuJoCo has one sliding coefficient, so it is ignored. |

Props are invisible to Viam's motion planner. For fixed scene geometry the planner should avoid, use [obstacles](#model-viam-labsmujoco-simobstacle).

### Lighting

`lighting` sets how cameras see the scene. Every key is optional; without it
the world has one directional light straight down (the "sun"), MuJoCo's
headlight, shadows, reflections, 4× anti-aliasing and a black sky.

```json
"lighting": {
  "lights": [
    {"type": "directional", "direction": [0.3, 0.2, -1], "color": [1, 0.95, 0.9]},
    {"type": "spot", "position": [0.5, 0, 1.5], "direction": [0, 0, -1], "cutoff_deg": 40, "shadows": false}
  ],
  "headlight": 0.5,
  "skybox": [[0.6, 0.75, 0.9], [0.2, 0.2, 0.25]]
}
```

| Name | Type | Default | Description |
|---|---|---|---|
| `lights` | list | the sun | The world's lights; setting it replaces the sun. `[]` leaves the world unlit except for the headlight and any scene lights. |
| `keep_default_light` | bool | `false` | Keep the sun when a [scene file](#model-viam-labsmujoco-simscene) brings its own lights (it is dropped otherwise). Not with `lights`. |
| `headlight` | float | `1` | Scales the light that follows each camera; 0 or more; `0` turns it off. |
| `skybox` | [[r, g, b], [r, g, b]] | none (black sky) | A gradient sky, top color then bottom, 0 to 1. A skybox from a scene file (or a robot's MJCF) wins (logged). For an image sky, use a scene file. |
| `shadows` | bool | `true` | Shadows on or off; off renders a little faster. |
| `shadow_size` | int | `4096` | Shadow-map resolution, 1 to 16384; higher is sharper. |
| `reflections` | bool | `true` | Reflections on or off; off renders a little faster. |
| `antialias` | int | `4` | Multisample anti-aliasing samples, 0 to 16; 0 or 1 is off. |

Each light:

| Name | Type | Default | Description |
|---|---|---|---|
| `type` | string | `"directional"` | `"directional"` (parallel rays, like the sun), `"spot"` or `"point"`. |
| `position` | [x, y, z] | see description | Meters, in the world frame. A directional light's shadows are cast around its position, so by default it sits 3 m up along its ray through the world origin (`[0, 0, 3]` for a light straight down); set it above the cell when the cell is far from the origin. Other lights, and a directional light that doesn't point down, default to `[0, 0, 3]`. |
| `direction` | [x, y, z] | `[0, 0, -1]` | Where it points; any length but zero. A point light ignores it. |
| `color` | [r, g, b] | `[0.7, 0.7, 0.7]` | 0 to 1; also its brightness. |
| `shadows` | bool | `true` | Whether it casts shadows. |
| `cutoff_deg` | float | `45` | `"spot"` only: the cone's half-angle, up to 90. |

Lights in a robot's MJCF (an arm, gripper or base) don't drop the sun.

### DoCommand

| Command | Fields | Returns |
|---|---|---|
| `status` | | `{playing, sim_time, real_time_factor, slips, warnings}` |
| `play` / `pause` | | `{"ok": true}` |
| `reset` | | `{"ok": true}`; arms go home and non-fixed props return to their spawn poses. |
| `set_prop_pose` | `prop`, `position`, `orientation_wxyz` | `{prop, position, orientation_wxyz}`; zeroes the prop's velocity. |
| `get_prop_pose` | `prop` | `{prop, pose: {position, orientation_wxyz}}` |
| `park_props` | `props`: list of names | `{poses: {name: {position, orientation_wxyz}}}`; returns props to their spawn poses. |
| `reimport` | | `{imported, from_frame, missing}`: runs `import_geometry` again, replacing the last import. `from_frame` names the imported components whose geometry came from their frame config rather than `GetGeometries`. |

`real_time_factor` is sim seconds per wall second over the last two seconds.
`slips` counts how often the sim fell more than 100 ms behind wall time and
re-anchored. `warnings` holds MuJoCo's warning counts, such as
`mjWARN_BADQACC`.

```json
{"command": "set_prop_pose", "prop": "pick_block", "position": [0.3, 0.05, 0.02], "orientation_wxyz": [1, 0, 0, 0]}
```

### Importing geometry from other modules

`import_geometry` lists components, of any module, whose geometry becomes fixed
and physical in the scene, so props and parcels rest on a
[`viam:workcell-components`](https://app.viam.com/module/viam/workcell-components)
pallet or bounce off a [`viam:obstacle-pack`](https://app.viam.com/module/viam/obstacle-pack)
obstacle:

```json
"attributes": { "import_geometry": ["pallet", "fence"] }
```

After the world starts, it asks viam-server for each component's geometries
(`GetGeometries`) and its pose in the frame system, and adds every box, sphere
and capsule as a fixed geom named `import/<component>/<label or index>`.
They are not dependencies: viam-server would otherwise rebuild the world, and
reset the sim, whenever one of them changed. A component that isn't there yet
(or a lost connection) is retried every 2 s for 60 s, then logged as an error.
mujoco-sim's own components (already in the scene) are skipped. Run `reimport`
after editing an imported component.

When a component doesn't implement `GetGeometries`, or returns no box, sphere
or capsule, the import falls back to the `geometry` in that component's
`frame` config, read from the machine's frame system and placed through the
frame's parent and pose, the same as the motion planner does. So the sim
matches what the motion planner sees. Many `viam:workcell-components` models
(`pallet`, `floor-decal`, ...) don't implement `GetGeometries`, and RDK never
asks a motor or sensor for its geometries either, so give those a frame
`geometry`:

```json
{
  "name": "pallet",
  "api": "rdk:component:generic",
  "model": "viam:workcell-components:pallet",
  "attributes": {},
  "frame": {
    "parent": "world",
    "translation": { "x": 900, "y": 0, "z": 0 },
    "geometry": {
      "type": "box",
      "x": 1200, "y": 800, "z": 144,
      "translation": { "x": 0, "y": 0, "z": 72 }
    }
  }
}
```

The geometry's `translation` is its centre in the component's frame: here the
box sits on the floor. A component with neither (no geometries, no frame
geometry) is logged once, right away, as a warning and skipped, with no retry:

```
import_geometry: pallet has no box, sphere or capsule geometry, from GetGeometries or its frame; skipped. Add a box, sphere or capsule geometry to its frame if it should be an obstacle
```

Which `viam:workcell-components` and `viam:obstacle-pack` parts become physical:

| Part | Imported |
|---|---|
| `workcell-components` `pallet`, `floor-decal` and other models without `GetGeometries` | Only with a frame `geometry` (otherwise a warning). |
| `workcell-components` models that report geometries | Yes. |
| `obstacle-pack` `obstacle` | Yes. |
| `obstacle-pack` `obstacle-from-mesh` | No: meshes aren't imported (warning). Use a mujoco-sim [obstacle](#model-viam-labsmujoco-simobstacle) with the mesh instead. |
| Anything whose shape is only in its `frame.geometry` config | Yes, from the frame. |

The import is a snapshot: moving a component later doesn't move its geoms
until `reimport`.

## Model viam-labs:mujoco-sim:arm

A simulated arm, placed by its standard Viam frame. Two are bundled, both from
the [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie);
any other arm works from its MJCF file and its Viam kinematics file, or from
its URDF alone (see [Your own arm](#your-own-arm)). The arm starts at its
MJCF's `home` keyframe (a URDF arm at all-zero joints). Joints are position
servos; moves are rate limited by `max_joint_deg_s`.

| `asset` | Arm | Home (degrees) | Kinematics |
|---|---|---|---|
| `lite6` | UFactory Lite6 with its wide gripper | `[0, 0, 90, 0, 90, 0]` | [viam-ufactory-xarm](https://github.com/viam-modules/viam-ufactory-xarm) |
| `ur5e` | Universal Robots UR5e, no gripper | `[-90, -90, 90, -90, -90, 0]` | [universal-robots](https://github.com/viam-modules/universal-robots) |

### Configuration

```json
{
  "name": "pick-arm",
  "api": "rdk:component:arm",
  "model": "viam-labs:mujoco-sim:arm",
  "frame": { "parent": "world" },
  "attributes": {
    "world": "sim-world",
    "asset": "lite6",
    "max_joint_deg_s": 30,
    "max_joint_deg_s2": 60
  }
}
```

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `world` | string | Yes | | The world component's name. |
| `asset` | string | One of `asset`, `mjcf`, `robot` | | A bundled arm: `"lite6"` or `"ur5e"`; with `robot`, a bundled limb: `"aloha_left"`, `"aloha_right"`, `"tiago_torso"`, `"tiago_arm_left"`, `"tiago_arm_right"` or `"tiago_head"`. Its defaults apply; any attribute below set in config wins. |
| `robot` | string | One of `asset`, `mjcf`, `robot` | | A [`viam-labs:mujoco-sim:robot`](#model-viam-labsmujoco-simrobot) this arm is a limb of. The arm then drives `joints` in that robot's file and attaches no file of its own. |
| `mjcf` | string | One of `asset`, `mjcf` | | Your arm's MJCF (`.xml`) or URDF (`.urdf`) file: an absolute path, `~/...`, or a path relative to the module's `assets/`. See [From a URDF](#from-a-urdf). |
| `mesh_dir` | string | No | `meshes/` beside the file if it exists, else the file's folder | A `.urdf` `mjcf`'s meshes: absolute, `~`, or relative to the file. `package://` paths are looked up there by file name. |
| `max_joint_deg_s` | float | No | `180` | Per-joint velocity limit on the commanded target, reported by `dynamics_metadata`. |
| `max_joint_deg_s2` | float | No | | Per-joint acceleration limit, reported by `dynamics_metadata` for trajectory planning. |
| `move_timeout_sec` | float | No | `30` | How long a move waits to arrive before it fails. |
| `path_tolerance_delta_deg` | float | No | `0.5` | How close every joint must be for a move to count as arrived. |
| `is_moving_velocity_threshold_rad_s` | float | No | `0.01` | Joints must also be slower than this to count as arrived. |
| `robot_control_freq_hz` | float | No | `50` | Rate at which `MoveThroughJointPositions` pushes targets, up to 1000. |
| `path_max_step_deg` | float | No | `2.0` | Largest per-joint step between pushed targets, up to 30. |
| `kinematics_url` | string | With an `.xml` `mjcf` | the bundled arm's file; a `.urdf` `mjcf` pruned to `flange_link` | The arm's Viam kinematics file, served by `GetKinematics`: `.json` for SVA, `.urdf`, `.xml` or `.xacro` for URDF. `http(s)://`, `file://`, or a path like `mjcf`. Read when the arm is configured. |
| `joints` | [string] | No | inferred | The arm's MJCF joint names, base to flange. |
| `flange_site` | string | No | `"attachment_site"` | The MJCF site at the flange, where Viam's kinematics chain ends. Not for a `.urdf`. |
| `flange_link` | string | With a `.urdf` that has several end links | the file's one end link | The URDF link at the flange. Only for a `.urdf`. |
| `home_keyframe` | string | No | `"home"` | The MJCF keyframe whose `ctrl` is the start pose; all zeros without one. |
| `base_offset` | pose | No | identity | From the arm's frame to the MJCF's world, for an MJCF whose base differs from its kinematics file. `{"translation": {"x", "y", "z"} mm, "orientation": {"o_x", "o_y", "o_z", "theta"} degrees}`. |
| `flange_offset` | pose | No | identity (`lite6`: 180° about Z) | From `flange_site` to Viam's end-effector frame, in the same form. |
| `collision` | string | No | `"kinematics"` with an SVA file that has geometries, else `"mesh"` | What MuJoCo collides the arm links with; see [Collision shapes](#collision-shapes). |
| `position` | [x, y, z] | No | `[0, 0, 0]` | Placement in meters, used when no frame is set. |
| `orientation_wxyz` | [w, x, y, z] | No | | Orientation, used when no frame is set. |
| `orientation_rpy_deg` | [r, p, y] | No | | Same as roll, pitch, yaw in degrees; `orientation_wxyz` wins if both are set. |
| `depth` | bool | No | `false` | `GetImages` also returns a `depth` image. |
| `depth_width` | int | No | half of `width` | Depth image and point cloud width in pixels. |
| `depth_height` | int | No | half of `height` | Depth image and point cloud height in pixels. |

The frame's `parent` is `world` or a base; see [Mounting on a base](#mounting-on-a-base).
An arm with `robot` is framed on its robot or on another arm of it; see
[Limbs](#limbs).

Joint values are degrees, and millimetres for a slide joint (TIAGo's torso
lift). A slide joint moves at up to 100 mm/s (fixed: `max_joint_deg_s` is for
hinges), counts as arrived within 1 mm and slower than 1 mm/s, and pushed
targets step at most 10 mm. The `*_radians` DoCommands stay SI: radians, and
metres for a slide.

### Mounting on a base

Frame the arm on a `viam-labs:mujoco-sim:base` to make a mobile manipulator.
The arm rides the base, and the frame's translation and orientation are its
offset from the base's frame (on the floor under the chassis, +Y forward).
The Husky's top plate is 383 mm up, so this puts a UR5e on it, 250 mm ahead
of the centre:

```json
{
  "name": "arm",
  "api": "rdk:component:arm",
  "model": "viam-labs:mujoco-sim:arm",
  "frame": { "parent": "husky", "translation": { "x": 0, "y": 250, "z": 385 } },
  "attributes": { "world": "sim-world", "asset": "ur5e" }
}
```

The base is a dependency of the arm: viam-server builds it first, and
rebuilds the arm (back at home) when the base is reconfigured.
`GetEndPosition` and `GetKinematics` stay in the arm's own frame wherever the
base drives. Use `drive: ideal` for a base that carries an arm: an arm move
leaves it within a millimetre of where it was. RDK doesn't follow the base
as it drives, so plan arm moves in the base's frame; see
[Mobile manipulation](#mobile-manipulation) and
[Things to know](#things-to-know).

### Supported API

| Method | Notes |
|---|---|
| `GetJointPositions` | Degrees (mm for a slide joint). |
| `MoveToJointPositions` | Returns when the arm arrives, or fails after `move_timeout_sec`. |
| `MoveThroughJointPositions` | Pushes waypoints at `robot_control_freq_hz`; used by the motion service. With extra `{"waitAtEnd": false}` (as the motion service's teleop sends) it returns at once and the next move continues from the commanded pose. |
| `MoveThroughJointPositionsStreamed` | Follows timestamped points in sim time; each batch plays straight after the last. |
| `GetEndPosition` | Viam's end-effector frame (the flange with `flange_offset`) in the arm's frame, in mm. |
| `GetKinematics` | The `kinematics_url` file; for a `.urdf` `mjcf` without one, the URDF from its root to `flange_link` and its collision meshes. |
| `IsMoving` | True while a `MoveToJointPositions` or `MoveThroughJointPositions` call is in flight. |
| `Stop` | Holds the current target, or the measured position if it interrupted a trajectory. |
| `MoveToPosition` | Not supported; use the motion service. |

### DoCommand

| Command | Fields | Returns |
|---|---|---|
| `dynamics_metadata` | | `{source, joint_names, max_velocity_rad_s, max_acceleration_rad_s2}` (m/s for a slide joint); the last only when `max_joint_deg_s2` is set. |
| `get_joint_positions_radians` | | `{values: [radians (m for a slide), one per joint]}` |
| `set_joint_targets_radians` | `values`: radians (m for a slide), one per joint | `{"accepted": true}`; latches the target and returns at once. |

### Your own arm

Give the arm `mjcf` and `kinematics_url` instead of `asset`. Everything else is
read from the two files. An arm described by a URDF needs only the URDF; see
[From a URDF](#from-a-urdf).

- **Joints.** The first N actuated hinge or slide joints in the MJCF's body
  tree, where N is the kinematics file's joint count. They must form one chain
  from the base; otherwise configuring fails and lists what it found. Set
  `joints` to choose them yourself.
- **Actuators.** The actuator whose transmission targets each joint, whatever
  its name. Each must be a position actuator (`<position>` or
  `<general biastype="affine">`) with gear 1 and gain equal to `kp`, because
  the arm writes joint positions to `ctrl`.
- **Home.** The `home_keyframe` keyframe's `ctrl` for those actuators, or its
  joint `qpos` when the keyframe sets no `ctrl`.
- **Gripper.** An actuator in the arm MJCF that drives none of the arm joints
  is its integrated gripper (see the gripper model).

Configuring also checks that the two files describe the same arm: the MJCF's
`flange_site`, with the offsets, must land within 5 mm and 2° of the
kinematics file's end effector at all-zero joints, at home, and at a mixed
pose. If not, the error names the joint position and the disagreement, and
when one fixed transform at the flange (or the base) explains it, it prints
the `flange_offset` (or `base_offset`) to set:

```
arm pick-arm: the MJCF flange and the kinematics file's end effector disagree by 0.7 mm and 180.0 deg at the mixed joint position [...] ...; try flange_offset: {"translation": {"x": 0.0, "y": 0.0, "z": -0.6}, "orientation": {"o_x": 0.0, "o_y": 0.0, "o_z": 1.0, "theta": -180.0}}
```

Vendor and Menagerie link lengths differ by about a millimetre, which passes.
Paths can point anywhere on the machine, including a registry package's
directory; a relative path is under this module's `assets/`.

### From a URDF

Point `mjcf` at the arm's `.urdf` (see [Bring your own robot](#bring-your-own-robot)
to get one from a xacro) and leave out `kinematics_url`:

- **Kinematics.** `GetKinematics` serves the URDF pruned to the links and
  joints from its root to `flange_link`, with its collision meshes (STL or
  PLY), so Viam's planner and the sim read the same file. A file with one end
  link (a link no joint hangs off) needs no `flange_link`; with several,
  configuring fails and lists them. An explicit `kinematics_url` still wins:
  an SVA file needs `joints` set, and a URDF one can't have mesh collisions
  (the planner gets no mesh bytes with it).
- **Joints and servos.** The arm's joints are the kinematics' joints. Each
  gets a position servo that reaches the joint's `<limit effort>` at about 2°
  of error, damped by the joint's inertia; a joint without an effort limit is
  an error. Every link is gravity-compensated.
- **Flange.** The module adds `attachment_site` at `flange_link`; a gripper
  attaches there.
- **Collision.** The URDF's own collision geoms. Links already touching at
  home are excluded from colliding with each other (a vendor's base and
  shoulder hulls often overlap a little, which would jam the first joint).
- **Planning speed.** The planner checks the URDF's full collision meshes,
  which can make planning slow with dense meshes. An SVA `kinematics_url`
  with simple shapes (and `joints`) plans faster.

### Collision shapes

With an SVA kinematics file that gives its links geometry (both bundled arms'
do), MuJoCo collides the arm with the same boxes, spheres and capsules Viam's
planner uses: each one is added to the MuJoCo body that carries it, named
`<arm>/viam_<link>`, and all of that body's own collision geoms are turned off.
The MJCF's link bodies must be named. Links with only a placeholder (smaller
than 5 mm, like the Lite6 base's 1 mm sphere) keep their MJCF collision geoms,
and a gripper in its own bodies is never changed. A gripper whose geoms sit
directly on the last link's body loses them too, so use `collision` `"mesh"`
for such an arm. Pairs of
links already touching at home are excluded from collision, as the planner
ignores collisions present at its start.

Where a planner shape is slimmer than the link, the visual mesh can sink a few
millimetres into an obstacle; where it is fatter, the arm stops short. Set
`collision` to `"mesh"` to collide the MJCF's own geoms instead.

## Model viam-labs:mujoco-sim:gripper

A gripper on a mujoco-sim arm, driven by one actuator: the arm MJCF's own
(the Lite6's), a bundled gripper, or your own gripper MJCF, which is attached
at the arm's `flange_site`. It is binary: `Grab` and `Open` latch the command
and return before the jaws move, like the real
Lite6 gripper. A suction gripper (an `<adhesion>` actuator) has no jaws: `Grab` turns the
vacuum on and `Open` turns it off. Adhesion pulls only through the cup's
contacts, so a violent move can tear the object off.

| `asset` | Gripper | On | Frame `translation.z` (TCP) | Frame `geometry` covering it |
|---|---|---|---|---|
| `lite6_gripper` (default) | The Lite6's own, in its MJCF | `lite6` | 81.1 mm | box 78 × 78 × 70 mm at `z: -35` |
| `robotiq_2f85` | Robotiq 2F-85 | `ur5e` or your own arm | 155.8 mm | box 150 × 80 × 170 mm at `z: -73` (fingers open along x) |
| `robotiq_epick` | Robotiq ePick (suction, one 45 mm cup) | `ur5e` or your own arm | 130 mm | box 90 × 90 × 130 mm at `z: -65` |
| `aloha` | ALOHA's own, in its robot file (open and closed swapped to match its actuator) | `aloha_left`, `aloha_right` | 0 mm (the arm's end effector is the gripper) | none: the arm's kinematics carry it |
| `tiago` | TIAGo++'s own, two coupled fingers on one actuator, in its robot file (open 45 mm a finger; open and closed swapped) | `tiago_arm_left`, `tiago_arm_right` | 0 mm (the arm's end effector is between the fingertips) | none: the arm's kinematics carry it |

Give the gripper's frame that geometry so the planner keeps the whole gripper
clear of obstacles; without it, a path can pass the arm but clip the gripper.
The 2F-85 and the ePick also switch the whole scene to elliptic friction cones
with `impratio` 10 (their own MJCF options, which grasping relies on) while
configured. The ePick's cup pulls with up to 100 N; it carries a 1 kg carton
through moves streamed at `max_joint_deg_s` 30, as MoveThroughJointPositions
streams them.

### Configuration

```json
{
  "name": "pick-gripper",
  "api": "rdk:component:gripper",
  "model": "viam-labs:mujoco-sim:gripper",
  "frame": {
    "parent": "pick-arm",
    "translation": { "x": 0, "y": 0, "z": 81.1 }
  },
  "attributes": {
    "world": "sim-world",
    "arm": "pick-arm",
    "sensorless": true
  }
}
```

The frame's `parent` is the arm. Its `z` puts the planner's tool point at the
fingertip center: use the TCP value from the table above, or for your own

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `world` | string | Yes | | The world component's name. |
| `arm` | string | Yes | | The arm this gripper belongs to. |
| `asset` | string | No | `"lite6_gripper"` | A bundled gripper, see above. An attached gripper on an arm whose MJCF already has one (the `lite6`) is an error. |
| `mjcf` | string | No | | Instead of `asset`: your gripper's MJCF (`.xml`) or URDF (`.urdf`), attached at the arm's flange site; a path like the arm's `mjcf`. An MJCF needs exactly one actuator, and its root bodies and jaw joint must be named; a suction gripper has an `<adhesion>` actuator on its cup body instead of a jaw. A URDF needs exactly one movable joint that isn't a `<mimic>`, with `<limit effort>` and a range: the module drives it with a position servo sized like a URDF arm's, and `<mimic>` fingers follow it. Use an MJCF for other designs. |
| `mesh_dir` | string | No | `meshes/` beside the file if it exists, else the file's folder | A `.urdf` `mjcf`'s meshes, as for the arm. |
| `actuator` | string | No | inferred | The gripper's actuator. |
| `open`, `closed` | float | No | the low and high ends of the actuator's `ctrlrange` | Position actuators: the `ctrl` targets. Force actuators (`<motor>`): their signs pick the direction. An error for suction. |
| `sensorless` | bool | No | `false` | `IsHoldingSomething` reports the latched grip instead of contacts, like the real Lite6 gripper. |
| `grip_force_n` | float | No | `5` | Force grippers: the closing and opening force, within the actuator's `ctrlrange` (the Lite6's is 10). Suction: the adhesion force, up to the actuator's gain times its `ctrlrange` maximum (the default). |

### Supported API

| Method | Notes |
|---|---|
| `Open` | Latches open and returns at once. |
| `Grab` | Latches closed and returns `true` at once. |
| `Stop` | Force grippers release the force; position grippers hold the jaw where it is; suction keeps the vacuum on. |
| `IsMoving` | True while the jaws move; always false for suction. |
| `IsHoldingSomething` | See below. |
| `GetCurrentInputs` / `GoToInputs` | One value: `0` open, `1` closed. |
| `GetKinematics` | Not supported, deliberately; the planner treats the gripper as a point. |

`IsHoldingSomething` always fills `meta`:

| Key | Meaning |
|---|---|
| `firm_grasp` | The gripper touches one object from two opposite sides. Suction: the vacuum is on and the cup touches an object that can move (a prop, not an obstacle). |
| `input` | The latched command, `0` or `1`. |
| `vacuum` | Suction only, instead of the jaw keys: the vacuum, 0 to 1 of the actuator's maximum. |
| `jaw` | The jaw joint's position, from `open` to `closed`: meters for a slide joint (the Lite6), radians for a hinge (the 2F-85). |
| `open`, `closed` | The jaw joint's limits, in the same units. |
| `jaw_m`, `open_m`, `closed_m` | The same three in meters, for slide jaws only (the Lite6). |

Its top-level answer is the latched grip when `sensorless` is true, and
`firm_grasp` otherwise. To wait for a grasp, poll until `meta.firm_grasp` is
true.

## Model viam-labs:mujoco-sim:camera

An RGB-D camera rendered offscreen. Place it on the world with its standard
frame and aim it at a point with `target`, or mount it on an arm's end effector
by making the arm its frame parent.

### Configuration

```json
{
  "name": "scene-cam",
  "api": "rdk:component:camera",
  "model": "viam-labs:mujoco-sim:camera",
  "frame": {
    "parent": "world",
    "translation": { "x": 1200, "y": 1200, "z": 1000 }
  },
  "attributes": {
    "world": "sim-world",
    "target": [0.3, 0, 0.15],
    "width": 1280,
    "height": 720
  }
}
```

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `world` | string | Yes | | The world component's name. |
| `target` | [x, y, z] | No | `[0, 0, 0]` when nothing sets the orientation | Point to aim at, in meters. Overrides the frame's orientation. |
| `fov_deg` | float | No | `60` | Vertical field of view, between 0 and 180. |
| `width` | int | No | `640` | Image width in pixels. |
| `height` | int | No | `480` | Image height in pixels. |
| `position` | [x, y, z] | No | `[1, 1, 1]` | Placement in meters, used when no frame is set. |
| `orientation_wxyz` | [w, x, y, z] | No | | Orientation, used when no frame is set. |
| `orientation_rpy_deg` | [r, p, y] | No | | Same as roll, pitch, yaw in degrees; `orientation_wxyz` wins if both are set. |

The frame's `parent` is `world` or another mujoco-sim component: an arm's end
effector or a base. A frame orientation is read in Viam's camera convention:
+Z forward, +Y down.

### Wrist camera

With an arm as its frame parent, the camera rides that arm's end effector, and
its frame is relative to the end-effector frame the frame system uses. With no
orientation it looks along the tool axis. `target` is rejected, because it
names a fixed point in the world. This one sits 60 mm to the side of the jaws
and 30 mm down the tool axis:

```json
{
  "name": "wrist-cam",
  "api": "rdk:component:camera",
  "model": "viam-labs:mujoco-sim:camera",
  "frame": {
    "parent": "pick-arm",
    "translation": { "x": 0, "y": 60, "z": 30 }
  },
  "attributes": { "world": "sim-world", "fov_deg": 70, "depth": true }
}
```

Mount it on the arm, not the gripper: the module doesn't know the gripper
frame's offset. Until the arm is configured, the camera's methods return an
error saying the camera is not in the scene.

### Supported API

| Method | Notes |
|---|---|
| `GetImages` | A JPEG named after the component and, with `depth`, a `depth` image in Viam's raw depth format (millimeters, 0 where nothing is in view). Both come from the same sim step. `filter_source_names` picks either one. Raw depth is uncompressed, 2 bytes a pixel: 150 KB a frame at the default 320 x 240, 600 KB at 640 x 480. |
| `GetProperties` | Intrinsics of the color image, computed from `fov_deg` and the image size. The depth image has the same vertical field of view at its own size. |
| `GetPointCloud` | Colored binary PCD in meters, in the camera's frame. Pixels with nothing in view are left out. Works whether or not `depth` is set. |

## Model viam-labs:mujoco-sim:base

A differential-drive or skid-steer base: a bundled Viam Rover 2 or Clearpath
Husky, or your own from an MJCF or URDF file. Its chassis and driven wheels
are read from the file. It drives either ideally, exactly as commanded with
the wheels turning to match, or through its wheels on the floor.

| Asset | What it is |
|---|---|
| `viam_rover` | A Viam Rover 2 with the dimensions from Viam's rover fragment: 381 mm wheel circumference, 356 mm track, two drive wheels and a rear caster. 0.762 m/s, 245 °/s. |
| `clearpath_husky` | A Clearpath Husky A200 from Clearpath's `husky_description` (BSD): four wheels, skid-steer, 990 x 571 mm, 56.6 kg. Clearpath's control limits: 1.0 m/s, 115 °/s (2.0 rad/s), 3.0 m/s², 6.0 rad/s². |
| `pal_tiago_base` | TIAGo++'s base (PAL Robotics, from the MuJoCo Menagerie, Apache-2.0), for the `tiago` robot to ride: 63.5 kg, 0.381 m/s and 98 °/s from its wheels' limits, `drive: ideal`. Its four omni wheels drive as a diff-drive: no sideways motion. |

### Configuration

```json
{
  "name": "rover",
  "api": "rdk:component:base",
  "model": "viam-labs:mujoco-sim:base",
  "frame": { "parent": "world" },
  "attributes": {
    "world": "sim-world",
    "asset": "viam_rover"
  }
}
```

A Husky driven through its wheels:

```json
{
  "name": "husky",
  "api": "rdk:component:base",
  "model": "viam-labs:mujoco-sim:base",
  "frame": { "parent": "world" },
  "attributes": {
    "world": "sim-world",
    "asset": "clearpath_husky",
    "drive": "wheels"
  }
}
```

[`examples/husky-lidar.json`](examples/husky-lidar.json) puts it in a walled
yard with a LIDAR on its top plate, odometry and an overview camera.

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `world` | string | Yes | | The world component's name. |
| `asset` | string | One of `asset`, `mjcf` | | A bundled base: `"viam_rover"` or `"clearpath_husky"`. |
| `mjcf` | string | One of `asset`, `mjcf` | | Your base's `.xml` (MJCF) or `.urdf` file: absolute, `~`, or relative to this module's `assets/`. |
| `mesh_dir` | string | No | `meshes/` beside the file if it exists, else the file's folder | Where a URDF's meshes are: absolute, `~`, or relative to the file. `package://` paths are looked up there by file name. |
| `drive` | string | No | `"ideal"` | `"ideal"` or `"wheels"`; see [Drive modes](#drive-modes). |
| `chassis` | string | No | the file's one top-level body | The chassis body, as named in the file. |
| `wheels` | object | No | inferred | The driven wheel joints, `{"left": ["joint", ...], "right": ["joint", ...]}`, as named in the file. |
| `forward_axis` | string | No | inferred | The chassis's forward axis: `"+x"`, `"-x"`, `"+y"` or `"-y"`. |
| `wheel_radius_m` | float | No | inferred | Wheel radius in meters. |
| `max_linear_mps` | float | Asset's, else from the file | | Top speed in meters per second; scales `SetPower`. Required when neither the asset nor the wheels' velocity actuators give it. |
| `max_angular_dps` | float | Asset's, else from the file | | Top yaw rate in degrees per second; scales `SetPower`. Same default. |
| `move_timeout_margin_sec` | float | No | `2` | Sim time `MoveStraight` and `Spin` may go without getting closer before they fail as stalled. |
| `position` | [x, y, z] | No | `[0, 0, 0]` | Placement in meters, used when no frame is set. |
| `orientation_wxyz` | [w, x, y, z] | No | | Orientation, used when no frame is set. |
| `orientation_rpy_deg` | [r, p, y] | No | | Same as roll, pitch, yaw in degrees; `orientation_wxyz` wins if both are set. |

A base carrying a robot or an arm keeps its own mass and yaw inertia: the
ideal drive is sized from the base's file alone. TIAGo's 28 kg upper body
(three times the base's yaw inertia) still drives and turns within its move
timeouts; a much heavier load may lag.

The frame's `parent` must be `world`. The base's frame is on the floor under
the chassis origin, +Y forward, +Z up, as in Viam's base frame, whatever axes
the file uses: a ROS base (+X forward) is turned a quarter turn left to fit.
Cameras, LIDARs and odometry all use this frame.

### Drive modes

- **`ideal`** (default). The chassis rides x and y slides and a yaw hinge,
  driven by stiff velocity actuators, so it goes exactly where it is told and
  can't tip, slip or fall. The file's actuators are dropped; the wheels and
  casters collide with nothing and turn by the distance their side travelled.
  The chassis's own geoms still collide: it pushes props and stops at walls.
- **`wheels`**. The chassis is free, and each driven wheel has a velocity
  actuator: the file's, or one the module adds with its torque capped at the
  wheel's share of traction (μ × mass × g × radius / wheel count). `SetPower`
  and `SetVelocity` turn the wheels at `(v ∓ ω × track / 2) / radius` and
  nothing more, so a skid-steer base under-turns, as the real one does (the
  Husky turns at about a tenth of the commanded rate). `MoveStraight` and
  `Spin` close the loop on the measured pose: they brake on a ramp into the
  target, `MoveStraight` holds its heading, and the yaw rate gets integral
  feedback, so a slipping base still turns at about the commanded rate. A
  file actuator's `ctrlrange` caps wheel speed, and so can cap how fast a
  skid-steer spins.

  The floor's friction is 1, and MuJoCo uses the larger of two touching
  geoms' frictions unless one has a higher `priority`. To give wheels their
  own friction, set `priority="1"` on their geoms, as the bundled bases do.

In both modes a move succeeds when the base reaches or passes its target
(measured along the starting heading, or in yaw), and fails as stalled when
the remaining distance or angle hasn't shrunk for `move_timeout_margin_sec`
of sim time.

### How the file is read

1. **Chassis.** The worldbody's one top-level body (a URDF's root link), or
   `chassis`. Its free joint and its pose in the file are dropped: the frame
   places it. Bodies fixed to it without joints (a URDF's fixed links) count
   as chassis.
2. **Driven wheels.** Hinge joints on bodies attached to the chassis whose
   axle is horizontal. If the file has actuators on joints, only actuated
   hinges count. A caster wheel under a swivel hangs off the swivel, so it
   isn't driven. All axles must be parallel.
3. **Forward.** Axles along the chassis's y mean +X forward (ROS); along x,
   +Y forward (Viam). A base that drives the other way sets `forward_axis`.
4. **Sides, track and radius.** A wheel is on the left when its joint is
   left of the chassis origin; each side needs one. The track is the distance
   between the sides' mean joint positions. The radius is the wheel body's
   cylinder or sphere, else its mesh's furthest point from the axle; all
   wheels must agree within 5%.
5. **Placement.** The chassis is lifted so its lowest collision geom (wheels
   and casters included) touches the floor.
6. **Speeds.** From the attributes, else the asset, else the driven wheels'
   velocity actuators' `ctrlrange` (ctrl × radius, and 2 × ctrl × radius /
   track).

The base logs its mass, wheels, sides, radius, track and forward axis when it
is built. Configuring fails, naming the attribute that fixes it, when the
file isn't `.xml` or `.urdf` or can't be read, a URDF mesh isn't in
`mesh_dir`, there isn't exactly one top-level body (`chassis`), a side has no
wheels or a `wheels` name isn't a hinge joint (`wheels`, with the hinge joints
listed), the axles aren't parallel (`wheels`, with each axle), forward can't be
told (`forward_axis`), a wheel's radius can't be found or the radii disagree
(`wheel_radius_m`), a driven wheel's own actuator isn't a velocity actuator
with `drive: wheels`, or no speed limits are given
(`max_linear_mps`/`max_angular_dps`).

### Your own base

See [Bring your own robot](#bring-your-own-robot) for the Husky from
Clearpath's xacro.

### Camera on the rover

A camera framed on the rover with no orientation looks along the rover's +Z,
straight up. To look forward, rotate the camera -90 degrees about X, which turns
its +Z (forward) to the rover's +Y and its +Y (image down) to the rover's -Z.
This one sits 120 mm ahead of the origin and 150 mm up:

```json
{
  "name": "rover-cam",
  "api": "rdk:component:camera",
  "model": "viam-labs:mujoco-sim:camera",
  "frame": {
    "parent": "rover",
    "translation": { "x": 0, "y": 120, "z": 150 },
    "orientation": {
      "type": "quaternion",
      "value": { "w": 0.7071068, "x": -0.7071068, "y": 0, "z": 0 }
    }
  },
  "attributes": { "world": "sim-world" }
}
```

### Supported API

| Method | Notes |
|---|---|
| `MoveStraight` | Returns when the base has gone the distance; a zero distance returns at once, and a zero velocity is rejected. Fails as stalled (see [Drive modes](#drive-modes)), and when stopped, superseded, reset or closed. |
| `Spin` | Same as `MoveStraight`, for an angle: a zero angle returns at once and a zero velocity is rejected. |
| `SetVelocity` | `linear.y` in mm/s and `angular.z` in degrees/s; other axes are ignored. Clamped to the maxima. |
| `SetPower` | `linear.y` and `angular.z`, each in [-1, 1] of the maxima. |
| `Stop` | Stops the base. |
| `IsMoving` | True while the base is commanded to move or is moving. |
| `GetProperties` | Width is the track, 0 turning radius, wheel circumference 2π × radius (Rover: 0.356 m and 0.381 m; Husky: 0.555 m and 1.037 m). |
| `GetGeometries` | One box around the chassis's collision geoms, not the wheels or casters, in the base's frame (Rover: 260 x 300 x 100 mm, centered 80 mm up). |

## Model viam-labs:mujoco-sim:lidar

A 2D LIDAR, like the rplidar module, or a multi-channel 3D one: it ray-casts a
scan and serves it as a point cloud. It is a camera that returns only point
clouds.

### Configuration

```json
{
  "name": "lidar",
  "api": "rdk:component:camera",
  "model": "viam-labs:mujoco-sim:lidar",
  "frame": {
    "parent": "rover",
    "translation": { "x": 0, "y": 0, "z": 160 }
  },
  "attributes": {
    "world": "sim-world"
  }
}
```

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `world` | string | Yes | | The world component's name. |
| `range_min_m` | float | No | `0.15` | Nearer hits are dropped. |
| `range_max_m` | float | No | `12` | Farther hits are dropped. |
| `angular_resolution_deg` | float | No | `1` | Ray spacing, 0.1 to 360. Rays are spread evenly round the full circle. |
| `channels` | int | No | `1` | Rings of rays, 1 to 128. More than 1 makes a 3D LIDAR. |
| `vertical_fov_deg` | [min, max] | No | `[0, 0]` | The lowest and highest ring's elevation in degrees above the scan plane, each within ±90. Rings are spread evenly between them; one ring uses the midpoint. |
| `position` | [x, y, z] | No | `[0, 0, 0.2]` | Placement in meters, used when no frame is set. |
| `orientation_wxyz` | [w, x, y, z] | No | | Orientation, used when no frame is set. |
| `orientation_rpy_deg` | [r, p, y] | No | | Same as roll, pitch, yaw in degrees; `orientation_wxyz` wins if both are set. |

The frame's `parent` is `world`, a mujoco-sim base or an arm. Like the camera,
it can't be framed on a gripper or another camera. The scan plane is the
frame's XY plane, and ray 0 points along +X. A mounted LIDAR never sees the
body it is mounted on (a base's chassis or an arm's last link) or anything
attached below it, such as a base's wheels and casters or a gripper.

A 3D LIDAR, like a 16-channel unit with a 30° vertical field of view:

```json
"attributes": {
  "world": "sim-world",
  "channels": 16,
  "vertical_fov_deg": [-15, 15],
  "angular_resolution_deg": 0.4,
  "range_max_m": 100
}
```

A scan casts `channels` × (360 / `angular_resolution_deg`) rays, at most
65,536 (a config error above that). The scan runs on the sim thread: about
65,000 rays take about 3.4 ms on a rover, longer where many rays hit meshes.

### Supported API

| Method | Notes |
|---|---|
| `GetPointCloud` | Binary PCD with `x y z` in meters, in the LIDAR's frame. `z` is 0 for a 2D LIDAR. Only hits are included. |
| `GetProperties` | `supports_pcd` is true. |
| `GetImages` | Not supported. |

## Model viam-labs:mujoco-sim:movement_sensor

Wheel odometry for a mujoco-sim base, like RDK's `wheeled-odometry`. It reads
the base's motion (its frame's velocity) from the sim and dead-reckons a pose
every physics step, so readings don't depend on how often you poll. It is
exact unless you add drift, in either drive mode.

### Configuration

```json
{
  "name": "odometry",
  "api": "rdk:component:movement_sensor",
  "model": "viam-labs:mujoco-sim:movement_sensor",
  "attributes": {
    "world": "sim-world",
    "base": "rover"
  }
}
```

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `world` | string | Yes | | The world component's name. |
| `base` | string | Yes | | The mujoco-sim base it rides on. |
| `linear_scale_error` | float | No | `0` | Reported speed is true speed × (1 + e), so `0.02` reads 2% long. Greater than -1. |
| `heading_drift_dps` | float | No | `0` | A constant yaw-rate bias in degrees per second, like a gyro's. It accumulates while parked too. |

Odometry starts at zero where the base is when the sensor starts, on a world
`reset`, or on `reset_odometry`: +Y is the base's forward there, +X its right,
yaw counter-clockwise. A config change restarts the sensor, so it zeroes too.

### Supported API

| Method | Notes |
|---|---|
| `GetLinearVelocity` | Body frame, m/s: `y` forward, `x` right. |
| `GetAngularVelocity` | `z`, degrees per second, counter-clockwise. |
| `GetOrientation` | Yaw only: orientation vector `(0, 0, 1)` with `theta` in degrees. |
| `GetCompassHeading` | Degrees clockwise from the starting heading. |
| `GetPosition` | Like `wheeled-odometry`: a point offset from latitude 0, longitude 0 by the odometry position (+Y north). Pass `extra: {"return_relative_pos_m": true}` for meters instead (`latitude` = y, `longitude` = x). Altitude is 0. |
| `GetReadings` | All of the above. |
| `GetProperties` | Everything except linear acceleration. |
| `GetLinearAcceleration` | Not supported. |

Readings error until the configured base is in the scene.

### DoCommand

`{"command": "reset_odometry"}` zeroes the pose.

For SLAM and navigation with nav-stack, see
[Mobile base, SLAM and navigation](#mobile-base-slam-and-navigation).

## Model viam-labs:mujoco-sim:imu

An inertial measurement unit: orientation, angular velocity and linear
acceleration, from MuJoCo's gyro and accelerometer at the sensor's frame.
Readings are exact unless you add noise.

### Configuration

```json
{
  "name": "torso-imu",
  "api": "rdk:component:movement_sensor",
  "model": "viam-labs:mujoco-sim:imu",
  "frame": {
    "parent": "husky",
    "translation": { "x": 0, "y": 0, "z": 300 }
  },
  "attributes": {
    "world": "sim-world",
    "gyro_noise_dps": 0.05,
    "accel_noise_mps2": 0.02,
    "seed": 1
  }
}
```

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `world` | string | Yes | | The world component's name. |
| `orientation_noise_deg` | float | No | `0` | Each orientation is turned by a random rotation of about this many degrees (the standard deviation). |
| `gyro_noise_dps` | float | No | `0` | Angular-velocity noise, a standard deviation in degrees per second on each axis. |
| `gyro_bias_dps` | [x, y, z] | No | `[0, 0, 0]` | A constant angular-velocity offset, degrees per second. |
| `accel_noise_mps2` | float | No | `0` | Acceleration noise, a standard deviation in m/s² on each axis. |
| `accel_bias_mps2` | [x, y, z] | No | `[0, 0, 0]` | A constant acceleration offset, m/s². |
| `seed` | int | No | `0` | Seeds the noise, so a run repeats; 0 or more. |
| `position`, `orientation_wxyz`, `orientation_rpy_deg` | | No | `[0, 0, 0]`, identity | Placement used when no frame is set. |

The frame's `parent` is `world`, a mujoco-sim base or an arm (its end
effector), as for the LIDAR.

### Supported API

| Method | Notes |
|---|---|
| `GetOrientation` | The sensor frame's orientation in the world frame, as an orientation vector in degrees. |
| `GetAngularVelocity` | Degrees per second, in the sensor's frame. |
| `GetLinearAcceleration` | m/s² in the sensor's frame, including gravity as a real accelerometer reports it: about +9.81 on the axis pointing up when it is still. |
| `GetReadings` | All three, from one physics step. |
| `GetProperties` | Orientation, angular velocity and linear acceleration. |
| `GetPosition`, `GetLinearVelocity`, `GetCompassHeading` | Not supported. |

Readings error until the frame's parent is in the scene.

## Model viam-labs:mujoco-sim:force_torque

A six-axis force-torque sensor at an arm's flange, like a wrist F/T sensor
between the arm and its gripper. It adds no length, mass or collision shape, so
the gripper and the arm's kinematics don't change. It measures what the tool
(the gripper, what it holds and what it touches) exerts on the flange, finger
and suction grippers alike.

### Configuration

```json
{
  "name": "wrist-ft",
  "api": "rdk:component:sensor",
  "model": "viam-labs:mujoco-sim:force_torque",
  "frame": { "parent": "pick-arm" },
  "attributes": {
    "world": "sim-world",
    "force_noise_n": 0.1,
    "torque_noise_nm": 0.005
  }
}
```

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `world` | string | Yes | | The world component's name. |
| `filter_s` | float | No | `0.02` | Low-pass filter time constant in seconds. `0` reads each physics step's raw value. |
| `force_noise_n` | float | No | `0` | Force noise, a standard deviation in N on each axis. |
| `torque_noise_nm` | float | No | `0` | Torque noise, a standard deviation in N·m on each axis. |
| `seed` | int | No | `0` | Seeds the noise, so a run repeats; 0 or more. |

Set the frame's `parent` to a mujoco-sim arm, with no translation or
orientation: the sensor's frame is then the arm's end-effector frame, which is
where the frame system and 3D scene put it.

### Supported API

| Method | Notes |
|---|---|
| `GetReadings` | `force` `{x, y, z}` in N and `torque` `{x, y, z}` in N·m, in the arm's end-effector frame and about its origin, minus any tare; `tared`. A hanging gripper reads its weight along +Z, the direction out of the flange. |

Readings error until the arm is in the scene.

### DoCommand

| Command | Returns |
|---|---|
| `tare` | `{"tared": true}`; later readings subtract the current one (the gripper's own weight, say). Reconfiguring the arm restarts the sensor, so tare again after. |
| `untare` | `{"tared": false}` |

## Model viam-labs:mujoco-sim:obstacle

A fixed box, sphere, capsule or mesh, such as a table, wall, bin or fixture. Its
shape is the frame's `geometry` or a `mesh`. RDK puts the shape in the frame
system, so Viam's motion planner avoids the same shape MuJoCo simulates:
configure it once.

### Configuration

```json
{
  "name": "table",
  "api": "rdk:component:generic",
  "model": "viam-labs:mujoco-sim:obstacle",
  "frame": {
    "parent": "world",
    "translation": { "x": 400, "y": 0, "z": 0 },
    "geometry": { "type": "box", "x": 600, "y": 800, "z": 20,
                  "translation": { "x": 0, "y": 0, "z": -10 } }
  },
  "attributes": { "world": "sim-world", "color": [0.6, 0.5, 0.4] }
}
```

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `world` | string | Yes | | The world component's name. |
| `color` | [r, g, b] | No | `[0.7, 0.7, 0.7]` | RGB, 0 to 1. |
| `static_friction` | float | No | `1.0` | Sliding friction, non-negative. |
| `texture` | string | No | | A `.png` or `.jpg` image on each face, as for props. Cameras see it; the 3D viewer uses `color`. |
| `mesh` | string | No | | A mesh file (STL, OBJ or PLY) instead of `frame.geometry`: a path, an assets-relative path or an https URL. |
| `scale` | float or [x, y, z] | No | `1` | Scales the mesh vertices. Mesh only. |
| `threshold` | float | No | `0.05` | CoACD concavity threshold, 0.01 to 1; lower gives more, tighter pieces and a slower first load. Mesh only. |

The frame's `parent` must be `world`, and it needs a `geometry` or a `mesh`, not
both. A geometry is `box` (`x`,
`y`, `z` in mm), `sphere` (`r`) or `capsule` (`r`, and `l` tip to tip along the
geometry's Z; `l` must be at least `2r`). The geometry's `translation` and
`orientation` offset it within the frame. Obstacles never move, and `reset`
leaves them where they are.

**Mesh obstacles.** Set `mesh` instead of `frame.geometry` (not both). The module
splits the mesh into convex pieces with CoACD; MuJoCo collides those pieces, and
the obstacle returns the same pieces from `GetGeometries`, which RDK uses when
the frame has no geometry. The planner sees them as `<name>:piece0`,
`<name>:piece1`, …. Needs a viam-server that builds a model from several
geometries (v1.14 or later); older versions plan against piece 0 only. The
bundled bin is about a dozen pieces, so the planner sees `bin:piece0`, `bin:piece1`, … The
mesh's origin is the frame's origin; the bin's is the centre of its underside.

```json
{
  "name": "bin",
  "api": "rdk:component:generic",
  "model": "viam-labs:mujoco-sim:obstacle",
  "frame": { "parent": "world", "translation": { "x": 320, "y": -220, "z": 0 } },
  "attributes": { "world": "sim-world", "mesh": "meshes/bin.stl", "color": [0.3, 0.5, 0.8] }
}
```

Use a capsule or box for round posts: Viam's geometry API has no cylinder yet
([viamrobotics/api#894](https://github.com/viamrobotics/api/pull/894)), and
viam-server can't pass a `cylinder` frame geometry to a module.

A goal inside an obstacle is refused at planning time. A path that only grazes
one can still fail in the sim: MuJoCo collides the arm links with the
planner's own shapes (see [Collision shapes](#collision-shapes)), but the
visual meshes and the gripper differ from them, and the Lite6 kinematics give
the gripper no geometry at all. The arm then stops short of its goal (`did not
converge on final waypoint`). For moves past obstacles, pass a collision
buffer to the motion service:

```python
await motion.move(component_name="pick-gripper", destination=goal,
                  extra={"collision_buffer_mm": 20})
```

The buffer applies to every obstacle, so leave it off for the final approach
onto a surface, such as placing into the bin. To have the planner see the gripper
itself, give its frame a geometry (the gripper table lists one for each bundled
gripper); this box covers the Lite6 gripper from the fingertips back to the
flange:

```json
"frame": {
  "parent": "pick-arm",
  "translation": { "x": 0, "y": 0, "z": 81.1 },
  "geometry": { "type": "box", "x": 78, "y": 78, "z": 70,
                "translation": { "x": 0, "y": 0, "z": -35 } }
}
```

### Supported API

| Method | Notes |
|---|---|
| `GetGeometries` | The frame geometry, or one PLY per convex piece labelled `piece<i>`. |

## Model viam-labs:mujoco-sim:robot

A robot file with several limbs: a bimanual cell, or a torso with arms. The
robot loads the MJCF file once; each limb is a
[`viam-labs:mujoco-sim:arm`](#model-viam-labsmujoco-simarm) with `robot` set
and the limb's `joints`, and grippers, cameras and sensors mount on those arms
as on any other. The module bundles
[ALOHA](https://github.com/google-deepmind/mujoco_menagerie/tree/main/aloha)
(two ViperX 300 arms with grippers, BSD-3-Clause, Trossen Robotics) and
TIAGo++'s upper body (`tiago`, riding the `pal_tiago_base` base; see
[Semi-humanoid: TIAGo++](#semi-humanoid-tiago)).

### Configuration

```json
{
  "name": "aloha",
  "api": "rdk:component:generic",
  "model": "viam-labs:mujoco-sim:robot",
  "frame": { "parent": "world" },
  "attributes": { "world": "sim-world", "asset": "aloha" }
},
{
  "name": "left-arm",
  "api": "rdk:component:arm",
  "model": "viam-labs:mujoco-sim:arm",
  "frame": { "parent": "aloha", "translation": { "x": -469, "y": -19, "z": 20 } },
  "attributes": { "world": "sim-world", "robot": "aloha", "asset": "aloha_left" }
},
{
  "name": "left-gripper",
  "api": "rdk:component:gripper",
  "model": "viam-labs:mujoco-sim:gripper",
  "frame": { "parent": "left-arm" },
  "attributes": { "world": "sim-world", "arm": "left-arm", "asset": "aloha" }
}
```

The right arm is the same with `asset: "aloha_right"`, translation
`{"x": 469, "y": -19, "z": 20}` and orientation
`{"type": "ov_degrees", "value": {"x": 0, "y": 0, "z": 1, "th": 180}}`.
[`examples/aloha-handover.json`](examples/aloha-handover.json) is the whole
cell, and [`examples/aloha_handover.py`](examples/aloha_handover.py) has the
left arm set the block down in the middle and the right arm carry it on.

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `world` | string | Yes | | The world component's name. |
| `asset` | string | One of `asset`, `mjcf` | | A bundled robot: `"aloha"` or `"tiago"`. |
| `mjcf` | string | One of `asset`, `mjcf` | | Your robot's `.xml` MJCF file: an absolute path, `~/...`, or a path relative to the module's `assets/`. URDF robot files aren't supported yet. |
| `home_keyframe` | string | No | the file's first keyframe | Sets the joints no arm or gripper drives. |

The robot's frame parent is `world` or a mujoco-sim base (any mujoco-sim
component with a Viam frame): the robot then rides it at the frame's offset
from the base's frame, and is built after it. While the base is missing, the
robot, its arms and grippers are out of the scene and their calls error, as
for an arm on a base. The file
can have actuators, tendons and equality constraints; mocap bodies, plugins,
flexes, skins and free joints are errors. Its names in the scene get the
prefix `<name>/`.

### Limbs

Each arm with `robot` set takes:

- `joints`: the limb's joint names in the file, base to flange, one chain;
- `kinematics_url`: its Viam kinematics, base to flange, with joint limits
  equal to the file's ranges;
- `flange_site`, `home_keyframe`, `base_offset`, `flange_offset` and the speed
  limits, as for any arm. The `aloha_left`, `aloha_right` and `tiago_*`
  presets set them.

The arm's frame parent is the robot, or another arm of the same robot (a
torso). On the robot, its translation and orientation must be the pose of its
chain base (the body its first joint hangs from) in the file, and that body
must be fixed to the robot. On another arm, they must be the chain base's pose
relative to that arm's end effector, and the chain base must be fixed to the
body that end effector is on: a jointed torso is its own arm (TIAGo's lift is
a 1-joint prismatic arm), and the limbs above it are framed on it, so RDK's
frame chain moves them with the torso as the sim does. A wrong frame is a
config error that gives the values to paste. An arm in a robot file uses the file's
own collision geoms (`collision: "kinematics"` isn't supported yet), and two
arms can't share a joint. A gripper on the arm finds its actuator below the
arm's last joint (an `actuator` you name must be one of those), as for an arm
with its gripper in its own MJCF.

### Supported API

| Method | Notes |
|---|---|
| `GetGeometries` | The file's fixed parts, in the robot's frame, except the bodies a limb hangs from (its arm's kinematics carry those). ALOHA and TIAGo have none (TIAGo's column is the torso arm's). |

If the robot is removed, its arms and grippers error until it's back; then the
arms return to their home and the grippers keep their last command.

## Model viam-labs:mujoco-sim:scene

Attaches a MuJoCo MJCF scene file (a room, a cell, a
[Menagerie](https://github.com/google-deepmind/mujoco_menagerie) scene) into the
world, placed by its frame. Cameras and physics see all of it, Viam's motion
planner sees its fixed geometry, and its free bodies become props the world's
prop commands move.

### Configuration

```json
{
  "name": "workbench",
  "api": "rdk:component:generic",
  "model": "viam-labs:mujoco-sim:scene",
  "frame": { "parent": "world", "translation": { "x": 0, "y": 0, "z": 0 } },
  "attributes": { "world": "sim-world", "mjcf": "scenes/workbench.xml" }
}
```

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `world` | string | Yes | | The world component's name. |
| `mjcf` | string | Yes | | The scene's `.xml` file: an absolute path, or a path relative to the module's `assets/` (the module bundles `scenes/workbench.xml`). Meshes and textures load relative to the file, as MuJoCo resolves them. |

The frame's `parent` must be `world`, and it must have no `geometry` (RDK would
plan against it instead of the scene). Its pose places the file's world origin.

What comes from the file:

- **Everything in its worldbody**: bodies, geoms, lights, and the materials,
  textures and skybox they use. Every name gets the prefix `<name>/`
  (`workbench/wall`). The world keeps its own `<option>`, `<size>`
  and `<visual>` settings; the file's own angle units still apply to it.
  Keyframes are dropped.
- **Lights and sky**: a file with any light turns off the world's default sun
  (unless the world sets `lighting.keep_default_light`; lights listed in
  `lighting.lights` stay). A file's skybox replaces the world's
  `lighting.skybox` (logged); with two such files, the first attached wins.
- **The floor**: a colliding plane directly in the file's worldbody replaces the world's
  floor, and the world's `floor_texture` is then ignored (logged). A plane on
  a body, such as a wall, doesn't.
- **Free bodies** become props named `<name>/<body>`
  (`workbench/block_a`): `set_prop_pose`, `get_prop_pose` and `park_props`
  move them, `reset` returns them to the file's pose, and the
  [world state store](#model-viam-labsmujoco-simworld_state) shows them in the
  3D scene. They grip and ride conveyors like config props. Every free body
  needs a name.
- **Passive joints** (hinge, slide, ball) simulate, with nothing driving
  them.

The motion planner gets every colliding geom of the fixed and jointed bodies,
at the file's starting pose: boxes, spheres and capsules as they are, cylinders
and meshes as convex meshes, ellipsoids as a box around them. Visual-only geoms
(`contype` and `conaffinity` 0), planes and height fields are left out.

A scene file can't have actuators, tendons, equality constraints, flexes,
skins, plugins or mocap bodies: robots come from the arm, gripper and base
components.

[`examples/scene-pick.json`](examples/scene-pick.json) is a Lite6 picking the workbench blocks into the bin.

## Model viam-labs:mujoco-sim:world_state

A world state store that publishes every movable (non-fixed) prop and every
live [spawner](#model-viam-labsmujoco-simspawner) parcel with its live pose
and shape, plus every [scene](#model-viam-labsmujoco-simscene)'s free bodies, so they show up and move in the Viam app's 3D scene viewer. It also
publishes the exact shapes of every [conveyor](#model-viam-labsmujoco-simconveyor)
(body and rails) and [scale](#model-viam-labsmujoco-simscale) platform: RDK
never asks a motor or sensor for its geometries, so they can't reach the viewer
through the frame system. Obstacles, arms and bases already appear through the
frame system. The world state store doesn't feed the motion service.

### Configuration

```json
{
  "name": "world-state",
  "api": "rdk:service:world_state_store",
  "model": "viam-labs:mujoco-sim:world_state",
  "attributes": { "world": "sim-world" }
}
```

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `world` | string | Yes | | The world component's name. |

### Supported API

| Method | Notes |
|---|---|
| `ListUUIDs` | One per movable prop and per conveyor or scale box, stable across restarts, and one per live parcel, new on every spawn. |
| `GetTransform` | The pose in `world` (mm, degrees) of the prop, the parcel or the conveyor or scale box's centre, its shape and `metadata.color` (0–255). |
| `StreamTransformChanges` | An ADDED event per prop and parcel, then, checked every 0.1 s, UPDATED events for those that moved more than 1 mm or 0.5°, ADDED for new parcels, conveyors and scales, and REMOVED for parcels back in their pool and conveyors and scales closed or removed. |

Cylinder props are sent as meshes, since the geometry API has no cylinder.

## Model viam-labs:mujoco-sim:conveyor

A belt conveyor, served as a motor. Parcels ride on the belt by friction, so
they speed up, slip, pile up against an obstacle and collide as they would on a
real belt.

### Configuration

```json
{
  "name": "belt",
  "api": "rdk:component:motor",
  "model": "viam-labs:mujoco-sim:conveyor",
  "frame": {
    "parent": "world",
    "translation": { "x": 600, "y": 0, "z": 450 },
    "geometry": { "type": "box", "x": 400, "y": 2000, "z": 450, "translation": { "x": 0, "y": 0, "z": -225 } }
  },
  "attributes": { "world": "sim-world", "length_m": 2.0, "width_m": 0.4 }
}
```

The frame is required, with parent `world`: it is the centre of the belt's top
surface, so `translation.z` is the belt's height. +Y is the direction of travel
at positive power. The infeed is the belt's −Y end and the outfeed its +Y end.
Legs reach down to the floor.

Give the frame a `geometry` box, or Viam's motion planner can't see the
conveyor: RDK (checked on v1.15) asks only arms, bases, cameras, gantries,
generics and grippers for their geometries, never a motor, so the conveyor's
`GetGeometries` is never called. The box runs from the floor to the belt's
top: `width_m` × `length_m` × the frame's `z`, in mm, centred `z`/2 below the
frame. Leave the rails out of it: a box up to the rail tops would enclose the
parcels on the belt, and every lift would start in collision. When the frame
has no geometry the module logs the one to paste. The Viam app's 3D scene
shows the exact shapes, rails included, through the
[world state store](#model-viam-labsmujoco-simworld_state).

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `world` | string | Yes | | The world component's name. |
| `length_m` | float | Yes | | Belt length, along +Y. |
| `width_m` | float | Yes | | Belt width. |
| `roller_diameter_m` | float | No | `0.05` | Converts RPM to belt speed: RPM × π × diameter / 60 m/s. |
| `max_speed_mps` | float | No | `0.5` | Belt speed at power 1, and the cap for `SetRPM` and `GoFor`. |
| `friction` | float | No | `1.0` | The belt's sliding friction; it wins over the parcels'. |
| `rails_m` | float | No | `0.04` | Side-rail height above the belt; `0` for none. |
| `color` | [r, g, b] | No | `[0.25, 0.27, 0.3]` | The frame's colour, 0 to 1. |

### Supported API

| Method | Notes |
|---|---|
| `SetPower` | Belt speed = power × `max_speed_mps`, until told otherwise. |
| `SetRPM` | Runs at that speed until told otherwise. |
| `GoFor` | Runs until the belt has travelled that many roller revolutions (π × diameter each), then stops; returns when done. The sign of RPM × revolutions picks the direction. |
| `GoTo` | `GoFor` the difference from the current position. |
| `GetPosition` | Belt travel in roller revolutions since the last `ResetZeroPosition`. |
| `ResetZeroPosition` | As RDK's encoded motors: the position reads −offset afterwards. |
| `Stop`, `IsPowered`, `IsMoving`, `GetProperties` | `IsPowered` reports the commanded speed as a fraction of `max_speed_mps`; `position_reporting` is true. |
| `GetGeometries` | A box from the floor to the belt's top, and a box per rail, in the conveyor's frame. viam-server never calls it on a motor; the world state store publishes these boxes. |

Speed changes ramp at 2 m/s², so parcels aren't flung. `Stop`, a new command, a
world `reset` or a reconfigure ends a `GoFor`/`GoTo` with an error. A world
`reset` stops the belt and zeroes its position.

The belt is a thin box on a slide joint, 5 cm longer than the conveyor; each
time it passes 5 cm it shifts back by 5 cm, so it can run forever. It overhangs
each end by up to 5 cm, outside the planner's box. It touches only props and
parcels: static geometry can't stall it, and arms meet the frame under it instead. A parcel held still on
a moving belt (against a stop, say) stays within about 0.6 mm, and a held pile
slows the belt slightly.

## Model viam-labs:mujoco-sim:spawner

Feeds box parcels onto a conveyor, served as a sensor. It also answers
[`viam:workcell-components`](https://app.viam.com/module/viam/workcell-components)
`box-detect`'s readings and commands, so code written for that infeed runs here.

### Configuration

```json
{
  "name": "feed",
  "api": "rdk:component:sensor",
  "model": "viam-labs:mujoco-sim:spawner",
  "attributes": {
    "world": "sim-world",
    "conveyor": "belt",
    "interval_s": 5,
    "classes": [
      { "name": "small", "weight": 2, "size_m": [0.1, 0.15], "mass_kg": [0.4, 0.8] },
      {
        "name": "large",
        "size_m": [[0.22, 0.3], [0.18, 0.24], [0.14, 0.2]],
        "mass_kg": [2.5, 3.0],
        "color": { "min": [0.55, 0.4, 0.25], "max": [0.8, 0.62, 0.42] }
      }
    ],
    "despawn": [{ "center": [0.6, 1.5, 0.25], "size": [1.0, 1.0, 0.5] }]
  }
}
```

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `world` | string | Yes | | The world component's name. |
| `conveyor` | string | Yes | | The `viam-labs:mujoco-sim:conveyor` it feeds. |
| `classes` | list | Yes | | Parcel classes, at least one; see below. |
| `interval_s` | float | No | `4` | Sim seconds between parcels. |
| `pool_size` | int | No | `12` | The most parcels live at once. |
| `seed` | int | No | random | The same seed gives the same parcels, again after a world `reset`. |
| `start_paused` | bool | No | `false` | Wait for `resume`. |
| `despawn` | list | No | `[]` | Boxes in world coordinates, `{"center": [x, y, z], "size": [x, y, z]}` in meters, such as a chute past the outfeed. A parcel whose centre enters one returns to the pool, as does one that falls below z = −0.5 m. |

Each class:

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | Yes | | Reported as the parcel's `class`. |
| `weight` | float | No | `1` | Relative chance of this class. |
| `size_m` | [low, high] or [[low, high] × 3] | Yes | | Box size in meters: one range for every side, or one per axis (x across the belt, y along it, z up). |
| `mass_kg` | [low, high] | Yes | | Mass range. |
| `color` | [r, g, b] or `{"min": [r, g, b], "max": [r, g, b]}` | No | `[0.6, 0.45, 0.3]` | A colour, or a range per channel. |
| `friction` | float | No | `1` | Sliding friction. |

Every `interval_s` of sim time (unless paused) the spawner picks a class by
weight, draws the parcel's size, mass and colour from its ranges, and puts it at
rest on the conveyor's infeed spot: at the −Y end, centred across the belt,
turned up to 10°. While a parcel is on the spot the spawn waits. A class whose
largest parcel is as wide as the belt (between the rails) or runs past the
belt's far end from the spot is never spawned: the spawner logs an error naming
the class and both sizes, once, and skips it. With every
parcel of the pool live (on the belt, in a bin, on the scale or in the gripper)
the spawn is skipped and `pool_exhausted` is true. Spawning doesn't rebuild the
scene: the pool's bodies are built once, parked below the floor, out of sight
of cameras and rays, and resized in place. A parked parcel never collides.

Live parcels keep their pose, size and mass across config edits of other
components. Reconfiguring the world, the conveyor or the spawner rebuilds the
spawner, which returns them all to the pool. A world `reset` does too, and restarts the timer and the random stream.
Parcels appear in the [world state store](#model-viam-labsmujoco-simworld_state).

### Supported API

| Method | Notes |
|---|---|
| `GetReadings` | `spawned` (since the last reset), `active` (live parcels), `paused`, `pool_exhausted`, `seconds_until_next`; and `box-detect`'s `enabled` (not paused), `box_present` (a parcel on the infeed spot) and `interval_seconds`. |

### DoCommand

| Command | Returns |
|---|---|
| `{"command": "spawn"}` | `{"spawned": name}` now, whatever the timer says; or `{"spawned": null, "reason": "infeed occupied" \| "pool exhausted" \| "class <name> does not fit the conveyor"}`. `"force": true` spawns onto an occupied spot. |
| `{"command": "pause"}` / `{"command": "resume"}` | `{"paused": bool}` |
| `{"command": "clear"}` | `{"cleared": n}`: every live parcel back to the pool. |
| `{"command": "get_parcels"}` | `{"parcels": [{name, class, pose: {position, orientation_wxyz}, size, mass_kg}]}`: world pose in meters, full size in meters. |
| `{"take": true}` | `{"taken": bool}`: `box-detect`'s; the parcel on the infeed spot returns to the pool. |
| `{"reset": true}` | `box-detect`'s; spawns one now, as `spawn`. |

## Model viam-labs:mujoco-sim:photo_eye

A through-beam photo-eye, served as a sensor: one ray per physics step, so it
counts every parcel that passes.

### Configuration

```json
{
  "name": "eye",
  "api": "rdk:component:sensor",
  "model": "viam-labs:mujoco-sim:photo_eye",
  "frame": {
    "parent": "belt",
    "translation": { "x": -250, "y": -400, "z": 30 }
  },
  "attributes": { "world": "sim-world", "range_m": 0.5 }
}
```

The beam starts at the frame's origin and runs along its +X. The frame's
`parent` is `world` or another mujoco-sim component, and the eye never sees its
parent or anything attached below it. To look across a conveyor, frame it on the
conveyor, as above: the beam then passes through the belt's rails and the belt.
An eye framed on the world would see the rail.

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `world` | string | Yes | | The world component's name. |
| `range_m` | float | No | `2` | Beam length. |

### Supported API

| Method | Notes |
|---|---|
| `GetReadings` | `blocked` (something within range), `distance_m` (to the hit, or `range_m`), `count` (times the beam became blocked since configure or reset). |
| `DoCommand` | `{"command": "reset"}` zeroes `count`; so does a world `reset`. |

## Model viam-labs:mujoco-sim:scale

A platform scale, served as a sensor. It weighs what presses on the platform:
the normal contact forces, filtered.

### Configuration

```json
{
  "name": "scale",
  "api": "rdk:component:sensor",
  "model": "viam-labs:mujoco-sim:scale",
  "frame": {
    "parent": "world",
    "translation": { "x": 0, "y": 550, "z": 450 },
    "geometry": { "type": "box", "x": 350, "y": 350, "z": 450, "translation": { "x": 0, "y": 0, "z": -225 } }
  },
  "attributes": { "world": "sim-world", "size_m": [0.35, 0.35, 0.45] }
}
```

The frame, parent `world`, is the centre of the platform's top face; the
platform is a box of `size_m` below it.

Give the frame that box as its `geometry` (`size_m` in mm, centred half its
height below the frame), or Viam's motion planner can't see the scale: RDK
never asks a sensor for its geometries, so the scale's `GetGeometries` is never
called. When the frame has no geometry the module logs the one to paste. The
Viam app's 3D scene shows the platform through the
[world state store](#model-viam-labsmujoco-simworld_state).

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `world` | string | Yes | | The world component's name. |
| `size_m` | [x, y, z] | Yes | | The platform box, meters. |
| `color` | [r, g, b] | No | `[0.6, 0.6, 0.62]` | 0 to 1. |

### Supported API

| Method | Notes |
|---|---|
| `GetReadings` | `weight_kg` (minus the tare) and `stable` (the weight moved less than 5 g over the last 0.3 s). The weight is filtered with a 0.1 s time constant, so a parcel reads stable about a second after it lands. |
| `DoCommand` | `{"command": "tare"}` sets the tare to the current weight and returns `{"tare_kg"}`. A world `reset` restarts the filter and keeps the tare. |
| `GetGeometries` | The platform box. viam-server never calls it on a sensor; the world state store publishes it. |

## Bring your own robot

A ROS robot's description is usually a xacro. Expand it to a URDF with the
Viam CLI (it runs xacro in Docker), then point a base's, arm's or gripper's
`mjcf` at the `.urdf`:

```sh
viam xacro convert --input-file urdf/robot.urdf.xacro --output-file robot.urdf
```

Don't add `--collapse-fixed-joints`: its output repeats each joint's
`<parent>`, `<child>` and `<origin>`, which MuJoCo refuses. A URDF's
`package://` meshes are looked up by file name in `mesh_dir` (default
`meshes/` beside the URDF if it exists, else its folder).

What to fix for realism:

- **Visuals.** MuJoCo drops a URDF's visuals (and can't load DAE meshes), so
  the robot renders its collision shapes. For the real look, convert the DAE
  visuals to OBJ or STL and add them to an MJCF.
- **Masses.** A link with geometry and no `<inertial>` gets its mass from
  geom density (water's), and configuring logs a warning naming those links.
  The expanded Husky's `base_link` comes out at 116 kg. Give those links
  `<inertial>`s.
- **Save it as MJCF to hand-tune** (friction, `priority`, masses, visuals),
  then point `mjcf` at the `.xml`:

  ```python
  import mujoco
  spec = mujoco.MjSpec.from_file("husky.urdf")
  spec.compiler.fusestatic = False  # keep base_link a body
  spec.meshdir, spec.strippath = "meshes", True
  open("husky.xml", "w").write(spec.to_xml())
  ```

### A base: the Clearpath Husky

1. Clone [husky/husky](https://github.com/husky/husky) and `cd
   husky_description`. Delete the six `accessories/` `xacro:include` lines
   near the top of `urdf/husky.urdf.xacro` (they pull packages, such as
   `lms1xx` and `realsense2_description`, that aren't in the ROS image).
2. Expand it:

   ```sh
   viam xacro convert --input-file urdf/husky.urdf.xacro --output-file husky.urdf
   ```

3. Point the base at it. Its meshes are in `meshes/` beside it, the default
   `mesh_dir`:

   ```json
   "attributes": {
     "world": "sim-world",
     "mjcf": "/path/to/husky_description/husky.urdf",
     "drive": "wheels",
     "max_linear_mps": 1.0,
     "max_angular_dps": 115
   }
   ```

### An arm: the Universal Robots UR5e

1. Clone [UniversalRobots/Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description)
   and expand the UR5e in it:

   ```sh
   viam xacro convert --input-file urdf/ur.urdf.xacro --output-file ur5e.urdf \
     --args ur_type:=ur5e --args name:=ur5e
   ```

2. Point the arm at it. The file ends in three links (`tool0`, `ft_frame` and
   `base`), so name the flange; its collision meshes are in
   `meshes/ur5e/collision`:

   ```json
   "attributes": {
     "world": "sim-world",
     "mjcf": "/path/to/Universal_Robots_ROS2_Description/ur5e.urdf",
     "mesh_dir": "meshes/ur5e/collision",
     "flange_link": "tool0"
   }
   ```

   The arm starts with all joints at zero. A bundled gripper (`robotiq_2f85`)
   attaches at `tool0` as on the bundled UR5e.

### A gripper

A URDF gripper attaches at the arm's flange like an MJCF one. It needs one
movable joint that isn't a `<mimic>` (the module drives it; `<mimic>` fingers
follow) with `<limit effort>` and a range. A gripper with several driven
joints, a tendon or suction needs an MJCF.

## De-palletizing

[`examples/depalletize.json`](examples/depalletize.json) is a UR5e with a
Robotiq ePick on a 400 mm pedestal, between a pallet of twelve 1 kg cartons
(190 × 150 × 150 mm, two layers of 3 × 2) and a conveyor. The pedestal,
pallet and conveyor are obstacles; the cartons are props, which the motion
planner does not see on its own.

[`examples/depalletize.py`](examples/depalletize.py) unloads the pallet over
one client session:

```sh
uv run python examples/depalletize.py <machine address> <api key id> <api key>
```

It restacks the cartons (`park_props`), then for each one, top layer first,
moves the cup onto its top face, grabs, lifts, carries it over the conveyor,
lowers and lets go, and reports whether it landed on the belt. The conveyor
doesn't move: the script moves each carton off its far end with
`set_prop_pose` to clear the drop spot. Run it again to start over.

Every `motion.move` passes a `WorldState`: the cartons still on the pallet as
obstacles, read with `get_prop_pose`. The one being picked is left out only for
the final straight-down approach onto it (otherwise the planner may route the
arm through it), and the ones under it only for the first 25 mm of a lift. While a carton is held, the
`WorldState` also carries it as a transform on the gripper, so the planner
keeps the carton clear of everything too:

```python
Transform(
    reference_frame="held_carton",
    pose_in_observer_frame=PoseInFrame(reference_frame="pick-gripper", pose=Pose(z=76, o_z=1)),
    physical_object=Geometry(
        center=Pose(o_z=1),  # RDK needs a centre pose, even the identity
        box=RectangularPrism(dims_mm=Vector3(x=190, y=150, z=146)),
        label="held_carton",
    ),
)
```

The box hangs below the tool point, just clear of the gripper's own planner
box, and is a few millimetres shorter than the carton so it doesn't touch the
surface the carton sat on. Use the same pattern for any object a gripper
carries.

## Parcel sorting

[`examples/parcel-sort.json`](examples/parcel-sort.json) is a sorting cell: a
UR5e with a Robotiq ePick on a 400 mm pedestal; a 2 m conveyor along +Y in
front of it; a spawner feeding small (0.4–0.8 kg), medium (1.2–1.8 kg) and
large (2.5–3 kg) parcels, whose sizes overlap and whose colours share one
range, so neither tells them apart; a photo-eye across the belt; a scale beside
it; and a light and a heavy bin (the bundled bin mesh, scaled up). Parcels that
reach the outfeed drop into a despawn chute.

The `belt` and `scale` frames carry `geometry` boxes, so the motion planner
keeps the arm off them: the belt's runs from the floor to the belt's top (not
the rail tops, which would enclose the parcels riding on it), the scale's
covers its platform. RDK never asks a motor or sensor for its shape, so without
these boxes the planner would route straight through both. The `world-state`
service shows their exact shapes, the belt's rails included, in the 3D scene.

[`examples/parcel_sort.py`](examples/parcel_sort.py) sorts them:

```sh
uv run python examples/parcel_sort.py <machine address> <api key id> <api key> [parcels]
```

For each parcel (10 by default) it runs the belt until the photo-eye's `count`
goes up, stops it, and finds the leading parcel with the spawner's
`get_parcels`. It picks the parcel by suction, sets it on the scale, waits for
`stable`, reads `weight_kg`, picks it again and drops it in the light (under
1 kg) or heavy bin. Binned parcels never despawn, so it lets go over the
quarter of the bin with the lowest pile, high enough to clear it. Every move
plans around the other parcels on the belt and in the bin, passed to the
motion service as obstacles, and carries a held parcel as a transform on the
gripper, sized from `get_parcels`, as in [De-palletizing](#de-palletizing).
Moves that carry a parcel add an `OrientationConstraint` (5°), so the cup
stays pointing down along the whole path, not only at its ends. At the end it reads `get_parcels` again and
reports any parcel that isn't in its right bin.

## Conveyor pick

[`examples/conveyor-pick.json`](examples/conveyor-pick.json) is the parcel
sorting cell without the scale, bins and photo-eye: the UR5e and ePick on
their pedestal, the belt and the spawner. The arm is faster
(`max_joint_deg_s` 180, `max_joint_deg_s2` 360) and the belt goes up to
1 m/s (`max_speed_mps` 1.0), so the arm can chase a parcel on it.

[`examples/conveyor_pick.py`](examples/conveyor_pick.py) picks one parcel
without stopping the belt, and is a way to compare how Viam's motion APIs
follow a moving target:

```sh
uv run python examples/conveyor_pick.py <machine address> <api key id> <api key> \
    <teleop|stream|move> [--belt POWER] [--lookahead S] [--csv OUT]
```

It parks the gripper over the start of the arm's reach, runs the belt at
`--belt` × 1 m/s (default 0.1), and waits for the leading parcel. Then, at a
steady rate, it reads the parcel's pose from the spawner's `get_parcels` and
sends a goal `--lookahead` seconds of belt travel ahead of it: 100 mm over it
until the gripper has been on target for three ticks, then down to its top,
`Grab`, and up 150 mm, still following the belt, to the edge of the reach.
The goal's height leads its own ramps by the lookahead too. It prints when
the gripper locked on and gripped, its error at the grab, and how far the
parcel rose; `--csv` writes every tick.

The modes send the same goals three ways:

| Mode | API | Rate | Lookahead |
|---|---|---|---|
| `teleop` | The motion service's `teleop_start`, `teleop_move` and `teleop_stop` DoCommands. RDK plans each goal in the background, with the belt's rails as obstacles, and runs the newest plan with `MoveThroughJointPositions` and `waitAtEnd` false. | 10 Hz | 0.4 s |
| `stream` | The motion service's `TempStreamArmJointPositions` session, which times a stream of joint targets with trajex. It takes joints, not poses, so the script solves IK itself on the module's UR5e kinematics, and nothing checks for collisions. Released viam-server builds refuse the session: it needs a build with the `viam_rdk_cgo_have_cxx20_rt` tag. | 20 Hz | 0.5 s |
| `move` | The motion service's `Move` (the arm leaves `MoveToPosition` to it), once per goal. Each call plans, runs and waits for the arm to stop. | as fast as `Move` returns | 1.0 s |

On a MacBook, one parcel each (viam-server v1.15.0, built with trajex):

| Mode | Belt | Locked on | Gripped | Error at grab (x, y, z mm) | Lifted |
|---|---|---|---|---|---|
| `teleop` | 0.1 m/s | 0.8 s | 1.8 s | 0.9, 1.6, 2.8 | 147 mm |
| `stream` | 0.1 m/s | 0.8 s | 1.9 s | 0.8, 8.6, 3.0 | 147 mm |
| `move` | 0.1 m/s | 5.2 s | 6.1 s | 0.1, -1.0, 0.9 | 144 mm |
| `teleop` | 0.3 m/s | 0.9 s | 2.0 s | 4.8, 4.8, 2.0 | 146 mm |
| `stream` | 0.3 m/s | 1.5 s | 2.8 s | 4.5, -29.9, 2.6 | 145 mm |
| `move` | 0.3 m/s | never | | | |

`move` stops between goals, about 1 s per goal at 0.1 m/s and 1.4 s at
0.3 m/s; the parcel crosses the reach in under 3 s at 0.3 m/s, too few goals
to lock on. `teleop` holds tightest: each plan starts from where the last
one is headed, so the arm doesn't stop. `stream` moves smoothly, but
trajex replans its timing whenever the target's speed changes, so it lags
most on the descent and grips further from the middle.

## Mobile base, SLAM and navigation

A simulated rover can map its world and navigate it with
[`viam-labs:nav-stack`](https://github.com/viam-labs/nav-stack), as a real Viam
Rover does. Viam's builtin motion service no longer plans for bases
(`MoveOnMap` and `MoveOnGlobe` were removed from RDK in late 2025), and the
builtin navigation service is GPS-only, so nav-stack is the path. It takes a
2D scan: keep the LIDAR's `channels` at 1.
[`examples/rover-nav.json`](examples/rover-nav.json) is the complete config
these steps build.

nav-stack reads three things from the sim: the LIDAR's point clouds, the
movement sensor's velocities, and the base's `SetVelocity`. It takes the
LIDAR's mount from the frame system and the rover's footprint from the base's
`GetGeometries` box.

### 1. Build the scene

1. A world with walls and obstacles as
   [obstacle](#model-viam-labsmujoco-simobstacle) components, like the 3 m
   pen in the example, so they're in the frame system and the 3D scene viewer
   too. Keep them tall enough for the LIDAR's scan plane to hit.
2. A `viam-labs:mujoco-sim:base` framed on `world`.
3. A `viam-labs:mujoco-sim:lidar` framed on the rover, above the chassis
   (`translation.z` over 130 mm; the example uses 160).
4. A `viam-labs:mujoco-sim:movement_sensor` with `base` set to the rover. Give
   it some drift (the example uses `linear_scale_error: 0.02` and
   `heading_drift_dps: 0.05`), or SLAM has nothing to correct.

An overhead `viam-labs:mujoco-sim:camera` aimed at the pen (with `target`) is a
handy way to watch the rover drive.

### 2. Add nav-stack

Add the module from the registry:

```json
{ "type": "registry", "name": "viam-labs_nav-stack", "module_id": "viam-labs:nav-stack", "version": "latest" }
```

- **Linux:** the registry build is `linux-any`. Its first-run script creates a
  venv with `python3 -m venv`, which needs the host's `python3.X-venv` package
  (`apt install python3.12-venv` on Ubuntu 24.04). Without it the first run
  fails and keeps failing, and viam-server holds back every config change
  until it succeeds. Install the package, then remove the module's partial
  `.venv` (or the module from the config and add it back).
  [viam-labs/nav-stack#100](https://github.com/viam-labs/nav-stack/pull/100)
  makes the first run install the package and recover on its own.
- **macOS:** there is no darwin build in the registry. Clone nav-stack, create
  its venv (`uv venv .venv --python 3.11 && VIRTUAL_ENV=.venv uv pip install -r
  requirements.txt`), and add it as a local module whose `executable_path` is
  the checkout's `run.sh`.

`latest` resolves to nav-stack's newest stable release. Its typed Navigation
methods (such as `GetNavigationStatus`) may be newer than that release; the
`DoCommand` interface below works on all of them.

### 3. Configure SLAM and navigation

```json
"services": [
  {
    "name": "slam",
    "api": "rdk:service:slam",
    "model": "viam-labs:nav-stack:slam",
    "attributes": {
      "base": "rover",
      "movement_sensor": "odometry",
      "lidar": "lidar",
      "movement_sensor_yaw_deg": 0,
      "mode": "mapping",
      "maps_dir": "/tmp/nav-maps",
      "active_map": "rover-pen"
    }
  },
  {
    "name": "nav",
    "api": "rdk:service:navigation",
    "model": "viam-labs:nav-stack:navigation",
    "attributes": { "slam_service": "slam", "base": "rover" }
  }
]
```

- `movement_sensor_yaw_deg: 0` is deliberate. This sensor already reports
  base-frame velocities (+Y forward), which nav-stack reads correctly; left
  out, nav-stack would take a yaw from the sensor's frame, if it has one, and
  rotate them a second time.
- `maps_dir` must be writable by viam-server; nav-stack creates it.
- Add a `viam-labs:nav-stack:nav-camera` component (`"navigation": "nav"`) to
  watch the map, plan and pose. It draws once nav-stack is localizing on a map;
  while mapping it shows a blank frame.
- nav-stack logs that the LIDAR "has no get_laser_scan; using get_point_cloud"
  once, on its first scan. That's expected: `get_laser_scan` is a MiR-specific
  shortcut, and point clouds are all the sim needs.

### 4. Record a map

With `mode: mapping`, nav-stack maps whenever the rover moves; drive it from
the base's control card in the Viam app or from the SDK. In the example pen, one
lap of driving out to four points and spinning a full turn at each (0.3 m/s,
45 °/s) takes about 80 s and maps every wall and both boxes. Then save the map
and switch to localizing on it:

```python
from viam.services.slam import SLAMClient

slam = SLAMClient.from_robot(robot, "slam")
await slam.do_command({"command": "save_map"})
await slam.do_command({"command": "start_localizing", "map": "rover-pen"})
```

To map again from scratch, call the world's `reset` (the rover returns to its
spawn pose and its odometry zeroes), then
`{"command": "start_mapping", "map": "<new name>"}`. Otherwise avoid `reset`
mid-session: nav-stack sees the teleport as a jump in the middle of its map.

### 5. Navigate

```python
from viam.services.navigation import NavigationClient

nav = NavigationClient.from_robot(robot, "nav")
await nav.do_command({"command": "navigate_to_point", "x": -1.1, "y": 0.9})
status = await nav.do_command({"command": "get_status"})  # "state": "succeeded" when done
```

- Goals are in nav-stack's map frame (meters), not MuJoCo's world frame. The
  map's origin is where mapping started, and its +X is the rover's forward
  there. Read the rover's map pose from `slam.get_position()` (millimetres and
  degrees, as Viam poses are).
- Keep goals clear of walls and obstacles by the rover's half-width plus
  nav-stack's `clearance_m` (0.13 + 0.2 m by default). A goal inside that band
  can't be reached, and nav-stack keeps trying rather than failing it.
- Heading drift is what SLAM corrects. The example's 0.05 °/s maps cleanly;
  0.5 °/s smears the walls into rotated copies. See nav-stack's README for
  tuning.

## Mobile manipulation

[`examples/husky-ur5e.json`](examples/husky-ur5e.json) is a Clearpath Husky
(`drive: ideal`) carrying a UR5e with a Robotiq 2F-85 on its top plate, with
odometry, a block on table A 2.8 m ahead, and table B 2.8 m to the left.
[`examples/mobile_pick.py`](examples/mobile_pick.py) moves the block over one
client session:

```sh
uv run python examples/mobile_pick.py <machine address> <api key id> <api key>
```

It resets the world, drives 2 m to table A, picks up the block, backs up,
turns left, drives 2 m to table B, puts the block down and reports where it
landed. Before each drive it moves the arm to a carry pose close over the
Husky, so the gripper and block stay clear of the tables. It checks every
`motion.move` result and stops with an error if a plan fails. Run it again to
start over.

RDK doesn't know the sim base has moved (see [Things to know](#things-to-know)),
so the script plans every move in the `husky` frame. At each table it reads
the Husky's pose from odometry (`get_position` with
`extra={"return_relative_pos_m": true}`, and `get_orientation`), moves the
block's pose from `get_prop_pose` and the tables' boxes into the `husky`
frame, and passes the destination and the `WorldState` in that frame. It
doesn't transform the held block: it sits inside the gripper's planner box,
so the planner already treats it as part of the gripper. The tables'
obstacle components are in the frame system too, but placed relative to where
RDK keeps the Husky: more than 2 m away, out of the arm's reach. Watch it
with `scene-cam`, not the 3D scene view.

## Semi-humanoid: TIAGo++

`tiago` is TIAGo++'s upper body as a robot file riding the `pal_tiago_base`
base, made from the
[MuJoCo Menagerie's `pal_tiago_dual`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/pal_tiago_dual)
(Apache-2.0, PAL Robotics; see [`assets/pal_tiago_dual/README.md`](assets/pal_tiago_dual/README.md)).
Its limbs are four arms: the torso lift (`tiago_torso`, one slide joint,
0–350 mm) framed on the robot, and the two 7-joint arms (`tiago_arm_left`,
`tiago_arm_right`) and the 2-joint head (`tiago_head`, pan and tilt, its end
effector the head camera: +Z along the view) framed on the torso. Every frame
offset is zero:

```json
{ "name": "tiago-base", "api": "rdk:component:base", "model": "viam-labs:mujoco-sim:base",
  "frame": { "parent": "world" }, "attributes": { "world": "sim-world", "asset": "pal_tiago_base" } },
{ "name": "tiago", "api": "rdk:component:generic", "model": "viam-labs:mujoco-sim:robot",
  "frame": { "parent": "tiago-base" }, "attributes": { "world": "sim-world", "asset": "tiago" } },
{ "name": "torso", "api": "rdk:component:arm", "model": "viam-labs:mujoco-sim:arm",
  "frame": { "parent": "tiago" }, "attributes": { "world": "sim-world", "robot": "tiago", "asset": "tiago_torso" } },
{ "name": "left-arm", "api": "rdk:component:arm", "model": "viam-labs:mujoco-sim:arm",
  "frame": { "parent": "torso" }, "attributes": { "world": "sim-world", "robot": "tiago", "asset": "tiago_arm_left" } },
{ "name": "head", "api": "rdk:component:arm", "model": "viam-labs:mujoco-sim:arm",
  "frame": { "parent": "torso" }, "attributes": { "world": "sim-world", "robot": "tiago", "asset": "tiago_head" } },
{ "name": "head-cam", "api": "rdk:component:camera", "model": "viam-labs:mujoco-sim:camera",
  "frame": { "parent": "head" }, "attributes": { "world": "sim-world" } },
{ "name": "left-gripper", "api": "rdk:component:gripper", "model": "viam-labs:mujoco-sim:gripper",
  "frame": { "parent": "left-arm" }, "attributes": { "world": "sim-world", "arm": "left-arm", "asset": "tiago" } }
```

It starts at its `home` keyframe: torso at 175 mm, upper arms down with the
forearms and grippers forward (each arm `[-6.9, 81.4, 81.4, 126.6, -6.3,
-54.4, 106.6]` degrees), head level. The arms are mirror images, so the same
joint values are mirror poses. The kinematics files' shapes don't touch
across limbs at home (RDK would ignore such a pair for the whole plan): the
torso's column starts above the base's box and its shell is narrowed to clear
the arms' shoulders. The base's box reaches up to TIAGo's antennas, so the
torso can't plan below about 60 mm.

[`examples/tiago-pick.json`](examples/tiago-pick.json) puts it 2 m from a
450 mm table with a block on it, with an 800 mm shelf to the left, odometry,
`world_state` and the builtin motion service.
[`examples/tiago_pick.py`](examples/tiago_pick.py) drives to the table, tilts
the head down and saves a head-camera image, picks the block with the left
gripper, drives to the shelf and puts it there, planning in the `tiago-base`
frame as `mobile_pick.py` does. The torso is in the left gripper's frame
chain, so the motion service moves it with the arm: down to reach the table,
up to reach the shelf (the script prints the torso before and after each
move).

## Things to know

- **Config edits reset the cell.** viam-server rebuilds a module resource on any
  config change (close, then new). Editing an arm attribute sends the arm back
  to home and reopens its gripper. Editing the world rebuilds every component
  that depends on it. A move in flight fails with
  `arm '<name>' was closed (removed or reconfigured)`. Leave the config alone
  during a run.
- **Parcels and props are invisible to the planner.** Pass the ones that matter
  to `motion.move` as `WorldState` obstacles, read with `get_parcels` or
  `get_prop_pose`, as `parcel_sort.py` and `depalletize.py` do.
- **Fixed props and obstacles never move.** `set_prop_pose` and `park_props`
  reject them, and `reset` leaves them where they are. Non-fixed props keep
  their pose across world edits; use `park_props` or `reset`.
- **No cylinder obstacles yet.** Viam's geometry API has no cylinder ([viamrobotics/api#894](https://github.com/viamrobotics/api/pull/894)), so use a capsule or box for a round post. Cylinder props work and appear in the scene viewer as meshes.
- **First load of a mesh is slow.** CoACD can take seconds to minutes on a dense mesh, and the whole module is unresponsive meanwhile. viam-server gives a resource about a minute (`VIAM_RESOURCE_CONFIGURATION_TIMEOUT`). If a mesh obstacle times out, RDK builds its frame without geometry and **the planner plans straight through it** until the next reconfigure. Decimate heavy CAD first (a few thousand triangles is plenty) or raise the timeout. The result is cached in the module's data dir, so restart once the log says "decomposed … into N convex pieces".
- **Mesh obstacles are missing from the `FrameSystemConfig` API.** RDK can't serialize a model built from geometries, so that call leaves them out (it blanks the part). The planner and the 3D scene viewer still see them (checked on viam-server v1.15).
- **Mesh and texture URLs are downloaded once** (they share the module's mesh cache). To pick up a changed file, change the URL (e.g. add `?v=2`).
- **A scene's mesh is one convex hull.** MuJoCo collides a mesh as its convex hull and the planner gets the same hull, so a whole cell exported as one mesh blocks the workspace. Split it into convex parts first.
- **Scene doors and drawers are planned where they start.** A hinged or sliding part that moves during a run stays at its starting pose for the planner.
- **Textures are for cameras.** The 3D scene viewer draws props and obstacles in their `color`. Spawner parcels take no texture. Images larger than 2048 px are scaled down.
- **Viam's planner collides meshes as surfaces.** A link entirely inside a piece, or a step that jumps past a thin wall, isn't caught. Keep walls at least 5 mm thick.
- **Arm self-collision is partial.** Link pairs touching at home are excluded,
  and the bundled Lite6 also excludes its wrist-link pairs, whose convex hulls
  overlap at reachable wrist angles. The gripper and links 3–6 still collide
  with the base and shoulder, which Viam's planner can't see.
- **Place components relative to `world`, or ride a base.** A base's frame
  must have parent `world`; a robot's is `world` or a base; an arm's is
  `world` or a base, or for a limb its robot or another arm of it; the
  gripper's is its arm; a camera's, LIDAR's or IMU's is `world` or another mujoco-sim
  component; a force-torque sensor's is its arm.
- **The frame system doesn't follow a base.** RDK keeps a base at its
  configured pose; it doesn't know the sim base has driven. Planning in the
  base's (or a carried arm's) frame stays right, but world-frame geometry
  (obstacles framed on `world`, world state store entries, imported
  geometry) is placed relative to the base's configured pose, so after a
  drive the planner sees it in the wrong place relative to the arm. The 3D
  scene view uses the same frame system: it draws the base and arm at their
  configured pose and props where they really are, so a held block appears a
  drive's length from the gripper. Keep world obstacles out of the arm's
  reach from the base's configured pose, and give the planner the real ones
  in the base's frame, as [`examples/mobile_pick.py`](examples/mobile_pick.py)
  does.
- **Base motion is ideal unless you ask for wheels.** With the default
  `drive: ideal` a base drives exactly as commanded, can't tip over, and
  contacts stop it. With `drive: wheels` it slips and skids like the real
  thing, but the movement sensor still reads its true motion: odometry drift
  is set by attributes, not by wheel slip.
- **A LIDAR scan is one instant.** All rays sample the same sim step, unlike a
  spinning sensor whose points span a rotation.
- **Depth is ideal.** It has no noise, holes or minimum range, unlike a real
  depth sensor. On macOS, MuJoCo warns that depth accuracy is limited; at 1 m
  it is still within 1 mm.
- **Kinematics are vendored.** `GetKinematics` serves the bundled arm's file
  (`assets/ufactory_lite6/lite6_kinematics.json` from
  viam-ufactory-xarm@73551eac, `assets/universal_robots_ur5e/ur5e_kinematics.json`
  from universal-robots@3baa567d). The `kinematics_url` attribute overrides it.
- **One-actuator grippers only.** No suction, and no grippers with more than
  one actuator.

## Run it locally

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.11 itself),
`make`, and `viam-server` for running on a machine.

```sh
make run.sh   # the entrypoint is generated, see Makefile
```

Point a machine's config at `run.sh` as a local module (see
`examples/lite6-pick.json`), start `viam-server`, then drive it with any Viam
client.

## Develop

```sh
make test
make lint
```

Tests run real MuJoCo; there is no mock backend. Tests use OSMesa on Linux
(`apt install libosmesa6`); macOS uses CGL.

The Lite6, UR5e and 2F-85 models are vendored from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(`assets/ufactory_lite6` and `assets/universal_robots_ur5e`, BSD-3-Clause;
`assets/robotiq_2f85`, BSD-2-Clause). See `NOTICE`.
(`assets/ufactory_lite6`, BSD-3-Clause).

To release, push a plain-semver tag such as `0.1.1`; the deploy workflow builds
and publishes it.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party
material.
