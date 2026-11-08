# viam-mujoco-sim

A proof-of-concept [Viam](https://www.viam.com) module that simulates robots in
[MuJoCo](https://mujoco.org). It is a drop-in sibling of
[`viam-labs:isaac-sim`](https://github.com/viam-modules/viam-isaac-sim): the
same core component attributes, the same world `DoCommand`s, and the standard
Viam frame for placement, so a config moves between engines by changing the
model triplet.

Physics runs on the CPU and the module is a few tens of MB, so it runs on a Mac
as well as on Linux. Cameras render offscreen with OpenGL: CGL on macOS, EGL
(the GPU) on Linux.

## Models

| Model | API | What it does |
|---|---|---|
| [`viam-labs:mujoco-sim:world`](#model-viam-labsmujoco-simworld) | `generic` | Owns the scene, props and step loop. Configure exactly one. |
| [`viam-labs:mujoco-sim:arm`](#model-viam-labsmujoco-simarm) | `arm` | A UFactory Lite6 placed by its frame, with joint control. |
| [`viam-labs:mujoco-sim:gripper`](#model-viam-labsmujoco-simgripper) | `gripper` | The Lite6's integrated binary gripper. |
| [`viam-labs:mujoco-sim:camera`](#model-viam-labsmujoco-simcamera) | `camera` | An RGB-D camera placed by its frame, on the world or an arm's end effector. |
| [`viam-labs:mujoco-sim:base`](#model-viam-labsmujoco-simbase) | `base` | A Viam Rover you can drive; ideal motion, visual wheels. |
| [`viam-labs:mujoco-sim:lidar`](#model-viam-labsmujoco-simlidar) | `camera` | A 2D LIDAR on the world, a base or an arm; point clouds only. |
| [`viam-labs:mujoco-sim:movement_sensor`](#model-viam-labsmujoco-simmovement_sensor) | `movement_sensor` | Wheel odometry for a mujoco-sim base, with optional drift. |
| [`viam-labs:mujoco-sim:obstacle`](#model-viam-labsmujoco-simobstacle) | `generic` | A fixed box, sphere or capsule that Viam's motion planner also sees. |
| [`viam-labs:mujoco-sim:world_state`](#model-viam-labsmujoco-simworld_state) | `world_state_store` | Streams movable props to the Viam app's 3D scene viewer. |

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
pick cell. Its `pick_block` starts parked out of reach at x = 2.5 m on purpose;
`scripted_pick.py` from
[viam-isaac-sim](https://github.com/viam-modules/viam-isaac-sim) places it with
`set_prop_pose`. [`examples/rover-lidar.json`](examples/rover-lidar.json) is a
rover cell: a rover with a LIDAR inside a walled pen, and
[`examples/rover-nav.json`](examples/rover-nav.json) adds odometry and
nav-stack for SLAM and navigation in it (see
[Mobile base, SLAM and navigation](#mobile-base-slam-and-navigation)). With an
arm and gripper configured, the builtin motion service plans for the arm as it
would for a real one.

## Model viam-labs:mujoco-sim:world

Owns the MuJoCo scene: the floor, light, props and the step loop. Configure
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

Each prop:

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | Yes | | Unique prop name, used by the prop commands. |
| `type` | string | No | `"cube"` | `"cube"`, `"sphere"`, `"cylinder"` or `"capsule"`. Each fills a box of `size × scale`: a sphere needs a uniform scale; a cylinder's or capsule's diameter is `size × scale[0]` (equal to `scale[1]`) and its height `size × scale[2]`, tip to tip for a capsule. A capsule's height must exceed its diameter (`scale[2] > scale[0]`). |
| `size` | float | No | `0.05` | Width in meters; see `type` for each shape. |
| `scale` | [x, y, z] | No | `[1, 1, 1]` | Per-axis scale of `size`. |
| `position` | [x, y, z] | No | `[0, 0, 0]` | Spawn position in meters. |
| `orientation_wxyz` | [w, x, y, z] | No | `[1, 0, 0, 0]` | Spawn orientation. |
| `mass` | float | No | `0.05` | Mass in kg. |
| `color` | [r, g, b] | No | `[0.7, 0.7, 0.7]` | RGB, 0 to 1. |
| `fixed` | bool | No | `false` | A fixed prop never moves. |
| `static_friction` | float | No | `1.0` | Sliding friction, non-negative. |
| `dynamic_friction` | float | No | | Accepted so isaac-sim configs carry over; MuJoCo has one sliding coefficient, so it is ignored. |

Props are invisible to Viam's motion planner. For fixed scene geometry the planner should avoid, use [obstacles](#model-viam-labsmujoco-simobstacle).

### DoCommand

| Command | Fields | Returns |
|---|---|---|
| `status` | | `{playing, sim_time, real_time_factor, slips, warnings}` |
| `play` / `pause` | | `{"ok": true}` |
| `reset` | | `{"ok": true}`; arms go home and non-fixed props return to their spawn poses. |
| `set_prop_pose` | `prop`, `position`, `orientation_wxyz` | `{prop, position, orientation_wxyz}`; zeroes the prop's velocity. |
| `get_prop_pose` | `prop` | `{prop, pose: {position, orientation_wxyz}}` |
| `park_props` | `props`: list of names | `{poses: {name: {position, orientation_wxyz}}}`; returns props to their spawn poses. |

`real_time_factor` is sim seconds per wall second over the last two seconds.
`slips` counts how often the sim fell more than 100 ms behind wall time and
re-anchored. `warnings` holds MuJoCo's warning counts, such as
`mjWARN_BADQACC`.

```json
{"command": "set_prop_pose", "prop": "pick_block", "position": [0.3, 0.05, 0.02], "orientation_wxyz": [1, 0, 0, 0]}
```

## Model viam-labs:mujoco-sim:arm

A UFactory Lite6, from the
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) model
with its wide gripper. The arm is placed by its standard Viam frame and starts
at home, `[0, 0, 90, 0, 90, 0]` degrees. Joints are position servos; moves are
rate limited by `max_joint_deg_s`.

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
| `asset` | string | Yes | | The robot. Only `"lite6"` is supported. |
| `max_joint_deg_s` | float | No | `180` | Per-joint velocity limit on the commanded target, reported by `dynamics_metadata`. |
| `max_joint_deg_s2` | float | No | | Per-joint acceleration limit, reported by `dynamics_metadata` for trajectory planning. |
| `move_timeout_sec` | float | No | `30` | How long a move waits to arrive before it fails. |
| `path_tolerance_delta_deg` | float | No | `0.5` | How close every joint must be for a move to count as arrived. |
| `is_moving_velocity_threshold_rad_s` | float | No | `0.01` | Joints must also be slower than this to count as arrived. |
| `robot_control_freq_hz` | float | No | `50` | Rate at which `MoveThroughJointPositions` pushes targets, up to 1000. |
| `path_max_step_deg` | float | No | `2.0` | Largest per-joint step between pushed targets, up to 30. |
| `kinematics_url` | string | No | vendored Lite6 file | Kinematics file served by `GetKinematics`: `.json` for SVA, `.urdf` for URDF. `http(s)://` or `file://`. |
| `position` | [x, y, z] | No | `[0, 0, 0]` | Placement in meters, used when no frame is set. |
| `orientation_wxyz` | [w, x, y, z] | No | | Orientation, used when no frame is set. |
| `orientation_rpy_deg` | [r, p, y] | No | | Same as roll, pitch, yaw in degrees; `orientation_wxyz` wins if both are set. |
| `depth` | bool | No | `false` | `GetImages` also returns a `depth` image. |
| `depth_width` | int | No | half of `width` | Depth image and point cloud width in pixels. |
| `depth_height` | int | No | half of `height` | Depth image and point cloud height in pixels. |

The frame's `parent` must be `world`.

### Supported API

| Method | Notes |
|---|---|
| `GetJointPositions` | Degrees. |
| `MoveToJointPositions` | Returns when the arm arrives, or fails after `move_timeout_sec`. |
| `MoveThroughJointPositions` | Pushes waypoints at `robot_control_freq_hz`; used by the motion service. |
| `MoveThroughJointPositionsStreamed` | Follows timestamped points in sim time. |
| `GetEndPosition` | The flange (link 6) in the arm's base frame, in mm. |
| `GetKinematics` | The vendored Lite6 SVA file, from [viam-ufactory-xarm](https://github.com/viam-modules/viam-ufactory-xarm). |
| `IsMoving` | True while a `MoveToJointPositions` or `MoveThroughJointPositions` call is in flight. |
| `Stop` | Holds the current target, or the measured position if it interrupted a trajectory. |
| `MoveToPosition` | Not supported; use the motion service. |

### DoCommand

| Command | Fields | Returns |
|---|---|---|
| `dynamics_metadata` | | `{source, joint_names, max_velocity_rad_s, max_acceleration_rad_s2}`; the last only when `max_joint_deg_s2` is set. |
| `get_joint_positions_radians` | | `{values: [six radians]}` |
| `set_joint_targets_radians` | `values`: six radians | `{"accepted": true}`; latches the target and returns at once. |

## Model viam-labs:mujoco-sim:gripper

The Lite6's integrated gripper, which comes with the arm. It is binary: open or
closed with a constant force. `Grab` and `Open` latch the command and return
before the jaws move, like the isaac-sim Lite6 gripper and the real one.

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

The frame's `parent` is the arm. A `z` of 81.1 mm puts the planner's tool
point at the fingertip center of the simulated gripper.

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `world` | string | Yes | | The world component's name. |
| `arm` | string | Yes | | The arm this gripper belongs to. |
| `asset` | string | No | `"lite6_gripper"` | The only option. |
| `sensorless` | bool | No | `false` | `IsHoldingSomething` reports the latched grip instead of contacts, like the real Lite6 gripper. |
| `grip_force_n` | float | No | `5` | Closing and opening force in newtons, greater than 0 and at most 10. |

### Supported API

| Method | Notes |
|---|---|
| `Open` | Latches open and returns at once. |
| `Grab` | Latches closed and returns `true` at once. |
| `Stop` | Releases the force. |
| `IsMoving` | True while the jaws move. |
| `IsHoldingSomething` | See below. |
| `GetCurrentInputs` / `GoToInputs` | One value: `0` open, `1` closed. |
| `GetKinematics` | Not supported, deliberately; the planner treats the gripper as a point. |

`IsHoldingSomething` always fills `meta`:

| Key | Meaning |
|---|---|
| `firm_grasp` | Both fingertips touch the same object. |
| `input` | The latched command, `0` or `1`. |
| `jaw_m` | Left finger position in meters, `open_m` to `closed_m`. |
| `open_m`, `closed_m` | The jaw limits. |

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

A Viam Rover 2, with the dimensions from Viam's rover fragment: a 381 mm wheel
circumference and a 356 mm track. It drives exactly as commanded; the wheels
are visual.

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

### Attributes

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `world` | string | Yes | | The world component's name. |
| `asset` | string | Yes | | The robot. Only `"viam_rover"` is supported. |
| `max_linear_mps` | float | No | `0.762` | Top speed in meters per second; scales `SetPower`. |
| `max_angular_dps` | float | No | `245` | Top yaw rate in degrees per second; scales `SetPower`. |
| `move_timeout_margin_sec` | float | No | `2` | Extra sim time `MoveStraight` and `Spin` get, beyond distance over speed, before they fail as stalled. |
| `position` | [x, y, z] | No | `[0, 0, 0]` | Placement in meters, used when no frame is set. |
| `orientation_wxyz` | [w, x, y, z] | No | | Orientation, used when no frame is set. |
| `orientation_rpy_deg` | [r, p, y] | No | | Same as roll, pitch, yaw in degrees; `orientation_wxyz` wins if both are set. |

The frame's `parent` must be `world`. +Y is forward, and the origin is at floor
level between the wheels.

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
| `MoveStraight` | Returns when the rover has gone the distance; a zero distance returns at once, and a zero velocity is rejected. Fails as stalled after distance / speed plus `move_timeout_margin_sec` of sim time, and when stopped, superseded, reset or closed. |
| `Spin` | Same as `MoveStraight`, for an angle: a zero angle returns at once and a zero velocity is rejected. |
| `SetVelocity` | `linear.y` in mm/s and `angular.z` in degrees/s; other axes are ignored. Clamped to the maxima. |
| `SetPower` | `linear.y` and `angular.z`, each in [-1, 1] of the maxima. |
| `Stop` | Stops the rover. |
| `IsMoving` | True while the rover is moving. |
| `GetProperties` | 0.356 m width, 0 turning radius, 0.381 m wheel circumference. |
| `GetGeometries` | The chassis box, 260 x 300 x 100 mm, centered 80 mm up. |

## Model viam-labs:mujoco-sim:lidar

A 2D LIDAR, like the rplidar module: it ray-casts a scan and serves it as a
point cloud. It is a camera that returns only point clouds.

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
| `position` | [x, y, z] | No | `[0, 0, 0.2]` | Placement in meters, used when no frame is set. |
| `orientation_wxyz` | [w, x, y, z] | No | | Orientation, used when no frame is set. |
| `orientation_rpy_deg` | [r, p, y] | No | | Same as roll, pitch, yaw in degrees; `orientation_wxyz` wins if both are set. |

The frame's `parent` is `world`, a mujoco-sim base or an arm. Like the camera,
it can't be framed on a gripper or another camera. The scan plane is the
frame's XY plane, and ray 0 points along +X. A mounted LIDAR never sees the
body it is mounted on, the rover chassis or an arm's last link, so mount it
clear of anything else on the robot.

### Supported API

| Method | Notes |
|---|---|
| `GetPointCloud` | Binary PCD with `x y z` in meters, in the LIDAR's frame. `z` is 0. Only hits are included. |
| `GetProperties` | `supports_pcd` is true. |
| `GetImages` | Not supported. |

## Model viam-labs:mujoco-sim:movement_sensor

Wheel odometry for a mujoco-sim base, like RDK's `wheeled-odometry`. It reads
the rover's motion from the sim and dead-reckons a pose every physics step, so
readings don't depend on how often you poll. It is exact unless you add drift.

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

## Model viam-labs:mujoco-sim:obstacle

A fixed box, sphere or capsule, such as a table, wall or fixture. Its frame's
`geometry` is its shape. RDK puts frame geometries in the frame system, so
Viam's motion planner avoids the same shape MuJoCo simulates: configure it once.

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

The frame's `parent` must be `world`, and it needs a `geometry`: `box` (`x`,
`y`, `z` in mm), `sphere` (`r`) or `capsule` (`r`, and `l` tip to tip along the
geometry's Z; `l` must be at least `2r`). The geometry's `translation` and
`orientation` offset it within the frame. Obstacles never move, and `reset`
leaves them where they are.

Use a capsule or box for round posts: Viam's geometry API has no cylinder yet
([viamrobotics/api#894](https://github.com/viamrobotics/api/pull/894)), and
viam-server can't pass a `cylinder` frame geometry to a module.

A goal inside an obstacle is refused at planning time. A path that only grazes
one can still fail in the sim: the planner checks the arm's kinematics
geometry, while MuJoCo collides the Lite6's links as convex hulls, which are
fatter, and the Lite6 kinematics give the gripper no geometry at all. The arm
then stops short of its goal (`did not converge on final waypoint`). For moves
past obstacles, pass a collision buffer to the motion service:

```python
await motion.move(component_name="pick-gripper", destination=goal,
                  extra={"collision_buffer_mm": 20})
```

The buffer applies to every obstacle, so leave it off for the final approach
onto a surface, such as placing on a pad. To have the planner see the gripper
itself, give its frame a geometry; this box covers the Lite6 gripper from the
fingertips back to the flange:

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
| `GetGeometries` | The frame geometry, labelled with the component's name. |

## Model viam-labs:mujoco-sim:world_state

A world state store that publishes every movable (non-fixed) prop with its
live pose and shape, so props show up and move in the Viam app's 3D scene
viewer. Obstacles, arms and bases already appear through the frame system. The
world state store doesn't feed the motion service.

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
| `ListUUIDs` | One per movable prop, stable across restarts. |
| `GetTransform` | The prop's pose in `world` (mm, degrees), its shape and `metadata.color` (0–255). |
| `StreamTransformChanges` | An ADDED event per prop, then UPDATED events for props that moved more than 1 mm or 0.5°, checked every 0.1 s. |

Cylinder props are sent as meshes, since the geometry API has no cylinder.

## Mobile base, SLAM and navigation

A simulated rover can map its world and navigate it with
[`viam-labs:nav-stack`](https://github.com/viam-labs/nav-stack), as a real Viam
Rover does. Viam's builtin motion service no longer plans for bases
(`MoveOnMap` and `MoveOnGlobe` were removed from RDK in late 2025), and the
builtin navigation service is GPS-only, so nav-stack is the path.
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

## Things to know

- **Config edits reset the cell.** viam-server rebuilds a module resource on any
  config change (close, then new). Editing an arm attribute sends the arm back
  to home and reopens its gripper. Editing the world rebuilds every component
  that depends on it. A move in flight fails with
  `arm '<name>' was closed (removed or reconfigured)`. Leave the config alone
  during a run.
- **Fixed props and obstacles never move.** `set_prop_pose` and `park_props`
  reject them, and `reset` leaves them where they are. Non-fixed props keep
  their pose across world edits; use `park_props` or `reset`.
- **No cylinder obstacles yet.** Viam's geometry API has no cylinder ([viamrobotics/api#894](https://github.com/viamrobotics/api/pull/894)), so use a capsule or box for a round post. Cylinder props work and appear in the scene viewer as meshes.
- **Arm self-collision is partial.** MuJoCo's convex hulls of the Lite6 links
  overlap at reachable wrist angles, so those wrist-link pairs are excluded.
  The gripper and links 3–6 still collide with the base and shoulder, which
  Viam's planner can't see.
- **Place components relative to `world`.** Arm and base frames must have
  parent `world`; the gripper's frame parent is its arm, and a camera's or
  LIDAR's is `world` or another mujoco-sim component.
- **The rover's motion is ideal.** It drives exactly as commanded, with no
  wheel slip; the wheels are visual. It can't tip over, and contacts stop it.
  The movement sensor's drift is set by attributes, not by physics.
- **A LIDAR scan is one instant.** All rays sample the same sim step, unlike a
  spinning sensor whose points span a rotation.
- **Depth is ideal.** It has no noise, holes or minimum range, unlike a real
  depth sensor. On macOS, MuJoCo warns that depth accuracy is limited; at 1 m
  it is still within 1 mm.
- **Kinematics are vendored.** `GetKinematics` serves
  `assets/ufactory_lite6/lite6_kinematics.json` (from
  viam-ufactory-xarm@73551eac). The `kinematics_url` attribute overrides it.

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

The Lite6 model is vendored from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(`assets/ufactory_lite6`, BSD-3-Clause).

To release, push a plain-semver tag such as `0.1.1`; the deploy workflow builds
and publishes it.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party
material.
