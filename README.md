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
`set_prop_pose`. With an arm and gripper configured, the builtin motion service
plans for the arm as it would for a real one.

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
      },
      {
        "name": "place_pad",
        "fixed": true,
        "size": 0.2,
        "scale": [1, 1, 0.05],
        "position": [0.32, -0.22, 0.005]
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
| `props` | list | No | `[]` | Cubes in the scene; see below. |

Each prop:

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | Yes | | Unique prop name, used by the prop commands. |
| `type` | string | No | `"cube"` | Only `"cube"` is supported. |
| `size` | float | No | `0.05` | Edge length in meters. |
| `scale` | [x, y, z] | No | `[1, 1, 1]` | Per-axis scale of `size`. |
| `position` | [x, y, z] | No | `[0, 0, 0]` | Spawn position in meters. |
| `orientation_wxyz` | [w, x, y, z] | No | `[1, 0, 0, 0]` | Spawn orientation. |
| `mass` | float | No | `0.05` | Mass in kg. |
| `color` | [r, g, b] | No | `[0.7, 0.7, 0.7]` | RGB, 0 to 1. |
| `fixed` | bool | No | `false` | A fixed prop never moves. |
| `static_friction` | float | No | `1.0` | Sliding friction, non-negative. |
| `dynamic_friction` | float | No | | Accepted so isaac-sim configs carry over; MuJoCo has one sliding coefficient, so it is ignored. |

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

The frame's `parent` is `world` or a `viam-labs:mujoco-sim:arm`. A frame
orientation is read in Viam's camera convention: +Z forward, +Y down.

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

## Things to know

- **Config edits reset the cell.** viam-server rebuilds a module resource on any
  config change (close, then new). Editing an arm attribute sends the arm back
  to home and reopens its gripper. Editing the world rebuilds every component
  that depends on it. A move in flight fails with
  `arm '<name>' was closed (removed or reconfigured)`. Leave the config alone
  during a run.
- **Fixed props never move.** `set_prop_pose` and `park_props` reject them, and
  `reset` leaves them where they are. Non-fixed props keep their pose across
  world edits; use `park_props` or `reset`.
- **Arm self-collision is partial.** MuJoCo's convex hulls of the Lite6 links
  overlap at reachable wrist angles, so those wrist-link pairs are excluded.
  The gripper and links 3–6 still collide with the base and shoulder, which
  Viam's planner can't see.
- **Place components relative to `world`.** Arm frames must have parent
  `world`; the gripper's frame parent is its arm, and a camera's is `world` or
  an arm.
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
