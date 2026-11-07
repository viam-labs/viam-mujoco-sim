# viam-mujoco-sim

A proof-of-concept [Viam](https://www.viam.com) module that simulates robots in
[MuJoCo](https://mujoco.org). It is a drop-in sibling of
[`viam-labs:isaac-sim`](../viam-isaac-sim): the same component attributes, the
same world `DoCommand`s, and the standard Viam frame for placement, so a config
moves between engines by changing the model triplet.

Physics runs on the CPU and the module is a few tens of MB, so it runs on a Mac
as well as on Linux. Cameras render offscreen with OpenGL: CGL on macOS, EGL
(the GPU) on Linux.

## Models

| Model | API | What it does |
|---|---|---|
| `viam-labs:mujoco-sim:world` | `generic` | Owns the scene, props and step loop. Configure exactly one. |
| `viam-labs:mujoco-sim:arm` | `arm` | Attaches a known arm (`asset: "lite6"`) at its frame and exposes joint control. |
| `viam-labs:mujoco-sim:gripper` | `gripper` | The Lite6's integrated binary gripper. |
| `viam-labs:mujoco-sim:camera` | `camera` | An RGB camera placed by its frame, aimed with `target`. |

See the docstring at the top of each file in `src/mujoco_module/models/` for
attributes, and `examples/lite6-pick.json` for a complete pick cell.

## Run it locally

```sh
make run.sh   # the entrypoint is generated, see Makefile
```

Point a machine's config at `run.sh` as a local module (see
`examples/lite6-pick.json`), start `viam-server`, then drive it with any Viam
client, for example `../viam-isaac-sim/examples/scripted_pick.py`.

## Things to know

- **Config edits reset the cell.** viam-server rebuilds a module resource on any
  config change (close, then new). Editing an arm attribute sends the arm back
  to home and reopens its gripper. Editing the world rebuilds every component
  that depends on it. A move in flight fails with
  `arm '<name>' was closed (removed or reconfigured)`. Leave the config alone
  during a run.
- **Prop poses survive world edits.** Changing a prop's `position` in the world
  config doesn't move an existing prop. Use the world's `park_props` or `reset`
  DoCommand.
- **Kinematics are vendored.** `GetKinematics` serves
  `assets/ufactory_lite6/lite6_kinematics.json` (from
  viam-ufactory-xarm@73551eac). The `kinematics_url` attribute overrides it.
- **Grab returns immediately.** Grab and Open latch the command and return
  before the jaws move, as in the isaac-sim Lite6 gripper. Poll
  IsHoldingSomething (`meta.firm_grasp`) to wait for a grasp.

## Develop

```sh
make test
make lint
```

Tests run real MuJoCo; there is no mock backend. On Linux without a display set
`MUJOCO_GL=osmesa` (`apt install libosmesa6`).

The Lite6 model is vendored from
[mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(`assets/ufactory_lite6`, BSD license).
