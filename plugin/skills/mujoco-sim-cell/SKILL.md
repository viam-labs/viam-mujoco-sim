---
name: mujoco-sim-cell
description: Build a working simulated robot cell on Viam with the viam-labs:mujoco-sim MuJoCo module from a plain-language description ("a UR5e picks blocks into a bin behind a wall"). Creates the machine, writes its config, runs viam-server locally, runs and checks a pick-and-place task, and iterates until it works. Use when the user wants to simulate, prototype or try out a robot arm, gripper, conveyor, gantry or mobile base cell with Viam, or wants a sim twin of an existing Viam machine to test its code.
---

# Build a mujoco-sim cell from a prompt

MODULE_VERSION: 0.18.0

Every config you write uses this registry module, pinned:

```json
{"type": "registry", "name": "viam-labs_mujoco-sim", "module_id": "viam-labs:mujoco-sim", "version": "<MODULE_VERSION>"}
```

Every template and doc you read comes from the same tag:
`https://raw.githubusercontent.com/viam-labs/viam-mujoco-sim/<MODULE_VERSION>/<path>`.
The README at that tag documents every model's attributes and DoCommands.

`cell.py` below means `uv run <this skill's directory>/scripts/cell.py`. Run `--help` for its usage. `check` and `pick` assume the world is named `sim-world`; pass `--world <name>` otherwise.

Prerequisites: `viam-server`, the `viam` CLI (logged in with `viam login`), `uv`, `git` (to clone asset repos), and the Viam MCP server (tools `create_machine`, `read_machine_metadata`, `read_machine_config`, `set_machine_config`, `read_machine_api`, `read_machine_logs`, `run_docommand`, `list_organizations`, `list_organization_locations`).

Keep a run report as you go and give it to the user at the end:

```
outcome: works | blocked: <why>
iterations: <n>   wall time: <min>
guesses: [every default or assumption you made, one line each]
```

## 1. Interpret

Turn the prompt into robot, end effector, objects, goal and sensors. Ask only about what you can't default, and say which defaults you chose. Add each default to `guesses`.

- An object with no bundled asset becomes a primitive. A plate, for example, becomes a thin fixed box prop.
- A named arm, gripper, object or environment that isn't bundled is sourced (step 2a), not substituted. Substitute only when no trusted source has it and the user doesn't give one, and record it as a scope change.
- Bundled arms: `lite6`, `ur5e`. Bundled grippers: `lite6_gripper` (the default, the Lite6's own), `robotiq_2f85`, `robotiq_epick`.
- Bundled bases: `viam_rover`, `clearpath_husky`, `pal_tiago_base`.
- Bundled robots: `aloha`, `tiago`.

## 2a. Custom assets

Only when something in the prompt isn't bundled: read `references/assets.md` (in this skill's directory) and follow it to source, place and check each asset before you compose. Custom assets need `MODULE_VERSION` 0.17.0 or later.

## 2b. Twin of a real machine

When the user names a real Viam machine to copy, read `references/twin.md` (in this skill's directory) and follow it instead of composing from scratch. It needs `MODULE_VERSION` 0.18.0 or later.

## 2. Compose

Fetch the closest example config and edit it:

| Example (`examples/…`) | What it is | Task script |
|---|---|---|
| `lite6-pick.json` | Lite6 and its gripper, a block, a bin mesh | none; use `cell.py pick` |
| `ur5e-pick.json` | UR5e + 2F-85, table, bin, wrist force-torque | none; use `cell.py pick` |
| `scene-pick.json` | Lite6 at an MJCF workbench scene | none |
| `vision-pick.json` / `vision-pick-suction.json` | UR5e, wrist camera, vision services, three bins | `vision_pick.py` |
| `gantry-sort.json` | XYZ gantry + ePick sorting into two bins | `gantry_sort.py` |
| `conveyor-pick.json` | UR5e + ePick picking off a running belt | `conveyor_pick.py` |
| `parcel-sort.json` | belt, spawner, photo-eye, scale, two bins | `parcel_sort.py` |
| `depalletize.json` | UR5e + ePick de-palletizing to a conveyor | `depalletize.py` |
| `rail-tending.json` | UR5e on a floor rail tending two machines | `rail_tending.py` |
| `husky-ur5e.json` | UR5e on a Husky between two tables | `mobile_pick.py` |
| `aloha-handover.json` | ALOHA bimanual handover | `aloha_handover.py` |
| `tiago-pick.json` | TIAGo++ table-to-shelf | `tiago_pick.py` |
| `rover-lidar.json`, `rover-nav.json`, `rover-avoid.json` | Viam Rover in a walled pen: LIDAR, nav, avoidance | `rover_avoid.py` |

Rules:
- **Replace the example's `local` module entry** with the registry entry above. Keep any other registry modules it lists.
- **Units differ.** Frame translations and geometry are in **mm**. Prop `position`, `size` and camera `target` are in **m**.
- Examples often park props out of reach, for example at x = 2.5 m. Place every prop where the task needs it, resting on its support: z = support top + half its height.
- **Props are invisible to the motion planner,** fixed or not. Every table, wall or bin the arm must avoid is an `obstacle` component. `cell.py pick` handles the prop being picked.
- **Always add an `overview` camera.** It is a `viam-labs:mujoco-sim:camera` on the world, about 1 m out and 0.7 m up from the work area, with `target` at the work area's center, 1280×720. It's how you see the cell.
- A gripper's frame `translation.z` is its TCP (fingertip center). Keep the example's value.
- The Lite6's own gripper needs a box `geometry` on its frame, 78×78×70 mm centered at z −35 mm, or the planner can clip it. `lite6-pick.json` doesn't have one, so add it. Check the README's gripper section for the others.

### Geometry reference

Measured in the sim at `MODULE_VERSION`, in mm. Check sizes against this before you compose, so you don't have to dig them out of meshes.

| Gripper | TCP (frame z) | Jaw gap, open → closed | Fingertips past the TCP | Takes |
|---|---|---|---|---|
| `lite6_gripper` | 81.1 | 39 → 21 | 0 (the tips are the TCP) | blocks about 22–38 wide; `--grasp-offset 0` clears the table |
| `robotiq_2f85` | 155.8 | 85 → 0 | 4.5 open, 17.7 closed (the pads swing down as they close) | blocks up to about 80 wide; raise `--grasp-offset` so the closed pads clear the support: 0.022 m worked on a 31 mm block |
| `robotiq_epick` | 130 | suction, one 45 mm cup | 0 | a flat top face at least the cup's size |

- **The open 2F-85 is about 151 mm across its jaws.** That's wider than the bundled bin's 140 mm inside, so let it go above the rim with `--release-gap`.
- **Bundled bin (`meshes/bin.stl`):** 150×150×80 outside, 5 mm walls and floor. Its origin is the center of its underside, so the inside floor is 5 mm above its frame.
- **Rest a table on the floor, never flush with it.** A box whose top is at z=0 (center at minus half its thickness) z-fights with the floor plane: camera images show the floor through it in jagged shards. Put its bottom on the floor (center z = half its thickness), then set the z of every prop and obstacle that rests on it (a bin, a fixture) from the table top. A bin left at z=0 inside the table's footprint has its floor buried in the table. Keep the arm's base on the table top or on the floor beside it.
- **`tcp_mm` from `measure` is the fingertips.** The bundled 2F-85 frame uses the pad centre, 155.8, about 4.5 mm short of its tips (`measure` says 160.3), so the table isn't wrong.
- **An object too wide for the gripper is a scope problem, not a tuning one.** Say so and offer another gripper (the 2F-85 on a UR5e) or a smaller object before you change anything.

For any other gripper or a mesh object, measure it: `cell.py measure --gripper <name>` or `--prop <name>`.

## 3. Provision

1. Confirm with the user before you create anything. If they didn't name an org and location, list them with `list_organizations` and `list_organization_locations` and ask.
2. Create the machine with `create_machine` and note its `machine_id`.
3. Run `cell.py creds --machine-id <id> --dir <scratch>/<machine>`. It creates the machine's API key with the `viam` CLI, finds its main part, and writes `viam.json` for viam-server and `cell.json` for everything else. Both hold secrets: keep them in the scratchpad and never print them. It prints the key's name and id; tell the user both, so they can revoke it later. Don't create keys yourself: the CLI's default key name is the user's login.
4. Start viam-server in the background, with output going to `<scratch>/<machine>/viam-server.log`:
   `viam-server -config <scratch>/<machine>/viam.json`

**Developer override:** only when the user asks to test unreleased module changes, use `{"type": "local", "name": "mujoco-sim", "executable_path": "<their checkout>/run.sh"}` in place of the registry entry. Read templates from their checkout too.

## 4. Bring up

Get the revision from `read_machine_config` (`parts[0].revision`), then write the whole config with `set_machine_config`. Then run `cell.py wait --cell <cell.json>`: it returns when the machine is running and every resource, module and package is ready, and on timeout lists what isn't with its error. Don't grep the log for readiness. It usually takes under a minute. **After every config write, run the world's `reset`** (`run_docommand` on the world with `{"command": "reset"}`). Rebuilding the scene keeps each prop's live pose, so a prop whose `position` you changed stays where it was until `reset`. The first start downloads the module, so allow up to 5 minutes (`wait`'s default timeout). If something isn't ready, its error says why: fix the config and write it again.

## 5. Look

- Look through the overview camera with `read_machine_api`, `method: "GetImages"`, `resource_name: "overview"`, `request: {"filter_source_names": ["color"]}`. The image comes back to you and is saved to a file; note that path for the hand-off.
- `cell.py check --cell <cell.json> --prop <name> --goal <its spawn x y z>` for every prop. It should be `at_rest` and `at_goal` true (at its spawn position). A prop that moved, sank or fell means the layout is wrong (overlap, or no support under it). Fix it before you run anything.

## 6. Run the task

- **A top-down pick and place of one prop:** run `cell.py pick --cell <cell.json> --gripper <name> --prop <name> --size <m> --place X Y Z`. `--place` is where the prop's center should end up. `--size` is the prop's edge; it's treated as a cube, so for any other shape pass its body's largest dimension (ignore a handle). It only shapes what the planner avoids and carries.
  - It passes the prop to the planner itself: as an obstacle on the approach, left out for the straight descent onto it, and as a box carried on the gripper while held, so the planner keeps the held prop clear of walls and rims. Near the prop while not holding it, it moves vertically in 10 mm hops, because the planner can curve a longer move through the prop. While holding, it makes as few moves as it can.
  - `--grasp-offset` (m above the prop's center) is the fix when the finger pads would hit the table. The 2F-85 needed about 0.022 on a 31 mm block. `--release-gap` (default 0.005) is how far above its goal the prop is let go.
  - A pick is about 25 moves, most of them short hops, so it takes a minute or two; it hasn't hung.
  - A move that doesn't plan within 60 s is tried again with the gripper turned 180° about its axis (a jaw gripper closes along the same line). That makes IK choose another arm configuration, and the Panda's first choice above a prop stalls the short hops. The turn is kept for the rest of the pick, so a held prop arrives rotated 180°.
  - On a planner refusal it prints the move, the errors from both tries (the colliding geometries are named in them), and which flag to change.
- **Anything else:** fetch the matching example script and write your own in the scratchpad. It should `sys.path.insert(0, "<this skill's directory>/scripts")` and then `import cell` for `cell.connect` and `cell.settle_check`, and end with a `settle_check` on every prop that has a goal. Give it its own PEP 723 header so `uv run` installs viam-sdk: `# /// script` / `# dependencies = ["viam-sdk>=0.85"]` / `# ///`.

The task works when every goal prop reports `at_goal: true` (within 2 cm and at rest). Then take the "after" look (`GetImages` again).

## 7. Iterate

Take a `GetImages` look on every attempt. On a failure, read the log (`read_machine_logs`), find the cause, change one thing, run `reset` on the world (`run_docommand` `{"command": "reset"}`), and run again. Typical causes:
- planning fails → a goal out of reach, or an obstacle in the way. Fix it with the grasp or release height, `--hover` or the goal, **not by changing the task's objects**. If an object's size must change, say so as a scope change in the report;
- your own script sweeps through a prop → props aren't obstacles unless you pass them in a `WorldState`. Approach in short vertical hops: the default planner refuses a `LinearConstraint` (`linear with cbirrt not allowed`);
- grab fails → the wrong height, or the object is too big for the gripper;
- the object drops → speed or friction.

Stop after 5 attempts at the same kind of failure and report what blocked it.

## 8. Hand off

Give the user:
- the machine's app link (its 3D scene tab shows the cell);
- the final config;
- the task command or script;
- the before and after images (the files `GetImages` saved);
- the run report.

Then stop viam-server unless they want it running, and ask whether to keep or delete the machine.

## Guardrails

- Confirm before you create a machine. Write config only to the machine you created, never into fragments.
- Always stop any viam-server you started.
- For a grip sent through MCP, use `disable_sessions`, and only for grips. It doesn't apply to grips from `cell.py` or your own scripts.
- Never print or commit `viam.json`, `cell.json` or keys.
