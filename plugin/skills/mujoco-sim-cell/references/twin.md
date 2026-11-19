# Sim twin of a real machine

Read this when the user names a real Viam machine to copy. The twin keeps every resource name, API and frame of the real machine, so the user's own code runs against it with only the address and credentials changed. It's for testing code, not for pixel-perfect looks.

## 1. Convert

Run `uv run cell.py twin --from-machine <id> --org-id <its organization ID> --dir <scratch>/twin --module-version <MODULE_VERSION>`. With `--live`, pass `--org-id`: when the machine's location is shared by several organizations, the CLI can't tell which one the read key belongs to. It reads the real config through the user's CLI login, fragments included, and changes nothing on the real machine. It prints the output paths, whether it was live, outcome counts and the number of flags.

Add `--live` only after asking the user: it creates an API key on the real machine (saved to `real.json` right after it's made, never printed; a rerun reuses it) and reads each camera's image and intrinsics and each arm's kinematics and joint positions. Nothing on the real machine moves. If the machine is offline, it falls back to the config and flags `unreachable`.

Read `twin-report.json`. Its `world` and `overview_camera` are the twin's world and overview camera; they are renamed if the real machine already uses `sim-world` or `twin-overview`. Pass `--world <world>` to `check`, `pick` and `measure`, and use that camera for SKILL.md's looks.

Each real resource has an outcome:
- `mapped`, `reframed`, `reparented`: a sim arm, gripper or camera. A `reframed` arm was mounted on something else (a rail); it now stands where that parent's zero puts it. Tell the user. A camera on a gripper is `reparented` onto its arm.
- `needs_asset`: a fake stands in. Source the arm or gripper with `assets.md`, then replace the fake (and the fakes on it) with mujoco-sim models under the same names and frames.
- `copied`, `faked`, `removed`, `unsupported`: as named. `removed` is data capture: sim images must never reach the real machine's datasets.
- `disabled`: disabled on the real machine, copied as it is and still disabled. The checks leave it out.

Flags (each with a note):
- `guessed`: a camera's size or fov had no source and took a default.
- `config_only`: a camera's size or fov came from config intrinsics, or an arm has no saved kinematics for `twin-check` to compare.
- `source_names_differ`: the real camera serves image names the twin doesn't, so the user's code may ask for one the twin lacks.
- `needs_review`: a local path or local module on the real host; a module it can't attribute to a resource, kept; a service that depends on a dropped resource; a camera with no frame; a fragment mod or import reported as "<op> not applied" (a fragment import with a prefix or repeated ids shows as `fragment_import`). Check each by hand.
- Secrets (module `env` and resource attributes whose name or value looks like a key, token, password or webhook) are removed from the twin config and flagged `needs_review` with their paths only, and `rdk:service:shell` is removed; never copy them back from `real-config.json`, and ask the user for a twin-specific value if one is needed (a removed value that turns out not to be a secret can be copied back by hand once checked).
- `data_capture_removed`: capture configs were dropped with the hardware.
- `live_failed`: a live read for that resource failed; its values come from the config.
- `unreachable`: the real machine couldn't be reached; everything comes from the config.
- `needs_review` "variable X has no value": variables take the values from their fragment import, else their defaults; one with neither is left out (a frame parent becomes the world). With `--live` the frames come from the real machine's frame system (`frames_from_live` lists them).

`erh:vmodutils` components are copied as they are; those whose model name contains "obstacle" are imported into the world as geometry.

- `reframed`: also appears on a faked or copied resource whose frame parent isn't in the twin; it was placed in the world, or on the nearest parent that survived. Check that is where it belongs.

## 2. Create the twin machine

Create it in the real machine's org and location after confirming with the user, then follow SKILL.md steps 3-4 with `<scratch>/twin/machine-config.json` as the whole config (provision, start viam-server, write the config, `wait`). Don't compose a new config: the converted one carries the real names and frames. Then, when the report's `files.joints` is set, run `cell.py twin-pose --cell <cell.json> --dir <scratch>/twin`. It prints `{"moved", "failed"}` (or `skipped` when there is no `joints.json`); look at `failed`. A world `reset` returns the arms home, not to the real pose; run `twin-pose` again after one. `twin-pose` drives the joints straight there without planning, so it can sweep through props: follow it with the world's `park_props` for every prop it may have hit. With a real gripper on, the hand can jam against a fixture on the way and the move times out: `reset` (arms go home), then `twin-pose` again. A URDF arm's home is all zeros, which can put a long gripper into the table: run `twin-pose` after every `reset`.

## 3. What the config lacks

Look at each `real_<camera>.jpg` (or photos from the user). Add what nothing in the config describes:
- tables and fixtures that aren't obstacles in the real config, as sim obstacles;
- workpieces, bins and parts, as props.

Size and place them from things of known size in view: the arm's base, the gripper, imported obstacles. When two estimates disagree by more than about 2 cm, ask the user to measure. Record each item in the run report as `from_photo` with its estimate.

- **The floor.** The real frames put the origin where the real machine has it, often the arm's base on a tabletop, so the sim floor (z = 0) cuts through the table and z-fights with its top. Set the world's `floor_z` to the real floor's height below the origin, from the photo or the user (a workbench is about -0.9 m). It's the real floor, not the bottom of a table obstacle: a tabletop slab in the real config is usually far thinner than the table is tall, and its legs go in a `scene` file down to the floor. It needs `MODULE_VERSION` 0.18.0 or later.
- **Saved poses locate things.** Joint positions saved in the config (an `arm-position-saver`, a teach pendant's poses) are exact: run them through the real kinematics (`kin_<arm>.json`) to find where the gripper touched a drawer, a bin or a fixture, and place it there rather than by eye.
- **Fixtures that move.** A drawer or door the real task opens belongs in a `scene` MJCF with a passive slide or hinge joint, not an obstacle. Exclude contacts between the moving body and the carcass it slides in, and don't give the file its own floor plane.
- **The arm in the 3D viewer.** The viewer draws an arm from its served kinematics. A real driver's kinematics file often has no geometry, so prefer serving the URDF you sourced (with a fixed link for any flange turn the real driver has; check it with `twin-check`) over `kinematics_url` pointing at the real file, and set `visual_models: true` for its visual meshes.
- **Props.** Spawn them apart: two props that overlap at spawn are thrown out of a bin. The world returns props to their spawn poses once `import_geometry` first lands, so props can rest on imported obstacles without a `reset`.

## 4. Look

Floor and table textures from crops of the real images (`texture`, `floor_texture`), lighting roughly matched. Stop when the overlay lines up; don't chase pixels.

## 5. Check

Run `uv run cell.py twin-check --cell <cell.json> --dir <scratch>/twin`. It prints JSON and exits 1 on a names, frames or kinematics failure; images never fail it.
- `names.missing`: the converter or an edit dropped a resource. Put it back under its real name.
- `frames.deltas`: a frame changed after conversion (over 1 mm or 0.5 degrees, or a different parent). Restore it from `twin-report.json`'s `expected_frames`.
- `kinematics`: always compared for each arm with saved real kinematics. Not ok means the joint count, a joint limit or a sample pose differs. Over 2 mm or 1 degree the twin may plan differently; tell the user the deltas. An arm with the note "fails while the twin arm is still a fake" was `needs_asset`: expect the failure until you replace the fake. For a bundled arm, a failure is a bug in the mapping: report it. `skipped` ("no real kinematics saved") usually means the real machine was offline; a `live_failed` flag on an arm means its read failed.
- `images[<camera>]`: `overlay` is the path to `side_<camera>.png`, the sim's edges in magenta over the real image; `twin` and `real` give each camera's `size` and `fov_deg`. An `error` means the twin camera gave no image. Read each overlay, move obstacles and props until the edges line up to about a gripper finger's width, then run `twin-check` again.

## 6. Run their code

If the user has a script, run it against the twin with only the address and credentials changed, and report what it did. Otherwise run `cell.py pick` on a prop placed from the photos.

## 7. Report

Mapped, faked, removed, unsupported and estimated items; kinematics deltas; which values came from live calls and which from the config alone. If `--live` made a key, give its name from `real.json`'s `key_name` (not the key, and never print `real.json`) and tell the user to revoke it. Then hand off as in SKILL.md step 8.
