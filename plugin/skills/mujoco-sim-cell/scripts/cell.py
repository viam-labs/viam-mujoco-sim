# /// script
# requires-python = ">=3.11"
# dependencies = ["viam-sdk>=0.85", "numpy", "pillow"]
# ///
"""Helpers for the mujoco-sim-cell skill.

  uv run cell.py creds --machine-id ID --dir DIR [--name KEY_NAME] [--org-id ORG]
      create a machine API key with the viam CLI (named, so not after your login), then write
      DIR/viam.json (for viam-server) and DIR/cell.json (address and key, for the rest); prints
      the key's name and id, never the key
  uv run cell.py wait --cell DIR/cell.json [--timeout S]
      wait until the machine runs and every resource, module and package is ready; print JSON,
      or on timeout list what isn't ready with its error and exit 1
  uv run cell.py check --cell DIR/cell.json --prop NAME [--goal X Y Z] [--world sim-world]
      print JSON: the prop's position (m), at_rest, and at_goal / error_m with --goal
  uv run cell.py pick --cell DIR/cell.json --gripper G --prop NAME --size M --place X Y Z
                      [--grasp-offset M] [--release-gap M] [--hover M]
      oracle top-down pick and place through the motion service, then the same check as `check`.
      The planner sees the prop as an obstacle on the approach and as carried on the gripper while
      held; moves near it while not holding it go straight up and down in 10 mm hops. A move that
      fails to plan within 60 s is tried again with the tool turned 180 degrees, and the turn is kept.
  uv run cell.py measure --cell DIR/cell.json (--gripper G | --prop NAME)
      a gripper's TCP, jaw gap and finger reach (mm), or a prop's size and top (m)

  uv run cell.py twin --from-machine ID --dir DIR --module-version V [--live] [--org-id ORG]
      a sim twin's config of a real machine (DIR/machine-config.json) and what was mapped, faked,
      copied, removed or flagged (DIR/twin-report.json). The config comes through your CLI login.
      --live also creates a key on the real machine (DIR/real.json; never printed) and saves its
      camera images, intrinsics, arm kinematics and joint positions. Nothing on it moves.
  uv run cell.py twin-pose --cell DIR/cell.json --dir TWIN_DIR
      move each twin arm to the real arm's saved joint positions
  uv run cell.py twin-check --cell DIR/cell.json --dir TWIN_DIR
      compare the running twin with what `twin` saved: resource names, frames, arm kinematics, and
      side_<camera>.png (the real image with the sim's edges over it). Exits 1 on a name, frame or
      kinematics failure; the images never fail it

Task scripts of your own: sys.path.insert(0, "<this dir>"); import cell;
then `async with cell.connect("DIR/cell.json") as robot:` and cell.judge(...).
"""

import argparse
import asyncio
import datetime
import json
import math
import os
import re
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

GOAL_TOL_M = 0.02  # spec: within 2 cm of the goal
REST_TOL_M = 0.002  # spec: two reads 0.5 s apart move less than 2 mm
REST_GAP_S = 0.5
DOWN = {"o_x": 0, "o_y": 0, "o_z": -1, "theta": 0}  # gripper pointing straight down
HOP_M = 0.01  # longest vertical move near the prop (10 mm hops worked in the baseline)
BREAKAWAY_M = 0.02  # first lift off the support, short so it stays straight
# Per planning try. RDK's default is 5 min, and a move that stalls in a bad IK branch uses all of it.
PLAN_TIMEOUT_S = 60


def judge(before, after, goal=None, goal_tol=GOAL_TOL_M, rest_tol=REST_TOL_M):
    """Two position reads REST_GAP_S apart, in m: at rest, and how far from the goal."""
    out = {"position": list(after), "at_rest": math.dist(before, after) < rest_tol}
    if goal is not None:
        out["error_m"] = round(math.dist(after, goal), 6)
        out["at_goal"] = out["at_rest"] and out["error_m"] < goal_tol
    return out


def hops(z_from, z_to, hop):
    """Heights (m) from z_from to z_to, no step longer than hop, ending on z_to. Short moves keep
    a vertical approach straight: the planner can curve a long one through the prop it can't see."""
    n = max(1, math.ceil(round(abs(z_to - z_from) / hop, 9)))
    return [round(z_from + (z_to - z_from) * k / n, 6) for k in range(1, n + 1)]


def pick_waypoints(prop, place, hover=0.1, grasp_offset=0.0, release_gap=0.005, hop=HOP_M):
    """Gripper TCP waypoints (mm) for a top-down pick of the prop centred at `prop` (m) to `place`
    (its centre at the goal, m): (x, y, z, target, action after arriving). The TCP grasps
    grasp_offset above the prop's centre and lets go release_gap above the goal. target says how
    the planner sees the prop: "avoid" an obstacle, "near" left out, "held" carried on the gripper."""
    def mm(v):
        return round(v * 1000, 3)

    (px, py, pz), (gx, gy, gz) = prop, place
    grasp, release = pz + grasp_offset, gz + release_gap + grasp_offset
    over_prop, over_goal = grasp + hover, gz + grasp_offset + hover

    def vertical(x, y, z_from, z_to, target, action):
        zs = hops(z_from, z_to, hop)
        return [(mm(x), mm(y), mm(z), target, action if i == len(zs) - 1 else None) for i, z in enumerate(zs)]

    return [
        (mm(px), mm(py), mm(over_prop), "avoid", None),
        *vertical(px, py, over_prop, grasp, "near", "grab"),
        # held: the planner sees the carried box, so no hops (every start and stop costs grip);
        # a short breakaway first, since collisions at a move's start (prop on table) are ignored
        (mm(px), mm(py), mm(grasp + BREAKAWAY_M), "held", None),
        (mm(px), mm(py), mm(over_prop), "held", "lifted"),
        (mm(gx), mm(gy), mm(over_goal), "held", None),
        (mm(gx), mm(gy), mm(release), "held", "open"),
        *vertical(gx, gy, release, over_goal, "near", None),
    ]


@asynccontextmanager
async def connect(cell_json, attempts=None, timeout=None):
    """Dial the machine in cell_json. attempts and timeout (s) make a machine that isn't there fail fast."""
    from viam.robot.client import RobotClient

    c = json.loads(Path(cell_json).read_text())
    opts = RobotClient.Options.with_api_key(api_key=c["key"], api_key_id=c["key_id"])
    if attempts:
        opts.dial_options.initial_connection_attempts = attempts
        opts.dial_options.max_reconnect_attempts = 0
    if timeout:
        opts.dial_options.timeout = opts.dial_options.initial_connection_attempt_timeout = timeout
    robot = await RobotClient.at_address(c["address"], opts)
    try:
        yield robot
    finally:
        await robot.close()


async def prop_position(world, prop):
    return (await world.do_command({"command": "get_prop_pose", "prop": prop}))["pose"]["position"]


async def settle_check(world, prop, goal=None):
    before = await prop_position(world, prop)
    await asyncio.sleep(REST_GAP_S)
    return judge(before, await prop_position(world, prop), goal)


def write_private(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)  # O_CREAT ignores the mode for an existing file
    with os.fdopen(fd, "w") as f:
        f.write(text)


def parse_key(output):
    """(key id, key) from `viam machines api-key create` output. The error never quotes the output."""
    found = dict(re.findall(r"^\s*key (id|value)\s*:\s*(\S+)", output, re.IGNORECASE | re.MULTILINE))
    found = {k.lower(): v for k, v in found.items()}
    if "id" not in found or "value" not in found:
        raise ValueError("couldn't read the key id and value from `viam machines api-key create`")
    return found["id"], found["value"]


def create_key(machine_id, name, org_id=None):
    """A new machine API key from the viam CLI, as (id, key)."""
    command = ["viam", "machines", "api-key", "create", "--machine-id", machine_id, "--name", name]
    if org_id:
        command += ["--org-id", org_id]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode:
        raise SystemExit(json.dumps({"failed": "api-key create", "error": done.stderr.strip()[-500:]}))
    return parse_key(done.stdout)


async def main_part(key_id, key, machine_id):
    from viam.app.viam_client import ViamClient
    from viam.rpc.dial import DialOptions

    async with await ViamClient.create_from_dial_options(DialOptions.with_api_key(key, key_id)) as client:
        parts = await client.app_client.get_robot_parts(machine_id)
    return next(p for p in parts if p.main_part)


async def creds(args):
    name = args.name or f"mujoco-sim-cell-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    key_id, key = await asyncio.to_thread(create_key, args.machine_id, name, args.org_id)
    part = await main_part(key_id, key, args.machine_id)
    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    viam_json = {"cloud": {"app_address": "https://app.viam.com:443", "id": part.id, "secret": part.secret}}
    write_private(d / "viam.json", json.dumps(viam_json))
    write_private(d / "cell.json", json.dumps({"address": part.fqdn, "key_id": key_id, "key": key}))
    print(json.dumps({"viam_json": str(d / "viam.json"), "cell_json": str(d / "cell.json"), "address": part.fqdn,
                      "part_id": part.id, "key_name": name, "key_id": key_id}))


def not_ready(status):
    """Everything in a GetMachineStatus reply short of ready, with its state and any error."""
    from viam.proto.robot import ModuleStatus, PackageStatus, ResourceStatus

    out = []
    if status.state != type(status).STATE_RUNNING:
        out.append({"what": "machine", "state": type(status).State.Name(status.state).removeprefix("STATE_")})

    def add(what, enum, item, ready):
        if item.state != ready:
            entry = {"what": what, "state": enum.Name(item.state).removeprefix("STATE_")}
            if item.error:
                entry["error"] = item.error
            out.append(entry)

    for r in status.resources:
        n = r.name
        add(f"{n.namespace}:{n.type}:{n.subtype}/{n.name}", ResourceStatus.State, r, ResourceStatus.STATE_READY)
    for m in status.modules:
        add(f"module {m.module_name}", ModuleStatus.State, m, ModuleStatus.STATE_READY)
    for p in status.packages:
        add(f"package {p.name}", PackageStatus.State, p, PackageStatus.STATE_READY)
    return out


UNREACHABLE_S = 60  # a machine that never answers this long is down, not starting


async def wait(args):
    """Poll the machine (reconnecting while viam-server starts) until nothing is short of ready."""
    import logging

    from viam import logging as viam_logging

    # each failed dial logs an error from the SDK and its Rust dialer; the JSON line is the answer
    os.environ.setdefault("RUST_LOG", "off")
    viam_logging.setLevel(logging.CRITICAL)
    start = time.monotonic()
    last, down_since = [{"what": "machine", "state": "unreachable"}], start
    while time.monotonic() - start < args.timeout:
        try:
            async with connect(args.cell) as robot:
                down_since = None
                while time.monotonic() - start < args.timeout:
                    last = not_ready(await robot.get_machine_status())
                    if not last:
                        print(json.dumps({"ready": True, "seconds": round(time.monotonic() - start)}))
                        return
                    await asyncio.sleep(2)
        except Exception as e:  # not up yet, or the connection dropped while it restarts
            down_since = down_since or time.monotonic()
            last = [{"what": "machine", "state": "unreachable", "error": f"{type(e).__name__}: {e}"[:300]}]
            if time.monotonic() - down_since > UNREACHABLE_S:
                raise SystemExit(json.dumps({
                    "ready": False, "seconds": round(time.monotonic() - start), "not_ready": last,
                    "hint": f"no answer for {UNREACHABLE_S} s: check viam-server is still running and read the "
                            "end of its log (a port already in use stops it at startup)"}))
            await asyncio.sleep(3)
    raise SystemExit(json.dumps({"ready": False, "seconds": round(time.monotonic() - start), "not_ready": last}))


async def check(args):
    from viam.components.generic import Generic

    async with connect(args.cell) as robot:
        print(json.dumps(await settle_check(Generic.from_robot(robot, args.world), args.prop, args.goal)))


PLAN_HINT = ("the error names the two geometries that collide: a gripper part vs the table or the prop's"
             " support → raise --grasp-offset; vs the goal surface → raise --release-gap; vs a wall or rim"
             " while carrying → raise --hover or move the goal")


def world_state(target, prop_mm, size_mm, gripper, grasp_offset_mm):
    """What the planner knows of the prop for a move: an obstacle (a sphere around the cube, whatever
    its yaw), nothing, or a cube on the gripper with its centre grasp_offset along the tool."""
    from viam.proto.common import (
        Geometry, GeometriesInFrame, Pose, PoseInFrame, RectangularPrism, Sphere, Transform, Vector3, WorldState,
    )

    if target == "avoid":
        x, y, z = prop_mm
        obstacle = Geometry(center=Pose(x=x, y=y, z=z, o_z=1), sphere=Sphere(radius_mm=size_mm * math.sqrt(3) / 2),
                            label="target")
        return WorldState(obstacles=[GeometriesInFrame(reference_frame="world", geometries=[obstacle])])
    if target == "held":
        held = Transform(
            reference_frame="held_target",
            pose_in_observer_frame=PoseInFrame(reference_frame=gripper, pose=Pose(z=grasp_offset_mm, o_z=1)),
            physical_object=Geometry(center=Pose(o_z=1), label="held_target",  # RDK needs a centre pose
                                     box=RectangularPrism(dims_mm=Vector3(x=size_mm, y=size_mm, z=size_mm))),
        )
        return WorldState(transforms=[held])
    return None


async def move_down(motion, gripper, xyz_mm, state, theta):
    """Move the gripper, pointing down, to xyz_mm in the world; return the tool yaw that planned.

    If the planner fails at `theta`, try once more turned 180 degrees about the tool axis: a jaw
    gripper closes along the same line either way, and the turn makes IK pick another arm
    configuration (the Panda's first choice above a prop stalls the short hops near it)."""
    from viam.proto.common import Pose, PoseInFrame

    x, y, z = xyz_mm
    errors = []
    for t in (theta, (theta + 180) % 360):
        goal = PoseInFrame(reference_frame="world", pose=Pose(x=x, y=y, z=z, **{**DOWN, "theta": t}))
        try:
            await motion.move(component_name=gripper, destination=goal, world_state=state,
                              extra={"timeout": PLAN_TIMEOUT_S})
            return t
        except Exception as e:  # mostly planner refusals; the error class tells a dropped connection apart
            errors.append(f"theta {t}: {type(e).__name__}: {e}")
    raise RuntimeError("; ".join(errors))


async def pick(args):
    from viam.components.generic import Generic
    from viam.components.gripper import Gripper
    from viam.services.motion import MotionClient

    async with connect(args.cell) as robot:
        world = Generic.from_robot(robot, args.world)
        gripper = Gripper.from_robot(robot, args.gripper)
        motion = MotionClient.from_robot(robot, "builtin")
        await gripper.open()
        prop = await prop_position(world, args.prop)
        waypoints = pick_waypoints(prop, args.place, args.hover, args.grasp_offset, args.release_gap)
        theta = DOWN["theta"]  # once a turn plans, keep it: the next moves start from that arm configuration
        try:
            for i, (x, y, z, target, action) in enumerate(waypoints):
                state = world_state(target, [v * 1000 for v in prop], args.size * 1000, args.gripper,
                                    args.grasp_offset * 1000)
                try:
                    theta = await move_down(motion, args.gripper, (x, y, z), state, theta)
                except RuntimeError as e:
                    raise SystemExit(json.dumps({"failed": "plan", "waypoint": i, "of": len(waypoints),
                                                 "tcp_mm": [x, y, z], "error": str(e), "hint": PLAN_HINT}))
                if action == "grab":
                    await gripper.grab()  # returns before the jaws close; give them up to 2 s
                    for _ in range(20):
                        if (await gripper.is_holding_something()).is_holding_something:
                            break
                        await asyncio.sleep(0.1)
                    await asyncio.sleep(0.3)
                elif action == "lifted":  # a sensorless gripper always "holds", so trust the prop's z
                    z_now = (await prop_position(world, args.prop))[2]
                    if z_now < prop[2] + args.hover / 2:
                        raise SystemExit(json.dumps({"failed": "lift", "prop": args.prop, "z_m": z_now}))
                elif action == "open":
                    await gripper.open()
                    await asyncio.sleep(0.5)
        finally:
            await gripper.open()
        print(json.dumps(await settle_check(world, args.prop, args.place)))


async def measure(args):
    from viam.components.generic import Generic
    from viam.components.gripper import Gripper

    async with connect(args.cell) as robot:
        if args.gripper:
            out = await Gripper.from_robot(robot, args.gripper).do_command({"command": "measure"})
        else:
            out = await Generic.from_robot(robot, args.world).do_command({"command": "measure", "prop": args.prop})
    print(json.dumps(out))


async def twin_cmd(args):
    import twin

    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    real, unapplied = await asyncio.to_thread(twin.read_real_config, args.from_machine)
    unresolved = [m for m in unapplied if m["op"] == "variable"]
    unapplied = [m for m in unapplied if m["op"] != "variable"]
    live = {}
    if args.live:
        import logging

        from viam import logging as viam_logging
        from viam.components.arm import Arm  # noqa: F401 - registers the API before dialing, so from_robot finds arms
        from viam.components.camera import Camera  # noqa: F401 - and cameras

        os.environ.setdefault("RUST_LOG", "off")  # as in `wait`: the JSON line is the answer
        viam_logging.setLevel(logging.CRITICAL)
        real_json = d / "real.json"
        try:
            saved = json.loads(real_json.read_text()) if real_json.exists() else {}
            if "key" not in saved:  # a key made on an earlier run is reused, never made twice
                name = f"mujoco-sim-twin-read-{datetime.datetime.now():%Y%m%d-%H%M%S}"
                key_id, key = await asyncio.to_thread(create_key, args.from_machine, name, args.org_id)
                saved = {"key_id": key_id, "key": key, "key_name": name}
                write_private(real_json, json.dumps(saved))
            if "address" not in saved:
                saved["address"] = (await main_part(saved["key_id"], saved["key"], args.from_machine)).fqdn
                write_private(real_json, json.dumps(saved))
            # A key made seconds ago may not be accepted yet, so a fresh key gets a few tries.
            async with connect(real_json, attempts=1 if saved.get("used") else 4, timeout=15) as robot:
                live = await twin.pull_live(robot, real, d)
            if not saved.get("used"):
                saved["used"] = True
                write_private(real_json, json.dumps(saved))
        except (Exception, SystemExit) as e:  # offline or no key: fall back to the config alone
            live = {"unreachable": str(e)[:300]}
            if "multiple orgs" in live["unreachable"]:  # the machine's location is shared: the CLI can't pick the org
                live["unreachable"] += " (pass --org-id with the real machine's organization ID)"
    real, from_live = twin.apply_live_frames(real, live.get("frames", {}))
    write_private(d / "real-config.json", json.dumps(real, indent=2))  # holds resolved values: IPs, serials
    config, report = twin.convert(real, live.get("cameras"), args.module_version)
    report["frames_from_live"] = from_live
    for v in unresolved:
        report["flags"].append({"name": v["path"], "flag": "needs_review", "note": f"variable {v['name']} has no "
                                "value: no import value or default; left out"})
    report["live"] = bool(live.get("files"))
    report["files"] = live.get("files", {})
    for name, error in live.get("errors", {}).items():
        report["flags"].append({"name": name, "flag": "live_failed", "note": error})
    if "unreachable" in live:
        report["flags"].append({"name": args.from_machine, "flag": "unreachable", "note": live["unreachable"]})
    for r in report["resources"]:
        if r["api"] == "rdk:component:arm" and r["name"] not in report["files"].get("kinematics", {}):
            report["flags"].append({"name": r["name"], "flag": "config_only",
                                    "note": "no real kinematics saved, so twin-check can't compare them"})
    for mod in unapplied:
        report["flags"].append({"name": mod["path"], "flag": "needs_review", "note": f"{mod['op']} not applied"})
    write_private(d / "machine-config.json", json.dumps(config, indent=2))
    (d / "twin-report.json").write_text(json.dumps(report, indent=2))
    counts = {}
    for r in report["resources"]:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    print(json.dumps({"config": str(d / "machine-config.json"), "report": str(d / "twin-report.json"),
                      "live": report["live"], "outcomes": counts, "flags": len(report["flags"])}))


async def twin_pose(args):
    from viam.components.arm import Arm
    from viam.proto.component.arm import JointPositions

    path = Path(args.dir) / "joints.json"
    if not path.exists():
        print(json.dumps({"moved": [], "failed": {}, "skipped": "no joints.json: the real machine wasn't read live"}))
        return
    joints = json.loads(path.read_text())
    moved, failed = [], {}
    async with connect(args.cell) as robot:
        for name, values in joints.items():
            try:
                await Arm.from_robot(robot, name).move_to_joint_positions(JointPositions(values=values))
                moved.append(name)
            except Exception as e:
                failed[name] = str(e)[:300]
    print(json.dumps({"moved": sorted(moved), "failed": failed}))


async def image_check(robot, twin, d, name, path, twin_config_cameras):
    from viam.components.camera import Camera

    cam = Camera.from_robot(robot, name)
    images, _ = await cam.get_images()
    colour = next((i for i in images if "dep" not in str(i.mime_type)), None)
    if colour is None:
        return {"error": "the twin camera returned no colour image"}
    sim = d / f"sim_{name}.png"
    sim.write_bytes(colour.data)
    side = d / f"side_{name}.png"
    twin.edge_overlay(d / path, sim, side)
    ip = (await cam.get_properties()).intrinsic_parameters
    twin_fov = twin.fov_deg(ip.height_px, ip.focal_y_px) if ip and ip.focal_y_px else None
    real_cam = next((c for c in twin_config_cameras if c["name"] == name), {}).get("attributes", {})
    return {"overlay": str(side),
            "twin": {"size": [ip.width_px, ip.height_px] if ip else None, "fov_deg": twin_fov},
            "real": {"size": [real_cam.get("width"), real_cam.get("height")], "fov_deg": real_cam.get("fov_deg")}}


async def twin_check(args):
    import logging

    import twin
    from viam import logging as viam_logging
    from viam.components.arm import Arm
    from viam.components.camera import Camera  # noqa: F401 - registers the API before dialing, so from_robot finds cameras

    from viam.robot.client import RobotClient  # noqa: F401 - creates the SDK's loggers, so the level below reaches them

    os.environ.setdefault("RUST_LOG", "off")  # stdout is the JSON line
    viam_logging.setLevel(logging.CRITICAL)

    d = Path(args.dir)
    report = json.loads((d / "twin-report.json").read_text())
    # The converter set each twin camera's size and fov from the real one, so its config holds the real values.
    twin_config_cameras = json.loads((d / "machine-config.json").read_text())["components"]
    result = {"kinematics": {}, "images": {}}
    async with connect(args.cell) as robot:
        live = {(f"{n.namespace}:{n.type}:{n.subtype}", n.name) for n in robot.resource_names}
        result["names"] = twin.compare_names(report, live)
        fs = {f.frame.reference_frame: (f.frame.pose_in_observer_frame.reference_frame,
                                        twin.proto_pose(f.frame.pose_in_observer_frame.pose))
              for f in await robot.get_frame_system_config()}
        result["frames"] = twin.compare_frames(report["expected_frames"], fs)
        for name, (path, fmt) in report.get("files", {}).get("kinematics", {}).items():
            outcome = next(r["outcome"] for r in report["resources"] if r["name"] == name)
            try:
                got_fmt, data = (await Arm.from_robot(robot, name).get_kinematics())[:2]
                result["kinematics"][name] = twin.compare_kinematics(
                    twin.chain((d / path).read_bytes(), fmt), twin.chain(data, int(got_fmt)))
            except Exception as e:
                result["kinematics"][name] = {"ok": False, "error": str(e)[:300]}
            if outcome == "needs_asset":  # the report is never rewritten once the agent resolves it
                result["kinematics"][name]["note"] = "fails while the twin arm is still a fake"
        for name, path in report.get("files", {}).get("images", {}).items():
            try:
                result["images"][name] = await image_check(robot, twin, d, name, path, twin_config_cameras)
            except Exception as e:
                result["images"][name] = {"error": str(e)[:300]}
    if not result["kinematics"]:
        result["kinematics"] = {"skipped": "no real kinematics saved"}
    ok = result["names"]["ok"] and result["frames"]["ok"] and all(
        v.get("ok", True) for v in result["kinematics"].values() if isinstance(v, dict))
    result["ok"] = ok
    print(json.dumps(result))
    if not ok:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("creds")
    c.add_argument("--machine-id", required=True)
    c.add_argument("--dir", required=True)
    c.add_argument("--name", help="the API key's name (default mujoco-sim-cell-<time>)")
    c.add_argument("--org-id", help="when the CLI can't tell the machine's org")
    w = sub.add_parser("wait")
    w.add_argument("--cell", required=True)
    w.add_argument("--timeout", type=float, default=300, help="seconds")
    for name in ("check", "pick"):
        s = sub.add_parser(name)
        s.add_argument("--cell", required=True)
        s.add_argument("--world", default="sim-world")
        s.add_argument("--prop", required=True)
        if name == "check":
            s.add_argument("--goal", type=float, nargs=3)
        if name == "pick":
            s.add_argument("--gripper", required=True)
            s.add_argument("--size", type=float, required=True, help="the prop's edge, m (a cube; else its largest dimension)")
            s.add_argument("--place", type=float, nargs=3, required=True, help="the prop's centre at the goal, m")
            s.add_argument("--hover", type=float, default=0.1, help="lift and approach height, m")
            s.add_argument("--grasp-offset", type=float, default=0.0,
                           help="grasp this far above the prop's centre, m: raise it when the fingers hit the table")
            s.add_argument("--release-gap", type=float, default=0.005,
                           help="let go with the prop this far above its goal, m")
    m = sub.add_parser("measure")
    m.add_argument("--cell", required=True)
    m.add_argument("--world", default="sim-world")
    which = m.add_mutually_exclusive_group(required=True)
    which.add_argument("--gripper", help="TCP, jaw gap and finger reach (mm)")
    which.add_argument("--prop", help="world-frame size and top (m)")
    t = sub.add_parser("twin")
    t.add_argument("--from-machine", required=True, help="the real machine's ID")
    t.add_argument("--dir", required=True)
    t.add_argument("--module-version", required=True, help="SKILL.md's MODULE_VERSION")
    t.add_argument("--live", action="store_true",
                   help="create a key on the real machine (ask the user first) and read its cameras and arms")
    t.add_argument("--org-id", help="with --live: the real machine's org, for its read key (needed when its location is shared)")
    for name in ("twin-pose", "twin-check"):
        s = sub.add_parser(name)
        s.add_argument("--cell", required=True, help="the twin's cell.json")
        s.add_argument("--dir", required=True, help="the dir `twin` wrote")
    args = p.parse_args()
    asyncio.run({"creds": creds, "wait": wait, "check": check, "pick": pick, "measure": measure, "twin": twin_cmd,
                 "twin-pose": twin_pose, "twin-check": twin_check}[args.cmd](args))


if __name__ == "__main__":
    main()
