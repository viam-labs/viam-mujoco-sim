"""Layered obstacle avoidance on the rover: lidar SLAM, a depth camera for what
the lidar's scan plane misses, and bumper range sensors for what appears too
close for the planner (README, "Mobile obstacle avoidance").

Usage: uv run python examples/rover_avoid.py <machine address> <api key id> <api key>
           [--run map|blind|depth|surprise ...]

Load examples/rover-avoid.json for map, depth and surprise, then
examples/rover-avoid-lidar.json for map and blind (each config maps first: after
a config switch SLAM restores a stale pose). With no --run, it runs everything
the loaded config supports.
"""

import argparse
import asyncio
import contextlib
import math
import time

from viam.components.base import Base, Vector3
from viam.components.generic import Generic
from viam.components.movement_sensor import MovementSensor
from viam.components.sensor import Sensor
from viam.robot.client import RobotClient
from viam.services.navigation import NavigationClient
from viam.services.slam import SLAMClient

WORLD, BASE, ODOMETRY, SLAM, NAV, DEPTH_CAM = "sim-world", "rover", "odometry", "slam", "nav", "depth-cam"
BUMPERS = ("bumper-left", "bumper-right")
MAP = "rover-avoid"
GOAL_MM = (0.0, 1200.0)  # world; the crate sits between the spawn and here
GOAL_REACHED_MM = 300.0  # nav-stack snaps a blocked goal up to 0.5 m and still says succeeded
COURSE = ("crate",)
SURPRISE = "surprise"
SURPRISE_PARK = (2.0, 0.0, 0.075)  # m, outside the pen
SURPRISE_HALF_M = 0.075
UPRIGHT = [1.0, 0.0, 0.0, 0.0]  # set_prop_pose needs an orientation (w, x, y, z)
ROVER_FRONT_M = 0.15  # the shell's front face ahead of the rover's centre
DROP_GAP_M = 0.25  # how far ahead of the rover's front the box drops in
STOP_M = 0.3  # a bumper reading under this stops the rover
RESUME_MARGIN_M = 0.1  # resume once past STOP_M by this (hysteresis, so it doesn't flap)
# the goal is 0.28 m from the north wall and the pitched-up bumpers see the wall:
# armed on the final approach, they would stop the rover short of its goal
ARM_BEYOND_M = 0.5
CLEAR_S = 1.0  # bumpers clear this long before resuming
MOVED_MM = 20  # a prop moved more than this was hit
LAP_MM = [(-600, 0), (-600, 1100), (1000, 1100), (1000, -900), (0, -900), (0, 0)]
LAP_SPEED_MM_S, SPIN_DEG_S, SPIN_S = 300.0, 45.0, 8.0
GOAL_TIMEOUT_S = 150.0
LOCALIZE_TOLERANCE = (150.0, 10.0)  # mm, degrees from the spawn
SAFETY_HZ = 20
RUNS = ("map", "blind", "depth", "surprise")
TERMINAL = ("succeeded", "failed", "canceled")
NOT_DRIVING = ("idle", "suspended", "canceled", None)
DROP_AFTER_MM = 500.0  # drop the box once the rover has come this far from the spawn
KEEP_CLEAR = {"crate": (0.0, 0.55), "box-a": (0.6, 0.7)}  # world centres (m) the box mustn't land on
KEEP_CLEAR_M = 0.4  # centre to centre
PEN_HALF_M = 1.3  # inside the walls' inner faces (1.48 m), less the box
MOVE_TIMEOUT_S = 30.0  # one lap leg or turn


def world_to_map(x_mm: float, y_mm: float) -> tuple[float, float]:
    """nav-stack's map frame: origin at the spawn (where mapping starts), +X the
    rover's forward there (world +Y), in meters."""
    return y_mm / 1000.0, -x_mm / 1000.0


def wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def moved(before: dict, after: dict, threshold_mm: float = MOVED_MM) -> dict[str, float]:
    """Props that moved more than threshold_mm, and by how much (mm)."""
    out = {}
    for name, p in before.items():
        d = math.dist(p, after[name]) * 1000.0
        if d > threshold_mm:
            out[name] = d
    return out


def drop_point(x_mm: float, y_mm: float, theta_deg: float) -> tuple[float, float, float]:
    """Where the surprise box's centre goes (world, m): DROP_GAP_M ahead of the
    rover's front along its heading. theta 0 faces world +Y, counterclockwise."""
    ahead = ROVER_FRONT_M + DROP_GAP_M + SURPRISE_HALF_M
    t = math.radians(theta_deg)
    return x_mm / 1000.0 - ahead * math.sin(t), y_mm / 1000.0 + ahead * math.cos(t), SURPRISE_HALF_M


def drop_ok(point: tuple[float, float, float]) -> bool:
    """The box lands inside the pen, clear of the crate and box-a (the rover
    detours round the crate, so the drop point is wherever it is heading)."""
    x, y = point[0], point[1]
    inside = abs(x) <= PEN_HALF_M and abs(y) <= PEN_HALF_M
    return inside and all(math.dist((x, y), c) > KEEP_CLEAR_M for c in KEEP_CLEAR.values())


def should_stop(distances: list[float]) -> bool:
    return min(distances) < STOP_M


def is_clear(distances: list[float]) -> bool:
    return min(distances) >= STOP_M + RESUME_MARGIN_M


def armed(status: dict) -> bool:
    """The bumpers may stop nav: it is driving to a goal more than ARM_BEYOND_M away
    (nav-stack 1.1.47 reports "active" while driving; match on what it isn't)."""
    remaining = status.get("distance_remaining_m")
    return status.get("state") not in TERMINAL + NOT_DRIVING and (remaining is None or remaining > ARM_BEYOND_M)


def runs_for(depth: bool) -> list[str]:
    return ["map", "depth", "surprise"] if depth else ["map", "blind"]


def check_runs(runs: list[str], depth: bool) -> str | None:
    """None if the loaded config supports every run, else what to load."""
    wrong = [r for r in runs if r not in runs_for(depth)]
    if not wrong:
        return None
    need = "examples/rover-avoid-lidar.json" if depth else "examples/rover-avoid.json"
    return f"{', '.join(wrong)} needs {need} on the machine"


async def rover_pose(odometry: MovementSensor) -> tuple[float, float, float]:
    """World pose (x, y mm, heading deg) from odometry, zero at the spawn (mobile_pick.py's husky_pose)."""
    position, _ = await odometry.get_position(extra={"return_relative_pos_m": True})
    orientation = await odometry.get_orientation()
    return position.longitude * 1000, position.latitude * 1000, orientation.theta


async def prop_poses(world: Generic, names) -> dict[str, tuple[float, float, float]]:
    out = {}
    for name in names:
        out[name] = tuple((await world.do_command({"command": "get_prop_pose", "prop": name}))["pose"]["position"])
    return out


async def turn_to(base: Base, odometry: MovementSensor, heading: float) -> None:
    deadline = time.monotonic() + MOVE_TIMEOUT_S
    while True:
        if time.monotonic() > deadline:
            await base.stop()
            raise RuntimeError(f"turn to {heading:.0f} deg timed out")
        _, _, theta = await rover_pose(odometry)
        error = wrap(heading - theta)
        if abs(error) < 1.0:
            break
        rate = 2.5 * error if abs(error) < 15 else math.copysign(SPIN_DEG_S, error)
        await base.set_velocity(Vector3(), Vector3(z=max(-SPIN_DEG_S, min(SPIN_DEG_S, rate))))
        await asyncio.sleep(0.05)
    await base.stop()


async def drive_to(base: Base, odometry: MovementSensor, x: float, y: float) -> None:
    """Closed loop on odometry: turn to face (x, y) mm, then drive there, steering."""
    x0, y0, _ = await rover_pose(odometry)
    await turn_to(base, odometry, math.degrees(math.atan2(-(x - x0), y - y0)))
    deadline = time.monotonic() + MOVE_TIMEOUT_S
    while True:
        if time.monotonic() > deadline:
            await base.stop()
            raise RuntimeError(f"drive to ({x:.0f}, {y:.0f}) mm timed out")
        px, py, theta = await rover_pose(odometry)
        dx, dy = x - px, y - py
        along = -dx * math.sin(math.radians(theta)) + dy * math.cos(math.radians(theta))
        if math.hypot(dx, dy) < 20 or along < 0:
            break
        error = wrap(math.degrees(math.atan2(-dx, dy)) - theta)
        await base.set_velocity(Vector3(y=min(LAP_SPEED_MM_S, max(60.0, along * 1.5))),
                                Vector3(z=max(-30.0, min(30.0, 2 * error))))
        await asyncio.sleep(0.05)
    await base.stop()


async def map_run(world, base, odometry, slam) -> str:
    await world.do_command({"command": "reset"})
    with contextlib.suppress(Exception):  # no map yet
        await slam.do_command({"command": "delete_map", "map": MAP})  # start_mapping would continue it
    started = time.monotonic()
    try:
        await slam.do_command({"command": "start_mapping", "map": MAP})
        for x, y in LAP_MM:
            await drive_to(base, odometry, x, y)
            if (x, y) != LAP_MM[-1]:  # a continuous spin: stop-and-go turns rotate the map
                await base.set_velocity(Vector3(), Vector3(z=SPIN_DEG_S))
                await asyncio.sleep(SPIN_S)
                await base.stop()
        await turn_to(base, odometry, 0.0)
    finally:
        await base.stop()
    await slam.do_command({"command": "save_map"})
    await slam.do_command({"command": "start_localizing", "map": MAP})
    return f"map: {MAP} saved after a {time.monotonic() - started:.0f} s lap"


async def localize_at_spawn(world, slam) -> None:
    """Reset the world and put SLAM's pose at the spawn: after a reset,
    start_localizing alone restores the last run's end pose (and still says tracking)."""
    await world.do_command({"command": "reset"})
    await slam.do_command({"command": "start_localizing", "map": MAP})
    await slam.do_command({"command": "set_initial_pose", "pose": {"x": 0, "y": 0, "theta": 0}, "refine": True})
    deadline = time.monotonic() + 15
    pose = None
    while time.monotonic() < deadline:
        pose = await slam.get_position()
        if math.hypot(pose.x, pose.y) < LOCALIZE_TOLERANCE[0] and abs(wrap(pose.theta)) < LOCALIZE_TOLERANCE[1]:
            return
        await asyncio.sleep(0.5)
    raise RuntimeError(f"SLAM did not localize at the spawn: pose {pose}")


class Safety:
    """Polls the bumpers; while nav is armed (nav_active, see armed()), a
    reading under STOP_M suspends nav (so it stops commanding velocity), then
    stops the base; once clear for CLEAR_S it resumes."""

    def __init__(self, nav, base, bumpers) -> None:
        self.nav, self.base, self.bumpers = nav, base, bumpers
        self.stops: list[float] = []  # bumper distance at each stop
        self.holding = False
        self.nav_active = False  # set by navigate(): only an active goal can trip it
        self.stopped = asyncio.Event()

    async def distances(self) -> list[float]:
        return [float((await b.get_readings())["distance"]) for b in self.bumpers]

    async def run(self) -> None:
        if not self.bumpers:  # the blind run: the lidar alone
            return
        clear_since = None
        while True:
            d = await self.distances()
            if not self.holding and self.nav_active and should_stop(d):
                await self.nav.do_command({"command": "suspend", "reason": "bumper"})
                with contextlib.suppress(Exception):  # suspended nav already stops; this is belt and braces
                    await self.base.stop()
                self.holding, clear_since = True, None
                self.stops.append(min(d))
                self.stopped.set()
            elif self.holding:
                clear_since = (clear_since or time.monotonic()) if is_clear(d) else None
                if clear_since and time.monotonic() - clear_since >= CLEAR_S:
                    await self.nav.do_command({"command": "resume"})
                    self.holding = False
                    self.stopped.clear()
            await asyncio.sleep(1 / SAFETY_HZ)


async def navigate(nav, safety: Safety, guard: asyncio.Task) -> tuple[str, float]:
    """Send the goal and poll to a terminal state. A terminal state counts only
    after nav has reported active or suspended (the first status after a new goal
    can still be the last run's), and not while the bumpers hold it. A failed
    bumper loop (guard) raises here."""
    gx, gy = world_to_map(*GOAL_MM)
    await nav.do_command({"command": "navigate_to_point", "x": gx, "y": gy})
    started = time.monotonic()
    seen_active = False
    try:
        while time.monotonic() - started < GOAL_TIMEOUT_S:
            if guard.done():
                guard.result()  # re-raises the bumper loop's failure
            status = await nav.do_command({"command": "get_status"})
            state = status.get("state")
            safety.nav_active = armed(status)
            seen_active = seen_active or state in ("active", "suspended")
            if state in TERMINAL and seen_active and not safety.holding:
                return state, time.monotonic() - started
            await asyncio.sleep(0.5)
        await nav.do_command({"command": "cancel"})
        return "held by bumpers" if safety.holding else "timeout", time.monotonic() - started
    finally:
        safety.nav_active = False


async def goal_run(name, world, odometry, slam, nav, base, bumpers, surprise=False) -> str:
    await localize_at_spawn(world, slam)
    before = await prop_poses(world, COURSE)
    safety = Safety(nav, base, bumpers)
    guard = asyncio.create_task(safety.run())
    trip = asyncio.create_task(navigate(nav, safety, guard))
    note = ""
    try:
        if surprise:
            note = await drop_surprise(world, odometry, safety, trip)
        state, seconds = await trip
        if guard.done() and guard.exception():  # a failed bumper read must not pass silently
            raise guard.exception()
    except BaseException:
        trip.cancel()
        with contextlib.suppress(Exception):  # best effort: keep the original error
            await nav.do_command({"command": "cancel"})
            await base.stop()
        raise
    finally:
        guard.cancel()
        with contextlib.suppress(BaseException):  # real failures surfaced above
            await guard
    hits = moved(before, await prop_poses(world, COURSE))
    stops = [round(d, 2) for d in safety.stops]
    x, y, _ = await rover_pose(odometry)
    off = math.dist((x, y), GOAL_MM)
    reached = "reached" if off <= GOAL_REACHED_MM else "NOT reached"
    return (f"{name}: {state} in {seconds:.0f} s, goal {reached} (odometry: {off:.0f} mm off); "
            f"moved {hits or 'nothing'}; bumper stops {stops if bumpers else 'off'}{note}")


async def drop_surprise(world, odometry, safety: Safety, trip: asyncio.Task) -> str:
    """Once the rover is DROP_AFTER_MM from the spawn, drop the box ahead of it; after the bumper
    stop, step it away. Returns a note for the run's line."""
    while math.hypot(*(await rover_pose(odometry))[:2]) < DROP_AFTER_MM:
        if trip.done():
            return f"; never got {DROP_AFTER_MM:.0f} mm from the spawn"
        await asyncio.sleep(0.05)
    x, y, theta = await rover_pose(odometry)
    point = drop_point(x, y, theta)
    if not drop_ok(point):
        return f"; no clear spot ahead of the rover at ({x:.0f}, {y:.0f}) mm, no drop"
    safety.stopped.clear()  # only a stop after the drop counts
    await world.do_command({"command": "set_prop_pose", "prop": SURPRISE, "position": list(point),
                            "orientation_wxyz": UPRIGHT})
    try:
        await asyncio.wait_for(safety.stopped.wait(), timeout=10)
        at_stop = (await prop_poses(world, [SURPRISE]))[SURPRISE]
        note = f"; surprise box moved {math.dist(at_stop, point) * 1000:.0f} mm by the stop"
    except TimeoutError:
        note = "; NO bumper stop within 10 s"
    await asyncio.sleep(1.0)  # the "person" steps away
    await world.do_command({"command": "set_prop_pose", "prop": SURPRISE, "position": list(SURPRISE_PARK),
                            "orientation_wxyz": UPRIGHT})
    return note


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("address")
    parser.add_argument("key_id")
    parser.add_argument("key")
    parser.add_argument("--run", action="append", choices=RUNS)
    args = parser.parse_args()
    opts = RobotClient.Options.with_api_key(api_key=args.key, api_key_id=args.key_id)
    async with await RobotClient.at_address(args.address, opts) as robot:
        depth = any(r.name == DEPTH_CAM for r in robot.resource_names)
        runs = args.run or runs_for(depth)
        problem = check_runs(runs, depth)
        if problem:
            raise SystemExit(problem)
        world = Generic.from_robot(robot, WORLD)
        base = Base.from_robot(robot, BASE)
        odometry = MovementSensor.from_robot(robot, ODOMETRY)
        slam = SLAMClient.from_robot(robot, SLAM)
        nav = NavigationClient.from_robot(robot, NAV)
        bumpers = [Sensor.from_robot(robot, b) for b in BUMPERS]
        for run in runs:
            try:
                if run == "map":
                    print(await map_run(world, base, odometry, slam), flush=True)
                else:
                    print(await goal_run(
                        # blind shows what the lidar alone misses: no bumpers
                        run, world, odometry, slam, nav, base, [] if run == "blind" else bumpers,
                        surprise=run == "surprise"), flush=True)
            except Exception as e:  # one failed run doesn't skip the rest
                print(f"{run}: ERROR {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
