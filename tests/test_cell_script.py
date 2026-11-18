"""plugin/skills/mujoco-sim-cell/scripts/cell.py: the goal check and the pick waypoints."""

import importlib.util
import stat
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "plugin" / "skills" / "mujoco-sim-cell" / "scripts" / "cell.py"


def load():
    spec = importlib.util.spec_from_file_location("cell", SCRIPT)
    cell = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cell)
    return cell


def test_judge_at_rest_and_at_goal():
    cell = load()
    r = cell.judge([0.30, 0.20, 0.02], [0.301, 0.20, 0.02], goal=[0.31, 0.20, 0.02])
    assert r["at_rest"] and r["at_goal"]
    assert abs(r["error_m"] - 0.009) < 1e-6


def test_judge_moving_or_off_goal():
    cell = load()
    assert not cell.judge([0.3, 0, 0.02], [0.3, 0.01, 0.02])["at_rest"]  # moved 1 cm in 0.5 s
    assert "at_goal" not in cell.judge([0.3, 0, 0.02], [0.3, 0, 0.02])  # no goal given
    assert not cell.judge([0.3, 0, 0.02], [0.3, 0, 0.02], goal=[0.4, 0, 0.02])["at_goal"]


def test_pick_waypoints_hop_only_where_the_planner_cannot_see_the_prop():
    cell = load()
    w = cell.pick_waypoints(prop=[0.3, 0.0, 0.02], place=[0.3, 0.2, 0.05], hover=0.1,
                            grasp_offset=0.01, release_gap=0.005, hop=0.05)
    # (x, y, z, target, action), mm. The TCP grasps 10 mm above the prop's centre (z 30).
    # target: "avoid" = an obstacle, "near" = left out, "held" = carried on the gripper.
    # Hops only while "near"; every move while held is checked, and each one costs grip.
    assert w == [
        (300.0, 0.0, 130.0, "avoid", None),     # over the prop, routed around it
        (300.0, 0.0, 80.0, "near", None),       # straight down in hops the planner can't curve
        (300.0, 0.0, 30.0, "near", "grab"),
        (300.0, 0.0, 50.0, "held", None),       # break away from the table, then one move up
        (300.0, 0.0, 130.0, "held", "lifted"),
        (300.0, 200.0, 160.0, "held", None),    # over the goal: its centre + offset + hover
        (300.0, 200.0, 65.0, "held", "open"),   # one move down; let go 5 mm above the goal
        (300.0, 200.0, 112.5, "near", None),    # back up in hops, clear of the placed prop
        (300.0, 200.0, 160.0, "near", None),
    ]


def test_hops_cover_the_whole_drop_in_steps_no_longer_than_hop():
    cell = load()
    assert cell.hops(0.1, 0.0, 0.03) == [0.075, 0.05, 0.025, 0.0]
    assert cell.hops(0.0, 0.0, 0.01) == [0.0]  # no drop: one move, to the end
    assert len(cell.hops(0.13, 0.03, 0.01)) == 10  # 0.1 / 0.01 is 10.000000000000002 in floats


def test_world_state_carries_the_held_prop_beyond_the_tcp():
    cell = load()
    held = cell.world_state("held", [0, 0, 0], 31, "g", 22).transforms[0]
    assert held.pose_in_observer_frame.reference_frame == "g"
    assert held.pose_in_observer_frame.pose.z == 22  # +z is along the tool: below the TCP when pointing down
    assert cell.world_state("near", [0, 0, 0], 31, "g", 22) is None
    avoid = cell.world_state("avoid", [300, 0, 20], 30, "g", 0).obstacles[0]
    assert avoid.reference_frame == "world" and avoid.geometries[0].center.x == 300


def test_write_private_is_owner_only(tmp_path):
    cell = load()
    path = tmp_path / "secret.json"
    path.write_text("old")
    path.chmod(0o644)
    cell.write_private(path, "x")
    assert path.read_text() == "x"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


class StallsAtTheta:
    """A motion service whose planner times out for goals at one tool yaw."""

    def __init__(self, bad_theta):
        self.bad_theta, self.calls = bad_theta, []

    async def move(self, component_name, destination, world_state=None, extra=None):
        self.calls.append((destination.pose.theta, extra))
        if destination.pose.theta == self.bad_theta:
            raise TimeoutError("cbirrt timeout context deadline exceeded")


def test_move_down_turns_the_tool_180_when_a_move_stalls_and_keeps_the_turn():
    import asyncio

    cell = load()
    motion = StallsAtTheta(bad_theta=0)
    assert asyncio.run(cell.move_down(motion, "g", (1, 2, 3), None, theta=0)) == 180
    assert asyncio.run(cell.move_down(motion, "g", (1, 2, 3), None, theta=180)) == 180
    assert [t for t, _ in motion.calls] == [0, 180, 180]
    assert all(extra == {"timeout": cell.PLAN_TIMEOUT_S} for _, extra in motion.calls)


def test_move_down_reports_both_tries_when_both_fail():
    import asyncio

    import pytest

    cell = load()

    class Refuses(StallsAtTheta):
        async def move(self, **kw):
            await super().move(**kw)
            raise ValueError(f"collision at {kw['destination'].pose.theta}")

    with pytest.raises(RuntimeError, match="theta 0.*theta 180"):
        asyncio.run(cell.move_down(Refuses(bad_theta=None), "g", (1, 2, 3), None, theta=0))


def test_not_ready_lists_everything_short_of_ready():
    from viam.proto.common import ResourceName
    from viam.proto.robot import GetMachineStatusResponse as Status
    from viam.proto.robot import ModuleStatus, PackageStatus, ResourceStatus

    cell = load()
    arm = ResourceName(namespace="rdk", type="component", subtype="arm", name="a")
    world = ResourceName(namespace="rdk", type="component", subtype="generic", name="w")
    status = Status(
        state=Status.STATE_INITIALIZING,
        resources=[ResourceStatus(name=world, state=ResourceStatus.STATE_READY),
                   ResourceStatus(name=arm, state=ResourceStatus.STATE_CONFIGURING, error="no site")],
        modules=[ModuleStatus(module_name="mujoco-sim", state=ModuleStatus.STATE_STARTING)],
        packages=[PackageStatus(name="p", state=PackageStatus.STATE_DOWNLOADING)],
    )
    assert cell.not_ready(status) == [
        {"what": "machine", "state": "INITIALIZING"},
        {"what": "rdk:component:arm/a", "state": "CONFIGURING", "error": "no site"},
        {"what": "module mujoco-sim", "state": "STARTING"},
        {"what": "package p", "state": "DOWNLOADING"},
    ]
    ready = Status(state=Status.STATE_RUNNING, resources=[ResourceStatus(name=world, state=ResourceStatus.STATE_READY)])
    assert cell.not_ready(ready) == []


def test_parse_key_reads_the_clis_output_and_never_echoes_it():
    import pytest

    cell = load()
    out = "Successfully created key:\nKey ID: 1234-abcd\nKey Value: s3cret\n\nKeep this key somewhere safe\n"
    assert cell.parse_key(out) == ("1234-abcd", "s3cret")
    with pytest.raises(ValueError) as error:
        cell.parse_key("Key Value: s3cret\n")
    assert "s3cret" not in str(error.value)
