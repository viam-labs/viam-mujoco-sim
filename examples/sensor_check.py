"""Sensor check: runs a pick-and-carry on the Husky + UR5e cell of
examples/husky-ur5e.json and checks the IMU and the wrist force-torque sensor
against the physics at each step.

Usage: uv run python examples/sensor_check.py <machine address> <api key id> <api key>

Take the address and an API key from the machine's CONNECT tab. The machine
needs the cell's torso-imu, plus a wrist-ft force-torque sensor on the arm:

  {"name": "wrist-ft", "api": "rdk:component:sensor", "model": "viam-labs:mujoco-sim:force_torque",
   "frame": {"parent": "arm"}, "attributes": {"world": "sim-world"}}

The checks:
  - IMU at rest: acceleration is gravity, +9.81 m/s^2 up; no rotation.
  - IMU while the Husky spins: the gyro's z is the commanded rate; afterwards
    the IMU's heading agrees with the wheel odometry's.
  - Force-torque, tared with the gripper empty and pointing down: lifting the
    block adds its weight along +Z (out of the flange); letting go takes it
    away again.
  - The world's step loop kept up (no new slips).

It exits non-zero if a check fails.
"""

import asyncio
import math
import sys

from viam.components.base import Base
from viam.components.generic import Generic
from viam.components.gripper import Gripper
from viam.components.movement_sensor import MovementSensor
from viam.components.sensor import Sensor
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

from mobile_pick import (
    ABOVE_MM,
    BACK_UP_MM,
    BLOCK,
    BLOCK_MM,
    CARRY_MM,
    DOWN,
    PLACE_MM,
    SPEED_MM_S,
    TABLE_TOP_MM,
    TABLES,
    TO_TABLE_A_MM,
    TO_TABLE_B_MM,
    box,
    husky_pose,
    on_table_b,
    to_husky,
)
from viam.proto.common import GeometriesInFrame, Pose, PoseInFrame, WorldState

G = 9.81
BLOCK_KG = 0.03  # as in husky-ur5e.json
SPIN_DEG, SPIN_DPS = 90, 45
SETTLE_S = 0.5  # the F/T filter (0.02 s) and the arm settle well within this


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str) -> None:
        self.rows.append((name, ok, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    def near(self, name: str, value: float, want: float, tol: float, unit: str) -> None:
        self.check(name, abs(value - want) <= tol, f"{value:.4f} {unit} (want {want:.4f} ± {tol} {unit})")

    @property
    def failed(self) -> int:
        return sum(not ok for _, ok, _ in self.rows)


async def force(ft: Sensor) -> tuple[float, float, float]:
    f = (await ft.get_readings())["force"]
    return f["x"], f["y"], f["z"]


async def sensor_check(robot: RobotClient, checks: Checks) -> None:
    world = Generic.from_robot(robot, "sim-world")
    base = Base.from_robot(robot, "husky")
    odometry = MovementSensor.from_robot(robot, "odometry")
    imu = MovementSensor.from_robot(robot, "torso-imu")
    ft = Sensor.from_robot(robot, "wrist-ft")
    gripper = Gripper.from_robot(robot, "gripper")
    motion = MotionClient.from_robot(robot, "builtin")
    tool = CARRY_MM

    async def move_husky(hx, hy, z, husky, step, *extra):
        nonlocal tool
        tables = [box(name, *table, husky) for name, table in TABLES.items()]
        state = WorldState(obstacles=[GeometriesInFrame(reference_frame="husky", geometries=[*tables, *extra])])
        destination = PoseInFrame(reference_frame="husky", pose=Pose(x=hx, y=hy, z=z, **DOWN))
        if not await motion.move(component_name="gripper", destination=destination, world_state=state):
            raise RuntimeError(f"motion.move failed: {step}")
        tool = (hx, hy, z)

    async def move(x, y, z, husky, step, *extra):
        await move_husky(*to_husky(x, y, husky), z, husky, step, *extra)

    async def carry(husky, step):
        await move_husky(tool[0], tool[1], CARRY_MM[2], husky, f"{step}: lift")
        await move_husky(*CARRY_MM, husky, f"{step}: pull in")

    async def block_mm():
        pose = (await world.do_command({"command": "get_prop_pose", "prop": BLOCK}))["pose"]
        return tuple(v * 1000 for v in pose["position"])

    await world.do_command({"command": "reset"})
    slips = (await world.do_command({"command": "status"}))["slips"]
    await ft.do_command({"command": "untare"})
    await gripper.open()
    await carry((0.0, 0.0, 0.0), "start")
    await asyncio.sleep(SETTLE_S)

    print("IMU at rest")
    accel = await imu.get_linear_acceleration()
    gyro = await imu.get_angular_velocity()
    checks.near("acceleration z (gravity, up)", accel.z, G, 0.05, "m/s²")
    checks.near("acceleration sideways", math.hypot(accel.x, accel.y), 0.0, 0.05, "m/s²")
    checks.near("angular velocity", math.sqrt(gyro.x**2 + gyro.y**2 + gyro.z**2), 0.0, 0.1, "°/s")

    print("force-torque, gripper empty and pointing down")
    fx, fy, fz = await force(ft)
    checks.check("gripper weight along +Z", fz > 0 and math.hypot(fx, fy) < 0.05 * fz,
                 f"({fx:.3f}, {fy:.3f}, {fz:.3f}) N = {fz / G:.3f} kg hanging")
    await ft.do_command({"command": "tare"})

    await base.move_straight(TO_TABLE_A_MM, SPEED_MM_S)
    husky = await husky_pose(odometry)
    x, y, z = await block_mm()
    await move(x, y, z + ABOVE_MM, husky, "above the block", box(BLOCK, (x, y, z), (BLOCK_MM,) * 3, husky))
    await move(x, y, z, husky, "down to the block")
    await asyncio.sleep(SETTLE_S)
    before = (await force(ft))[2]
    await gripper.grab()
    await asyncio.sleep(1.0)  # Grab returns at once; let the jaws close
    if not (await gripper.is_holding_something()).is_holding_something:
        raise RuntimeError("the gripper is not holding the block")
    await move(x, y, z + ABOVE_MM, husky, "lift the block")
    await asyncio.sleep(SETTLE_S)
    print("force-torque, holding the block")
    lifted = (await force(ft))[2]
    checks.near("tared, on the table before the grasp", before, 0.0, 0.05, "N")
    checks.near("lifting adds the block's weight", lifted, BLOCK_KG * G, 0.05, "N")
    await carry(husky, "before backing up")
    await asyncio.sleep(SETTLE_S)
    checks.near("still the block's weight over the Husky", (await force(ft))[2], BLOCK_KG * G, 0.05, "N")

    await base.move_straight(-BACK_UP_MM, SPEED_MM_S)
    print(f"IMU while the Husky spins {SPIN_DEG}° at {SPIN_DPS} °/s")
    rates: list[float] = []
    spinning = asyncio.Event()

    async def sample() -> None:
        while not spinning.is_set():
            rates.append((await imu.get_angular_velocity()).z)
            await asyncio.sleep(0.05)

    async def spin() -> None:
        try:
            await base.spin(SPIN_DEG, SPIN_DPS)
        finally:
            spinning.set()

    await asyncio.gather(spin(), sample())
    turning = sorted(r for r in rates if abs(r) > SPIN_DPS / 2)  # samples during the turn, not the edges
    median = turning[len(turning) // 2] if turning else 0.0
    checks.near(f"gyro z, median of {len(turning)} samples", median, SPIN_DPS, 0.05 * SPIN_DPS, "°/s")
    imu_yaw = (await imu.get_orientation()).theta
    odo_yaw = (await odometry.get_orientation()).theta
    checks.near("IMU heading minus odometry heading", math.remainder(imu_yaw - odo_yaw, 360.0), 0.0, 0.5, "°")
    checks.near("IMU heading after the spin", imu_yaw, SPIN_DEG, 1.0, "°")

    await base.move_straight(TO_TABLE_B_MM, SPEED_MM_S)
    husky = await husky_pose(odometry)
    x, y = PLACE_MM
    z = TABLE_TOP_MM + BLOCK_MM / 2
    await move(x, y, z + ABOVE_MM, husky, "above table B")
    await move(x, y, z + 2, husky, "down onto table B")
    await gripper.open()
    await asyncio.sleep(1.0)
    await move(x, y, z + ABOVE_MM, husky, "retreat from table B")
    await asyncio.sleep(SETTLE_S)
    print("force-torque, after letting go")
    checks.near("back to the tare", (await force(ft))[2], 0.0, 0.05, "N")
    bx, by, bz = await block_mm()
    checks.check("block on table B", on_table_b(bx, by, bz), f"({bx:.0f}, {by:.0f}, {bz:.0f}) mm")

    status = await world.do_command({"command": "status"})
    checks.check("no new step-loop slips", status["slips"] == slips,
                 f"{status['slips'] - slips} new, real-time factor {status['real_time_factor']:.3f}")
    await ft.do_command({"command": "untare"})


async def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    address, api_key_id, api_key = sys.argv[1:]
    robot = await RobotClient.at_address(
        address, RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id)
    )
    checks = Checks()
    try:
        await sensor_check(robot, checks)
    finally:
        await robot.close()
    print(f"\n{len(checks.rows) - checks.failed}/{len(checks.rows)} checks passed")
    if checks.failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
