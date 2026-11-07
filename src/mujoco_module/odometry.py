"""Wheel odometry for a viam-labs:mujoco-sim:base.

Truth comes from the base's joints (spawn-frame x/y slides and a yaw hinge,
see BaseHandle). Optional, deterministic drift is applied, and the drifted
velocities are dead-reckoned on the sim thread every physics step, so readings
don't depend on how often a client polls.

Odometry frame: origin at the base's pose when odometry was last zeroed,
+Y forward at that pose, +X right, yaw counter-clockwise.
"""

import math
from dataclasses import dataclass
from typing import Any

from .handles import BASE_JOINTS
from .sim import Contribution, SimManager

# kellydunn/golang-geo's radius, which RDK's wheeled-odometry uses.
EARTH_RADIUS_KM = 6371.0


def odometry_geo_point(x_m: float, y_m: float) -> tuple[float, float]:
    """(latitude, longitude) in degrees of the point x_m east and y_m north of
    (0, 0), as RDK's wheeled-odometry computes it with golang-geo's
    PointAtDistanceAndBearing (distance in km, bearing clockwise from +Y)."""
    angular = math.hypot(x_m, y_m) / 1000.0 / EARTH_RADIUS_KM
    bearing = math.atan2(x_m, y_m)
    lat = math.asin(math.sin(angular) * math.cos(bearing))
    lng = math.atan2(math.sin(bearing) * math.sin(angular), math.cos(angular))
    return math.degrees(lat), math.degrees(lng)


@dataclass(frozen=True)
class OdometryReading:
    x: float  # m, odometry frame
    y: float  # m
    yaw: float  # rad, counter-clockwise, unwrapped
    lateral: float  # m/s, body frame +X (right), drifted
    forward: float  # m/s, body frame +Y, drifted
    yaw_rate: float  # rad/s, drifted


class OdometryHandle:
    def __init__(self, sim: SimManager, name: str, attrs: dict[str, Any]) -> None:
        self._sim = sim
        self.name = name
        self.base = str(attrs["base"])
        self._scale = 1.0 + float(attrs.get("linear_scale_error", 0.0))
        self._bias = math.radians(float(attrs.get("heading_drift_dps", 0.0)))
        self._zero()
        self._ids: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        self._ids_generation = -1
        self._callback_key = f"odom:{name}"
        # No geometry: the contribution is only for init, the reset hook (costs a rebuild on add/close).
        sim.set_contribution(
            name, Contribution(build=lambda _spec: None, init=lambda _m, _d: self._zero())
        )
        sim.run(lambda: sim.step_callbacks.__setitem__(self._callback_key, self._on_step))

    def close(self) -> None:
        self._sim.run(lambda: self._sim.step_callbacks.pop(self._callback_key, None))
        self._sim.remove_contribution(self.name)

    def reset_odometry(self) -> None:
        self._sim.run(self._zero)

    def read(self) -> OdometryReading:
        def snapshot() -> OdometryReading:
            self._sim.require_world()
            if self._truth() is None:
                raise RuntimeError(
                    f"movement sensor {self.name!r}: base {self.base!r} is not in the "
                    "scene; configure that viam-labs:mujoco-sim:base"
                )
            return OdometryReading(
                self._x, self._y, self._yaw, self._lateral, self._forward, self._yaw_rate
            )

        return self._sim.run(snapshot)

    # -- sim thread ------------------------------------------------------

    def _zero(self) -> None:
        self._x = self._y = self._yaw = 0.0
        self._lateral = self._forward = self._yaw_rate = 0.0

    def _ids_now(self) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
        model = self._sim.model
        if model is None:
            return None
        if self._ids_generation != self._sim.generation:
            self._ids_generation = self._sim.generation
            try:
                joints = [model.joint(f"{self.base}/{joint}") for joint in BASE_JOINTS]
            except KeyError:
                self._ids = None
            else:
                self._ids = (
                    tuple(int(j.qposadr[0]) for j in joints),
                    tuple(int(j.dofadr[0]) for j in joints),
                )
        return self._ids

    def _truth(self) -> tuple[float, float, float] | None:
        """True (lateral, forward, yaw rate) of the base, body frame."""
        ids = self._ids_now()
        if ids is None or self._sim.data is None:
            return None
        qpos, dof = ids
        data = self._sim.data
        yaw = float(data.qpos[qpos[2]])
        xdot, ydot, wdot = (float(data.qvel[i]) for i in dof)
        c, s = math.cos(yaw), math.sin(yaw)
        return xdot * c + ydot * s, -xdot * s + ydot * c, wdot

    def _on_step(self, dt: float) -> None:
        truth = self._truth()
        if truth is None:
            return  # base not in the scene; resume when it is
        lateral, forward, yaw_rate = truth
        self._lateral = lateral * self._scale
        self._forward = forward * self._scale
        self._yaw_rate = yaw_rate + self._bias
        self._yaw += self._yaw_rate * dt
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        self._x += (self._lateral * c - self._forward * s) * dt
        self._y += (self._lateral * s + self._forward * c) * dt
