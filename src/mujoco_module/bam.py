"""BAM actuator physics for a robot file's servos: Rhoban's better-actuator-models
(https://github.com/Rhoban/bam, Apache-2.0), as pollen-robotics/microduck_rl runs it on CPU MuJoCo.

At spec time the matching position actuators become voltage-limited torque motors with their joint
damping and friction zeroed, because BAM computes friction itself each step (rewrite_spec, a port
of microduck_rl@0daab16 scripts/infer_policy.py load_mujoco_with_bam). Each physics step,
BamDrive.step runs bam.mujoco.MujocoController.update(), which turns a position target into motor
torque through the servo's voltage control law, back-EMF and load-dependent friction.

The model is recompiled whenever a contribution is added, so BamDrive binds a fresh controller to
the current model on the first step (or io call) after each rebuild. q_target restarts at the pose
then, which follows every contribution's init.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .sim import SimManager

# BAM's stiff friction-loss constraint (bam.mjlab.BamActuator stiff_frictionloss), as infer_policy.py
STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)
# The step BAM's XL330 fit, the duck's contact solref and armature are tuned at (body_server.py)
FIT_DT = 0.005
_STRINGS = ("motor", "bam_model", "joints")


@dataclass(frozen=True)
class BamConfig:
    motor: str = "xl330"
    bam_model: str = "m6"
    kp_fw: float = 200.0  # the servo firmware's position gain register
    vin: float = 7.4  # battery volts
    vin_drop_gain: float = 0.1  # sag: vin - gain * sum|motor torque|; 0 turns it off
    vin_min: float = 6.0  # floor on the sagged voltage
    joints: str = r"^(?!passive_).*"  # actuators whose target joint matches are BAM's

    @classmethod
    def from_attrs(cls, attrs: Any, label: str) -> BamConfig:
        if not isinstance(attrs, dict):
            raise ValueError(f"{label}: actuators must be an object")
        attrs = dict(attrs)
        if attrs.pop("model", None) != "bam":
            raise ValueError(f'{label}: actuators.model must be "bam"')
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(attrs) - known)
        if unknown:
            raise ValueError(f"{label}: unknown actuators keys {unknown}; known: {sorted(known)}")
        try:
            cfg = cls(**{k: (str(v) if k in _STRINGS else float(v)) for k, v in attrs.items()})
            for key in ("vin", "kp_fw"):
                if not np.isfinite(getattr(cfg, key)) or getattr(cfg, key) <= 0:
                    raise ValueError(f"{key} must be a positive number")
            for key in ("vin_drop_gain", "vin_min"):
                if not np.isfinite(getattr(cfg, key)) or getattr(cfg, key) < 0:
                    raise ValueError(f"{key} must be a non-negative number")
            re.compile(cfg.joints)
        except (TypeError, ValueError, re.error) as error:
            raise ValueError(f"{label}: actuators: {error}") from None
        try:
            cfg.load()
        except FileNotFoundError:
            raise ValueError(f"{label}: BAM has no motor {cfg.motor!r} model {cfg.bam_model!r}") from None
        return cfg

    def load(self) -> Any:
        """A fresh bam.model.Model with this config's firmware gain and battery voltage."""
        from bam.model import load_model

        model = load_model(motor_name=self.motor, model=self.bam_model)
        model.actuator.kp = self.kp_fw
        model.actuator.vin = self.vin
        model.actuator.max_current = None  # training runs without the firmware current limit
        return model


def rewrite_spec(spec: mujoco.MjSpec, cfg: BamConfig, bam_model: Any) -> list[str]:
    """Make the actuators whose target joint matches cfg.joints BAM's torque motors, in place, and
    return their names in file order. Must run before the spec compiles."""
    force = bam_model.actuator.vin * bam_model.kt.value / bam_model.R.value
    pattern = re.compile(cfg.joints)
    names = []
    for act in spec.actuators:
        if not pattern.match(act.target):
            continue
        joint = spec.joint(act.target)
        if joint is None:
            raise ValueError(f"actuator {act.name!r} drives {act.target!r}, which isn't a joint")
        act.set_to_motor()
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = (-force, force)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        act.gear = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        joint.damping = np.zeros(np.shape(joint.damping))
        joint.frictionloss = 0.0
        joint.solref_friction = STIFF_SOLREF_FRICTION
        joint.solimp_friction = STIFF_SOLIMP_FRICTION
        names.append(act.name)
    if not names:
        raise ValueError(f"no actuator's joint matches actuators.joints {cfg.joints!r}")
    return names


class BamDrive:
    """Drives a robot's rewritten actuators (scene names, prefixed) with BAM. Sim thread only."""

    def __init__(self, cfg: BamConfig, names: Sequence[str]) -> None:
        self.cfg = cfg
        self.names = tuple(names)
        self.torque = True
        self._ctrl: Any = None
        self._generation = -1
        self._failed_generation = -1

    def present(self, sim: SimManager) -> bool:
        return sim.model is not None and mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.names[0]
        ) >= 0

    def ensure_bound(self, sim: SimManager) -> Any:
        """The controller bound to sim's current model, rebound after a rebuild (q_target then
        holds the current pose). RuntimeError while the robot is out of the scene."""
        if self._ctrl is None or self._generation != sim.generation:
            if not self.present(sim):
                raise RuntimeError("the robot is not in the scene")
            from bam.mujoco import MujocoController

            # MujocoController.__init__ sets the armature, then calls mj_setConst(model, data),
            # which resets data.qpos to qpos0. On sim.data that would teleport every joint in the
            # world. Construct it on scratch data, then point it at the live data.
            ctrl = MujocoController(
                self.cfg.load(), list(self.names), sim.model, mujoco.MjData(sim.model),
                vin_drop_gain=self.cfg.vin_drop_gain or None, vin_min=self.cfg.vin_min,
            )
            ctrl.mujoco_data = sim.data
            ctrl.last_ts = sim.data.time
            ctrl.reset(sim.data.qpos)
            self._ctrl, self._generation = ctrl, sim.generation
        return self._ctrl

    def step(self, sim: SimManager) -> None:
        """The step callback's body: never raises (step_once drops a callback that raises)."""
        if not self.present(sim):
            return
        try:
            ctrl = self.ensure_bound(sim)
            ctrl.update()
        except Exception:
            if self._failed_generation != sim.generation:  # once per generation, not per step
                self._failed_generation = sim.generation
                LOGGER.exception("BAM step failed for %s; skipping it", self.names[0])
            return
        if not self.torque:  # limp: no motor torque; BAM's friction on the back-driven gears stays
            sim.data.ctrl[ctrl.act_indexes] = 0.0

    def set_targets(self, sim: SimManager, targets: np.ndarray) -> None:
        self.ensure_bound(sim).q_target[:] = targets

    def set_torque(self, sim: SimManager, on: bool) -> None:
        ctrl = self.ensure_bound(sim)
        if on and not self.torque:
            ctrl.reset(sim.data.qpos)  # don't fling the robot at a stale target
        self.torque = on

    def targets(self, sim: SimManager) -> np.ndarray:
        return self.ensure_bound(sim).q_target.copy()
