"""Game-agnostic closed-loop evaluation abstractions for plan-conditioned NitroGen.

The whole point of the System-2 (VLM plan) -> System-1 (NitroGen DiT) hierarchy is that a
plan should STEER behavior when the same observation admits many valid futures. Proxy metrics
(velocity-MSE, stick-delta) cannot validate that; only a CLOSED LOOP in a real game can. These
abstractions decouple the four moving parts so any FOSS game, any policy, and any success
detector plug in without touching each other:

    Scenario  --(defines)-->  initial state + the PLAN given to the policy + success criterion
    GameEnv   --(renders)-->  RGB frames the policy sees; --(exposes)--> privileged state
    Policy    --(reads frame + plan)-->  a NitroGen-layout action chunk (H,25)
    Detector  --(reads state/frame)-->   did the scenario's objective complete?
    EpisodeRunner ties them in a loop; EvalProtocol runs suites and aggregates metrics.

The canonical action is NitroGen's 25-dim gamepad vector per control step (buttons[0:21],
j_left[21:23] in [-1,1] with -x=left/-y=up, j_right[23:25]); a chunk is (H,25). Each GameEnv
maps that canonical gamepad onto its own input system, so policies stay game-agnostic.

KEY EXPERIMENTAL PROTOCOL (why this exists): "same start, different goals". Build several
Scenarios that share an initial state but carry DIFFERENT plans+objectives. If the policy
reaches goal A under plan A and goal B under plan B from the IDENTICAL start, the plan
causally controls behavior among valid choices -> the planning capability is real, measured
in a game rather than a proxy. This is the legitimate counterfactual test (both outcomes are
valid; the plan is not fighting the frame, it is selecting among many valid futures).
"""
from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

# ---- canonical action layout (NitroGen gamepad) ---------------------------------------
ACTION_DIM = 25
JLX, JLY = 21, 22  # left-stick x,y indices (x: -left/+right, y: -up/+down)
JRX, JRY = 23, 24  # right-stick x,y
N_BUTTONS = 21


@dataclass
class Observation:
    """What the loop produces each control step. `frame` is what the POLICY sees (RGB, the
    masked game area, like training). `state` is PRIVILEGED ground truth the DETECTOR may use
    (player position, map/room id, items, health, flags) — never given to the policy."""
    frame: np.ndarray                      # (H, W, 3) uint8 RGB
    state: dict[str, Any] = field(default_factory=dict)
    step_idx: int = 0
    done: bool = False                     # env-terminal (death, level end, time limit)


@dataclass
class Scenario:
    """A single closed-loop test case. The PLAN is what we feed the policy's System-2; the
    OBJECTIVE + success_spec are what the detector checks. For counterfactual/selection tests,
    several scenarios share `init` (same start) but differ in plan + objective."""
    id: str
    plan: str                              # natural-language plan given to the policy (System-2)
    objective: str                         # human-readable goal (for logs/VLM-judge)
    init: Any = None                       # env-specific start spec (savestate id / scenario key)
    success_spec: dict[str, Any] = field(default_factory=dict)  # detector config
    max_steps: int = 200                   # control steps before timeout (200 * 0.6s = 2 min)
    cfg_scale: float = 1.0                 # plan-CFG guidance for this scenario (policy may use)
    group: str = ""                        # scenarios sharing a `group` share an init (same start)


@dataclass
class EpisodeResult:
    scenario_id: str
    success: bool
    steps: int
    reason: str = ""                       # "objective_met" / "timeout" / "env_done" / "error"
    trajectory: list[dict] = field(default_factory=list)  # per-step state snapshots (light)
    extra: dict[str, Any] = field(default_factory=dict)


# ---- interfaces -----------------------------------------------------------------------
class GameEnv(abc.ABC):
    """A controllable game behind a uniform interface. Implementations map the canonical
    25-dim gamepad onto their own input system and expose privileged state."""

    name: str = "game"
    action_hz: float = 30.0                # NitroGen control rate (1 chunk = 18 steps = 0.6s)

    @abc.abstractmethod
    def reset(self, scenario: Scenario) -> Observation:
        """Load the scenario's initial state; return the first observation."""

    @abc.abstractmethod
    def step(self, action_chunk: np.ndarray) -> Observation:
        """Apply an (H,25) NitroGen-layout action chunk (each row held for the right number of
        engine frames to match action_hz) and return the resulting observation. May be called
        with a single (25,) action too."""

    def close(self) -> None:
        """Release resources (process, sockets, windows)."""

    # convenience: split a chunk into per-step rows
    @staticmethod
    def iter_steps(action_chunk: np.ndarray):
        a = np.asarray(action_chunk, dtype=np.float32)
        if a.ndim == 1:
            a = a[None]
        for row in a:
            yield row


class Policy(abc.ABC):
    """Maps an observation (+ a plan set at reset) to a NitroGen-layout action chunk."""

    @abc.abstractmethod
    def reset(self, scenario: Scenario) -> None:
        """Begin a new episode under scenario.plan (encode plan tokens, clear history)."""

    @abc.abstractmethod
    def act(self, obs: Observation) -> np.ndarray:
        """Return an (H, 25) action chunk for this observation."""


class SuccessDetector(abc.ABC):
    """Decides whether the scenario's objective has been achieved. May read privileged state
    (StatePredicateDetector) and/or the frame (VLMJudgeDetector)."""

    @abc.abstractmethod
    def reset(self, scenario: Scenario) -> None:
        ...

    @abc.abstractmethod
    def update(self, obs: Observation) -> None:
        """Observe one step; accumulate toward a success/failure decision."""

    @abc.abstractmethod
    def status(self) -> tuple[bool, bool]:
        """Return (done, success). done=True ends the episode early (objective met OR
        unrecoverable failure)."""


# ---- the closed loop ------------------------------------------------------------------
class EpisodeRunner:
    """Runs one scenario: reset env+policy+detector, then loop act->step->detect until the
    detector says done, the env terminates, or max_steps is hit. The policy owns its own
    System-2 cadence internally (it decides when to re-invoke the VLM); the runner only drives
    the System-1 control loop and the success check."""

    def __init__(self, snapshot_every: int = 1, on_step: Optional[Callable] = None):
        self.snapshot_every = snapshot_every
        self.on_step = on_step             # optional callback(obs, action) for logging/video

    def run(self, env: GameEnv, policy: Policy, detector: SuccessDetector,
            scenario: Scenario) -> EpisodeResult:
        traj: list[dict] = []
        try:
            obs = env.reset(scenario)
            policy.reset(scenario)
            detector.reset(scenario)
            detector.update(obs)
            for t in range(scenario.max_steps):
                done, success = detector.status()
                if done:
                    return EpisodeResult(scenario.id, success, t,
                                         "objective_met" if success else "failed", traj)
                action = policy.act(obs)
                if self.on_step is not None:
                    self.on_step(obs, action)
                obs = env.step(action)
                detector.update(obs)
                if t % self.snapshot_every == 0:
                    traj.append({"t": t, **{k: obs.state.get(k) for k in obs.state}})
                if obs.done:
                    done, success = detector.status()
                    return EpisodeResult(scenario.id, success, t + 1,
                                         "objective_met" if success else "env_done", traj)
            done, success = detector.status()
            return EpisodeResult(scenario.id, success, scenario.max_steps,
                                 "objective_met" if success else "timeout", traj)
        except Exception as e:  # noqa: BLE001 — surface env/policy errors as a failed episode
            return EpisodeResult(scenario.id, False, len(traj), f"error:{type(e).__name__}:{e}", traj)
