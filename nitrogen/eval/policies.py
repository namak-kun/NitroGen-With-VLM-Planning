"""Baseline / test policies. The real NitroGen closed-loop policy (plan-CFG sampling, async
System-2 replanning) lives in planner_poc/eval_policy.py to keep heavy model deps out of the
core package; these light policies let us test the harness + envs without a GPU."""
from __future__ import annotations

import numpy as np

from .core import ACTION_DIM, JLX, JLY, Observation, Policy, Scenario


def neutral_chunk(h: int = 18) -> np.ndarray:
    """An all-neutral action chunk: buttons 0, sticks centered (0.5 in [0,1] gamepad encoding
    is NOT used here — NitroGen sticks are [-1,1] with 0=center)."""
    a = np.zeros((h, ACTION_DIM), dtype=np.float32)
    return a


class NullPolicy(Policy):
    """Does nothing (neutral inputs). Baseline: any plan-conditioned success above this is the
    policy actually doing something."""

    def __init__(self, horizon: int = 18):
        self.h = horizon

    def reset(self, scenario: Scenario) -> None:
        pass

    def act(self, obs: Observation) -> np.ndarray:
        return neutral_chunk(self.h)


class ScriptedDirectionPolicy(Policy):
    """Pushes the left stick in a fixed cardinal direction, optionally chosen from the plan
    text (so we can smoke-test 'plan selects direction' end-to-end without a model). Used to
    validate envs + detectors + the same-start/different-goal protocol before the real policy."""

    DIRS = {"left": (JLX, -1.0), "right": (JLX, +1.0), "up": (JLY, -1.0), "down": (JLY, +1.0)}

    def __init__(self, horizon: int = 18, magnitude: float = 1.0):
        self.h = horizon
        self.mag = magnitude
        self._axis_sign = None

    def reset(self, scenario: Scenario) -> None:
        text = (scenario.plan or "").lower()
        self._axis_sign = next((v for k, v in self.DIRS.items() if k in text), None)

    def act(self, obs: Observation) -> np.ndarray:
        a = neutral_chunk(self.h)
        if self._axis_sign is not None:
            axis, sign = self._axis_sign
            a[:, axis] = sign * self.mag
        return a
