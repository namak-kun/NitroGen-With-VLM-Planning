"""Baseline / test policies. The real NitroGen closed-loop policy (plan-CFG sampling, async
System-2 replanning) lives in planner_poc/eval_policy.py to keep heavy model deps out of the
core package; these light policies let us test the harness + envs without a GPU."""
from __future__ import annotations

import numpy as np

from .core import ACTION_DIM, JLX, JLY, JRX, JRY, Observation, Policy, Scenario


def neutral_chunk(h: int = 18) -> np.ndarray:
    """An all-neutral action chunk: buttons 0, sticks CENTERED at 0.5.

    STICK CONVENTION (load-bearing): the trained NitroGen model emits joysticks in [0,1] with
    0.5 = NEUTRAL (verified: a null/unconditioned chunk gives stick_x≈0.50), and every GameEnv's
    action_to_keys thresholds against 0.5±t. So neutral is 0.5 (NOT 0). Buttons are 0/1."""
    a = np.zeros((h, ACTION_DIM), dtype=np.float32)
    a[:, JLX] = a[:, JLY] = 0.5
    a[:, JRX] = a[:, JRY] = 0.5
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
    validate envs + detectors + the same-start/different-goal protocol before the real policy.

    Sticks are [0,1] with 0.5 neutral (matching the model + envs): left=0.5-mag, right=0.5+mag,
    up=0.5-mag, down=0.5+mag."""

    DIRS = {"left": (JLX, -1.0), "right": (JLX, +1.0), "up": (JLY, -1.0), "down": (JLY, +1.0)}

    def __init__(self, horizon: int = 18, magnitude: float = 0.5):
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
            a[:, axis] = 0.5 + sign * self.mag
        return a
