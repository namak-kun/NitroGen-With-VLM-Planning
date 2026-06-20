"""A tiny self-contained grid world that implements GameEnv — NO external game needed. Its
sole purpose is to validate the whole eval harness (Scenario -> Env -> Policy -> Detector ->
Runner -> metrics) and especially the "same start, different goals" protocol, before we invest
in standing up a real FOSS game (doukutsu-rs / zelda3).

The agent is a dot on an NxN grid. The canonical 25-dim gamepad's LEFT STICK moves it (x: +
right/- left, y: + down/- up, matching NitroGen's convention -y=up). Several goal cells exist;
a Scenario's success_spec uses reach_region over the privileged state keys 'x','y'. Identical
start + different goal cells = the counterfactual selection test in miniature.
"""
from __future__ import annotations

import numpy as np

from ..core import JLX, JLY, GameEnv, Observation, Scenario


class DummyGridEnv(GameEnv):
    name = "dummy_grid"

    def __init__(self, size: int = 16, cell_px: int = 16, speed: float = 0.6):
        self.size = size
        self.cell_px = cell_px
        self.speed = speed                 # cells moved per full-deflection control step
        self.x = size / 2.0
        self.y = size / 2.0
        self._goals: dict = {}
        self._step = 0

    def reset(self, scenario: Scenario) -> Observation:
        init = scenario.init or {}
        self.x = float(init.get("x", self.size / 2.0))
        self.y = float(init.get("y", self.size / 2.0))
        self._goals = init.get("goals", {})  # name -> (cx, cy) for rendering markers
        self._step = 0
        return self._obs()

    def step(self, action_chunk: np.ndarray) -> Observation:
        for row in self.iter_steps(action_chunk):
            dx = float(row[JLX]) * self.speed
            dy = float(row[JLY]) * self.speed
            self.x = float(np.clip(self.x + dx, 0, self.size - 1))
            self.y = float(np.clip(self.y + dy, 0, self.size - 1))
        self._step += 1
        return self._obs()

    def _obs(self) -> Observation:
        # render a simple RGB frame: dark grid, colored goal cells, white agent dot
        n, c = self.size, self.cell_px
        img = np.zeros((n * c, n * c, 3), dtype=np.uint8)
        img[:] = (20, 20, 28)
        palette = [(200, 80, 80), (80, 200, 80), (80, 120, 220), (220, 200, 80)]
        for i, (name, (gx, gy)) in enumerate(self._goals.items()):
            col = palette[i % len(palette)]
            y0, x0 = int(gy) * c, int(gx) * c
            img[y0:y0 + c, x0:x0 + c] = col
        ax, ay = int(self.x) * c, int(self.y) * c
        img[ay:ay + c, ax:ax + c] = (240, 240, 240)
        state = {"x": self.x, "y": self.y, "step": self._step}
        return Observation(frame=img, state=state, step_idx=self._step, done=False)
