"""Success detectors: decide whether a Scenario's objective completed.

Two families, mirroring how robotics/embodied-AI handle success detection:
  * StatePredicateDetector — reads PRIVILEGED engine state (player pos, map id, items, flags)
    and evaluates a predicate. This is the "emulator-RAM / sim-state" approach: unambiguous for
    BOUNDED objectives ("reach region X", "obtain item Y", "enter map Z"). Cheap + exact.
    Preferred whenever the objective is expressible as a state predicate.
  * VLMJudgeDetector — shows a VLM the recent frames + the instruction and asks "completed?".
    For FUZZY goals not expressible as a state predicate. Soft signal (planner and judge may
    share blind spots), so use as a secondary metric, not ground truth.

Both are game-agnostic; the GAME-specific part is only which state keys exist, declared in the
Scenario.success_spec.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable

import numpy as np

from .core import Observation, Scenario, SuccessDetector


class StatePredicateDetector(SuccessDetector):
    """Succeeds when a predicate over the privileged state becomes (and stays) true.

    success_spec keys (all optional, combined with AND unless `mode='any'`):
      - reach_region: (key_x, key_y, cx, cy, radius) -> within `radius` of (cx,cy) of state[x,y]
      - reach_map:    (key, value)                   -> state[key] == value (e.g. map/room id)
      - obtain:       (key, value)                   -> state[key] >= value (item count/flag)
      - flag_true:    key                            -> bool(state[key]) is True
      - predicate:    callable(state) -> bool        -> arbitrary user predicate
      - fail_if:      callable(state) -> bool        -> immediate failure (e.g. death)
      - hold_steps:   int (default 1)                -> predicate must hold this many steps
      - mode:         'all' (default) | 'any'
    """

    def __init__(self):
        self._spec: dict[str, Any] = {}
        self._held = 0
        self._done = False
        self._success = False

    def reset(self, scenario: Scenario) -> None:
        self._spec = dict(scenario.success_spec)
        self._held = 0
        self._done = False
        self._success = False

    def _check_one(self, name: str, val: Any, state: dict) -> bool:
        if name == "reach_region":
            kx, ky, cx, cy, r = val
            if kx not in state or ky not in state:
                return False
            return (float(state[kx]) - cx) ** 2 + (float(state[ky]) - cy) ** 2 <= r * r
        if name == "reach_map":
            key, value = val
            return state.get(key) == value
        if name == "obtain":
            key, value = val
            return key in state and float(state[key]) >= float(value)
        if name == "flag_true":
            return bool(state.get(val))
        if name == "predicate":
            return bool(val(state))
        return False

    def update(self, obs: Observation) -> None:
        if self._done:
            return
        state = obs.state
        fail_if = self._spec.get("fail_if")
        if fail_if is not None and fail_if(state):
            self._done, self._success = True, False
            return
        checks = [(k, v) for k, v in self._spec.items()
                  if k in ("reach_region", "reach_map", "obtain", "flag_true", "predicate")]
        if not checks:
            return
        results = [self._check_one(k, v, state) for k, v in checks]
        ok = any(results) if self._spec.get("mode") == "any" else all(results)
        self._held = self._held + 1 if ok else 0
        if self._held >= int(self._spec.get("hold_steps", 1)):
            self._done, self._success = True, True

    def status(self) -> tuple[bool, bool]:
        return self._done, self._success


class VLMJudgeDetector(SuccessDetector):
    """Asks a VLM whether the instruction was completed, from the recent frames. Lazy: only
    queries the VLM every `query_every` steps and at episode end (VLM calls are expensive).

    success_spec keys:
      - judge_instruction: str (defaults to scenario.objective)
      - query_every: int (default 25 control steps ~ 15s)
      - n_frames: int (default 4) recent frames to show the judge
    `judge_fn(frames: list[np.ndarray], instruction: str) -> bool` is injected (so this stays
    decoupled from any specific VLM; planner_poc supplies a Gemma/Qwen-backed judge).
    """

    def __init__(self, judge_fn: Callable[[list, str], bool]):
        self.judge_fn = judge_fn
        self._instruction = ""
        self._every = 25
        self._n = 4
        self._buf: deque = deque(maxlen=4)
        self._t = 0
        self._done = False
        self._success = False

    def reset(self, scenario: Scenario) -> None:
        spec = scenario.success_spec
        self._instruction = spec.get("judge_instruction", scenario.objective)
        self._every = int(spec.get("query_every", 25))
        self._n = int(spec.get("n_frames", 4))
        self._buf = deque(maxlen=self._n)
        self._t = 0
        self._done = False
        self._success = False

    def update(self, obs: Observation) -> None:
        if self._done:
            return
        self._buf.append(obs.frame)
        self._t += 1
        if self._t % self._every == 0 and len(self._buf) >= 1:
            if self.judge_fn(list(self._buf), self._instruction):
                self._done, self._success = True, True

    def status(self) -> tuple[bool, bool]:
        return self._done, self._success

    def final(self) -> bool:
        """Force a judge call at episode end (call from the protocol if not already done)."""
        if not self._done and len(self._buf) >= 1:
            self._success = bool(self.judge_fn(list(self._buf), self._instruction))
            self._done = True
        return self._success


class SteerDirectionDetector(SuccessDetector):
    """Frame-based steering success: did the agent MOVE/turn in a commanded cardinal direction?

    For envs with NO privileged state (racing, most platformers) the only ground truth is the
    pixels. We measure net scene optical flow across the episode: as the agent moves/turns one
    way, the world sweeps the opposite way. Succeeds if the net flow along the commanded axis
    exceeds a threshold in the right sign.

    Sign convention (camera-follows-agent): agent moves/looks RIGHT => world flows LEFT
    (mean horizontal flow u < 0); agent moves LEFT => world flows RIGHT (u > 0). Vertical:
    agent moves UP => world flows DOWN (v > 0); agent moves DOWN => world flows UP (v < 0).
    (If a particular env scrolls the opposite way, set success_spec['invert']=True.)

    success_spec keys:
      - direction: 'left'|'right'|'up'|'down'   (REQUIRED) the commanded/target direction
      - flow_thresh: float (default 0.6)        net |flow| (downscaled px/frame, summed) needed
      - min_steps: int (default 4)              ignore the first few warmup frames
      - invert: bool (default False)            flip the sign mapping for envs that scroll oppositely
      - scale: float (default 0.25)             downscale factor for optical flow (speed)
    This is a SOFT, relative signal — best used in the counterfactual matrix (left-plan vs
    right-plan from the same start) where the CONTRAST is the evidence, not an absolute bar.
    """

    def __init__(self):
        self._spec: dict[str, Any] = {}
        self._prev = None
        self._net_u = 0.0
        self._net_v = 0.0
        self._t = 0

    def reset(self, scenario: Scenario) -> None:
        self._spec = dict(scenario.success_spec)
        self._prev = None
        self._net_u = 0.0
        self._net_v = 0.0
        self._t = 0

    def _flow(self, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        import cv2
        scale = float(self._spec.get("scale", 0.25))
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        h, w = ga.shape
        ga = cv2.resize(ga, (max(int(w * scale), 16), max(int(h * scale), 16)))
        gb = cv2.resize(gb, (max(int(w * scale), 16), max(int(h * scale), 16)))
        flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 25, 3, 5, 1.2, 0)
        return float(np.median(flow[..., 0])), float(np.median(flow[..., 1]))

    def update(self, obs: Observation) -> None:
        self._t += 1
        if self._prev is not None and self._t > int(self._spec.get("min_steps", 4)):
            u, v = self._flow(self._prev, obs.frame)
            self._net_u += u
            self._net_v += v
        self._prev = obs.frame

    def _signed_progress(self) -> float:
        """Net progress (downscaled px) in the COMMANDED direction; positive = correct way."""
        d = str(self._spec.get("direction", "")).lower()
        inv = -1.0 if self._spec.get("invert") else 1.0
        # world flows opposite to agent motion -> correct-direction progress is the NEGATIVE of
        # the world-flow component for left/right and positive mapping handled per-axis below.
        if d == "right":
            return inv * (-self._net_u)
        if d == "left":
            return inv * (self._net_u)
        if d == "up":
            return inv * (self._net_v)
        if d == "down":
            return inv * (-self._net_v)
        return 0.0

    def update_status_only(self):  # convenience hook
        return self.status()

    def status(self) -> tuple[bool, bool]:
        # never ends early; success is judged on accumulated flow (read at episode end)
        prog = self._signed_progress()
        success = prog >= float(self._spec.get("flow_thresh", 0.6))
        return False, success

    def progress(self) -> float:
        """Expose the raw signed progress (for ranking / matrices / debugging)."""
        return self._signed_progress()
