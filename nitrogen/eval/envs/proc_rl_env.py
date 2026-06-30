"""proc_rl_env.py -- RL reward wrapper for the native FOSS proc envs (TheXTech, Solarus) that export
SEMANTIC ground-truth state via the de-forked readers. Complements retro_rl_env.py (emulator/Genesis):
the proc envs give clean semantic reward but NO frame-exact save/load (reset = relaunch, slow) -> good
for PPO-style RL, not save-state GRPO.

Reward = per-game function of consecutive read_state() dicts. Verified state fields:
  thextech: {x,y,vx,vy,lives,dead,in_menu,won,beat_code}   -> reward = Δx (advance right)
  solarus : {x,y,life,map,in_menu}                         -> reward = movement + map-change bonus

Usage:
  from nitrogen.eval.envs.proc_rl_env import ProcRLEnv
  env = ProcRLEnv("thextech"); obs = env.reset()
  obs, reward, done, info = env.step(action_chunk)   # (H,25); env applies A rows
"""
from __future__ import annotations

import os
import sys

import numpy as np

_R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from run_poc import make_env_factory


def _num(d, k, default=0.0):
    v = (d or {}).get(k, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def reward_thextech(sb, sa):
    """Advance RIGHT. reward = Δx (scaled); big bonus for level win; done on death/win/menu-strand."""
    dx = _num(sa, "x") - _num(sb, "x")
    r = 0.01 * dx
    done = False
    if _num(sa, "won") > 0 or _num(sa, "beat_code") not in (0.0,):
        r += 10.0; done = True
    if _num(sa, "dead") > 0:
        r -= 5.0; done = True
    if _num(sa, "in_menu") > 0:           # stranded in test-pause menu = terminal (no menu affordance)
        done = True
    return r, done


def reward_solarus(sb, sa):
    """Explore: reward movement + new-area (map change); penalize life loss; done on death."""
    dpos = ((_num(sa, "x") - _num(sb, "x")) ** 2 + (_num(sa, "y") - _num(sb, "y")) ** 2) ** 0.5
    r = 0.01 * dpos
    if str((sa or {}).get("map")) != str((sb or {}).get("map")):
        r += 5.0                           # reached a new area
    dlife = _num(sa, "life") - _num(sb, "life")
    if dlife < 0:
        r += 0.5 * dlife                   # took damage
    done = _num(sa, "life") <= 0
    return r, done


REWARD_FNS = {
    "thextech": reward_thextech,
    "thextech_get_flower": reward_thextech,
    "solarus_zelda": reward_solarus,
}

# buttons to force off (menu/pause) so a spurious press can't strand the agent
MENU_MASK = {"thextech": (19,), "thextech_get_flower": (19,), "solarus_zelda": (19,)}


class ProcRLEnv:
    def __init__(self, env_name: str, A: int = 2, objective: str = "make progress", **env_kw):
        self.env_name = env_name
        self.A = A
        self.objective = objective
        self.reward_fn = REWARD_FNS.get(env_name)
        if self.reward_fn is None:
            raise ValueError(f"no reward fn for {env_name}; have {list(REWARD_FNS)}")
        # Optional level override (THEXTECH_LEVEL env var) → headroom test on a full level (vs short bonus1).
        level = os.environ.get("THEXTECH_LEVEL")
        if env_name in ("thextech", "thextech_get_flower") and level:
            from nitrogen.eval.envs.thextech import TheXTechEnv
            self.env = TheXTechEnv(boot_wait=env_kw.get("boot_wait", 15.0),
                                   freeze_during_inference=env_kw.get("freeze", True), level=level)
        else:
            self.env = make_env_factory(env_name, **env_kw)()
        self.H = 18
        self.per_row = self.env.chunk_seconds / self.H
        self._prev = {}
        self._steps = 0

    def reset(self) -> np.ndarray:
        sc = Scenario(self.env_name, plan="", objective=self.objective, max_steps=10_000)
        obs = self.env.reset(sc)
        self._prev = obs.state or {}
        self._steps = 0
        return obs.frame

    def step(self, action_chunk: np.ndarray):
        a = np.asarray(action_chunk, dtype=np.float32)
        rows = a[None] if a.ndim == 1 else a
        chunk = rows[: self.A].copy()
        for b in MENU_MASK.get(self.env_name, ()):
            chunk[:, b] = 0.0
        captured = self.env.apply_chunk_capture(chunk, self.per_row)  # A x (row, frame, state)
        frame = captured[-1][1]
        state = captured[-1][2] or {}
        reward, done = self.reward_fn(self._prev, state)
        self._prev = state
        self._steps += self.A
        info = {"steps": self._steps, "state": state}
        return frame, float(reward), bool(done), info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "thextech"
    env = ProcRLEnv(name)
    obs = env.reset()
    print(f"{name} reset frame {obs.shape} state={env._prev}")
    right = np.full((18, 25), 0.5, np.float32); right[:, 21] = 1.0; right[:, 18] = 1.0  # right + jump
    tot = 0.0
    for t in range(6):
        obs, r, done, info = env.step(right); tot += r
        print(f"  step{t} reward={r:+.3f} state={info['state']} done={done}")
        if done:
            break
    print(f"total reward (right+jump): {tot:+.3f}")
    env.close()
