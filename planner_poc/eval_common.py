"""eval_common.py — shared SURVIVAL-AWARE advance for all eval rollouts (fixes the 'plays through death into
menus' bug). The naive (reward_var_end - reward_var_start) is corrupted when the actor dies mid-rollout: the
level resets, progress (screen_x/xpos) jumps back to ~start, and continued stepping measures menu/respawn motion
instead of how far the attempt actually got.

survival_advance() steps chunk-by-chunk, captures the env's (frame, reward, done, info) return, and STOPS at the
first death. Death is detected robustly across env types (RetroRLEnv lives, IntegrationEnv health) by:
  (1) info['died'] / done == True  (env's own game-over signal), OR
  (2) a PROGRESS RESET: reward_var drops by > reset_drop in one chunk (level reset on respawn).
Advance = running max(reward_var) - start  (furthest progress WHILE ALIVE; robust to minor backtracking).
Returns (advance, survived_chunks, died).
"""
from __future__ import annotations
import numpy as np


def _step_done_info(env, rows):
    """Step the env with an action chunk and normalize the return to (done, info)."""
    out = env.step(rows)
    done, info = False, {}
    if isinstance(out, tuple):
        if len(out) >= 3:
            done = bool(out[2])
        if len(out) >= 4 and isinstance(out[3], dict):
            info = out[3]
    return done, info


def _lives(env):
    """Best-effort LIVES counter (the death signal). Prefer a dedicated 'lives' var (RetroRLEnv + the MMX
    integration both expose it); fall back to env.life_var ONLY if it is literally 'lives' (NOT 'health', which
    decrements on every hit, not on death). Returns None when unavailable."""
    for name in ("lives",):
        try:
            v = env._var(name)
            if v is not None:
                return float(v)
        except Exception:
            pass
    if getattr(env, "life_var", None) == "lives":
        try:
            return float(env._var("lives"))
        except Exception:
            pass
    return None


def survival_advance(pol, env, state, plan, chunks, A, cfg, null=False, reset_drop=40.0, repaint=True):
    """Roll the actor from `state`; return (advance, survived_chunks, died). advance is survival-aware:
    running-max progress BEFORE any death/level-reset, so post-death menu/respawn motion is never counted.
    Death = (a) a LIVES decrement [primary; catches deaths near level-start where progress barely drops],
    (b) info['died']/done, or (c) a PROGRESS RESET (reward_var drops > reset_drop, e.g. level restart)."""
    env.reset(); env.load_state(state)
    if repaint and env.frame().mean() < 1.0:          # MMX/mGBA don't repaint until stepped
        env._emu_step([], 1)
    x0 = env._var(env.reward_var)
    best = 0.0; survived = 0; died = False; prev = x0
    lives_seen = _lives(env)
    for c in range(chunks):
        ch = pol._sample_chunk(env.frame(), plan, cfg, plan_frames=[env.frame()], null=null)
        done, info = _step_done_info(env, ch[:A])
        rv = env._var(env.reward_var)
        lv = _lives(env)
        lost_life = lives_seen is not None and lv is not None and 0 < (lives_seen - lv) <= 3
        if info.get("died") or lost_life or rv < prev - reset_drop:   # game-over / death / level reset -> stop
            died = True
            break
        best = max(best, rv - x0)
        survived = c + 1
        prev = rv
        if lv is not None and lives_seen is not None:
            lives_seen = max(lives_seen, lv)          # allow 1-ups to raise the counter (no false death)
        if done:
            break
    return float(best), survived, died

