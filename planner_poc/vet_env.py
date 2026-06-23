"""Vet a candidate game env: does it boot, can we control it, and is it ~in-distribution for the
FROZEN base NitroGen DiT? Run this on a new MyGameEnv BEFORE investing in scenarios/reward.

Checks:
  (1) BOOT + CAPTURE: env starts headless, frames are real content (std > threshold).
  (2) CONTROL: a non-trivial action chunk changes the frame more than neutral (input reaches game).
  (3) FREEZE: with freeze_during_inference, two grabs without stepping are ~identical.
  (4) IN-DISTRIBUTION: run base NitroGen UNCONDITIONED on a few frames; report the action it
      produces (accel/steer/top buttons). Sane genre-appropriate actions => behaviorally
      in-distribution => a good env (visual skin can differ; control+physics matching is what
      matters). Garbage/neutral => OOD in a dimension that matters -> deprioritize.

Usage:
  PYTHONPATH=. .venv/bin/python planner_poc/vet_env.py <module.path:ClassName> [k=v ...]
  e.g. PYTHONPATH=. .venv/bin/python planner_poc/vet_env.py \
         nitrogen.eval.envs.supertuxkart:SuperTuxKartEnv ai=1
"""
import importlib
import sys

import numpy as np

REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
from nitrogen.shared import BUTTON_ACTION_TOKENS

JLX, JLY = 21, 22
I_RTRIG, I_LTRIG = 16, 9


def load_env(spec, kwargs):
    mod, cls = spec.split(":")
    EnvCls = getattr(importlib.import_module(mod), cls)
    kw = {}
    for kv in kwargs:
        k, v = kv.split("=")
        try:
            v = int(v)
        except ValueError:
            try:
                v = float(v)
            except ValueError:
                v = {"true": True, "false": False}.get(v.lower(), v)
        kw[k] = v
    return EnvCls(freeze_during_inference=True, **kw)


def main():
    spec = sys.argv[1]
    env = load_env(spec, sys.argv[2:])
    import time
    print(f"=== VETTING {spec} ===")
    obs = env.reset()
    f0 = obs.frame.astype(float)
    print(f"(1) BOOT+CAPTURE: frame {obs.frame.shape}, std={f0.std():.1f} "
          f"-> {'OK real content' if f0.std() > 8 else 'BLACK/STATIC (check launch/window)'}")

    # (3) freeze
    g1 = env._grab().astype(float); time.sleep(1.0); g2 = env._grab().astype(float)
    fz = np.abs(g2 - g1).mean()
    print(f"(3) FREEZE: frozen-gap diff={fz:.2f} -> {'OK frozen' if fz < 3 else 'NOT frozen (speedhack?)'}")

    # (2) control: neutral vs a strong action
    neutral = np.zeros((18, 25), np.float32)
    strong = np.zeros((18, 25), np.float32); strong[:, JLX] = -1.0; strong[:, I_RTRIG] = 1.0
    a = env.step(neutral).frame.astype(float)
    b = env.step(strong).frame.astype(float)
    c = env.step(strong).frame.astype(float)
    dn = np.abs(b - a).mean(); ds = np.abs(c - b).mean()
    print(f"(2) CONTROL: neutral->strong frame deltas {dn:.1f}, {ds:.1f} "
          f"-> {'OK input controls game' if max(dn, ds) > 5 else 'NO visible control (check action_to_keys/window focus)'}")

    # (4) in-distribution: base NitroGen unconditioned
    frames = [a, b, c]
    try:
        from eval_policy import NitroGenPolicy
        import torch
        pol = NitroGenPolicy(f"{REPO}/runs/stage2_2b_override/plan_stage1_3000.pt", qwen="Qwen/Qwen3.5-2B")
        pol.mm_mode = False
        print("(4) IN-DISTRIBUTION (base NitroGen unconditioned):")
        for i, f in enumerate(frames):
            torch.manual_seed(0)
            act = pol._sample_chunk(f.astype(np.uint8), "", 1.0, null=True)
            accel = float(act[:, I_RTRIG].mean()); steer = float(act[:, JLX].mean())
            bmean = act[:, :21].mean(0); top = np.argsort(bmean)[::-1][:2]
            tops = [(BUTTON_ACTION_TOKENS[j], round(float(bmean[j]), 2)) for j in top if bmean[j] > 0.1]
            print(f"    frame{i}: accel(RT)={accel:.2f} steer_x={steer:.2f} top_buttons={tops}")
        print("    => sane genre-appropriate actions (e.g. accel in racing, jump in platformer)")
        print("       = behaviorally in-distribution = GOOD env. Garbage/all-neutral = deprioritize.")
    except Exception as e:
        print(f"(4) IN-DISTRIBUTION: skipped ({repr(e)[:80]})")

    env.save_frame("/tmp/vet_frame.png")
    env.close()
    print("saved /tmp/vet_frame.png; vetting done.")


if __name__ == "__main__":
    main()
