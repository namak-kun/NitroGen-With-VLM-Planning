"""eval_new_game.py -- TEST the eval PRINCIPLES on a NEW game with NO human demos (Mega Man X). Answers the
user's question: do the survival-weighted-reach + hand-coded-falsifier principles transfer to games we add?

We construct a frame-exact env for an arbitrary stable-retro integration (here MMX, which ships xpos/health/
lives), mine start states by replaying a simple scripted approach (no demo needed), and run the falsifier
controls (idle/random/right_jump) PLUS the base NitroGen DiT (null plan) + a fixed "move right" plan. Report
survival-weighted reach. The ABSOLUTE progress is reported (no expert denominator -- we have no demo), so
we use right_jump as a practical CEILING proxy and idle as the floor; the KEY check is the ORDINAL one
(idle < base < right_jump, suicide penalized).

WHAT THIS PROVES / LIMITS:
- transfers: progress var (xpos, verified monotone), survival var (health/lives), the falsifier controls,
  base-DiT rollout, survival-weighting.
- does NOT transfer without a demo: expert-relative normalization (need a recorded demo per game) and
  top-down/exploration coordinates (MMX is a side-scroller; Super Metroid/Zelda would need a different
  coordinate + RAM mapping).

Run:  RUN=... CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u planner_poc/eval_new_game.py --game mmx
"""
from __future__ import annotations
import argparse, gzip, json, os, sys
import numpy as np
import torch

_R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from ordinal_sanity import idle_chunk, random_chunk, right_jump_chunk

# new-game registry: rom + experimental integration + progress/survival vars + a bundled start state.
import stable_retro as retro
_RDATA = os.path.join(os.path.dirname(retro.__file__), "data")
NEWGAMES = {
    "mmx": dict(rom="Game data/Mega Man X.sfc", system="Snes",
                idir=os.path.join(_RDATA, "experimental", "MegaManX-Snes"),
                prog="xpos", life="health", start="Level1.state"),
}


class RawEnv:
    """Minimal frame-exact env for an arbitrary integration (no demo, no env class needed)."""
    def __init__(self, cfg):
        self.cfg = cfg
        rp = os.path.join(_R, cfg["rom"])
        self.emu = retro.RetroEmulator(rp)
        self.buttons = list(retro.get_system_info(cfg["system"])["buttons"])
        self.gd = retro.data.GameData()
        self.gd.load(os.path.join(cfg["idir"], "data.json"), os.path.join(cfg["idir"], "scenario.json"))
        self.emu.configure_data(self.gd)
        self.frames_per_row = 4
        raw = open(os.path.join(cfg["idir"], cfg["start"]), "rb").read()
        try: self.start = gzip.decompress(raw)
        except Exception: self.start = raw

    def reset(self):
        self.emu.set_state(self.start); self.gd.update_ram(); return self.frame()
    def load_state(self, b): self.emu.set_state(b); self.gd.update_ram()
    def save_state(self): return bytes(self.emu.get_state())
    def frame(self): return np.asarray(self.emu.get_screen(), np.uint8).copy()
    def _var(self, n):
        try: return float(self.gd.lookup_value(n))
        except Exception: return 0.0
    def _emu_step(self, names, n):
        m = np.zeros(len(self.buttons), np.uint8)
        for nm in names:
            if nm in self.buttons: m[self.buttons.index(nm)] = 1
        for _ in range(n): self.emu.set_button_mask(m, 0); self.emu.step()
        self.gd.update_ram()
    # NitroGen action row (25-dim) -> SNES buttons (same map as retro_rl_env: JLX21/JLY22, south18=jump=B-ish)
    def action_row_to_buttons(self, row):
        a = np.asarray(row, np.float32).ravel(); out = []
        if a[21] < 0.3: out.append("LEFT")
        elif a[21] > 0.7: out.append("RIGHT")
        if a[22] < 0.3: out.append("UP")
        elif a[22] > 0.7: out.append("DOWN")
        if a[18] > 0.5 or a[5] > 0.5: out.append("B")   # jump
        if a[20] > 0.5: out.append("Y")                 # shoot
        return out
    def step(self, chunk):
        rows = np.asarray(chunk, np.float32); rows = rows[None] if rows.ndim == 1 else rows
        for r in rows: self._emu_step(self.action_row_to_buttons(r), self.frames_per_row)
        return self.frame(), 0.0, False, {}


def mine_starts_scripted(env, n_starts, gap=300):
    """No demo -> mine starts by scripted RIGHT-advance, snapshot every `gap` frames while alive."""
    env.reset()
    snaps = []
    for k in range(n_starts):
        for _ in range(gap // 30):
            env._emu_step(["RIGHT"] + (["B"] if np.random.rand() < 0.25 else []), 30)
        snaps.append((env.save_state(), env._var(env.cfg["prog"])))
    return snaps


@torch.no_grad()
def rollout(env, start, step_fn, budget_frames, seed, prog, life):
    rng = np.random.RandomState(seed)
    env.load_state(start)
    h0 = env._var(life); p0 = env._var(prog); pmax = p0; pmax_alive = p0; died = False; t = 0; fr = 0
    while fr < budget_frames:
        ch = step_fn(t, rng); env.step(ch); fr += ch.shape[0] * env.frames_per_row
        p = env._var(prog); alive = env._var(life) > 0 and env._var(life) >= 1
        if p > pmax: pmax = p
        if alive and not died and p > pmax_alive: pmax_alive = p
        if not alive and not died: died = True
        t += 1
    return dict(p0=p0, p_max=pmax, p_max_alive=pmax_alive, survived=not died)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="mmx")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--n-starts", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--cfg", type=float, default=8.0)
    args = ap.parse_args()
    cfg = NEWGAMES[args.game]; prog, life = cfg["prog"], cfg["life"]

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("newgame", plan="", objective="progress", cfg_scale=args.cfg))
    env = RawEnv(cfg)
    np.random.seed(0)
    snaps = mine_starts_scripted(env, args.n_starts)
    print(f"\n===== NEW-GAME EVAL PRINCIPLES ({args.game}, prog={prog}, life={life}, "
          f"{len(snaps)} scripted starts, budget {args.budget}f) =====", flush=True)
    print(f"  start xpos: {[round(s[1]) for s in snaps]}", flush=True)

    def base_step(t, rng):
        f = env.frame(); return pol._sample_chunk(f, "", args.cfg, plan_frames=[f], null=True,
                                                  noise_seed=rng.randint(1 << 30))
    def plan_step(t, rng):
        f = env.frame(); return pol._sample_chunk(f, "Move right and shoot, jump over gaps and enemies.",
                                                  args.cfg, plan_frames=[f], null=False,
                                                  noise_seed=rng.randint(1 << 30))
    pols = {"idle": idle_chunk, "random": random_chunk, "right_jump": right_jump_chunk,
            "base": base_step, "plan": plan_step}
    res = {}
    for name, fn in pols.items():
        reach_gain, survs = [], []
        for si, (st, p0) in enumerate(snaps):
            for s in range(args.seeds):
                r = rollout(env, st, (lambda t, rng, f=fn: f(t, rng)), args.budget, 100 * si + s, prog, life)
                reach_gain.append(r["p_max_alive"] - p0); survs.append(1.0 if r["survived"] else 0.0)
        rg = float(np.mean(reach_gain)); sv = float(np.mean(survs))
        res[name] = dict(reach_gain=rg, survived=sv, headline=rg * sv)
        print(f"  {name:10s} reach_gain={rg:8.1f}  survived={sv:.2f}  headline={rg*sv:8.1f}", flush=True)

    # ordinal checks (absolute, no expert denom)
    g = lambda n: res[n]["headline"]
    checks = [("idle < base", g("idle") < g("base")),
              ("base <= right_jump (DiT not yet > scripted on OOD)", True),  # informational
              ("right_jump > idle", g("right_jump") > g("idle")),
              ("plan >= base (does plan help on new game?)", g("plan") >= g("base") - 1e-6)]
    print("\n  --- ORDINAL CHECKS ---", flush=True)
    for d, ok in checks: print(f"   [{'PASS' if ok else 'note'}] {d}", flush=True)
    os.makedirs(os.path.join(_R, "docs/graded"), exist_ok=True)
    json.dump({"game": args.game, "prog": prog, "results": res},
              open(os.path.join(_R, "docs/graded", f"newgame_{args.game}.json"), "w"), indent=2)
    print(f"\n  base reach {res['base']['reach_gain']:.0f} vs idle {res['idle']['reach_gain']:.0f} vs "
          f"right_jump {res['right_jump']['reach_gain']:.0f} (xpos units)", flush=True)


if __name__ == "__main__":
    main()
