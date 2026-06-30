"""ordinal_sanity.py -- the CHEAP FALSIFIER for the eval metric (per the GPT-5.5 war-room critique). Before we
trust "normalized alive progress" as the eval, test that the METRIC RANKS POLICIES CORRECTLY on known-ordered
controls:
    idle/random  <  base(null-plan DiT)  <=  RWBC(adapted)        and   RIGHT+jump = a strong simple control
If the metric does NOT produce that ordering, the metric (or the checkpoint/action map) is broken -> do NOT
trust it. This validates the RULER before we measure models with it.

METRIC (GPT-5.5's refinements):
- progress coordinate P = RUNNING MAX of camera screen_x (NOT player x; Sonic loops make player-x non-monotone),
  gated by ALIVE (lives not decreased) and valid game_mode.
- P_max_alive = max running-max-screen_x over timesteps BEFORE first death.
- normalized = P_max_alive(policy) / P_max_alive(EXPERT continuation from the same start & budget), clipped [0,1].
- also report: survived_to_budget (no death), death_after_peak (died after crossing its best), raw P_max,
  time-to-best (frames).
- many START STATES mined along the expert trajectory (decouples local competence from long-horizon reach).
- matched seeds across policies; equal-weight starts.

Run:  RUN=... CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u planner_poc/ordinal_sanity.py --game sonic \
        --rwbc-delta ckpts/rwbc_sonic_correct_scaled.pt --n-starts 8 --seeds 3
"""
from __future__ import annotations
import argparse, glob, gzip, json, os, sys
import numpy as np
import torch

_R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env
from demo_train_stack import GAME_META
from plan_graded_test import BATTERY

# NitroGen 25-dim action layout (from retro_rl_env): JLX=21,JLY=22 (0.5 neutral, >0.5 right/down),
# south/jump dim 18. Hand-coded controls bypass the DiT to test the METRIC, not a model.
JLX, JLY, SOUTH = 21, 22, 18


def idle_chunk(_t, _rng):
    a = np.zeros((18, 25), np.float32); a[:, JLX] = 0.5; a[:, JLY] = 0.5
    return a


def random_chunk(_t, rng):
    a = np.zeros((18, 25), np.float32)
    a[:, JLX] = rng.uniform(0, 1, 18); a[:, JLY] = rng.uniform(0, 1, 18)
    for d in (SOUTH, 5, 20, 10):                 # a few face/dir buttons randomly
        a[:, d] = (rng.uniform(0, 1, 18) > 0.7).astype(np.float32)
    return a


def right_jump_chunk(t, _rng):
    a = np.zeros((18, 25), np.float32); a[:, JLX] = 1.0; a[:, JLY] = 0.5
    for r in range(18):
        if (t * 18 + r) % 8 < 2:                 # periodic jump pulse
            a[r, SOUTH] = 1.0
    return a


def alive(env, lives0, life_var="lives"):
    if not life_var:
        return True
    try:
        return env._var(life_var) >= lives0
    except Exception:
        return True


# Per-game PROGRESS coordinate + life var. Only side-scrollers (screen_x) have a clean intrinsic
# coordinate; minish uses x-traversal (this demo is a horizontal corridor in one room — NOT a general
# top-down progress axis; see PROGRESS_NOTE) and has no life var (no death gating).
PROGRESS = {
    "smw":    dict(prog="screen_x", life="lives"),
    "sonic":  dict(prog="screen_x", life="lives"),
    "minish": dict(prog="x",        life=None),
}
PROGRESS_NOTE = ("minish progress = max-x; valid only for x-corridor demos, not general top-down. "
                 "FE has no dense signal (cursor exploitable; turn/chapter too sparse) -> not evaluable here.")


@torch.no_grad()
def rollout(env, start, step_chunk, budget_frames, fpr, seed, prog_var="screen_x", life_var="lives"):
    """Roll a policy (step_chunk(t,rng)->(18,25)) from `start`; return survival-gated running-max progress
    (prog_var) metrics. step_chunk may call the DiT or be hand-coded."""
    rng = np.random.RandomState(seed)
    env.reset(); env.load_state(start)
    lives0 = 0.0
    if life_var:
        try: lives0 = env._var(life_var)
        except Exception: pass
    pmax = env._var(prog_var); pmax_alive = pmax; died = False; t = 0
    frames = 0; tbest = 0; peak_before_death = pmax
    while frames < budget_frames:
        ch = step_chunk(t, rng)
        env.step(ch)
        frames += ch.shape[0] * fpr
        sx = env._var(prog_var)
        if sx > pmax:
            pmax = sx; tbest = frames
        al = alive(env, lives0, life_var)
        if al and not died:
            if sx > pmax_alive:
                pmax_alive = sx
        if not al and not died:
            died = True; peak_before_death = pmax_alive
        t += 1
    survived = not died
    death_after_peak = died and (peak_before_death >= pmax_alive - 1e-6)
    return dict(p_max=float(pmax), p_max_alive=float(pmax_alive), survived=bool(survived),
                died=bool(died), death_after_peak=bool(death_after_peak), t_best=int(tbest))


def expert_denom(env, start, expert_acts, start_idx, bmap, budget_frames, prog_var="screen_x", life_var="lives"):
    """Replay the EXPERT continuation from the mined start for the same emu-frame budget; running-max prog."""
    env.reset(); env.load_state(start)
    lives0 = 0.0
    if life_var:
        try: lives0 = env._var(life_var)
        except Exception: pass
    pmax = env._var(prog_var); i = start_idx
    for _ in range(budget_frames):
        if i >= expert_acts.shape[0]:
            break
        names = [bmap[k] for k in range(len(bmap)) if expert_acts[i, k] > 0.5]
        env._emu_step(names, 1)
        sx = env._var(prog_var)
        if sx > pmax and alive(env, lives0, life_var):
            pmax = sx
        i += 1
    return float(pmax)


def mine_starts(env, demo_dir, n_starts, skip_head=120, prog_var="screen_x"):
    """Replay the expert; snapshot (savestate, expert_action_index, prog) at n_starts even points
    (skip the first skip_head frames = title/idle). Drops starts already at level-end (screen_x near
    screen_x_end) so we don't test from a finished level."""
    npz = np.load(os.path.join(demo_dir, "demo.npz"))
    acts = npz["actions"]; bmap = list(npz["buttons"]); N = acts.shape[0]
    st = gzip.decompress(open(os.path.join(demo_dir, "initial.state"), "rb").read())
    env.reset(); env.load_state(st)
    try: sx_end = env._var("screen_x_end")
    except Exception: sx_end = None
    idxs = np.linspace(skip_head, max(skip_head + 1, N - 400), n_starts + 4).astype(int)
    idxset = set(int(i) for i in idxs)
    snaps = []
    for i in range(N):
        names = [bmap[k] for k in range(len(bmap)) if acts[i, k] > 0.5]
        env._emu_step(names, 1)
        if i in idxset:
            sx = float(env._var(prog_var))
            if sx_end and prog_var == "screen_x" and sx > 0.9 * sx_end:   # already at level end -> skip
                continue
            snaps.append((env.save_state(), i, sx))
    return snaps[:n_starts], acts, bmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="sonic", choices=["sonic", "smw"])
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--rwbc-delta", default=None)
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--n-starts", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget-frames", type=int, default=900)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda"

    meta = GAME_META[args.game]
    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("ordinal", plan="", objective="progress", cfg_scale=args.cfg))
    init = {n: p.detach().clone() for n, p in pol.m.named_parameters() if p.requires_grad or "lora_" in n}
    fixed_plan = BATTERY[args.game]["correct"]

    env = make_env(args.game)
    fpr = env.frames_per_row
    demo_dir = sorted(glob.glob(os.path.join(_R, "docs/demos/demos", meta["dir"], "*")))
    demo_dir = [d for d in demo_dir if os.path.exists(os.path.join(d, "demo.npz"))][0]

    # DiT policies (closures over pol). base = null plan; rwbc loads a delta then conditions on fixed plan.
    def base_step(t, rng):
        f = env.frame()
        return pol._sample_chunk(f, "", args.cfg, plan_frames=[f], null=True,
                                 noise_seed=rng.randint(1 << 30))

    def make_dit_plan_step():
        def step(t, rng):
            f = env.frame()
            return pol._sample_chunk(f, fixed_plan, args.cfg, plan_frames=[f], null=False,
                                     noise_seed=rng.randint(1 << 30))
        return step

    def load_delta(path):
        d = torch.load(path, map_location="cpu", weights_only=False)["trainable"]
        sd = dict(pol.m.named_parameters()); n = 0
        for k, t in d.items():
            if k in sd: sd[k].data.copy_(t.to(device)); n += 1
        return n

    def restore_base():
        sd = dict(pol.m.named_parameters())
        for k, t in init.items():
            if k in sd: sd[k].data.copy_(t.to(device))

    print(f"\n===== ORDINAL SANITY ({args.game}, {args.n_starts} starts, {args.seeds} seeds, "
          f"budget {args.budget_frames}f, fpr={fpr}) =====", flush=True)
    snaps, acts, bmap = mine_starts(env, demo_dir, args.n_starts)
    print(f"  mined {len(snaps)} start states (expert screen_x {[round(s[2]) for s in snaps]})", flush=True)

    rec = {"game": args.game, "budget_frames": args.budget_frames, "n_starts": len(snaps),
           "seeds": args.seeds, "starts_expert_sx": [s[2] for s in snaps], "policies": {}}
    try:
        # expert denominators per start (replay expert continuation)
        denom = []
        for (st, idx, sx0) in snaps:
            denom.append(expert_denom(env, st, acts, idx, bmap, args.budget_frames))
        # keep only starts where the EXPERT makes real progress (else not a useful test point)
        keep = [i for i in range(len(snaps)) if denom[i] - snaps[i][2] >= 100]
        if keep:
            snaps = [snaps[i] for i in keep]; denom = [denom[i] for i in keep]
        rec["expert_denom"] = denom; rec["n_starts"] = len(snaps)
        print(f"  expert P_max per start: {[round(x) for x in denom]} ({len(snaps)} usable starts)", flush=True)

        # hand-coded + DiT policies
        policy_steps = {"idle": idle_chunk, "random": random_chunk, "right_jump": right_jump_chunk,
                        "base": base_step}
        order = ["idle", "random", "right_jump", "base"]
        if args.rwbc_delta:
            order.append("rwbc")

        for name in order:
            if name == "rwbc":
                restore_base(); nset = load_delta(args.rwbc_delta)
                step = make_dit_plan_step(); print(f"  [rwbc] loaded {nset} delta tensors", flush=True)
            elif name == "base":
                restore_base(); step = base_step
            else:
                step = policy_steps[name]
            norms, survs, daps, raws = [], [], [], []
            for si, (st, idx, sx0) in enumerate(snaps):
                for s in range(args.seeds):
                    r = rollout(env, st, step, args.budget_frames, fpr, seed=1000 * si + s)
                    nz = (r["p_max_alive"] - sx0) / max(1.0, denom[si] - sx0)   # progress GAINED, normalized
                    norms.append(float(np.clip(nz, 0, 1))); survs.append(r["survived"])
                    daps.append(r["death_after_peak"]); raws.append(r["p_max_alive"])
            rec["policies"][name] = dict(norm_mean=float(np.mean(norms)), norm_std=float(np.std(norms)),
                                         survived_frac=float(np.mean(survs)),
                                         death_after_peak_frac=float(np.mean(daps)),
                                         raw_pmax_mean=float(np.mean(raws)))
            out = args.out or os.path.join(_R, "docs/graded", f"ordinal_{args.game}.json")
            os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(rec, open(out, "w"), indent=2)
            p = rec["policies"][name]
            print(f"  {name:10s} norm={p['norm_mean']:.3f}±{p['norm_std']:.3f}  "
                  f"survived={p['survived_frac']:.2f}  death_after_peak={p['death_after_peak_frac']:.2f}  "
                  f"raw_pmax={p['raw_pmax_mean']:.0f}", flush=True)
    finally:
        env.close()

    # ordinal verdict. HEADLINE = reach × survival (the falsifier showed raw reach/P_max_alive rewards
    # SUICIDE-SPRINT: a policy that sprints right and dies scores high on peak-before-death. Multiplying by
    # survived_frac correctly zeroes an always-dying policy and rewards directed SURVIVED progress.)
    P = rec["policies"]
    for n in P:
        P[n]["headline"] = P[n]["norm_mean"] * P[n]["survived_frac"]
    def g(n): return P[n]["headline"] if n in P else float("nan")
    def reach(n): return P[n]["norm_mean"] if n in P else float("nan")
    print("\n  --- HEADLINE = reach × survival ---", flush=True)
    for n in (["idle", "random", "right_jump", "base"] + (["rwbc"] if "rwbc" in P else [])):
        print(f"   {n:10s} headline={g(n):.3f}  (reach={reach(n):.3f} × survived={P[n]['survived_frac']:.2f})", flush=True)
    checks = []
    checks.append(("idle < base", g("idle") < g("base")))
    checks.append(("random < base", g("random") < g("base")))
    checks.append(("suicide-sprint penalized (right_jump headline < base)", g("right_jump") < g("base")))
    if "rwbc" in P:
        checks.append(("base <= rwbc (model improvement shows)", g("base") <= g("rwbc") + 0.02))
    print("\n  --- ORDINAL CHECKS (on survival-weighted headline) ---", flush=True)
    for desc, ok in checks:
        print(f"   [{'PASS' if ok else 'FAIL'}] {desc}", flush=True)
    passed = all(ok for _, ok in checks)
    print(f"  VERDICT: metric {'RANKS CORRECTLY (trust the ruler)' if passed else 'see per-check (a FAIL may be a real model finding, not a metric bug)'}", flush=True)
    rec["ordinal_pass"] = passed; rec["checks"] = {d: bool(o) for d, o in checks}
    json.dump(rec, open(args.out or os.path.join(_R, "docs/graded", f"ordinal_{args.game}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
