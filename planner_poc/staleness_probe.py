"""staleness_probe.py — quantify the STALE-PLAN + PLANNER-VARIANCE gap the demo-fit work assumed away.

Our Δ_plan evals used a FIXED ORACLE plan held fresh every chunk. Deployment instead has: (1) the LIVE VLM
generating the plan (planner quality + flip variance), and (2) the plan going STALE between re-plans. This
probe evaluates the SAME model (base or demo-fit delta) from fixed demo save-states under 4 plan regimes and
reports game-progress advance for each:
  null      : no plan (the floor)
  oracle    : BATTERY[game]['correct'], re-encoded every chunk (our ceiling / what demo-fit was trained on)
  live      : VLM generate_plan() every A chunks (the REAL deployment plan; measures planner quality+flip)
  stale     : VLM generate_plan() ONCE at t=0, held the whole rollout (measures pure staleness of the text)
Gaps: (oracle-live)=planner-quality cost; (live-stale)=staleness cost; (live-null)=does the LIVE plan help.
Run with --delta to load a demo-fit plan-head and see if it lifts the LIVE/STALE cases too (the real test).
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "planner_poc"))
from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env, SYS, INSTR, _syskey
from plan_graded_test import BATTERY
from demo_bc import demo_start_states, GAME_CFG
from play_annotated_horizon import _first_sentence
from eval_common import _lives, _step_done_info

# SYS prompts for the new games (right-running platformers -> thextech-style)
SYS = dict(SYS)
SYS.setdefault("thextech", SYS["thextech"])


def syskey(game):
    if game == "sonic":
        return "sonic"
    return "thextech"   # smw/mmx/smbas = right-running platformers


def _plan_text(pol, frames, sk):
    return _first_sentence(pol.pl.generate_plan(frames[-4:], pol.device, instruction=INSTR,
                           system=SYS[sk], max_new_tokens=32) or "move right")


def _encode(pol, frame, text, aug=0):
    """Frame-grounded plan hidden (h,kpm). aug>0: average encode over `aug` jittered frames (MEMO-style
    augmentation consistency); jitter = small pixel rolls. Same text -> same token length -> averageable."""
    if aug <= 1:
        return pol.pl.encode_multimodal([frame], text or ".", pol.device, text_only=pol.mm_text_only)
    hs = []
    h0, kpm = None, None
    for j in range(aug):
        dy, dx = ((j % 2) * 2 - 1) * (j // 2 + 1), ((j + 1) % 2 * 2 - 1) * (j // 2 + 1)
        fj = np.roll(np.asarray(frame), (dy, dx), axis=(0, 1))
        h, kpm = pol.pl.encode_multimodal([fj], text or ".", pol.device, text_only=pol.mm_text_only)
        hs.append(h)
    L = min(h.shape[1] for h in hs)
    return torch.stack([h[:, :L] for h in hs]).mean(0), kpm[:, :L]


def _chunk_dir(ch):
    """Action-direction signature of a chunk (H,25): (x in {L,N,R}, jump bool). For the flip-rate metric."""
    mx = float(np.asarray(ch)[:, 21].mean())
    xdir = "R" if mx > 0.55 else ("L" if mx < 0.45 else "N")
    jump = bool((np.asarray(ch)[:, 18] > 0.5).mean() > 0.3)
    return (xdir, jump)


@torch.no_grad()
def _pooled_token(pol, frame, text, override):
    """Pooled K-plan-token vector for (frame, text/override). The short-path output; its chunk-to-chunk drift =
    plan-token (in)stability. cached (fixed override) -> ~0 drift; fresh (re-grounded) -> >0; fresh_ema -> less."""
    d = pol._prep(frame, text, plan_frames=[frame], plan_hidden_override=override)
    dc = dict(d); dc["plan_dropped"] = torch.tensor([False], device=pol.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pt, _ = pol.m.compute_plan_tokens(dc)        # (1,K,d)
    return pt.float().mean(1)[0]                       # (d,)


@torch.no_grad()
def rollout(pol, env, st, mode, game, chunks=16, A=2, cfg=8.0, replan_every=2, reset_drop=40.0,
            ema=0.7, aug=4):
    """Survival-aware advance + action-direction flip-rate under a plan regime. Modes:
      null   : no plan (floor).                  oracle : BATTERY correct text, re-grounded every chunk (ceiling).
      live   : regenerate text every replan_every, re-ground every chunk (deployment).
      cached : generate text ONCE at t0, encode the K-token plan_hidden ONCE at t0, REUSE it every chunk
               (the TRUE fully-stale floor -- no frame re-grounding).
      fresh  : generate text ONCE at t0, RE-ENCODE the plan_hidden on the LIVE frame every chunk (forward-pass
               TTA = the von Oswald 'short path is already TTT' test; vs cached isolates re-grounding's value).
      fresh_ema : fresh + EMA the plan_hidden across chunks (beta=ema) + aug-average over `aug` jittered frames.
    Returns (advance, flip_rate)."""
    env.reset(); env.load_state(st)
    if env.frame().mean() < 1.0:
        env._emu_step([], 1)
    x0 = env._var(env.reward_var)
    sk = syskey(game); hist = [env.frame()]
    null = (mode == "null")
    plan = ""
    if mode == "oracle":
        plan = BATTERY[game]["correct"]
    elif mode in ("stale", "cached", "fresh", "fresh_ema"):
        plan = _plan_text(pol, hist, sk)            # the t=0 VLM text (held the whole rollout)
    override = None
    if mode == "cached":
        override = _encode(pol, env.frame(), plan)  # encode ONCE at t0, reuse
    ema_h = None
    best = 0.0; prev = x0; lives_seen = _lives(env); toks = []
    for t in range(chunks):
        if mode == "live" and t % replan_every == 0:
            plan = _plan_text(pol, hist, sk)
        if mode == "fresh_ema":
            h_fresh, kpm = _encode(pol, env.frame(), plan, aug=aug)
            ema_h = h_fresh if ema_h is None else (ema * ema_h + (1 - ema) * h_fresh)
            override = (ema_h, kpm)
        cur_override = override if mode in ("cached", "fresh_ema") else None
        if not null:
            toks.append(_pooled_token(pol, env.frame(), plan, cur_override))
        # cached: fixed override; fresh/oracle/live/null: re-ground via plan_frames every chunk
        ch = pol._sample_chunk(env.frame(), plan, cfg, plan_frames=[env.frame()], null=null,
                               plan_hidden_override=cur_override)
        done, info = _step_done_info(env, ch[:A]); hist.append(env.frame())
        rv = env._var(env.reward_var); lv = _lives(env)
        lost_life = lives_seen is not None and lv is not None and 0 < (lives_seen - lv) <= 3
        if info.get("died") or lost_life or rv < prev - reset_drop:   # death/level-reset -> stop counting
            break
        best = max(best, rv - x0); prev = rv
        if lv is not None and lives_seen is not None:
            lives_seen = max(lives_seen, lv)
        if done:
            break
    # plan-token DRIFT = mean cosine distance between consecutive chunks' pooled K-tokens (short-path instability)
    drift = 0.0
    if len(toks) > 1:
        cd = [1.0 - float(torch.nn.functional.cosine_similarity(a, b, dim=0))
              for a, b in zip(toks, toks[1:])]
        drift = float(np.mean(cd))
    return float(best), float(drift)


def eval_modes(pol, game, states, modes, seed=0, ema=0.7, aug=4):
    env = make_env(GAME_CFG[game]["env"])
    adv = {m: [] for m in modes}; flip = {m: [] for m in modes}
    try:
        for st in states:
            for m in modes:
                torch.manual_seed(seed)
                a, f = rollout(pol, env, st, m, game, ema=ema, aug=aug)
                adv[m].append(a); flip[m].append(f)
    finally:
        env.close()
    return ({m: float(np.mean(v)) for m, v in adv.items()},
            {m: float(np.mean(v)) for m, v in flip.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="smw")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--delta", default=None, help="optional demo-fit plan-head delta to load")
    ap.add_argument("--n-starts", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ema", type=float, default=0.7)
    ap.add_argument("--aug", type=int, default=4)
    ap.add_argument("--modes", nargs="+",
                    default=["null", "oracle", "live", "cached", "fresh", "fresh_ema"])
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=8.0)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("stale", plan="", objective="progress", cfg_scale=8.0))
    tag = "BASE"
    if args.delta:
        d = torch.load(args.delta, map_location="cpu", weights_only=False)["trainable"]
        msd = pol.m.state_dict()
        for k, v in d.items():
            if k in msd:
                msd[k] = v.to(msd[k].device, msd[k].dtype)
        pol.m.load_state_dict(msd, strict=False); tag = os.path.basename(args.delta)

    states = demo_start_states(GAME_CFG[args.game]["demo_glob"])[:args.n_starts]
    modes = args.modes
    adv, drift = eval_modes(pol, args.game, states, modes, seed=args.seed, ema=args.ema, aug=args.aug)
    print(f"\n===== STALENESS / FORWARD-PASS TTA [{args.game}] ({tag}, n={len(states)}, seed={args.seed}) =====")
    print(f"  {'mode':10s} {'advance':>9s} {'tok-drift':>10s}")
    for m in modes:
        print(f"  {m:10s} {adv[m]:+9.1f} {drift[m]:10.4f}")
    def g(a, b):
        return adv.get(a, float('nan')) - adv.get(b, float('nan'))
    print("  --- key gaps (von Oswald 'short path is already TTT' test) ---")
    if "fresh" in adv and "cached" in adv:
        denom = (adv.get("oracle", 0) - adv.get("cached", 0)) or 1e-9
        rec = 100 * (adv["fresh"] - adv["cached"]) / denom
        print(f"  fresh-cached (re-grounding value) = {g('fresh','cached'):+.1f}  "
              f"=> recovers {rec:.0f}% of (oracle-cached)")
    if "live" in adv and "fresh" in adv:
        print(f"  live-fresh  (is regenerating text worth it) = {g('live','fresh'):+.1f}")
    if "oracle" in adv and "fresh" in adv:
        print(f"  oracle-fresh (text-quality ceiling gap)     = {g('oracle','fresh'):+.1f}")
    if "fresh_ema" in drift and "fresh" in drift:
        fr = drift["fresh_ema"] / (drift["fresh"] or 1e-9)
        print(f"  tok-drift: cached {drift.get('cached',0):.4f} (expect ~0) | fresh {drift['fresh']:.4f} -> "
              f"fresh_ema {drift['fresh_ema']:.4f}  (EMA ratio {fr:.2f})")
    if args.out:
        import json
        json.dump({"game": args.game, "tag": tag, "seed": args.seed, "n": len(states),
                   "advance": adv, "drift": drift}, open(args.out, "w"), indent=1)
        print(f"  -> {args.out}")



if __name__ == "__main__":
    main()
