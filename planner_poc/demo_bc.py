"""demo_bc.py -- EXPERT behavioral cloning from human gold demos to bootstrap an OOD actor past the
competence floor (the fix self-imitation RWBC can't do; see plan.md competence-band finding).

Human demos (demos-*.zip -> docs/demos/) give exact (frame, 12-button-action) expert pairs, ROM-matched to
our Game data ROMs, with loadable start/end save-states (gzip'd #!s9xsnp:0009 = our stable-retro format).
This maps the 12 console buttons -> NitroGen's 25-dim action, chunks the demo into H=18 @stride2 chunks, BC's
the DiT-LoRA on (frame -> expert action chunk), and evals from the demo initial.state vs the pre-BC actor.

Mapping (NitroGen 25-dim = buttons[0:21] + j_left[21:23] + j_right[23:25], 0.5=neutral):
  SNES B->south(18) Y->west(20) SELECT->back(0) START->start(19) A->east(5) X->north(10) L->lshoulder(7)
  R->rshoulder(14); d-pad LEFT/RIGHT->j_left x (0/1), UP/DOWN->j_left y (0/1).  (inverse of snes_env map.)

Run:
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python planner_poc/demo_bc.py --game smw --steps 600
"""
from __future__ import annotations

import argparse
import glob
import gzip
import os
import sys
import copy

import numpy as np
import torch

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from rwbc_actor_adapt import make_env, build_batch, eval_reward
from plan_graded_test import BATTERY
from eval_common import survival_advance

# SNES 12-button order (demo): B,Y,SELECT,START,UP,DOWN,LEFT,RIGHT,A,X,L,R
SNES_BTN_TO_NITRO = {0: 18, 1: 20, 2: 0, 3: 19, 8: 5, 9: 10, 10: 7, 11: 14}  # face/shoulder/start/select
JLX, JLY = 21, 22

GAME_CFG = {
    "smw": dict(demo_glob="docs/demos/demos/SuperMarioWorld-Snes/*", env="smw"),
    "sonic": dict(demo_glob="docs/demos/demos/SonicTheHedgehog2-Genesis/*", env="sonic"),
    "mmx": dict(demo_glob="docs/demos/demos/MegaManX-Snes/*", env="mmx"),
    "smbas": dict(demo_glob="docs/demos/demos/SuperMarioAllStars-Snes/*", env="smbas"),
}
GENERIC_PLAN = "move right, run and jump over obstacles to advance"

# R9 (war-room consensus): the ducking collapse is a DATA artifact -- one constant terse plan makes plain BC
# ignore the plan and emit the marginal advance policy, dropping rare situational actions (duck=1.7%->0%).
# FIX = per-chunk situational plans where the plan LABEL IS DERIVED FROM THE CHUNK'S OWN ACTION (no VLM/narration
# needed; the action labels itself -> generalizes to every game). Thresholds/text are the GPT-5.5+Opus consensus.
SIT_PLANS = {
    "duck":    "Press down to duck under the enemy or hazard, then keep moving right.",
    "retreat": "Turn around and move left, back away from the hazard.",
    "wait":    "Stop and wait, hold still until the path ahead is clear.",
    "jump":    "Jump now to clear the gap or enemy ahead, then keep moving right.",
}  # "advance" falls back to the per-game BATTERY['correct'] plan


def label_chunk(a: np.ndarray) -> str:
    """Action-derived situational label for an (H,25) demo chunk (map_action dims: DOWN=a[:,22]>0.5,
    LEFT=a[:,21]<0.5, RIGHT=a[:,21]>0.5, JUMP/south=a[:,18]>0.5). Priority duck>retreat>wait>jump>advance so
    'jump' never swallows duck+jump chunks (both agents' ordering)."""
    down = float((a[:, 22] > 0.5).mean())
    left = float((a[:, 21] < 0.5).mean())
    right = float((a[:, 21] > 0.5).mean())
    jump = float((a[:, 18] > 0.5).mean())
    if down >= 3 / 18:                 return "duck"
    if left > right and left > 0.30:   return "retreat"
    if right < 0.15 and jump < 0.15:   return "wait"
    if jump >= 0.40:                   return "jump"
    return "advance"


_TRIM = None
def _trim_seconds(demo_dir):
    """Tail-trim (seconds) for a demo dir, from docs/demos/demo_trim.json (death/restart/idle tails)."""
    global _TRIM
    if _TRIM is None:
        tp = os.path.join(_R, "docs/demos/demo_trim.json")
        try:
            import json as _j
            _TRIM = {k: v for k, v in _j.load(open(tp)).items() if not k.startswith("_")}
        except Exception:
            _TRIM = {}
    key = "/".join(demo_dir.rstrip("/").split("/")[-2:])    # '<Game>/<timestamp>'
    return float(_TRIM.get(key, 0.0))


def map_action(snes12: np.ndarray) -> np.ndarray:
    """12 binary SNES buttons -> NitroGen 25-dim action row."""
    a = np.zeros(25, np.float32); a[JLX] = a[JLY] = 0.5; a[23] = a[24] = 0.5
    for s_idx, n_dim in SNES_BTN_TO_NITRO.items():
        if snes12[s_idx]:
            a[n_dim] = 1.0
    if snes12[6]:  a[JLX] = 0.0   # LEFT
    if snes12[7]:  a[JLX] = 1.0   # RIGHT
    if snes12[4]:  a[JLY] = 0.0   # UP
    if snes12[5]:  a[JLY] = 1.0   # DOWN
    return a


def load_demo_chunks(demo_glob, H=18, stride=2, chunk_stride=18, max_chunks=None, plan=GENERIC_PLAN,
                     situational=False, advance_plan=None):
    """Return list of {frame(HxWx3 uint8), action(H,25), plan, label} expert chunks across all demos for a game.
    If situational, each chunk's plan is DERIVED FROM ITS OWN ACTION (label_chunk) -> SIT_PLANS, with 'advance'
    falling back to advance_plan (the per-game BATTERY['correct'])."""
    samples = []
    for d in sorted(glob.glob(demo_glob)):
        npz = os.path.join(d, "demo.npz")
        if not os.path.exists(npz):
            continue
        z = np.load(npz, allow_pickle=True)
        obs, acts = z["observations"], z["actions"]            # (N+1,H,W,3), (N,12)
        fps = 60.0
        trim = int(_trim_seconds(d) * fps)                     # drop death/restart/idle tail before chunking
        if trim > 0:
            acts = acts[:max(0, len(acts) - trim)]
        span = H * stride
        i = 0
        while i + span <= len(acts):
            chunk = np.stack([map_action(acts[i + stride * k]) for k in range(H)])  # (18,25)
            if situational:
                lab = label_chunk(chunk)
                p = SIT_PLANS.get(lab, advance_plan or plan)
            else:
                lab, p = "advance", plan
            samples.append({"frame": np.asarray(obs[i]), "plan": p, "label": lab,
                            "action": chunk.astype(np.float32), "reward": 1.0})
            i += chunk_stride
    if max_chunks:
        samples = samples[:max_chunks]
    return samples


def add_residual_targets(pol, samples, cfg):
    """Populate base_null/base_null2 (frozen base-DiT null action) per chunk so build_batch uses the per-dim
    frame-counterfactual residual as a SUPERVISED weight on the (trusted) human action target. R7: the residual
    belongs here (trusted demo target), not as an RWBC self-imitation gate."""
    for i, s in enumerate(samples):
        f = s["frame"]
        s["base_null"] = np.asarray(pol._sample_chunk(f, "", cfg, plan_frames=[f], null=True,
                                                      noise_sigma=0.6, noise_seed=7 * i + 1), np.float32)
        s["base_null2"] = np.asarray(pol._sample_chunk(f, "", cfg, plan_frames=[f], null=True,
                                                       noise_sigma=0.6, noise_seed=7 * i + 2), np.float32)


def demo_start_states(demo_glob):
    """Loadable initial states (gunzip'd) for eval, one per demo."""
    out = []
    for d in sorted(glob.glob(demo_glob)):
        sp = os.path.join(d, "initial.state")
        if os.path.exists(sp):
            out.append(gzip.decompress(open(sp, "rb").read()))
    return out


@torch.no_grad()
def eval_from_states(pol, env, states, plan, chunks, A, cfg, null=False, survival=True):
    """Mean screen_x advance running the actor from each demo start state (plan-conditioned, or null).
    survival=True (default) uses the death-aware advance (survival_advance): running-max progress BEFORE any
    death/level-reset, so a rollout that dies mid-way and respawns/menu-navigates is not counted as 'advance'
    (the old end-minus-start metric was corrupted by death). Set survival=False for the legacy raw metric."""
    deltas = []
    for st in states:
        if survival:
            adv, _surv, _died = survival_advance(pol, env, st, plan, chunks, A, cfg, null=null)
            deltas.append(adv)
            continue
        env.reset(); env.load_state(st)
        if env.frame().mean() < 1.0:        # some emulators (MMX/mGBA) don't repaint until stepped
            env._emu_step([], 1)
        x0 = env._var(env.reward_var)
        for _ in range(chunks):
            ch = pol._sample_chunk(env.frame(), plan, cfg, plan_frames=[env.frame()], null=null)
            env.step(ch[:A])
        deltas.append(float(env._var(env.reward_var) - x0))
    return float(np.mean(deltas)), deltas


# R9 expressiveness battery: does the bridge press DOWN when the plan says to duck? The base bridge is steerable
# (terse 1.2% -> "press down" 10.8%); the collapsed demo-fit emits 0%. Measure DOWN on BOTH channels: the stick
# (JLY=dim 22, what map_action encodes human DOWN as) AND the dpad_down button (dim 1, what the orig probe used).
DUCK_PLANS = [("terse", None), ("dodge", "duck to dodge the bullet ahead"),
              ("press_down", "press down to duck under it")]


@torch.no_grad()
def duck_probe(pol, env, states, cfg, advance_plan, n_chunks=12, stick_thr=0.6):
    """For each plan in DUCK_PLANS, roll the actor from each start state and report mean DOWN-press rate.
    DPAD channel (dim1 > 0.5) is the clean headline (binary button, matches the orig probe); the STICK channel
    (JLY dim22) rests at 0.5 so it needs a >0.6 threshold to exclude neutral jitter (0.5-thresh gave a noisy
    non-monotone 45%). map_action encodes the human DOWN as the STICK, so training should lift the stick channel.
    'terse' uses the advance plan; returns {tag: (stick%, dpad%)}."""
    out = {}
    for tag, ptxt in DUCK_PLANS:
        plan = advance_plan if ptxt is None else ptxt
        srates, drates = [], []
        for st in states:
            env.reset(); env.load_state(st)
            if env.frame().mean() < 1.0:
                env._emu_step([], 1)
            rows = []
            for _ in range(n_chunks):
                ch = np.asarray(pol._sample_chunk(env.frame(), plan, cfg, plan_frames=[env.frame()]), np.float32)
                rows.append(ch)
                env.step(ch[:2])
            allrows = np.concatenate(rows, 0)                        # (n_chunks*H, 25)
            srates.append(float((allrows[:, 22] > stick_thr).mean()))   # stick DOWN (jitter-excluded)
            drates.append(float((allrows[:, 1] > 0.5).mean()))          # dpad_down button
        out[tag] = (100 * float(np.mean(srates)), 100 * float(np.mean(drates)))
    return out


@torch.no_grad()
def build_anchor_pool(pol, samples, device, plans, n=32, seed=0, text_only=True):
    """Encode n (demo-frame x random anchor-plan) pairs -> padded (n,L,d) plan_hidden + (n,L) kpm for the KL-anchor
    (P0A1): a functional L2 on the plan-TOKEN output to the PRE-fit plan_head over a BROAD plan distribution incl
    duck. Holding the duck plan->token map fixed preserves duck-on-command WITHOUT a label taxonomy."""
    rng = np.random.RandomState(seed)
    hs, kpms = [], []
    for _ in range(n):
        s = samples[rng.randint(len(samples))]
        p = plans[rng.randint(len(plans))]
        h, kpm = pol.pl.encode_multimodal([s["frame"]], p or ".", device, text_only=text_only)
        hs.append(h[0]); kpms.append(kpm[0])
    L = max(h.shape[0] for h in hs); d = hs[0].shape[-1]
    ph = torch.zeros(n, L, d, device=device, dtype=hs[0].dtype)
    pk = torch.ones(n, L, dtype=torch.bool, device=device)
    for i, (h, kpm) in enumerate(zip(hs, kpms)):
        ph[i, :h.shape[0]] = h.to(device); pk[i, :h.shape[0]] = kpm.to(device)
    return ph, pk


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", default="smw", choices=sorted(GAME_CFG))
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--chunk-stride", type=int, default=18)
    ap.add_argument("--eval-chunks", type=int, default=16)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--train", choices=["plan_head", "lora", "both"], default="both",
                    help="which params to fit (R7 pivot: plan_head or both, NOT lora-only)")
    ap.add_argument("--lora-only", action="store_true", help="(legacy) train only LoRA")
    ap.add_argument("--use-correct-plan", action="store_true",
                    help="condition on BATTERY[game]['correct'] instead of the generic plan")
    ap.add_argument("--residual-mask", action="store_true",
                    help="weight the demo-BC loss by the per-dim frame-counterfactual residual (supervised target weight)")
    ap.add_argument("--residual-floor", type=float, default=0.05)
    ap.add_argument("--residual-scale", type=float, default=0.3)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--save-delta", default=None, help="save trained delta (trainable params) here")
    ap.add_argument("--eval-starts", type=int, default=8, help="cap # demo start states used for eval (SMB1 has 30 -> slow)")
    ap.add_argument("--situational-plans", action="store_true",
                    help="R9 fix: per-chunk plan DERIVED FROM THE CHUNK'S ACTION (duck/retreat/wait/jump/advance)")
    ap.add_argument("--rare-oversample", type=float, default=0.0,
                    help="if >0, batch-sample so rare situational chunks (duck/retreat/wait) hit ~this fraction")
    ap.add_argument("--duck-probe", action="store_true",
                    help="report DOWN-press rate vs plan explicitness (terse/dodge/press_down) PRE+POST")
    ap.add_argument("--kl-anchor", type=float, default=0.0,
                    help="P0A1 objective-fix: lambda for functional L2 of plan-tokens to the PRE-fit plan_head over "
                         "a broad plan distribution (incl duck) -> preserves expressiveness taxonomy-free")
    ap.add_argument("--kl-anchor-n", type=int, default=32, help="anchor pool size (frame x plan pairs)")
    args = ap.parse_args()
    device = "cuda"
    np.random.seed(args.seed_offset); torch.manual_seed(args.seed_offset)
    cfg = GAME_CFG[args.game]
    plan = BATTERY[args.game]["correct"] if args.use_correct_plan else GENERIC_PLAN
    print(f"[demo-bc] plan = {plan!r}", flush=True)

    samples = load_demo_chunks(cfg["demo_glob"], chunk_stride=args.chunk_stride, plan=plan,
                               situational=args.situational_plans, advance_plan=plan)
    print(f"[demo-bc] {args.game}: {len(samples)} expert chunks", flush=True)
    if args.situational_plans:
        from collections import Counter
        lc = Counter(s["label"] for s in samples)
        print(f"  situational label mix: {dict(lc)}", flush=True)
    # sanity on the mapping: mean stick + button rates
    acts = np.stack([s["action"] for s in samples])
    print(f"  expert chunk stats: j_left_x mean={acts[:,:,JLX].mean():.2f} (RIGHT-ish>0.5)  "
          f"south/jump rate={ (acts[:,:,18]>0.5).mean():.2f}  east/run rate={(acts[:,:,5]>0.5).mean():.2f}")

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("demo_bc", plan="", objective="progress", cfg_scale=args.cfg))
    m = pol.m
    train_lora = args.lora_only or args.train in ("lora", "both")
    train_plan = (not args.lora_only) and args.train in ("plan_head", "both")
    for n, p in m.named_parameters():
        p.requires_grad_((train_lora and "lora_" in n) or (train_plan and n.startswith("plan_head.")))
    train_params = [p for p in m.parameters() if p.requires_grad]
    print(f"  trainable {sum(p.numel() for p in train_params)/1e6:.2f}M (lora={train_lora} plan_head={train_plan})", flush=True)

    if args.residual_mask:
        print("  computing frame-counterfactual residual targets (base-DiT null)...", flush=True)
        add_residual_targets(pol, samples, args.cfg)

    plan_head_ref = pool_h = pool_kpm = None
    if args.kl_anchor > 0:
        plan_head_ref = copy.deepcopy(m.plan_head).eval()
        for p in plan_head_ref.parameters():
            p.requires_grad_(False)
        anchor_plans = list(SIT_PLANS.values()) + [t for _, t in DUCK_PLANS if t] + [plan]
        pool_h, pool_kpm = build_anchor_pool(pol, samples, device, anchor_plans, n=args.kl_anchor_n,
                                             seed=args.seed_offset, text_only=pol.mm_text_only)
        print(f"  KL-anchor ON (lambda={args.kl_anchor}, pool={pool_h.shape[0]} over {len(anchor_plans)} plans)", flush=True)

    env = make_env(cfg["env"])
    try:
        states = demo_start_states(cfg["demo_glob"])[:args.eval_starts]
        # action-level null-invariance ref: masked-null + plan-head-only => the null chunk on a fixed frame+seed
        # must stay BIT-IDENTICAL after training (the rollout-advance null number is just sampling noise, not this).
        nz_frames = [samples[i]["frame"] for i in range(min(4, len(samples)))]
        null_ref = np.stack([np.asarray(pol._sample_chunk(f, "", args.cfg, plan_frames=[f], null=True,
                             noise_seed=99), np.float32) for f in nz_frames])
        null_pre, _ = eval_from_states(pol, env, states, "", args.eval_chunks, args.A, args.cfg, null=True)
        pre, _ = eval_from_states(pol, env, states, plan, args.eval_chunks, args.A, args.cfg)
        print(f"  PRE-BC from {len(states)} demo starts: null={null_pre:+.1f}  plan={pre:+.1f}  "
              f"Δ_plan(pre)={pre-null_pre:+.1f}", flush=True)
        if args.duck_probe:
            dp_pre = duck_probe(pol, env, states, args.cfg, plan)
            print("  DUCK-PROBE PRE (DOWN% stick>0.6/dpad): " +
                  "  ".join(f"{t}={s:.1f}/{d:.1f}" for t, (s, d) in dp_pre.items()), flush=True)

        # rare-action oversampling: upweight situational (duck/retreat/wait) chunks toward args.rare_oversample
        if args.rare_oversample > 0 and args.situational_plans:
            rare = np.array([s["label"] in ("duck", "retreat", "wait") for s in samples], bool)
            n_rare = int(rare.sum())
            w = np.ones(len(samples), np.float64)
            if 0 < n_rare < len(samples):
                tgt = args.rare_oversample
                w[rare] = (tgt / n_rare); w[~rare] = ((1 - tgt) / (len(samples) - n_rare))
            w /= w.sum()
            print(f"  rare-oversample: {n_rare}/{len(samples)} rare chunks -> target frac {args.rare_oversample}", flush=True)
        else:
            w = None

        opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.0)
        for step in range(args.steps):
            m.train()
            idx = np.random.choice(len(samples), size=min(args.bs, len(samples)),
                                   replace=(w is not None), p=w)
            batch = build_batch(pol, [samples[i] for i in idx], device,
                                residual_floor=args.residual_floor, residual_scale=args.residual_scale)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m(batch)["loss"]
                anchor_val = 0.0
                if args.kl_anchor > 0:
                    aix = np.random.choice(pool_h.shape[0], size=min(args.bs, pool_h.shape[0]), replace=False)
                    ai = torch.as_tensor(aix, device=device)
                    cur_tok = m.plan_head(pool_h[ai], key_padding_mask=pool_kpm[ai])[0]
                    with torch.no_grad():
                        ref_tok = plan_head_ref(pool_h[ai], key_padding_mask=pool_kpm[ai])[0]
                    a_loss = torch.nn.functional.mse_loss(cur_tok.float(), ref_tok.float())
                    loss = loss + args.kl_anchor * a_loss
                    anchor_val = float(a_loss.detach())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
            if step % 50 == 0 or step == args.steps - 1:
                extra = f"  anchor {anchor_val:.4f}" if args.kl_anchor > 0 else ""
                print(f"   step {step:3d} loss {float(loss.detach()):.4f}{extra}", flush=True)
        m.eval()

        if args.save_delta:
            sd = {n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad}
            torch.save({"trainable": sd, "game": args.game, "plan": plan,
                        "train": args.train, "residual": args.residual_mask}, args.save_delta)
            print(f"  saved delta ({len(sd)} tensors) -> {args.save_delta}", flush=True)

        null_post, _ = eval_from_states(pol, env, states, "", args.eval_chunks, args.A, args.cfg, null=True)
        post, dl = eval_from_states(pol, env, states, plan, args.eval_chunks, args.A, args.cfg)
        print(f"\n===== DEMO-BC RESULT ({args.game}) =====")
        print(f"  null:    PRE {null_pre:+.1f} -> POST {null_post:+.1f}")
        print(f"  plan:    PRE {pre:+.1f} -> POST {post:+.1f}")
        print(f"  Δ_plan:  PRE {pre-null_pre:+.1f} -> POST {post-null_post:+.1f}  "
              f"(plan-channel improvement Δ={ (post-null_post)-(pre-null_pre):+.1f})")
        print(f"  per-state POST plan: {[round(x,1) for x in dl]}")
        print(f"  => RESULT: plan-advance {'IMPROVED' if post > pre + 15 else 'flat/down'}  "
              f"Δ_plan {'WIDENED' if (post-null_post)>(pre-null_pre)+10 else 'unchanged'}")
        if args.duck_probe:
            dp_post = duck_probe(pol, env, states, args.cfg, plan)
            print("  DUCK-PROBE POST (DOWN% stick>0.6/dpad): " +
                  "  ".join(f"{t}={s:.1f}/{d:.1f}" for t, (s, d) in dp_post.items()), flush=True)
            # headline the DPAD channel (clean binary, matches the orig finding); monotone in plan-explicitness
            td = dp_post["terse"][1]; dd = dp_post["dodge"][1]; pdd = dp_post["press_down"][1]
            pss = dp_post["press_down"][0]
            mono = pdd >= dd >= td
            print(f"  => DUCK pass: press_down dpad={pdd:.1f}% stick={pss:.1f}%  monotone(dpad)={'Y' if mono else 'N'}"
                  f"  (terse dpad={td:.1f}); compare PRE base + collapse 0.0%")
        # action-level null-invariance: recompute the null chunk on the SAME fixed frames+seed -> max|diff| ~ 0
        null_now = np.stack([np.asarray(pol._sample_chunk(f, "", args.cfg, plan_frames=[f], null=True,
                            noise_seed=99), np.float32) for f in nz_frames])
        print(f"  null-invariance (action-level max|diff|): {np.abs(null_now-null_ref).max():.2e} (expect ~0)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
