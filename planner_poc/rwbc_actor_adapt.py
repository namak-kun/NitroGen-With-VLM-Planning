"""rwbc_actor_adapt.py -- THE critical experiment (per the GPT-5.5 rubber-duck): can the DiT-LoRA acquire
better closed-loop behavior from EMULATOR/STATE reward, without collapsing? We use REWARD-WEIGHTED
BEHAVIOR CLONING (the cheapest actor-RL): roll out btn_s600 on a GT-reward env, keep the HIGH-return
action chunks, and BC the DiT-LoRA (+plan-head) on (frame, plan -> full 18-step chunk) via the model's
own flow-matching loss. Then re-evaluate reward. If reward goes UP without collapse -> actor adaptation
works (the pivotal positive result). If not -> the actor is the wall (pivotal negative result).

NOTE: this is a SCOPED proof-of-concept (small data, few steps, single start state) to see signal, not a
full RL run. Anti-collapse: only LoRA + plan-head train (base DiT frozen); low LR; eval reward as the
metric (not training loss). reward-weighting: keep top-quantile chunks, weight loss by normalized return.

Run (one GPU, ~10-20 min):
  CUDA_VISIBLE_DEVICES=0 env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc \
    QWEN=Qwen/Qwen3.5-2B .venv/bin/python planner_poc/rwbc_actor_adapt.py --env thextech \
      --collect-eps 8 --chunks 14 --top-frac 0.4 --steps 200 --lr 1e-4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from play_annotated_horizon import _first_sentence

SYS = {"thextech": "You are the planner for a Mario-style platformer. Goal: advance RIGHT, jump "
                   "platforms, avoid hazards. Output ONE short imperative plan (max 10 words).",
       "sonic": "You are the planner for Sonic, a 2D platformer. Goal: move RIGHT, jump gaps/enemies. "
                "Output ONE short plan (max 10 words).",
       "solarus": "You are the planner for a top-down Zelda-like. Goal: explore, move to new areas, "
                  "fight enemies. Output ONE short plan (max 10 words)."}
INSTR = "In one sentence say what to do next using concrete directions (left,right,up,down,jump)."


def _syskey(name):
    if name == "sonic":
        return "sonic"
    if "solarus" in name or "minish" in name:
        return "solarus"
    return "thextech"


def make_env(name):
    if name == "sonic":
        from nitrogen.eval.envs.retro_rl_env import RetroRLEnv
        return RetroRLEnv()
    if name == "smw":
        from nitrogen.eval.envs.retro_rl_env import RetroRLEnv
        return RetroRLEnv(rom_path="Game data/Super Mario World.sfc",
                          game="SuperMarioWorld-Snes-v0", system="Snes", reward_var="screen_x")
    if name in ("minish", "minish_cap"):
        from nitrogen.eval.envs.gba_env import GbaRLEnv
        return GbaRLEnv(game="minish_cap")
    if name == "mmx":
        from nitrogen.eval.envs.new_demo_envs import make_mmx
        return make_mmx()
    if name == "smbas":
        from nitrogen.eval.envs.new_demo_envs import make_smbas
        import os, json
        cfgp = os.path.join("tmp", "retro_data", "smbas_progress.json")
        kw = {k: v for k, v in json.load(open(cfgp)).items() if not k.startswith("_")} if os.path.exists(cfgp) else {}
        return make_smbas(**kw)
    if name.startswith("gba_"):
        from nitrogen.eval.envs.gba_env import make_gba_env
        return make_gba_env(name)
    from nitrogen.eval.envs.proc_rl_env import ProcRLEnv
    return ProcRLEnv(name)


def collect(pol, env, name, eps, chunks, A, cfg, seed0=0, plan_temp=0.0, explore_sigma=0.0, fixed_plan=None,
            reward_mode="local", gamma=0.95, residual_mask=False):
    """Roll out on a REUSED env (caller owns construct/close); return dicts {frame, plan, action(H,25),
    reward}. plan_temp>0 SAMPLES plans (exploration). explore_sigma>0 makes the ACTOR sampler stochastic
    (late-step SDE noise) so the kept top-frac chunks are the BEST EXPLORED actions -> self-imitation can
    discover beyond the greedy plateau (the DDPO-lite bridge), no log-prob/PG needed. fixed_plan != None
    skips the (noisy) planner and conditions on ONE clean plan string (best-of-K finding: a good fixed plan
    ~= oracle, and avoids wrong-direction plan noise polluting the BC data).

    reward_mode: "local" = each chunk keeps its own per-step env reward (DEFAULT, unchanged). "rtg" =
    RETURN-TO-GO: each chunk is scored by the discounted sum of FUTURE rewards in its episode
    (G_t = r_t + gamma*G_{t+1}). This fixes the credit-assignment failure that collapses obstacle games:
    a risky forward-sprint chunk that causes a DEATH 2-3 chunks later inherits that -death_penalty (discounted)
    and is correctly DROPPED by the top-frac filter, instead of being kept for its local +screen_x. This is the
    AWR/GRPO return-based advantage (the env death/stuck penalties only become effective once propagated)."""
    sk = _syskey(name)
    samples = []
    for ep in range(eps):
        torch.manual_seed(seed0 + ep); np.random.seed(seed0 + ep)
        cur = env.reset(); hist = [cur]; plan = fixed_plan or "move right"
        ep_samples, ep_rewards = [], []
        for t in range(chunks):
            if fixed_plan is None and t % A == 0:
                plan = _first_sentence(pol.pl.generate_plan(hist[-4:], pol.device, instruction=INSTR,
                                       system=SYS[sk], max_new_tokens=32, temperature=plan_temp) or "move right")
            chunk = pol._sample_chunk(cur, plan, cfg, plan_frames=[cur], null=False,
                                      noise_sigma=explore_sigma, noise_seed=(seed0 + ep) * 1000 + t)  # (H,25)
            obs, reward, done, info = env.step(chunk[:A])
            srow = {"frame": np.asarray(cur), "plan": plan,
                    "action": np.asarray(chunk, np.float32), "reward": float(reward)}
            if residual_mask:   # cache FROZEN base-DiT null actions for the per-dim frame-counterfactual mask
                sb = (seed0 + ep) * 1000 + t
                srow["base_null"] = np.asarray(pol._sample_chunk(cur, "", cfg, plan_frames=[cur], null=True,
                                                                 noise_sigma=0.6, noise_seed=sb), np.float32)
                srow["base_null2"] = np.asarray(pol._sample_chunk(cur, "", cfg, plan_frames=[cur], null=True,
                                                                  noise_sigma=0.6, noise_seed=sb + 777), np.float32)
            ep_samples.append(srow)
            ep_rewards.append(float(reward))
            cur = obs; hist.append(cur)
            if done:
                break
        if reward_mode == "rtg":                      # propagate future reward (incl. death) back to each chunk
            G = 0.0
            for i in range(len(ep_samples) - 1, -1, -1):
                G = ep_rewards[i] + gamma * G
                ep_samples[i]["reward"] = G
        samples.extend(ep_samples)
    return samples


def build_batch(pol, samples, device, residual_floor=0.05, residual_scale=0.3):
    """Build a model.forward batch from a list of samples (frame, plan, action(H,25)). The target
    action is ALREADY packed (H,25), so we build actions/actions_mask directly (the tokenizer only
    emits them in training mode and expects raw buttons/sticks).

    If a sample carries 'base_null' (frozen base-DiT null action) we set actions_mask to the per-dim
    FRAME-COUNTERFACTUAL RESIDUAL m = clip((|action - base_null| - |base_null - base_null2|)/scale, floor, 1):
    the plan/LoRA gets gradient ONLY where the reward-selected chunk DEVIATES from what the base DiT already
    emits from the frame (the type-A residual). On frame-determined dims (e.g. sprint-right) m->floor, so the
    plan-head can't be dragged toward the suicide-sprint that collapses obstacle games. null-invariance is
    untouched (mask only reweights the conditional flow-BC target)."""
    tok_keys = ["images", "dropped_images", "vl_token_ids", "sa_token_ids", "vl_attn_mask"]
    rows = {k: [] for k in tok_keys + ["actions", "actions_mask"]}
    plan_h, plan_kpm = [], []
    for s in samples:
        pv = pol.ip([s["frame"]], return_tensors="pt")["pixel_values"][0].numpy()
        ex = pol.tok.encode({"frames": pv[None], "dropped_frames": np.zeros((1,), bool)})
        for k in tok_keys:
            rows[k].append(np.asarray(ex[k]))
        acts = np.asarray(s["action"], np.float32)            # (H,25), already packed
        rows["actions"].append(acts)
        if "base_null" in s:
            b1 = np.asarray(s["base_null"], np.float32)
            b2 = np.asarray(s.get("base_null2", b1), np.float32)
            sig = np.abs(acts - b1) - np.abs(b1 - b2)          # residual minus null-vs-null noise floor
            m = np.clip(sig / residual_scale, residual_floor, 1.0).astype(np.float32)
            rows["actions_mask"].append(m)
        else:
            rows["actions_mask"].append(np.ones_like(acts, dtype=np.float32))
        h, kpm = pol.pl.encode_multimodal([s["frame"]], s["plan"] or ".", device, text_only=pol.mm_text_only)
        plan_h.append(h[0]); plan_kpm.append(kpm[0])
    batch = {k: torch.as_tensor(np.stack(rows[k])).to(device) for k in rows}
    batch["images"] = batch["images"].float()
    batch["actions"] = batch["actions"].float()
    batch["actions_mask"] = batch["actions_mask"].float()
    n = len(samples)
    batch["embodiment_id"] = torch.zeros(n, dtype=torch.long, device=device)
    batch["game_id"] = torch.zeros(n, dtype=torch.long, device=device)
    batch["has_real_action"] = torch.ones(n, device=device)
    batch["plan_dropped"] = torch.zeros(n, dtype=torch.bool, device=device)
    # pad plan hiddens to common length
    L = max(h.shape[0] for h in plan_h)
    d = plan_h[0].shape[-1]
    ph = torch.zeros(n, L, d, device=device, dtype=plan_h[0].dtype)
    pk = torch.ones(n, L, dtype=torch.bool, device=device)
    for i, (h, kpm) in enumerate(zip(plan_h, plan_kpm)):
        ph[i, :h.shape[0]] = h.to(device); pk[i, :h.shape[0]] = kpm.to(device)
    batch["plan_hidden"] = ph; batch["plan_key_padding_mask"] = pk
    return batch


def eval_reward(pol, env, name, eps, chunks, A, cfg, seed0=100, fixed_plan=None):
    sk = _syskey(name)
    tot = []
    for ep in range(eps):
        torch.manual_seed(seed0 + ep); np.random.seed(seed0 + ep)
        cur = env.reset(); hist = [cur]; plan = fixed_plan or "move right"; r_ep = 0.0
        for t in range(chunks):
            if fixed_plan is None and t % A == 0:
                plan = _first_sentence(pol.pl.generate_plan(hist[-4:], pol.device, instruction=INSTR,
                                       system=SYS[sk], max_new_tokens=32) or "move right")
            chunk = pol._sample_chunk(cur, plan, cfg, plan_frames=[cur], null=False)
            obs, reward, done, info = env.step(chunk[:A]); r_ep += reward
            cur = obs; hist.append(cur)
            if done:
                break
        tot.append(r_ep)
    return float(np.mean(tot)), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="thextech")
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--collect-eps", type=int, default=8)
    ap.add_argument("--chunks", type=int, default=14)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--top-frac", type=float, default=0.4, help="keep top-return fraction of chunks")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--eval-eps", type=int, default=4)
    ap.add_argument("--lora-only", action="store_true", help="train ONLY DiT-LoRA (freeze plan-head) — the duck's actor-only warmup")
    ap.add_argument("--anchor", type=float, default=0.0, help="L2 anchor of trainable params to their init (anti-collapse KL proxy)")
    ap.add_argument("--plan-temp", type=float, default=0.0, help="temperature for plan sampling during COLLECTION (exploration; 0=greedy)")
    ap.add_argument("--explore-sigma", type=float, default=0.0, help="actor-side stochastic exploration noise during COLLECTION (DDPO-lite; 0=greedy)")
    ap.add_argument("--fixed-plan", default=None, help="condition on ONE fixed plan string (skip the noisy planner)")
    ap.add_argument("--use-correct-plan", action="store_true", help="set fixed-plan = plan_graded_test.BATTERY[env]['correct'] (clean near-oracle plan)")
    ap.add_argument("--save-delta", default=None, help="save trained LoRA(+plan-head) delta here AFTER training (segfault-safe)")
    ap.add_argument("--reward-mode", choices=["local", "rtg"], default="local",
                    help="local=per-chunk env reward (default); rtg=return-to-go (propagates future death/stuck "
                         "back to the chunks that caused it -> fixes obstacle-game RWBC collapse)")
    ap.add_argument("--gamma", type=float, default=0.95, help="discount for --reward-mode rtg")
    ap.add_argument("--seed-offset", type=int, default=0, help="offset added to collect (base 0) and eval (base 100) seeds for multi-seed robustness")
    ap.add_argument("--residual-mask", action="store_true", help="weight the flow-BC loss by the per-dim frame-counterfactual residual (gradient only where the chunk DEVIATES from the base-DiT null prior); collapse-safe plan-head training")
    ap.add_argument("--residual-floor", type=float, default=0.05, help="min mask value for --residual-mask")
    ap.add_argument("--residual-scale", type=float, default=0.3, help="residual normalizer for --residual-mask (smaller=stronger gating)")
    ap.add_argument("--advantage-norm", action="store_true", help="weight kept chunks by group-relative advantage A=(R-mean)/std (AWR/GRPO) instead of min-max reward")
    args = ap.parse_args()
    device = "cuda"
    fixed_plan = args.fixed_plan
    if args.use_correct_plan and fixed_plan is None:
        from plan_graded_test import BATTERY
        key = {"smw": "smw", "sonic": "sonic", "minish": "minish"}.get(args.env, "smw")
        fixed_plan = BATTERY[key]["correct"]
        print(f"[fixed-plan] using correct plan for {args.env}: {fixed_plan!r}")

    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.mm_mode = True; pol.mm_text_only = True
    pol.reset(Scenario("rwbc", plan="", objective="progress", cfg_scale=args.cfg))
    m = pol.m

    # which params train: LoRA (+plan-head unless --lora-only). Base DiT frozen -> anti-collapse.
    for n, p in m.named_parameters():
        train_it = ("lora_" in n) or (not args.lora_only and n.startswith("plan_head."))
        p.requires_grad_(train_it)
    train_params = [p for n, p in m.named_parameters() if p.requires_grad]
    init = [p.detach().clone() for p in train_params]   # for the L2 anchor
    n_train = sum(p.numel() for p in train_params)
    print(f"RWBC actor-adapt on {args.env} | trainable {n_train/1e6:.2f}M "
          f"({'LoRA-only' if args.lora_only else 'LoRA+plan-head'}) lr={args.lr} anchor={args.anchor}\n")

    print("[1/4] baseline reward (pre-RWBC)...", flush=True)
    env = make_env(args.env)             # ONE env for the whole run (emulator cores segfault on repeat
    try:                                 # construction in a long-lived CUDA process; reset() per episode).
        base_r, base_list = eval_reward(pol, env, args.env, args.eval_eps, args.chunks, args.A, args.cfg, seed0=100 + args.seed_offset, fixed_plan=fixed_plan)
        print(f"   baseline mean reward = {base_r:+.3f}  {[round(x,2) for x in base_list]}\n")

        print(f"[2/4] collecting {args.collect_eps} eps (plan_temp={args.plan_temp}, reward_mode={args.reward_mode})...", flush=True)
        samples = collect(pol, env, args.env, args.collect_eps, args.chunks, args.A, args.cfg,
                          seed0=args.seed_offset,
                          plan_temp=args.plan_temp, explore_sigma=args.explore_sigma, fixed_plan=fixed_plan,
                          reward_mode=args.reward_mode, gamma=args.gamma, residual_mask=args.residual_mask)
        rewards = np.array([s["reward"] for s in samples])
        thr = np.quantile(rewards, 1 - args.top_frac)
        keep = [s for s in samples if s["reward"] >= thr]
        print(f"   collected {len(samples)} chunks; keep top {args.top_frac} (reward>={thr:.3f}) -> {len(keep)} chunks "
              f"(reward mean {np.mean([s['reward'] for s in keep]):+.3f}) "
              f"[residual-mask={args.residual_mask} adv-norm={args.advantage_norm}]\n")

        print(f"[3/4] reward-weighted BC: {args.steps} steps lr={args.lr}...", flush=True)
        opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.0)
        R = np.array([s["reward"] for s in keep])
        if args.advantage_norm:                       # AWR/GRPO group-relative advantage, positive-clamped
            rw = np.clip((R - R.mean()) / (R.std() + 1e-6), 0.0, None) + 0.1
        else:
            rw = (R - R.min()) / (R.max() - R.min() + 1e-6) + 0.2
        for step in range(args.steps):
            m.train()
            idx = np.random.choice(len(keep), size=min(args.bs, len(keep)), replace=False)
            batch = build_batch(pol, [keep[i] for i in idx], device,
                                residual_floor=args.residual_floor, residual_scale=args.residual_scale)
            w = torch.tensor(rw[idx], device=device, dtype=torch.float32).mean()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = m(batch)
                loss = out["loss"] * w
            if args.anchor > 0:
                anc = sum(((p - p0) ** 2).sum() for p, p0 in zip(train_params, init))
                loss = loss + args.anchor * anc
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step()
            if step % 25 == 0 or step == args.steps - 1:
                print(f"   step {step:3d} loss {float(out['loss'].detach()):.4f}", flush=True)
            # intermediate reward checkpoints to SEE the trajectory (collapse vs improve)
            if args.steps >= 100 and step in (49, 99) and step != args.steps - 1:
                m.eval()
                mid_r, _ = eval_reward(pol, env, args.env, max(2, args.eval_eps // 2), args.chunks, args.A, args.cfg, seed0=100 + args.seed_offset, fixed_plan=fixed_plan)
                print(f"   [mid step {step}] reward = {mid_r:+.3f}", flush=True)
        m.eval()

        # save the trained delta BEFORE post-eval (post-eval rolls the env -> segfault risk; don't lose weights)
        if args.save_delta:
            sd = {n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad}
            torch.save({"trainable": sd, "env": args.env, "fixed_plan": fixed_plan,
                        "base_reward": base_r, "lora_only": args.lora_only}, args.save_delta)
            print(f"   saved trained delta ({len(sd)} tensors) -> {args.save_delta}", flush=True)

        print(f"\n[4/4] post-RWBC reward...", flush=True)
        post_r, post_list = eval_reward(pol, env, args.env, args.eval_eps, args.chunks, args.A, args.cfg, seed0=100 + args.seed_offset, fixed_plan=fixed_plan)
    finally:
        env.close()
    print(f"   post mean reward = {post_r:+.3f}  {[round(x,2) for x in post_list]}")
    print(f"\n===== RESULT: reward {base_r:+.3f} -> {post_r:+.3f}  (Δ={post_r-base_r:+.3f}) =====")
    print(f"   {'ACTOR ADAPTS (reward up)' if post_r>base_r+1e-2 else 'no improvement / collapse'} "
          f"on {args.env} via reward-weighted BC of DiT-LoRA+plan-head.")


if __name__ == "__main__":
    main()
