"""rl_rollout_demo.py -- faithful end-to-end check of the RL planner OBSERVATION format on a LIVE env.

Unlike the synthetic `rl_planner_prompt.py --demo` (gray frames), this drives the REAL actor (NitroGen
DiT) in a real env, steps each 18-row chunk as S=3 SUB-CHUNKS of 6 rows and captures the frame AFTER
each sub-chunk, then invokes the frozen System-2 planner every A=2 chunks using the sub-chunk-interleaved
prompt (rl_planner_prompt.build_rl_messages) in THINK MODE (budget-forced). It prints, per replan cycle,
the clean plan + a think snippet + cumulative GT reward -- proving the user's "input format, revisited"
works on real env frames and is ready to wire into the RL loop.

Run (free GPU; Sonic = fast frame-exact emulator):
  RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B'
  $RUN CUDA_VISIBLE_DEVICES=1 .venv/bin/python planner_poc/rl_rollout_demo.py --env sonic --cycles 3
  # platformer with semantic reward: --env thextech   (slower; proc env relaunches)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.eval.core import Scenario
from eval_policy import NitroGenPolicy
from play_annotated_horizon import summarize_actions, _first_sentence
from rl_planner_prompt import build_rl_messages, generate_rl_plan, RLPlannerConfig

# game_info + control-genre per env (the System-2 grounding).
GAME_INFO = {
    "sonic": ("Game: Sonic the Hedgehog 2 (Genesis). Objective: advance RIGHT through the level, "
              "as far as possible, avoiding hazards.", "platformer"),
    "smw": ("Game: Super Mario World (SNES). Objective: advance RIGHT and reach the level goal; "
            "JUMP over enemies and gaps.", "platformer"),
    "thextech": ("Game: a Super Mario-style 2D platformer. Objective: advance RIGHT and reach the "
                 "level exit; JUMP over obstacles and gaps.", "platformer"),
    "solarus_zelda": ("Game: a top-down Zelda-like. Objective: explore and move toward new areas "
                      "(doors/exits) in any of LEFT/RIGHT/UP/DOWN.", "topdown"),
    "gba_minish_cap": ("Game: The Minish Cap (GBA). Objective: top-down exploration; move toward doors "
                      "and new rooms in LEFT/RIGHT/UP/DOWN.", "topdown"),
    "gba_pokemon_emerald": ("Game: Pokemon Emerald (GBA). Objective: explore the map.", "topdown"),
    "gba_fire_emblem_sacred_stones": ("Game: Fire Emblem Sacred Stones (GBA). Objective: move the cursor "
                                     "and make tactical progress.", "srpg"),
}


def make_env(name):
    if name == "sonic":
        from nitrogen.eval.envs.retro_rl_env import RetroRLEnv
        return RetroRLEnv()
    if name == "smw":
        from nitrogen.eval.envs.retro_rl_env import RetroRLEnv
        return RetroRLEnv(rom_path="Game data/Super Mario World.sfc",
                          game="SuperMarioWorld-Snes-v0", system="Snes", reward_var="screen_x")
    if name.startswith("gba_"):
        from nitrogen.eval.envs.gba_env import make_gba_env
        return make_gba_env(name)
    # NOTE/GOTCHA: ProcRLEnv.step() applies only the first self.A rows of whatever chunk it is given
    # (rows[:self.A]) — so the S=3 sub-chunks of 6 rows below are NOT fully applied on proc envs
    # (TheXTech/Solarus): only the first A rows of each 6-row sub-chunk take effect. The sub-chunk
    # frame capture is therefore only FAITHFUL on the emulator envs (retro_rl_env/mgba/snes), which
    # apply every row passed to step(). To use a proc env here, set env.A = subchunk_len first (or step
    # row-by-row). Left as-is because Sonic (emulator, frame-exact) is the intended demo target.
    from nitrogen.eval.envs.proc_rl_env import ProcRLEnv
    return ProcRLEnv(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="sonic", choices=sorted(GAME_INFO))
    ap.add_argument("--ckpt", default="ckpts/btn_s600_full.pt")
    ap.add_argument("--qwen", default=os.environ.get("QWEN", "Qwen/Qwen3.5-2B"))
    ap.add_argument("--cycles", type=int, default=3, help="number of A-chunk replan cycles")
    ap.add_argument("--cfg", type=float, default=8.0, help="DiT CFG guidance scale")
    ap.add_argument("--think-budget", type=int, default=200)
    ap.add_argument("--plan-budget", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.0, help="plan sampling temp (RL exploration)")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--seed-plan", default="move right")
    args = ap.parse_args()

    ginfo, genre = GAME_INFO[args.env]
    rlcfg = RLPlannerConfig()                 # K=8, A=2, S=3
    A, S = rlcfg.num_chunks, rlcfg.intra_chunk_rate
    sub_len = rlcfg.subchunk_len              # 18 // 3 = 6
    H = rlcfg.action_horizon                  # 18

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pol = NitroGenPolicy(args.ckpt, qwen=args.qwen, default_cfg=args.cfg)
    pol.reset(Scenario("rl_rollout_demo", plan="", objective="progress", cfg_scale=args.cfg))

    env = make_env(args.env)
    # Proc envs (TheXTech/Solarus) cap step() at env.A rows; set it to the sub-chunk length so each
    # env.step(sub) applies the FULL 6-row sub-chunk (emulator envs apply all rows regardless).
    if hasattr(env, "A"):
        try:
            env.A = sub_len
        except Exception:
            pass
    cur = env.reset()
    plan = args.seed_plan
    cumr = 0.0
    plan_hist = []
    print(f"# RL rollout demo on {args.env} | A={A} chunks/replan, S={S} subchunks, sub_len={sub_len}, "
          f"cfg={args.cfg}, think={'OFF' if args.no_think else 'ON'}")

    done = False
    for cyc in range(args.cycles):
        window_f0 = np.asarray(cur).copy()
        chunk_records = []
        for ci in range(A):
            chunk = pol._sample_chunk(cur, plan, args.cfg, plan_frames=[cur], null=False)  # (18,25)
            subs = []
            for si in range(S):
                sub = chunk[si * sub_len:(si + 1) * sub_len]
                if sub.shape[0] == 0:
                    break
                obs, r, d, info = env.step(sub)
                cumr += float(r)
                summary = summarize_actions([row for row in sub], thresh=0.2)
                subs.append((summary, np.asarray(obs).copy()))
                cur = obs
                done = done or bool(d)
                if done:
                    break
            chunk_records.append({"subs": subs})
            if done:
                break

        # invoke the System-2 planner with the SUB-CHUNK interleaved observation.
        sysd, content, images, render = build_rl_messages(
            window_f0, chunk_records, genre=genre, game_info=ginfo, cfg=rlcfg,
            prev_plans=plan_hist, include_prev_plans=bool(plan_hist))
        res = generate_rl_plan(pol.pl, sysd, content, images, device,
                               enable_thinking=not args.no_think, think_budget=args.think_budget,
                               plan_budget=args.plan_budget, temperature=args.temperature)
        new_plan = _first_sentence(res["plan"]) or plan
        think_snip = (res["think"] or "").replace("\n", " ")[:160]
        print(f"\n[cycle {cyc}] cumR={cumr:+.2f} forced_close={res['forced']}")
        print(f"  prev_plan : {plan}")
        print(f"  THINK     : {think_snip}{'…' if len(res.get('think','')) > 160 else ''}")
        print(f"  NEW PLAN  : {new_plan}")
        plan_hist = (plan_hist + [plan])[-2:]
        plan = new_plan
        if done:
            print(f"\n[done] env terminated at cycle {cyc}, cumR={cumr:+.2f}")
            break

    env.close()
    print(f"\n# FINAL cumR={cumr:+.2f} over {cyc+1} replan cycle(s). "
          f"RL planner format ran end-to-end on real {args.env} frames.")


if __name__ == "__main__":
    main()
