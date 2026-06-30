"""Compare the WINNING grounded prompt with THINKING ON vs OFF, on the bench scenarios. Non-think =
generate_plan (direct). Think = budget-forced reasoning (benchmark_think.generate) then the grounded
plan. Shows the reasoning trace + both plans so we can see whether thinking changes/justifies the move.

Run: env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=. QWEN=Qwen/Qwen3.5-2B \
        .venv/bin/python planner_poc/compare_think_grounded.py
"""
import sys

import torch

import os; REPO = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
import benchmark_prompts as BP
import benchmark_think as BT
from nitrogen.planner import PlanEncoder, PlannerConfig

# the winning grounded instruction from the sweep (A_broad_concrete)
GROUND = ("Give just your one-sentence recommendation for what the agent should broadly do next (no "
          "frame-by-frame description). Use concrete control or game-specific terms so the direction is "
          "clear -- e.g. 'move RIGHT', 'go UP', 'JUMP', 'ATTACK' -- not vague words like 'forward'.")


def main():
    pl = PlanEncoder(PlannerConfig(backbone_name_or_path=BP.QWEN)); pl.load()
    dev = "cuda"
    if next(pl.backbone.parameters()).device != torch.device(dev):
        pl.backbone.to(dev)
    for name, genre, globs, history, typ in BP.CASES:
        frames = BP.imgs(globs)
        sys_p = BP.BASE_SYS + " " + BP.CONTROLS[genre]
        instr = f"{history}\n\n{GROUND}"
        print(f"\n===== {name} ({genre}, type={typ}) =====")
        # NON-THINKING (what the rollouts currently use)
        nt = pl.generate_plan(frames, dev, instruction=instr, system=sys_p, max_new_tokens=40)
        print(f"  NO-THINK : {nt!r}")
        # THINKING (budget-forced): reuse benchmark_think.generate with our grounded sys/instr
        BT.SYS, BT.INSTR = sys_p, instr
        th = BT.generate(pl, frames, dev, enable_thinking=True, max_new_tokens=256)
        print(f"  THINK    : plan={th['plan']!r}  ({th['n_think_tok']} think tok, "
              f"{'forced-close' if th['forced'] else 'self-closed'}, {th['dt']:.1f}s)")
        print(f"     reasoning[:240]: {th['think'][:240]!r}", flush=True)


if __name__ == "__main__":
    main()
