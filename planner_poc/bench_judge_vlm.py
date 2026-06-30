"""bench_judge_vlm.py -- the frozen-Qwen VLM-as-JUDGE over a bench_rollout manifest.

For each re-plan cycle it shows the judge the BEFORE frame, the AFTER frame (A chunks later), the
agent's PLAN for that interval, and a summary of the inputs actually executed, plus the game objective
and control scheme. The judge returns a STRUCTURED verdict (JSON):

  plan_sensible : 0|1|2   -- is the plan a sensible goal for this state? (0 nonsense, 2 clearly right)
  progress      : -1|0|1  -- did the state move TOWARD the objective between before/after?
  plan_followed : 0|1|2   -- did the executed actions / after-frame match the plan?
  failure       : planner|actor|both|none  -- attribute any failure (bad plan vs bad execution)
  reason        : short string

This is the candidate RL reward signal. The SAME schema is judged by the GPT-5.5 / Gemini strong models
(strong-model subagents) so we can measure agreement (does the cheap VLM-judge track strong models?).

Run (one GPU):
  CUDA_VISIBLE_DEVICES=3 env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc \
    QWEN=Qwen/Qwen3.5-2B .venv/bin/python planner_poc/bench_judge_vlm.py --manifest docs/bench/thextech/manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
from PIL import Image

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.planner import PlanEncoder, PlannerConfig

QWEN = os.environ.get("QWEN", "Qwen/Qwen3.5-2B")

JUDGE_SYS = (
    "You are an expert game analyst evaluating an AI agent that plays 2D video games. For one short "
    "interval you are given: the game OBJECTIVE, the CONTROL scheme, the agent's PLAN for the interval, "
    "a summary of the inputs it actually executed, the BEFORE frame (start of the interval) and the "
    "AFTER frame (end of the interval). Judge strictly from the evidence."
)

# The judge schema + instruction (shared verbatim with the strong-model subagents for apples-to-apples).
JUDGE_INSTR = (
    "Answer ONLY with a single JSON object, no prose, with EXACTLY these keys:\n"
    '  "plan_sensible": 0|1|2   (is the PLAN a sensible goal for the BEFORE state + objective? '
    "0=nonsense/ungrounded, 1=plausible, 2=clearly correct)\n"
    '  "progress": -1|0|1        (did the state move TOWARD the objective from BEFORE to AFTER? '
    "-1=worse, 0=none, 1=progress)\n"
    '  "plan_followed": 0|1|2   (did the executed inputs + AFTER frame match the PLAN? '
    "0=ignored, 1=partial, 2=followed)\n"
    '  "failure": "planner"|"actor"|"both"|"none"   (attribute any failure: "planner"=plan was wrong, '
    '"actor"=plan was right but execution failed, "both", or "none" if it succeeded)\n'
    '  "reason": "<one short sentence>"\n'
    "Output the JSON now:"
)


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {"_parse_error": True, "raw": text[:300]}
    try:
        return json.loads(m.group(0))
    except Exception:
        # lenient: pull keys individually
        out = {"_parse_error": True, "raw": text[:300]}
        for k, pat in [("plan_sensible", r'"plan_sensible"\s*:\s*(-?\d)'),
                       ("progress", r'"progress"\s*:\s*(-?\d)'),
                       ("plan_followed", r'"plan_followed"\s*:\s*(-?\d)'),
                       ("failure", r'"failure"\s*:\s*"(\w+)"')]:
            mm = re.search(pat, text)
            if mm:
                out[k] = mm.group(1)
        return out


def judge_cycle(pl, base_dir, cyc, device, max_new_tokens=160):
    before = Image.open(os.path.join(base_dir, cyc["before"])).convert("RGB")
    after = Image.open(os.path.join(base_dir, cyc["after"])).convert("RGB")
    ctx = (f"OBJECTIVE: {cyc.get('objective','')}\n"
           f"CONTROLS: {cyc.get('controls','')}\n"
           f"PLAN: {cyc['plan']}\n"
           f"EXECUTED INPUTS: {cyc['action_summary']} (stick {cyc.get('stick')}, "
           f"buttons {cyc.get('buttons_held')})\n"
           f"GROUND-TRUTH STATE before={cyc.get('state_before')} after={cyc.get('state_after')}\n")
    content = [{"type": "text", "text": ctx + "\nBEFORE frame:"},
               {"type": "image"},
               {"type": "text", "text": "AFTER frame:"},
               {"type": "image"},
               {"type": "text", "text": JUDGE_INSTR}]
    msgs = [{"role": "system", "content": JUDGE_SYS},
            {"role": "user", "content": content}]
    text = pl.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = pl.processor(text=[text], images=[before, after], return_tensors="pt").to(device)
    out = pl.backbone.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    gen = pl.processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    verdict = _extract_json(gen)
    verdict["_judge"] = "qwen:" + os.path.basename(QWEN)
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--qwen", default=QWEN)
    ap.add_argument("--out", default=None, help="default: <manifest dir>/judge_vlm.json")
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(args.manifest))
    man = json.load(open(args.manifest))
    for c in man["cycles"]:
        c.setdefault("objective", man.get("objective", ""))
        c.setdefault("controls", man.get("controls", ""))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl = PlanEncoder(PlannerConfig(backbone_name_or_path=args.qwen)); pl.load()
    if pl.processor is None:
        raise RuntimeError("VL processor missing (install torchvision).")
    if next(pl.backbone.parameters()).device != torch.device(device):
        pl.backbone.to(device)

    print(f"VLM-judge {args.qwen} on {man['env']} ({len(man['cycles'])} cycles)", flush=True)
    judgments = []
    for cyc in man["cycles"]:
        v = judge_cycle(pl, base_dir, cyc, device)
        judgments.append({"i": cyc["i"], "plan": cyc["plan"], **v})
        print(f"  c{cyc['i']:02d} sens={v.get('plan_sensible')} prog={v.get('progress')} "
              f"follow={v.get('plan_followed')} fail={v.get('failure')} :: {str(v.get('reason'))[:70]}",
              flush=True)

    out = args.out or os.path.join(base_dir, "judge_vlm.json")
    with open(out, "w") as f:
        json.dump({"env": man["env"], "genre": man["genre"], "judge": "qwen:" + os.path.basename(args.qwen),
                   "judgments": judgments}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
