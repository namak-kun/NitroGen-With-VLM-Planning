"""build_plan_jobs.py -- prepare per-game job files for the plan-authoring subagents.

Reads docs/demo_plans/<game>/segments.jsonl, picks a strided subset (spans the whole demo set), and writes
docs/demo_plans/<game>/job_input.json = a compact list the subagent consumes:
  [{seg_id, game_context, frames:{f0,mid,end ABS paths}, action_trace:[ "s1_1: <summary>", ... 6 ]}]
The subagent views the 3 frames + reads the action_trace (privileged: it knows the buttons pressed) and
writes docs/demo_plans/<game>/plans.jsonl = {seg_id, plan}.

Run: $RUN .venv/bin/python planner_poc/build_plan_jobs.py --game smw --cap 24
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--root", default="docs/demo_plans")
    ap.add_argument("--cap", type=int, default=24, help="max segments (strided across all demos)")
    ap.add_argument("--shards", type=int, default=1, help="split into N shard job files")
    args = ap.parse_args()
    gdir = os.path.join(_R, args.root, args.game)
    rows = [json.loads(l) for l in open(os.path.join(gdir, "segments.jsonl"))]
    if len(rows) > args.cap:
        step = len(rows) / args.cap
        rows = [rows[int(i * step)] for i in range(args.cap)]
    jobs = []
    for r in rows:
        fmap = {f["tag"]: os.path.join(gdir, f["path"]) for f in r["frames"]}
        jobs.append({
            "seg_id": r["seg_id"],
            "game_context": r["game_context"],
            "frames": {"start_f0": fmap.get("f0"), "mid_f1_3": fmap.get("f1_3"), "end_f2_3": fmap.get("f2_3")},
            "action_trace": [f'{s["label"]}: {s["action_summary"]}' for s in r["subchunks"]],
        })
    if args.shards <= 1:
        out = os.path.join(gdir, "job_input.json")
        json.dump(jobs, open(out, "w"), indent=2)
        print(f"[plan-jobs] {args.game}: {len(jobs)} segments -> {out}")
    else:
        for i in range(args.shards):
            shard = jobs[i::args.shards]
            out = os.path.join(gdir, f"job_input_{i}.json")
            json.dump(shard, open(out, "w"), indent=2)
            print(f"[plan-jobs] {args.game} shard {i}: {len(shard)} segments -> {out}")
    print(f"  output expected at: {os.path.join(gdir, 'plans.jsonl')} (merge shard plans_*.jsonl)")


if __name__ == "__main__":
    main()
