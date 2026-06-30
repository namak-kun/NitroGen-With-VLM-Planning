"""bench_aggregate.py -- combine the three judges (Qwen VLM, GPT-5.5, Gemini-3.5-Flash) over the bench
rollouts and report (1) inter-judge AGREEMENT (does the cheap Qwen VLM-judge track the strong models?
-> feasibility of VLM-as-RL-reward) and (2) the PLANNER/ACTOR/BOTH failure taxonomy by genre.

Reads, per env dir under --root: manifest.json, judge_vlm.json, judge_gpt55.json, judge_gemini.json
(any subset; missing judges are skipped). Each judge_*.json has {judgments:[{i, plan_sensible, progress,
plan_followed, failure, reason}]}. Writes a summary json + prints a human report.

Run: .venv/bin/python planner_poc/bench_aggregate.py --root docs/bench --out docs/bench/AGGREGATE.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

ORD = ["plan_sensible", "progress", "plan_followed"]
CAT = ["failure"]
JUDGES = {"vlm": "judge_vlm.json", "gpt55": "judge_gpt55.json", "gemini": "judge_gemini.json"}


def _num(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _load_judge(path):
    if not os.path.exists(path):
        return None
    try:
        d = json.load(open(path))
    except Exception:
        return None
    return {j["i"]: j for j in d.get("judgments", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="docs/bench")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env_dirs = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                      if os.path.exists(os.path.join(d, "manifest.json")))
    summary = {"envs": {}, "agreement": {}, "failure_taxonomy": {}}

    # agreement accumulators: pairwise vlm-vs-strong on shared cycles
    pair_ord = defaultdict(lambda: defaultdict(lambda: {"exact": 0, "within1": 0, "n": 0}))  # strong->field
    pair_fail = defaultdict(lambda: {"match": 0, "n": 0})
    fail_by_genre = defaultdict(Counter)
    fail_by_judge = defaultdict(Counter)
    prog_by_genre_judge = defaultdict(lambda: defaultdict(list))

    for ed in env_dirs:
        man = json.load(open(os.path.join(ed, "manifest.json")))
        env, genre = man["env"], man.get("genre", "?")
        judges = {name: _load_judge(os.path.join(ed, fn)) for name, fn in JUDGES.items()}
        present = [n for n, v in judges.items() if v]
        summary["envs"][env] = {"genre": genre, "cycles": len(man["cycles"]), "judges": present}

        all_i = sorted({c["i"] for c in man["cycles"]})
        for i in all_i:
            # failure taxonomy: prefer strong-model consensus, else vlm
            for jn in present:
                jr = judges[jn].get(i)
                if not jr:
                    continue
                f = str(jr.get("failure", "")).lower()
                if f in ("planner", "actor", "both", "none"):
                    fail_by_judge[jn][f] += 1
                    if jn == "gpt55" or (jn == "gemini" and "gpt55" not in present):
                        fail_by_genre[genre][f] += 1
                p = _num(jr.get("progress"))
                if p is not None:
                    prog_by_genre_judge[genre][jn].append(p)
            # agreement vlm vs each strong
            v = judges["vlm"].get(i) if judges["vlm"] else None
            if not v:
                continue
            for strong in ("gpt55", "gemini"):
                s = judges[strong].get(i) if judges[strong] else None
                if not s:
                    continue
                for f in ORD:
                    a, b = _num(v.get(f)), _num(s.get(f))
                    if a is None or b is None:
                        continue
                    cell = pair_ord[strong][f]
                    cell["n"] += 1
                    cell["exact"] += int(a == b)
                    cell["within1"] += int(abs(a - b) <= 1)
                fa, fb = str(v.get("failure", "")).lower(), str(s.get("failure", "")).lower()
                if fa and fb:
                    pair_fail[strong]["n"] += 1
                    pair_fail[strong]["match"] += int(fa == fb)

    # finalize agreement
    for strong, fields in pair_ord.items():
        summary["agreement"][f"vlm_vs_{strong}"] = {
            f: {"exact": round(c["exact"] / c["n"], 3) if c["n"] else None,
                "within1": round(c["within1"] / c["n"], 3) if c["n"] else None, "n": c["n"]}
            for f, c in fields.items()}
        pf = pair_fail[strong]
        summary["agreement"][f"vlm_vs_{strong}"]["failure_match"] = (
            {"rate": round(pf["match"] / pf["n"], 3) if pf["n"] else None, "n": pf["n"]})

    # ---- GROUND-TRUTH anchor: each judge's progress vs gt.progress_gt (objective) ----
    gt_acc = defaultdict(lambda: {"exact": 0, "within1": 0, "fp": 0, "n": 0})  # judge -> stats
    for ed in env_dirs:
        man = json.load(open(os.path.join(ed, "manifest.json")))
        gtmap = {c["i"]: c.get("gt", {}) for c in man["cycles"]}
        judges = {name: _load_judge(os.path.join(ed, fn)) for name, fn in JUDGES.items()}
        for name, jm in judges.items():
            if not jm:
                continue
            for i, g in gtmap.items():
                pg = g.get("progress_gt")
                jp = _num(jm.get(i, {}).get("progress")) if i in jm else None
                if pg is None or jp is None:
                    continue
                s = gt_acc[name]
                s["n"] += 1
                s["exact"] += int(jp == pg)
                s["within1"] += int(abs(jp - pg) <= 1)
                s["fp"] += int(jp == 1 and pg <= 0)   # false-positive progress (rewards non-progress)
    summary["ground_truth_anchor"] = {
        name: {"progress_exact": round(s["exact"] / s["n"], 3),
               "progress_within1": round(s["within1"] / s["n"], 3),
               "false_positive_progress": round(s["fp"] / s["n"], 3), "n": s["n"]}
        for name, s in gt_acc.items() if s["n"]}

    summary["failure_taxonomy"] = {
        "by_genre": {g: dict(c) for g, c in fail_by_genre.items()},
        "by_judge": {j: dict(c) for j, c in fail_by_judge.items()},
        "mean_progress_by_genre_judge": {
            g: {j: round(sum(v) / len(v), 3) for j, v in d.items() if v}
            for g, d in prog_by_genre_judge.items()},
    }

    out = args.out or os.path.join(args.root, "AGGREGATE.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- human report ----
    print("\n===== BENCH AGGREGATE =====")
    print(f"envs: {len(summary['envs'])}")
    for env, e in summary["envs"].items():
        print(f"  {env:20s} genre={e['genre']:10s} cycles={e['cycles']} judges={e['judges']}")
    print("\n-- VLM-judge vs strong-model agreement (does cheap judge track strong?) --")
    for k, v in summary["agreement"].items():
        print(f"  {k}:")
        for f in ORD:
            if f in v:
                print(f"      {f:14s} exact={v[f]['exact']} within1={v[f]['within1']} (n={v[f]['n']})")
        if "failure_match" in v:
            print(f"      failure_match  rate={v['failure_match']['rate']} (n={v['failure_match']['n']})")
    if summary.get("ground_truth_anchor"):
        print("\n-- GROUND-TRUTH anchor: judge progress vs actual Δstate (objective) --")
        for name, s in summary["ground_truth_anchor"].items():
            print(f"  {name:8s} exact={s['progress_exact']} within1={s['progress_within1']} "
                  f"false_pos_progress={s['false_positive_progress']} (n={s['n']})")
    print("\n-- failure taxonomy by genre (strong-model attribution) --")
    for g, c in summary["failure_taxonomy"]["by_genre"].items():
        print(f"  {g:10s} {dict(c)}")
    print("\n-- mean progress by genre x judge --")
    for g, d in summary["failure_taxonomy"]["mean_progress_by_genre_judge"].items():
        print(f"  {g:10s} {d}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
