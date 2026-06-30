"""How FACTUAL are the factual plans? Does the frozen DiT already explain the real action o
from the frame alone (=> factual plan is a no-op shortcut), or is there headroom the plan fills?

NitroGen is GENERATIVE (flow matching): p(action|frame) is multimodal, so MSE(o, o_hat) is never
0 even for a perfect model (o is one sample, o_hat another). The right questions:
  (1) HEADROOM: is the frame branchy? spread_null = MSE between two independent NULL samples.
  (2) BASE ADHERENCE: err_null = MSE(o, o_hat_null) -- does base DiT land near the real o?
  (3) DOES THE PLAN HELP: err_plan = MSE(o, o_hat_plan) with the FACTUAL plan. If err_plan <<
      err_null, the plan concentrates the prediction toward the real action (factual learning is
      real). If err_plan ~= err_null, the plan is ignored / adds nothing (the shortcut).

Stick X (dim 0, [0,1], 0.5=neutral) is the direction axis we care about; also report full 25-dim.

Run: PYTHONPATH=. .venv/bin/python planner_poc/diag_factual_adherence.py \
        runs/stage2_2b_a1b48/plan_stage1_1000.pt 40
"""
import json
import os
import sys

import numpy as np
import torch

import os; REPO = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
from PIL import Image
from eval_policy import NitroGenPolicy
from nitrogen.training.actions import load_chunk_actions, assemble_chunk, chunk_dominant_dir

CKPT = sys.argv[1] if len(sys.argv) > 1 else "runs/stage2_2b_a1b48/plan_stage1_1000.pt"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
LOOKUP = os.environ.get("LOOKUP", "/tmp/stage2_plan_lookup_mm_a4.json")
FRAMES_CC = os.environ.get("FRAMES_CC", "/tmp/frames_cc")
ROOTS = ["/tmp/stage1_big", "/tmp/stage1_more"]
QWEN = os.environ.get("QWEN", "Qwen/Qwen3.5-2B")
H, STRIDE = 18, 2


def pack_real(rc):
    """assemble_chunk -> model action layout [j_left, j_right, buttons], joysticks [0,1]."""
    jl = (np.asarray(rc["j_left"], dtype=np.float32) + 1) / 2.0
    jr = (np.asarray(rc["j_right"], dtype=np.float32) + 1) / 2.0
    bt = np.asarray(rc["buttons"], dtype=np.float32)
    return np.concatenate([jl, jr, bt], axis=-1)  # (H, 25)


def uuid_dir(meta_roots, uuid):
    for r in meta_roots:
        for sub in (uuid.replace("_actions", ""), uuid):
            pass
    return None


def main():
    lookup = json.load(open(LOOKUP))
    # index chunk dirs by uuid
    import glob
    dir_by_uuid = {}
    for r in ROOTS:
        for md in glob.glob(f"{r}/**/metadata.json", recursive=True):
            try:
                u = json.load(open(md))["uuid"]
            except Exception:
                continue
            dir_by_uuid[u] = os.path.dirname(md)

    gp = [(u, e) for u, e in lookup.items()
          if e.get("is_gameplay") and e.get("chunk_starts") and u in dir_by_uuid
          and os.path.exists(os.path.join(FRAMES_CC, f"{u}__{e['frame_offsets'][0]}.png"))]
    import random
    random.seed(0); random.shuffle(gp)
    gp = gp[:N]
    print(f"diagnostic on {len(gp)} gameplay chunks | ckpt={CKPT}")

    pol = NitroGenPolicy(CKPT, qwen=QWEN, default_cfg=1.0)
    pol.mm_mode = True; pol.mm_text_only = True

    rows = []
    for i, (u, e) in enumerate(gp):
        d = dir_by_uuid[u]
        pq = os.path.join(d, "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(d, "actions_raw.parquet")
        try:
            a = load_chunk_actions(pq)
        except Exception:
            continue
        cs0 = e["chunk_starts"][0]
        rc = assemble_chunk(a["buttons"], a["j_left"], a["j_right"], cs0, H, STRIDE)
        if rc is None:
            continue
        o = pack_real(rc)                                  # (H,25)
        o_dir = chunk_dominant_dir(rc)
        f0 = e["frame_offsets"][0]
        frame = np.asarray(Image.open(os.path.join(FRAMES_CC, f"{u}__{f0}.png")).convert("RGB"))
        plan_frames = [np.asarray(Image.open(os.path.join(FRAMES_CC, f"{u}__{o2}.png")).convert("RGB"))
                       for o2 in e["frame_offsets"]
                       if os.path.exists(os.path.join(FRAMES_CC, f"{u}__{o2}.png"))]

        torch.manual_seed(1)
        on_a = pol._sample_chunk(frame, "", 1.0, plan_frames=plan_frames, null=True)
        torch.manual_seed(2)
        on_b = pol._sample_chunk(frame, "", 1.0, plan_frames=plan_frames, null=True)
        torch.manual_seed(1)
        op = pol._sample_chunk(frame, e["plan"], 1.0, plan_frames=plan_frames, null=False)

        def mse(x, y, sl=slice(None)):
            return float(np.mean((x[:, sl] - y[:, sl]) ** 2))

        rows.append({
            "dir": o_dir,
            "err_null": mse(o, on_a), "spread_null": mse(on_a, on_b), "err_plan": mse(o, op),
            "err_null_stick": mse(o, on_a, slice(0, 4)),
            "spread_null_stick": mse(on_a, on_b, slice(0, 4)),
            "err_plan_stick": mse(o, op, slice(0, 4)),
            "o_sx": float(o[:, 0].mean()), "on_sx": float(on_a[:, 0].mean()), "op_sx": float(op[:, 0].mean()),
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(gp)}", flush=True)

    import statistics as st
    def avg(k):
        return st.mean(r[k] for r in rows)
    # neutral-hedge check: |stick_x - 0.5| for real vs null vs plan (0 == neutral)
    def avg_absdev(key):
        return st.mean(abs(r[key] - 0.5) for r in rows)
    print(f"\n=== FACTUAL ADHERENCE ({len(rows)} chunks) ===")
    print(f"  FULL 25-dim:  err_null={avg('err_null'):.4f}  spread_null={avg('spread_null'):.4f}  err_plan={avg('err_plan'):.4f}")
    print(f"  STICK 0:4  :  err_null={avg('err_null_stick'):.4f}  spread_null={avg('spread_null_stick'):.4f}  err_plan={avg('err_plan_stick'):.4f}")
    print(f"\n  headroom (spread_null): higher => frame branchier => more room for the plan")
    print(f"  plan effect (err_null - err_plan): {avg('err_null')-avg('err_plan'):+.4f} full, "
          f"{avg('err_null_stick')-avg('err_plan_stick'):+.4f} stick "
          f"(positive => plan concentrates toward real o)")
    # stick-x sign agreement (did the prediction go the real direction?)
    def signmatch(pred_key):
        c = t = 0
        for r in rows:
            if abs(r["o_sx"] - 0.5) < 0.05:  # skip near-neutral truth
                continue
            t += 1
            c += int(np.sign(r[pred_key]-0.5) == np.sign(r["o_sx"]-0.5))
        return c, t
    cn, tn = signmatch("on_sx"); cp, tp = signmatch("op_sx")
    print(f"\n  stick-X direction match (non-neutral truth): null={cn}/{tn}, plan={cp}/{tp}")
    print(f"\n  NEUTRAL-HEDGE check |stick_x - 0.5| (0=neutral, 0.5=full deflection):")
    print(f"    real o = {avg_absdev('o_sx'):.3f}   null = {avg_absdev('on_sx'):.3f}   plan = {avg_absdev('op_sx'):.3f}")
    print(f"    => if null << real, the base model HEDGES toward neutral on under-determined frames")


if __name__ == "__main__":
    main()
