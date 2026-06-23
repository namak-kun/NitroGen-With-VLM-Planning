"""EXP-050 knob-1: precompute FRAME-CONDITIONED plan-encoder hidden states.

For each Stage-2 chunk, run the frozen Qwen3.5-VL planner over [before_frame, after_frame] +
the (frame-grounded) plan text and save the last-layer hidden states. These replace the
text-only PlanHiddenCache: the resampler then cross-attends over BOTH the frames and the plan
text, so the K plan tokens become frame-specific (the fix for the EXP-042/049 text-only
collinearity / data wall). Backbone + (uuid, window) are deterministic -> compute once.

Frames come from frames_cc (before=window, after=window+H*stride), the SAME pair the plans were
generated from (gen_stage2_lookup_mm.py records `before_idx`/`after_idx`). Plan text from the mm
lookup. ZERO new downloads.

Out: torch.save({uuid: {"h": (L,d) fp16, "mask": (L,) bool}}) -> /tmp/stage2_mm_hidden.pt
Run: PYTHONPATH=. .venv/bin/python planner_poc/cache_mm_hidden.py
"""
import json
import os
import sys

import torch
from PIL import Image

REPO = "/home/t-nagupta/NitroGen"
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
from nitrogen.planner import PlanEncoder, PlannerConfig

DEVICE = "cuda"
LOOKUP = os.environ.get("LOOKUP", "/tmp/stage2_plan_lookup_mm.json")
OUT = os.environ.get("OUT", "/tmp/stage2_mm_hidden.pt")
FRAMES_CC = os.environ.get("FRAMES_CC", "/tmp/frames_cc")
QWEN = os.environ.get("QWEN", f"{REPO}/ckpts/qwen35-0.8b")  # MUST match the training planner backbone
MAX_SIDE = int(os.environ.get("MAX_SIDE", "512"))
TEXT_ONLY = os.environ.get("TEXT_ONLY", "0") == "1"   # EXP-052: keep only text-token hiddens
# TEACHER P+ (EXP-044): augment plan text with the action summary so the encoder hidden carries
# the DIRECTION the base plan text lacks (probe: base hiddens left/right cos ~0.999). Used to build
# the teacher mm-hidden; the teacher's tokens are then distilled into the base-text student.
PLAN_AUG = os.environ.get("PLAN_AUG", "0") == "1"
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))


def _resize(im):
    w, h = im.size
    s = MAX_SIDE / max(w, h)
    return im.resize((max(1, int(w * s)), max(1, int(h * s)))) if s < 1 else im


def main():
    lookup = json.load(open(LOOKUP))
    items = list(lookup.items())
    if NUM_SHARDS > 1:
        items = items[SHARD_INDEX::NUM_SHARDS]
    pl = PlanEncoder(PlannerConfig(backbone_name_or_path=QWEN)); pl.load()
    out = {}
    done = skip = 0
    for uuid, e in items:
        if not e.get("is_gameplay", True):
            skip += 1
            continue
        # N-frame author window (gen_stage2_lookup_mm records frame_offsets); fall back to the
        # before/after pair for legacy lookups. Must MATCH the frames the plan was authored over.
        offs = e.get("frame_offsets") or [e.get("before_idx"), e.get("after_idx")]
        paths = [os.path.join(FRAMES_CC, f"{uuid}__{o}.png") for o in offs]
        if not (all(os.path.exists(p) for p in paths) and e.get("plan")):
            skip += 1
            continue
        frames = [_resize(Image.open(p).convert("RGB")) for p in paths]
        plan_text = e["plan"]
        if PLAN_AUG:
            summ = e.get("action_summary")
            if summ:
                plan_text = plan_text + " To do this I take the following actions: " + summ + "."
        h, kpm = pl.encode_multimodal(frames, plan_text, DEVICE, text_only=TEXT_ONLY)
        out[uuid] = {"h": h[0].half().cpu(), "mask": kpm[0].cpu()}
        done += 1
        if done % 20 == 0:
            print(f"  {done} cached (L={out[uuid]['h'].shape[0]}) | skip={skip}", flush=True)
    out_path = OUT if NUM_SHARDS == 1 else f"{OUT}.shard{SHARD_INDEX}"
    torch.save(out, out_path)
    Ls = [v["h"].shape[0] for v in out.values()]
    print(f"\nDONE: {len(out)} mm-hidden sets (d={next(iter(out.values()))['h'].shape[-1]}, "
          f"L min/mean/max={min(Ls)}/{sum(Ls)//len(Ls)}/{max(Ls)}) -> {out_path}; skipped {skip}")


if __name__ == "__main__":
    main()
