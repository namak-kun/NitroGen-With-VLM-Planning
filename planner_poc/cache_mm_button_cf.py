"""Broaden Job-1 steering BEYOND directions to BUTTONS/ACTIONS (jump/accelerate/attack/brake).

Analogous to cache_mm_cf.py (directional override) but the counterfactual target is a BUTTON press,
not a stick direction. Button presses are frame-agnostic motor primitives (the single-chunk override
design boundary), so we pair each gameplay frame_i with a chunk j whose DOMINANT button is a target
button B, use a TERSE synthetic plan for B ("jump", "accelerate", ...), and store the override action_j
(which presses B). This teaches the plan token to OVERRIDE the frame prior and command the button.

Terse single-concept button plans are clean (cf. the syntext finding that short "move left" separates
at 1.0 while long tactical plans dilute the direction word); buttons are NOT scene-symmetric the way
left/right is, so button override is expected to be at least as learnable as direction override.

Out: torch.save({uuid_i: {h:(L,d)fp16, mask:(L,), buttons,j_left,j_right (action_j),
                          cf_button:int, cf_name:str, cf_uuid, plan:str}}) -> /tmp/stage2_mm_button_cf_2b.pt
Run: env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=. QWEN=Qwen/Qwen3.5-2B TEXT_ONLY=1 \
        LOOKUP=/tmp/stage2_plan_lookup_mm_clean.json INDEX=/tmp/stage2_index_a4.pt \
        .venv/bin/python planner_poc/cache_mm_button_cf.py
"""
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
from PIL import Image

import os; REPO = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, REPO); sys.path.insert(0, REPO + "/planner_poc")
from nitrogen.planner import PlanEncoder, PlannerConfig

DEVICE = "cuda"
LOOKUP = json.load(open(os.environ.get("LOOKUP", "/tmp/stage2_plan_lookup_mm_clean.json")))
INDEX = torch.load(os.environ.get("INDEX", "/tmp/stage2_index_a4.pt"),
                   map_location="cpu", weights_only=False)
OUT = os.environ.get("OUT", "/tmp/stage2_mm_button_cf_2b.pt")
FRAMES_CC = os.environ.get("FRAMES_CC", "/tmp/frames_cc")
QWEN = os.environ.get("QWEN", "Qwen/Qwen3.5-2B")
MAX_SIDE = int(os.environ.get("MAX_SIDE", "512"))
SEED = int(os.environ.get("SEED", "0"))
TEXT_ONLY = os.environ.get("TEXT_ONLY", "1") == "1"
DOM_THRESH = float(os.environ.get("DOM_THRESH", "0.5"))   # button pressed in >= this frac of steps
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "1"))
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))

# controller-button -> terse action plan (synthetic, single concept). Indices = BUTTON_ACTION_TOKENS.
BUTTON_PLANS = {
    16: ["accelerate", "speed up", "go fast"],          # RIGHT_TRIGGER (racing throttle; base-DiT floors it)
    18: ["jump", "jump now", "leap up"],                # SOUTH (A) = jump/confirm
    20: ["attack", "shoot", "fire"],                    # WEST (X) = attack/shoot
    9:  ["brake", "slow down", "stop"],                # LEFT_TRIGGER = brake
    5:  ["dash", "dodge", "use action"],               # EAST (B) = secondary action
}
PLAN_PHRASINGS = os.environ.get("PHRASINGS", "0") == "1"   # vary phrasing per pair (else use [0])


def _resize(im):
    w, h = im.size
    s = MAX_SIDE / max(w, h)
    return im.resize((max(1, int(w * s)), max(1, int(h * s)))) if s < 1 else im


def dominant_button(buttons):
    """Return the controller-button index that is pressed in >= DOM_THRESH of the chunk steps and is
    the most-pressed of the eligible target buttons, else None."""
    frac = np.asarray(buttons, dtype=np.float32).mean(0)   # (21,)
    best, best_f = None, DOM_THRESH
    for b in BUTTON_PLANS:
        if frac[b] >= best_f:
            best, best_f = b, frac[b]
    return best


def main():
    rng = random.Random(SEED)
    # group INDEX chunks by their dominant target-button (the override-action pool).
    by_button = {}
    for u, e in INDEX.items():
        b = dominant_button(e["buttons"])
        if b is not None:
            by_button.setdefault(b, []).append(u)
    print(f"button pools (dominant >= {DOM_THRESH}): "
          f"{ {f'{b}:{BUTTON_PLANS[b][0]}': len(v) for b, v in by_button.items()} }")

    # gameplay uuids that have a frame pair + appear in INDEX (so their own dir/buttons are known).
    uuids = [u for u, e in LOOKUP.items() if e.get("is_gameplay") and u in INDEX]
    targets = [b for b in BUTTON_PLANS if by_button.get(b)]

    pl = PlanEncoder(PlannerConfig(backbone_name_or_path=QWEN)); pl.load()
    out, skip = {}, 0
    work = uuids[SHARD_INDEX::NUM_SHARDS] if NUM_SHARDS > 1 else uuids
    for i, u in enumerate(work):
        e = LOOKUP[u]
        offs = e.get("frame_offsets") or [e.get("before_idx"), e.get("after_idx")]
        paths = [os.path.join(FRAMES_CC, f"{u}__{o}.png") for o in offs]
        if not all(os.path.exists(p) for p in paths):
            skip += 1; continue
        # avoid pairing a frame with a chunk whose dominant button == its own (keep it counterfactual).
        own_b = dominant_button(INDEX[u]["buttons"])
        cands = [b for b in targets if b != own_b] or targets
        cf_b = rng.choice(cands)
        uj = rng.choice(by_button[cf_b])
        plan = rng.choice(BUTTON_PLANS[cf_b]) if PLAN_PHRASINGS else BUTTON_PLANS[cf_b][0]
        frames = [_resize(Image.open(p).convert("RGB")) for p in paths]
        h, kpm = pl.encode_multimodal(frames, plan, DEVICE, text_only=TEXT_ONLY)
        out[u] = {
            "h": h[0].half().cpu(), "mask": kpm[0].cpu(),
            "buttons": np.asarray(INDEX[uj]["buttons"], dtype=np.float32),
            "j_left": np.asarray(INDEX[uj]["j_left"], dtype=np.float32),
            "j_right": np.asarray(INDEX[uj]["j_right"], dtype=np.float32),
            "cf_button": int(cf_b), "cf_name": BUTTON_PLANS[cf_b][0],
            "cf_uuid": uj, "plan": plan,
        }
        if (i + 1) % 40 == 0:
            print(f"  [shard {SHARD_INDEX}/{NUM_SHARDS}] {i+1}/{len(work)} cached", flush=True)
    out_path = OUT if NUM_SHARDS == 1 else f"{OUT}.shard{SHARD_INDEX}"
    torch.save(out, out_path)
    print(f"\nDONE: {len(out)} button-cf pairs -> {out_path}; skipped {skip}")
    print("  cf_button dist:", dict(Counter(v["cf_name"] for v in out.values())))


if __name__ == "__main__":
    main()
