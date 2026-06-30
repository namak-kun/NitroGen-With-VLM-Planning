"""EXP-050 data hygiene: tag each Stage-2 chunk as active gameplay vs non-gameplay.

Twitch/YouTube source frames are contaminated: stream intro cards, title/logo overlays, menus,
character-select, loading screens, facecams. Plans generated from those are garbage ("Approach
the Elden Ring logo", "Select Juego Libre") and would poison the frame-grounded planner. The VLM
itself can tell gameplay from non-gameplay, so we use it as a binary gate per uuid (reading the
SAME 'before' frame the plan/teacher/mm-hidden caches use) and write {uuid: is_gameplay} so
training can filter on it. Keeps everything on disk; filters at train/cache time.

Out: writes the boolean back INTO the mm lookup ({uuid: {... , is_gameplay}}) and a summary.
Run: PYTHONPATH=. .venv/bin/python planner_poc/tag_gameplay.py
"""
import json
import os
import re
import sys

import torch
from PIL import Image

import os; sys.path.insert(0, os.path.join(os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"), "planner_poc"))
import os; sys.path.insert(0, os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning"))
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL = os.environ.get("VLM", "Qwen/Qwen3.5-9B")
LOOKUP = os.environ.get("LOOKUP", "/tmp/stage2_plan_lookup_mm.json")
FRAMES_CC = os.environ.get("FRAMES_CC", "/tmp/frames_cc")
MAX_SIDE = int(os.environ.get("MAX_SIDE", "512"))
FORCE = os.environ.get("FORCE", "0") == "1"   # re-tag entries that already carry is_gameplay

GATE = (
    "Look at this single frame from a game stream. Is a live video game being ACTIVELY PLAYED "
    "and VISIBLE anywhere in the frame -- a player-controlled character or vehicle in the game "
    "world? Answer YES even if the game is in a sub-window or bordered by stream overlays (chat "
    "boxes, webcam/facecam, banners, channel branding) -- as long as actual gameplay is visible. "
    "Answer NO ONLY if there is NO active gameplay visible at all: i.e. the frame is purely a "
    "title/logo/branding screen, main menu, pause/options menu, character or level select, "
    "map/inventory screen, loading screen, or a full-screen non-interactive cutscene. "
    "Answer with exactly one word: YES or NO."
)


def _resize(im):
    w, h = im.size
    s = MAX_SIDE / max(w, h)
    return im.resize((max(1, int(w * s)), max(1, int(h * s)))) if s < 1 else im


def main():
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    lookup = json.load(open(LOOKUP))
    kept = dropped = miss = 0
    for i, (uuid, e) in enumerate(lookup.items()):
        if not FORCE and "is_gameplay" in e:   # idempotent: already tagged (cheap 2nd pass)
            kept += int(e["is_gameplay"]); dropped += int(not e["is_gameplay"])
            continue
        bi = e.get("before_idx")
        bp = os.path.join(FRAMES_CC, f"{uuid}__{bi}.png")
        if not os.path.exists(bp):
            e["is_gameplay"] = False; miss += 1; continue
        img = _resize(Image.open(bp).convert("RGB"))
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": GATE}]}]
        try:
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt",
                                           enable_thinking=False).to(model.device)
        except TypeError:
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=8, do_sample=False)
        ans = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        ans = re.sub(r"<think>.*?</think>", "", ans, flags=re.S)
        is_gp = bool(re.search(r"\byes\b", ans, re.I)) and not bool(re.search(r"\bno\b", ans, re.I))
        e["is_gameplay"] = is_gp
        kept += int(is_gp); dropped += int(not is_gp)
        if (i + 1) % 20 == 0:
            json.dump(lookup, open(LOOKUP, "w"), indent=2)
            print(f"  {i+1}/{len(lookup)} | kept={kept} dropped={dropped}", flush=True)
    json.dump(lookup, open(LOOKUP, "w"), indent=2)
    print(f"\nDONE: gameplay={kept}, non-gameplay={dropped}, missing-frame={miss} "
          f"(kept {100*kept/max(1,len(lookup)):.0f}%) -> {LOOKUP}")
    # also dump the dropped list for inspection
    drop = {u: e["plan"] for u, e in lookup.items() if not e.get("is_gameplay")}
    json.dump(drop, open("/tmp/stage2_dropped_nongameplay.json", "w"), indent=2)
    print(f"dropped plans -> /tmp/stage2_dropped_nongameplay.json ({len(drop)})")


if __name__ == "__main__":
    main()
