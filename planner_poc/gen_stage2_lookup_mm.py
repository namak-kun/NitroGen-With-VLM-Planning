"""EXP-050 knob-2: regenerate richer, frame-grounded Stage-2 plans.

Unlike gen_stage2_lookup.py (one frame), the planner here sees TWO boundary frames -- the
decision-point frame (before the chunk) and the frame ~1s later (after the chunk's actions) --
so it can perceive MOTION / progress / being stuck, which a single frame cannot show.

Thinking is DISABLED: the Stage-2 dataset plans carry no reasoning traces, so the planner that
produces them must not emit one either (otherwise train/inference plan distributions diverge).

We re-anchor Stage-2 to a cross-chunk window (default start=250) because frames_cc already has
the before(=s) + after(=s+H*stride) boundary pair for every uuid -- so this needs ZERO new
downloads. The action summary + the chosen window are recorded so the teacher-token + mm-hidden
caches use the EXACT same frames/chunk.

Out: {uuid: {plan, game, action_summary, window, before_idx, after_idx}}
Run: PYTHONPATH=. .venv/bin/python planner_poc/gen_stage2_lookup_mm.py
"""
import glob
import json
import os
import re
import sys

import torch
from PIL import Image

sys.path.insert(0, "/home/t-nagupta/NitroGen/planner_poc")
sys.path.insert(0, "/home/t-nagupta/NitroGen")
from transformers import AutoProcessor, AutoModelForImageTextToText
from nitrogen.training.actions import load_chunk_actions, assemble_chunk
from action_summary import summarize_chunk
from vtt_align import parse_vtt, window_text, best_vtt

MODEL = os.environ.get("VLM", "Qwen/Qwen3.5-9B")
OUT = os.environ.get("OUT", "/tmp/stage2_plan_lookup_mm_a4.json")
FRAMES_CC = os.environ.get("FRAMES_CC", "/tmp/frames_cc")
ROOTS = os.environ.get("ROOTS", "/tmp/stage1_big,/tmp/stage1_more").split(",")
H, STRIDE, ACTION_SHIFT = 18, 2, 3
A = int(os.environ.get("A", "4"))                       # chunks the plan spans (cross-chunk targets)
NUM_FRAMES = int(os.environ.get("NUM_FRAMES", str(A)))  # frames shown to the author over the window (1/chunk default -> motion-at-now + 1/chunk). DENSITY KNOB: raise for half-chunk (needs matching cc extraction + realign).
TRANSCRIPT_DIR = os.environ.get("TRANSCRIPT_DIR", "/tmp/transcripts")
USE_TRANSCRIPT = os.environ.get("USE_TRANSCRIPT", "1") == "1"
WINDOWS = [int(x) for x in os.environ.get("WINDOWS", "250,100,400").split(",")]  # preference order
MAX_SIDE = int(os.environ.get("MAX_SIDE", "512"))  # resize longest side (token budget)
SPAN_SEC = (NUM_FRAMES - 1) * H * STRIDE / 60.0     # wall-clock span the frames cover

TACTICAL = (
    "You are a gameplay strategist. You are shown a SHORT SEQUENCE of frames from a game in time "
    f"order (oldest first, newest last), spanning about {SPAN_SEC:.1f} seconds of play. Compare them "
    "to judge what is happening -- is the character making progress, stuck, looping, or under threat? "
    "You are also given a summary of the player's inputs over that span, and (when available) what the "
    "player SAID while playing (live commentary -- often off-topic chit-chat; use it ONLY if it clearly "
    "clarifies the immediate intent, otherwise ignore it). Output ONE short tactical PLAN (max 14 words) "
    "describing the SPECIFIC immediate GOAL for THIS scene, grounded in what you actually SEE (objects, "
    "layout, enemies, gaps, doors, items). Describe INTENT, never controls -- do NOT mention buttons, "
    "sticks, or raw directions like 'press B' or 'move left'. Be concrete to this scene, not generic. "
    "Output ONLY the plan, nothing else."
)

# DIRECTIONAL variant (DIRECTIONAL=1): ALLOWS scene-anchored spatial direction (left/right/up/down
# relative to what's visible), because left/right is otherwise absent from intent-only plans and is
# scene-symmetric (the model can't recover it). Still intent, not raw controls.
TACTICAL_DIR = (
    "You are a gameplay strategist. You are shown a SHORT SEQUENCE of frames from a game in time "
    f"order (oldest first, newest last), spanning about {SPAN_SEC:.1f} seconds of play. Compare them "
    "to judge what is happening -- progress, stuck, looping, threat? You are also given the player's "
    "input summary and (when available) their live commentary (often chit-chat; use only if it clearly "
    "clarifies intent). Output ONE short tactical PLAN (max 16 words) describing the SPECIFIC immediate "
    "GOAL for THIS scene, grounded in what you SEE. IMPORTANT: anchor the goal SPATIALLY -- say WHERE "
    "the target is and which way to head relative to the scene (e.g. 'move LEFT toward the door', "
    "'climb to the platform ABOVE', 'chase the enemy on the RIGHT', 'drop DOWN into the pit'). Use "
    "left/right/up/down when they are spatially meaningful for the goal. Do NOT mention buttons or "
    "controller inputs. Be concrete to this scene. Output ONLY the plan, nothing else."
)
if os.environ.get("DIRECTIONAL", "0") == "1":
    TACTICAL = TACTICAL_DIR


def _resize(im):
    w, h = im.size
    s = MAX_SIDE / max(w, h)
    return im.resize((max(1, int(w * s)), max(1, int(h * s)))) if s < 1 else im


def pick_window(acts, uuid):
    """Find a window start s such that all NUM_FRAMES author frames exist on disk AND there is
    room for all A action-chunks after their starts. Returns (s, frame_offsets, frame_paths,
    chunk_starts) or None."""
    T = acts["buttons"].shape[0]
    chunk_span = (A - 1) * H * STRIDE          # last chunk-start offset (frames)
    for s in WINDOWS:
        frame_offsets = [s + i * H * STRIDE for i in range(NUM_FRAMES)]
        frame_paths = [os.path.join(FRAMES_CC, f"{uuid}__{o}.png") for o in frame_offsets]
        if not all(os.path.exists(p) for p in frame_paths):
            continue
        chunk_starts = [s + a * H * STRIDE + ACTION_SHIFT for a in range(A)]
        last_cs = s + chunk_span + ACTION_SHIFT
        if last_cs + H * STRIDE >= T:
            continue
        # every chunk in the span must assemble (real ground-truth target per block)
        if any(assemble_chunk(acts["buttons"], acts["j_left"], acts["j_right"], cs, H, STRIDE) is None
               for cs in chunk_starts):
            continue
        return s, frame_offsets, frame_paths, chunk_starts
    return None


def transcript_window(m, s, end_frame):
    """Transcript text overlapping the window [s, end_frame] (frame indices into the slice).
    Returns '' if no captions or disabled. Capped to keep the prompt tight."""
    if not USE_TRANSCRIPT:
        return ""
    ov = m["original_video"]
    p = best_vtt(TRANSCRIPT_DIR, ov["video_id"])
    if not p:
        return ""
    try:
        cues = parse_vtt(p)
    except Exception:
        return ""
    t0 = float(ov["start_time"]) + s / 60.0
    t1 = float(ov["start_time"]) + end_frame / 60.0
    txt = window_text(cues, t0, t1).strip()
    return txt[:400]


def main():
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    lookup = json.load(open(OUT)) if os.path.exists(OUT) else {}
    mds = []
    for root in ROOTS:
        mds += sorted(glob.glob(f"{root}/**/metadata.json", recursive=True))
    print(f"scanning {len(mds)} chunks; {len(lookup)} already labeled; model={MODEL}", flush=True)
    done = 0
    for md in mds:
        m = json.load(open(md)); uuid = m["uuid"]; game = m.get("game", "?")
        if uuid in lookup:
            continue
        pq = os.path.join(os.path.dirname(md), "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(os.path.dirname(md), "actions_raw.parquet")
        try:
            a = load_chunk_actions(pq)
        except Exception:
            continue
        pw = pick_window(a, uuid)
        if pw is None:
            continue
        s, frame_offsets, frame_paths, chunk_starts = pw
        # per-chunk action summaries across the A-chunk span (so the author sees the whole trajectory)
        per_chunk = []
        for a_i, cs in enumerate(chunk_starts):
            rc_a = assemble_chunk(a["buttons"], a["j_left"], a["j_right"], cs, H, STRIDE)
            per_chunk.append(summarize_chunk(rc_a) if rc_a is not None else "n/a")
        summary = per_chunk[0]  # backward-compat field (chunk 0)
        span_summary = " | ".join(f"~{a_i * H * STRIDE / 60.0:.1f}s: {sm}"
                                  for a_i, sm in enumerate(per_chunk))
        imgs = [_resize(Image.open(p).convert("RGB")) for p in frame_paths]
        transcript = transcript_window(m, s, frame_offsets[-1])
        ctx = (f"Game: {game}\nPlayer's inputs over this window (context only): {span_summary}")
        if transcript:
            ctx += f"\nWhat the player said while playing (context only, may be off-topic): \"{transcript}\""
        msgs = [{"role": "user", "content":
                 [{"type": "image", "image": im} for im in imgs]
                 + [{"type": "text", "text": TACTICAL + "\n\n" + ctx}]}]
        try:
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt",
                                           enable_thinking=False).to(model.device)
        except TypeError:
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=48, do_sample=False)
        full = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        # no-think: strip any stray <think>..</think> defensively, take the plan line
        txt = re.sub(r"<think>.*?</think>", "", full, flags=re.S).strip()
        txt = txt.replace("<think>", "").replace("</think>", "").strip()
        plan = ([l.strip() for l in txt.splitlines() if l.strip()] or [txt])[-1].strip().strip('"')
        lookup[uuid] = {"plan": plan, "game": game,
                        "action_summary": summary, "span_summary": span_summary,
                        "window": s, "before_idx": s, "after_idx": frame_offsets[-1],
                        "frame_offsets": frame_offsets, "chunk_starts": chunk_starts,
                        "num_chunks": A, "used_transcript": bool(transcript)}
        done += 1
        if done % 5 == 0:
            json.dump(lookup, open(OUT, "w"), indent=2)
            ntr = sum(1 for v in lookup.values() if v.get("used_transcript"))
            print(f"  {done} (tr={ntr}) | [{game[:14]}] w={s} {plan}", flush=True)
    json.dump(lookup, open(OUT, "w"), indent=2)
    print(f"\nDONE: {len(lookup)} plans -> {OUT}")


if __name__ == "__main__":
    main()
