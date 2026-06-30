"""generate_grounded_plan.py -- build the INTERLEAVED grounded plan-teacher prompt and generate a
broad, control-grounded plan with the frozen planner VLM.

This is the (reconstructed) "grounded interleaved" format @namak-kun designed (the original file was
lost with the dev box, never committed). Unlike generate_plan()/objective_label_demo.py (which feed a
flat frame window + one text instruction), this assembles a MULTI-CHUNK interleaved prompt so the VLM
sees, per action-chunk, BOTH what the screen looked like AND what inputs were taken:

    [SYSTEM] role + Controls scheme (per game/genre) + OUTPUT INSTRUCTIONS
    [USER]
      "Initial game state (before any actions):"   <frame_0>           # 1 extra "before" frame
      "Chunk 1 -- frames:"  <frame> <frame> ...                        # a few frames spanning chunk 1
      "Chunk 1 -- actions taken: <action summary>"                     # action details, interleaved
      "Chunk 2 -- frames:"  <frame> ...
      "Chunk 2 -- actions taken: <action summary>"
       ...
      <final OUTPUT INSTRUCTION>

The VLM can't read gamepad inputs from pixels (EXP-036/037), so the interleaved ACTION DETAILS ground
the plan by construction. The output is a BROAD, free-form natural-language plan (NOT action-by-action)
that uses concrete control-mapped terms (left/right/up/down/jump/attack) and AVOIDS ambiguous words
like "forward" that are meaningless in some views (e.g. top-down).

Inputs are NitroGen chunk dirs (each with actions_*.parquet) + their extracted frames. Frame source is
flexible: pass --frames-dir with files named "<uuid>__<offset>.png" (frames_cc convention) OR a chunk
dir that itself contains f*.png frames. If no frames are found it still emits the prompt with action
details only (text-only grounding), so it is usable even before the frame caches are rebuilt.

Run:
  env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$PWD:$PWD/planner_poc QWEN=Qwen/Qwen3.5-2B \
    .venv/bin/python planner_poc/generate_grounded_plan.py \
      --chunk-dir "/path/to/<video>_chunk_0030" [more chunk dirs ...] \
      --frames-dir /path/to/frames --genre platformer
  # or dry-run the assembled prompt without loading the VLM:
  ... --print-prompt-only
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

import os; _R = os.environ.get("NITROGEN_REPO", "/home/t-nagupta/NitroGen-With-VLM-Planning")
sys.path.insert(0, _R); sys.path.insert(0, os.path.join(_R, "planner_poc"))

from nitrogen.training.actions import (
    assemble_chunk,
    load_chunk_actions,
    summarize_chunk,
)

QWEN = os.environ.get("QWEN", "Qwen/Qwen3.5-2B")

# ---- Controls schemes (per genre) -- the "grounded" vocabulary the plan must stick to. -------------
# Mirrors planner_poc/play_annotated_horizon.py:CONTROLS so the two stay consistent.
CONTROLS = {
    "racing": "Controls: steer LEFT or RIGHT, ACCELERATE, BRAKE.",
    "platformer": "Controls: move LEFT or RIGHT, look UP, crouch DOWN, JUMP, ATTACK. To reach a higher "
                  "ledge you must JUMP (moving sideways alone will not climb).",
    "topdown": "Controls: move LEFT, RIGHT, UP or DOWN, ATTACK, ACTION to interact.",
    "shmup": "Controls: move LEFT, RIGHT, UP or DOWN, FIRE.",
    "generic": "Controls: move LEFT, RIGHT, UP or DOWN, plus action buttons (JUMP/ATTACK/ACTION).",
}

# ---- The system header: role + controls + OUTPUT INSTRUCTIONS. -------------------------------------
SYS_HEADER = (
    "You are an expert game-playing assistant acting as the high-level planner for an agent. "
    "You are shown the game state before any actions, then for each subsequent time-chunk the frames "
    "spanning that chunk together with the inputs the player actually took during it. Use the inputs "
    "to understand cause and effect (you cannot read the controller from the image alone)."
)

# The OUTPUT INSTRUCTIONS block (what @namak-kun calls the "output format"): a broad, free-form plan
# grounded in the control vocabulary, no ambiguous direction words.
OUTPUT_INSTRUCTIONS = (
    "Now output ONE short, broad plan in plain natural language for what the agent should do next and "
    "briefly why. Describe the goal in a general sense -- do NOT list actions step by step or per "
    "chunk. Ground every direction in the listed controls: use concrete terms like LEFT, RIGHT, UP, "
    "DOWN, JUMP, ATTACK. Do NOT use ambiguous words like 'forward', 'onward', 'explore', or 'the "
    "image' -- e.g. in a top-down game 'forward' is meaningless, say UP/DOWN/LEFT/RIGHT instead. "
    "Output ONLY the plan sentence, nothing else."
)


def _load_frame(path: str) -> Image.Image | None:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _frames_for_chunk(uuid: str, chunk_dir: str, frames_dir: str | None,
                      start_idx: int, action_horizon: int, frame_stride: int,
                      per_chunk: int) -> list[Image.Image]:
    """Find up to `per_chunk` frames spanning one action chunk. Tries, in order:
      1. frames_dir/<uuid>__<offset>.png   (frames_cc naming; offsets across the chunk span)
      2. chunk_dir/f*.png                  (per-chunk frame dump; evenly sampled)
    Returns [] if none found (the prompt then uses action details only for that chunk)."""
    span = action_horizon * frame_stride
    offsets = [int(start_idx + round(j * (span - 1) / max(1, per_chunk - 1)))
               for j in range(per_chunk)] if per_chunk > 1 else [start_idx]
    frames: list[Image.Image] = []
    if frames_dir:
        for o in offsets:
            im = _load_frame(os.path.join(frames_dir, f"{uuid}__{o}.png"))
            if im is not None:
                frames.append(im)
    if not frames:
        local = sorted(glob.glob(os.path.join(chunk_dir, "f*.png")))
        if local:
            sel = np.linspace(0, len(local) - 1, num=min(per_chunk, len(local))).round().astype(int)
            frames = [im for im in (_load_frame(local[i]) for i in sel) if im is not None]
    return frames


def build_interleaved(chunk_dirs: list[str], frames_dir: str | None, genre: str,
                      action_horizon: int = 18, frame_stride: int = 2, action_shift: int = 3,
                      per_chunk_frames: int = 3, start_idx: int = 3):
    """Assemble (system_text, content, images) for the interleaved grounded prompt.

    `content` is a list of {"type": "image"} / {"type": "text", "text": ...} items in the Qwen
    chat-content format (interleaved); `images` is the parallel list of PIL images (one per image
    item, in order). Returns also a plain-text RENDER of the prompt for --print-prompt-only / logging.
    """
    controls = CONTROLS.get(genre, CONTROLS["generic"])
    system_text = SYS_HEADER + "\n" + controls

    content: list[dict] = []
    images: list[Image.Image] = []
    render_lines: list[str] = [f"[SYSTEM]\n{system_text}", "", "[USER]"]

    def add_text(t: str):
        content.append({"type": "text", "text": t})
        render_lines.append(t)

    def add_image(im: Image.Image, tag: str):
        content.append({"type": "image"})
        images.append(im)
        render_lines.append(f"<image: {tag}>")

    # 1) initial "before" frame: the first frame of the first chunk's span.
    first = chunk_dirs[0]
    uuid0 = os.path.basename(first.rstrip("/"))
    before = _frames_for_chunk(uuid0, first, frames_dir, start_idx, action_horizon, frame_stride,
                               per_chunk=1)
    add_text("Initial game state (before any actions):")
    if before:
        add_image(before[0], f"{uuid0} initial")

    # 2) per-chunk: a few frames + the action details (interleaved).
    for ci, cdir in enumerate(chunk_dirs, 1):
        uuid = os.path.basename(cdir.rstrip("/"))
        pq = os.path.join(cdir, "actions_processed.parquet")
        if not os.path.exists(pq):
            pq = os.path.join(cdir, "actions_raw.parquet")
        summary = "(actions unavailable)"
        if os.path.exists(pq):
            a = load_chunk_actions(pq)
            rc = assemble_chunk(a["buttons"], a["j_left"], a["j_right"], start_idx,
                                action_horizon, frame_stride)
            if rc is not None:
                summary = summarize_chunk(rc)
        frames = _frames_for_chunk(uuid, cdir, frames_dir, start_idx, action_horizon, frame_stride,
                                   per_chunk=per_chunk_frames)
        add_text(f"Chunk {ci} -- frames:")
        for k, im in enumerate(frames):
            add_image(im, f"{uuid} chunk{ci} f{k}")
        add_text(f"Chunk {ci} -- actions taken: {summary}")

    # 3) final output instruction.
    add_text(OUTPUT_INSTRUCTIONS)

    return system_text, content, images, "\n".join(render_lines)


def generate(system_text: str, content: list[dict], images: list[Image.Image],
             device: str, max_new_tokens: int = 48) -> str:
    """Run the frozen planner VLM on the interleaved [system + content + images] message."""
    from nitrogen.planner import PlanEncoder, PlannerConfig
    pl = PlanEncoder(PlannerConfig(backbone_name_or_path=QWEN))
    pl.load()
    if next(pl.backbone.parameters()).device != torch.device(device):
        pl.backbone.to(device)
    if pl.processor is None:
        raise RuntimeError("generate_grounded_plan requires a VL processor (Qwen3VLProcessor).")
    msgs = [{"role": "system", "content": system_text},
            {"role": "user", "content": content}]
    text = pl.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    proc_kwargs = {"text": [text], "return_tensors": "pt"}
    if images:
        proc_kwargs["images"] = images
    inp = pl.processor(**proc_kwargs).to(device)
    out = pl.backbone.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    gen = pl.processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    plan = gen.strip().strip('"').strip()
    return plan.split("\n")[0].strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk-dir", nargs="+", required=True,
                    help="one or more NitroGen chunk dirs (each with actions_*.parquet), in time order")
    ap.add_argument("--frames-dir", default=None,
                    help="dir with <uuid>__<offset>.png frames (frames_cc convention); optional")
    ap.add_argument("--genre", default="generic", choices=sorted(CONTROLS),
                    help="control scheme to ground the plan in")
    ap.add_argument("--per-chunk-frames", type=int, default=3, help="frames shown per action chunk")
    ap.add_argument("--start-idx", type=int, default=3, help="first frame index within each chunk (=action_shift)")
    ap.add_argument("--action-horizon", type=int, default=18)
    ap.add_argument("--frame-stride", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--print-prompt-only", action="store_true",
                    help="assemble + print the prompt render, do NOT load the VLM")
    args = ap.parse_args()

    system_text, content, images, render = build_interleaved(
        args.chunk_dir, args.frames_dir, args.genre,
        action_horizon=args.action_horizon, frame_stride=args.frame_stride,
        per_chunk_frames=args.per_chunk_frames, start_idx=args.start_idx)

    n_imgs = len(images)
    print(f"# interleaved grounded prompt: {len(args.chunk_dir)} chunk(s), {n_imgs} image(s), "
          f"genre={args.genre}\n")
    print(render)
    print()
    if args.print_prompt_only:
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = generate(system_text, content, images, device, max_new_tokens=args.max_new_tokens)
    print(f"\n===> GROUNDED PLAN: {plan}")


if __name__ == "__main__":
    main()
