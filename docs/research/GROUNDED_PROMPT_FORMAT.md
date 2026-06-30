# Grounded interleaved prompt/teacher format (reconstruction)

## Status: MISSING from the repo
The file that built this exact format was **never committed** and is lost with the old dev box.
Verified 2026-06-26: no `generate_grounded*`/interleaved-format `.py` on disk, in git history (any
branch), in untracked files, in git stashes, or in `git fsck` dangling objects. The pieces exist
scattered (see "Where the pieces live" below) but no single file assembles the interleaved format.

`planner_poc/play_annotated_horizon.py` generates the `docs/horizon_play*` videos and has the
per-genre `Controls:` header + a recent-actions SUMMARY, but it is NOT the interleaved per-chunk
teacher format — it shows a recent-frame window + one action summary, not `[frames][action details]`
interleaved per chunk with a separate initial state frame.

---

## The prompt template @namak-kun described (INPUT)
An interleaved multi-chunk grounded prompt:

```
[SYSTEM / HEADER]
  - Controls: <which buttons/sticks do what for this game/genre>
  - Output instructions: <what to produce + format>

[INITIAL STATE]
  - 1 extra frame = the game state BEFORE any actions are taken

[PER ACTION CHUNK i = 1..N]   (interleaved)
  - a few frames for chunk i (the frames spanning that 18-step action chunk)
  - action details for chunk i (what inputs were taken during that chunk —
    e.g. the action summary: "moved RIGHT 12, JUMP 6 of 18")

[FINAL]
  - output instruction / the model emits its answer
```

Key properties:
- **Controls + output instructions** in a small header up front.
- **1 initial "before" frame** establishing pre-action state.
- **A few frames PER action chunk** (not a flat recent-window) — lets the model see what each
  chunk's actions did to the screen.
- **Action details interleaved IN THE MIDDLE** — the actual inputs per chunk are given as text
  between the frame groups (grounding by construction; the VLM can't read inputs from pixels,
  EXP-036/037).

## OUTPUT format
The model outputs a **plan**, with these properties (per @namak-kun 2026-06-26):
- **Free-form natural language** describing what has to be done.
- **Broad / high-level** — NOT action-by-action, not per-chunk. A general sense of the goal.
- **Grounded in the controls** — must use concrete control-mapped terms (left/right/up/down/jump/
  attack/etc.) and AVOID ambiguous words like **"forward"** that don't make sense in some views
  (e.g. "forward" is meaningless in a top-down game). The "grounded" requirement is about
  vocabulary discipline, not granularity.

### "Output instructions" (the header's instruction block — what @namak-kun meant by "output format")
The small header includes an OUTPUT-INSTRUCTIONS block that directs the model to emit exactly the
above: a short, broad, free-form plan grounded in the listed controls, no ambiguous directional
words. (cf. the existing `GROUND_INSTR` in play_annotated_horizon.py:76 and `objective_label_demo.py`
INSTR — same spirit: "name the concrete direction... rather than vague words like 'forward'/'explore'".)

---

## Where the pieces live now (to rebuild)
- Controls header (per genre): `planner_poc/play_annotated_horizon.py:69` (`CONTROLS`, `GENRE_OF`).
- Grounded output instruction: `play_annotated_horizon.py:76` (`GROUND_INSTR`); objective-style
  prompt in `planner_poc/objective_label_demo.py` (`SYS`/`INSTR`).
- Per-chunk ACTION DETAILS (the "action details in the middle"):
  `nitrogen/training/actions.py::summarize_chunk` (re-exported by `planner_poc/action_summary.py`)
  — turns an 18-step chunk into "moved RIGHT 12, JUMP 6 of 18".
- Frames + action-summary context assembly: `planner_poc/gen_stage2_lookup_mm.py` (TACTICAL prompt,
  records `frame_offsets`/`before_idx`/`after_idx`); `planner_poc/s2_reaction_plans.py` (per-chunk
  ctx loop). VLM multimodal encode: `nitrogen/planner.py::encode_multimodal` /  `generate_plan`.
- Action layout for decoding details: buttons[0:21], j_left[21:23], j_right[23:25]; sticks [0,1],
  0.5=neutral; -x=left, -y=up.

## REVISED input format (2026-06-26, for RL training — supersedes the simple version above)
Order:
```
intro + task
game info
controls

frame f0                       # initial state, before any actions
s1_1                           # subchunk of actions = 1/3 of a chunk (intra-chunk sample rate = 3)
f1_1                           # frame AFTER subchunk s1_1
s1_2
f1_2
s1_3
f1_3                           # frame after the ENTIRE chunk 1 has elapsed
s2_1                           # chunk 2 (A=2: planner invoked every 2 chunks)
f2_1
s2_2
f2_2
s2_3
f2_3                           # last frame (planner fires here)

final output instructions
```
- **Subchunk granularity:** each 18-action chunk is split into 3 subchunks (6 actions each); a frame
  is shown after each subchunk → finer cause/effect than the "few frames per chunk" v1. (intra-chunk
  sample rate assumed 3 for now.)
- **THINK MODE (crucial):** RL-train the planner WITH thinking enabled, so model analysis stays in the
  reasoning trace and does NOT leak into the plan output (otherwise the plan gets diluted by analysis).
  NOTE: this CONTRADICTS the Stage-2 caches which DISABLED thinking (train/infer match). For RL we WANT
  think mode; the plan is extracted from the post-think output. Reconcile when building the RL planner.
- **K=8 plan tokens per chunk, A=2** (planner invoked every 2 chunks). Last frame = f2_3.
- **Experience in prompt (ABLATE):** optionally include the LAST TWO PLANS in the prompt to incentivize
  exploration / non-repetition. Open question: how to do context management (how much history, how to
  summarize outcomes). User unsure — flagged to ablate.
