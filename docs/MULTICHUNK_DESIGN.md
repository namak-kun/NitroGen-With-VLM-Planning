# Multi-Chunk & Long-Horizon Plan Conditioning — Design Notes

Author: drafted overnight 2026-06-17 for @namak-kun. Status: DESIGN ONLY (untested).
Context: NitroGen emits chunks of 18 actions; our planner fires sparsely. We want a
plan that is "a long string of actions" (e.g. 20+ low-level actions, or a semantic
goal) to influence MULTIPLE consecutive chunks differently. EXP-011 showed even
WITHIN one chunk the model can't yet sequence (SEQ "left then right" fails), so this
doc covers both intra-chunk and inter-chunk, since they share machinery.

---

## 0. Problem restatement (your words, formalized)

A plan may specify more low-level actions than fit in one 18-step chunk (e.g. "go
left for 20 steps then right for 20"). It therefore HAS to be divided across chunks.
Two sub-problems:
- **(P1) Intra-chunk temporal:** within one 18-step chunk, route plan-meaning to
  positions (steps 0–8 = left, 9–17 = right). EXP-011: currently fails.
- **(P2) Inter-chunk phase:** across chunks i, i+1, …, the SAME sparse plan must
  produce DIFFERENT behavior per chunk (chunk i = left, chunk i+1 = right). There is
  no shared clock today, so a static plan token conditions all chunks identically.

Your specific construction: take a long action string; let the FIRST 18 match the
streamer's real actions (so chunk-0 is functionally a NULL plan), and subsequent
chunks diverge per the plan. You correctly note this is **biased** (chunk-0 always
"do what the streamer did") and that a real env would help but is hard.

---

## 1. Why it's hard without an environment

To supervise "chunk i+1 should do X given the plan and the frame at the start of
chunk i+1", you need the FRAME at the start of chunk i+1. But that frame is the
*result of executing chunk i* — which requires either:
- (a) a real/sim environment to roll forward (infra-heavy, what we're avoiding), or
- (b) the streamer's actual future frames (we HAVE these! the video continues), or
- (c) a world model that predicts the next frame from (frame, actions) (Genie-style).

**Key realization: option (b) is free.** The dataset video continues, so for any
chunk we already have the real frames at t, t+18, t+36, … and the real actions in
each. We do NOT need a live env to get multi-chunk (frame, action) sequences — only
to get COUNTERFACTUAL multi-chunk sequences (what the frames WOULD be if the agent
followed the plan instead of the streamer). That distinction is the crux.

---

## 2. Three regimes of supervision (increasing difficulty)

### Regime R0 — Real multi-chunk, real plan (NO counterfactual) — FREE, do first
Use consecutive real chunks from one video as a multi-chunk sequence, paired with a
plan that DESCRIBES what the streamer actually did over those chunks (a "post-hoc"
plan, e.g. from the transcript or auto-generated from the actions: "move left, then
jump, then right"). Train the planner to emit a plan whose conditioning reproduces
the real multi-chunk action sequence, with the cursor/phase advancing per chunk.
- Pros: zero env, zero counterfactual bias, real frame transitions. Directly trains
  P2 (inter-chunk phase) and P1 (intra-chunk) on REAL data.
- Cons: only teaches "follow the plan you were given that matches reality" — it does
  NOT teach the agent to OVERRIDE the streamer (the counterfactual skill). But it is
  the cleanest way to install the *machinery* (cursor, phase, multi-chunk
  conditioning) before adding counterfactual pressure.

### Regime R1 — Counterfactual first chunk only (your "first-18-match" idea)
Chunk-0 target = streamer real (null-like); but condition on a plan that says "do
something different starting next chunk." Problem: we can't supervise chunk-1's
counterfactual target without the counterfactual frame. So R1 degenerates: either
chunk-1 also uses the real frame+real action (then it's just R0 with a delayed,
still-real plan), OR we need a world model. So **R1 isn't a distinct trainable regime
without (b)+plan-relabeling or (c) a world model.** Your bias intuition is right:
forcing chunk-0 = streamer makes the plan ignorable for chunk-0; the model can learn
"plan affects later chunks only," but supervising "later" still needs frames.

### Regime R2 — Full counterfactual multi-chunk (needs world model or env)
Roll the agent forward under the plan using a learned world model (predict frame_{i+1}
from frame_i + chunk_i), generating counterfactual frames, and supervise each chunk
toward the plan's intent. This is the powerful-but-hard regime. Ties to your "exploit
NitroGen's implicit state prediction" hypothesis (see §5).

**Recommendation:** build the MACHINERY on R0 (free, real), validate the cursor/phase
mechanism, THEN layer counterfactual pressure via R2 once a (even crude) world model
exists. Skip R1 as a standalone — it's R0-with-a-relabeled-plan.

---

## 3. Architecture: K×A blocks + cursor (chosen design)

(Already logged in EXPERIMENTS; expanded here.)

- Planner fires once per A chunks. Resampler emits **K×A** plan tokens, reshaped to A
  blocks of K. A per-chunk **cursor a∈[0,A)** selects block a for chunk i's injection.
- No new DiT phase signal needed: physically swapping the injected K tokens per chunk
  gives different conditioning. The existing `_PLAN_TOKEN` injection path is reused;
  only the tokenizer/model pick block a.
- Training (R0): a multi-chunk example = A consecutive real chunks + their frames +
  one plan text. For chunk i (i∈[0,A)), inject block i, target = real action chunk i.
  The resampler learns to put chunk-i's meaning in block i; the cursor enforces
  alignment. SupCon/contrastive can push blocks apart (as we did for directions).
- Inference: run planner once, then for A chunks feed block 0,1,…,A−1 with the live
  frame each chunk. Refresh the plan (re-run planner) every A chunks (Helix-style).

### Alternative to K×A: a scalar phase + single plan token set
Feed `t_plan = i/A ∈ [0,1]` into the DiT via AdaLN (like the flow timestep). One K-set
of tokens encodes the whole "program"; the phase scalar tells the DiT how far along.
Cheaper (K not K×A) but asks the DiT to internally index the program — harder to
learn, and the DiT cross-attention may need unfreezing/LoRA. K×A is more explicit and
likely easier; phase-scalar is more elegant if it works. **Try K×A first.**

---

## 4. Intra-chunk temporal (P1) — fix BEFORE inter-chunk

EXP-011: SEQ within a chunk doesn't work with frozen DiT + static tokens. Likely fixes
(running EXP-012 tonight to test the first):
1. **Unfreeze DiT (or LoRA on its cross-attention)** so it can learn position-
   dependent plan extraction (action-token i attends to plan tokens differently than
   token j). The DiT already has positional embeddings on the 18 action tokens — it
   CAN distinguish positions; it just needs the capacity/gradient to route.
2. **SEQ-heavy sampling** so temporal targets are dense (done in EXP-012).
3. **Sub-chunk plan blocks**: even within a chunk, give the resampler 2 sub-blocks
   (early/late) — a mini K×A with A=2 inside one chunk. Most explicit.
4. **Explicit temporal tag** on plan tokens (positional embedding added to the K
   tokens marking "first-half token" vs "second-half token").

P1 is the cheaper testbed for the same routing problem as P2; solve it first.

---

## 5. The "exploit NitroGen's implicit state prediction" hypothesis (your idea)

NitroGen's DiT outputs a coherent 18-action chunk, which means it *implicitly* models
how the state evolves over those 18 steps (otherwise the actions wouldn't be
self-consistent). Hypotheses to TEST (none assumed true):
- **H1 (intra-chunk readout):** the DiT's hidden states contain a predictable
  trajectory of latent state. If we could decode frame_{i+1} (or its SigLIP features)
  from the DiT internals after chunk i, we'd get a free 1-step world model → enables R2
  offline. Test: train a small probe from DiT hidden states → next-frame SigLIP
  features on REAL consecutive chunks (we have the targets!). If it predicts well,
  NitroGen already "contains" a world model.
- **H2 (horizon extrapolation):** does the model's behavior over 18 actions compose
  with itself over 36 (i.e. is chunking roughly Markov-consistent)? Test: compare
  (a) one 36-step rollout vs (b) two chained 18-step rollouts re-encoding the predicted
  frame; measure divergence. Low divergence ⇒ chaining chunks via a world model is
  sound.
Both are concrete, env-free experiments using the real future frames as ground truth.

---

## 6. Concrete next steps (ordered)
1. (running) EXP-012: unfreeze DiT + SEQ-heavy → does intra-chunk SEQ (P1) emerge?
2. If yes: add sub-chunk blocks / temporal tags to strengthen P1.
3. Build R0 multi-chunk dataset (A consecutive real chunks + frames + a post-hoc plan
   from the actions/transcript). Implement K×A blocks + cursor. Train; verify the
   cursor produces per-chunk-different conditioning that matches each real chunk.
4. Probe H1 (decode next-frame features from DiT internals) → if good, a free world
   model → enables R2 counterfactual multi-chunk.
5. Only then attempt counterfactual multi-chunk (R2) and the "override the streamer
   across chunks" skill.

## 7. Honest assessment
- The biased "first-18-match-streamer" construction alone won't teach cross-chunk
  override without future counterfactual frames; it mainly teaches "ignore plan for
  chunk 0." Useful as a curriculum step, not a solution.
- The cheapest real progress is R0 (machinery on real multi-chunk data) + the H1 probe
  (free world model from DiT internals). Both need NO live env and use the real video
  future we already download.
- A real env remains the gold standard for R2 but is the infra cost we're deferring;
  the world-model route (H1/Genie) is the principled synthetic substitute.
