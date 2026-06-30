# ARCH WARROOM R5 — Opus 4.8: the S1/S2 ATTRIBUTION problem in human narration

**TL;DR stance.** The verbal S1/S2 split is NOT cleanly recoverable line-by-line — and you shouldn't
try, because a person's report of *why* they acted is a post-hoc theory, not a causal trace (Nisbett &
Wilson 1977, *Telling more than we can know*). But the split you actually NEED — **frame-determined vs
frame-underdetermined** — IS recoverable, automatably, by measuring it on **actions, not words**. Rank:
**(c) frame-counterfactual filter as the LABELER ∘ (a) decode-free VLM re-abstraction as the ENCODER**,
with **(b) temporal abstraction as a PRIOR** and **(d) as the honest floor**. GPT-5.5's filter is the
rigorous operational form of my "frame-underdetermined" instinct — adopt it.

## Q1 — Recoverable? Yes — reattribute ACTIONS, not WORDS. (c)∘(a), (b) prior, (d) floor.

The owner can't tag each line; that's Nisbett & Wilson, not laziness. So abandon per-token semantic
labels. (a)–(d) aren't rivals — they act on different objects and COMPOSE:

- **(c) frame-counterfactual filter — the only falsifiable definition, and the labeler.** A narrated
  action is type-B iff the FROZEN base DiT already reproduces it from the frame alone:
  `_sample_chunk(frame, "", null=True)` vs the streamer's real action (base_dit_perdim.py: buttons AUC
  0.82–0.96), or equivalently ‖v_c−v_u‖≈0 at deploy (eval_policy.py:237–239). This OPERATIONALIZES
  "frame-underdetermined" exactly. Keep it.
- **(a) decode-free re-abstraction — the encoder (my push).** Don't let the bridge ingest raw narration;
  it will parrot type-B. Feed frames+narration to the frozen Qwen and take
  `encode_multimodal(text_only=True)` (EXP-052, planner.py:217–225): the ~10 text tokens have already
  cross-attended the frames, so they carry plan authority + grounding without a decode and without
  parroting the reaction. Narration becomes a GROUNDING HINT, never the plan itself.
- **(b) temporal abstraction — a prior, not a method.** The slow objective = slowest-varying component
  over the A-chunk window (options framework, Sutton–Precup–Singh 1999). Dedup across seconds: persistent
  intent survives, per-second flicker dies.
- **(d) the floor** for the residual (c)+(a) still can't attribute. Crucially, what's irreducibly
  unrecoverable from OFFLINE narration is exactly what needs ON-POLICY labels — DAgger without a human
  oracle (Ross et al. 2011, arXiv:1011.0686): the emulator, not the transcript, supplies it.

**Stance:** (c) is the supervision MASK; (a) the supervision INPUT; (b) regularizes; (d) the remainder.
The split is recoverable at the level of the action DISTRIBUTION, not the prose.

## Q2 — Supervision target: the frame-counterfactual RESIDUAL, not the narrated action.

Training plan tokens to reproduce the narrated action leaks (B): the DiT already makes it. Right target =
the part of the action the frame UNDERdetermines. Concrete:
- **Input:** o = text_only re-abstracted objective (a).
- **Loss:** residual-weighted flow-BC, `L = Σ_i m_i·‖v_θ(a_i, plan) − u_i‖²`, with
  `m_i = clip(‖a_real,i − a_base,i‖)` (or per-dim `1 − AUC`) = the base-DiT frame-counterfactual residual
  (c). Where the base already nails it, m_i→0 → the plan gets NO gradient → this RESPECTS the verified
  "near-zero plan gradient on factual data" instead of fighting it; null-invariance is untouched.
- **Plus** the existing `distill_weight` InfoNCE/CLIP term (planner.py:43) to de-collinearize the K
  tokens so they carry DISTINCT objectives, not a collapsed mean.

**SMW vs Sonic falls out for free.** m_i is large at SMW decision points (enter-pipe? / chase-or-not /
pit routing) and ≈0 for Sonic "run right." The SAME loss auto-allocates plan-gradient to SMW and starves
Sonic — reproducing the dissociation (Δ_plan +0.20 vs −0.18) with NO per-game knob. **The residual mask
IS the type-A fraction.**

## Q3 — (C) confabulation + (D) flair, and prior contamination.

The residual mask is robust by construction: flair spin-jumps the DiT does anyway → m_i low → starved; a
one-off "vain attempt left" has no consistent frame→objective map → InfoNCE finds no stable positive →
averaged out as variance. **Route the owner's own flags as GOLD weak labels:** "no real reason" / "for
flair" → re-abstract to a NULL objective → the existing learned null plan absorbs them (null-invariance
is the confabulation sink). Popular-game-prior guardrail: SMW is memorized by Qwen, so its objective may
be PRIOR, not GROUNDED in this frame. The frame-counterfactual filter is ALSO the contamination detector
— if the VLM objective doesn't shrink the base-DiT residual toward the real action, it's ungrounded prior
→ down-weight. And EVAL on obscure/homebrew/custom levels (the seed's own plan) where the VLM has no
prior. Never CE-train on narration tokens — that bakes confabulation straight into the bridge.

## Q4 — Architecture: TIMESCALE split, not hard semantic S1/S2.

The owner's inability to separate S1/S2 in retrospect is EVIDENCE, not a measurement gap: at 1s
granularity the clean two-system story is itself a confabulated narrative (Nisbett & Wilson). So the
narration UNDERMINES a hard SEMANTIC split (reflex-vs-deliberate — even the actor can't tag it) but
SUPPORTS a TIMESCALE split, which is what the data actually has: a fast per-chunk action stream + a slow
per-A-chunk objective. That is precisely Helix (fast visuomotor + slow VLM latent,
figure.ai/news/helix) and π0.5 (high-level subtask → low-level flow chunk, pi.website/blog/pi05).

**Challenge to the project framing:** if "System-2" is read as a module that owns every *deliberate*
action, the data says NO. Let the markov DiT own the FULL action distribution (including
deliberate-LOOKING but frame-determined moves) and let the VLM own ONLY the slow frame-underdetermined
objective. The boundary is **frame-determinacy (measured by c) at the slow timescale**, not a cognitive
reflex/plan partition. Read that way, the project's frozen-DiT + slow-VLM-plan IS already a correct
timescale split; the SMW/Sonic dissociation is an information-ALLOCATION result (SMW: slow channel
informative; Sonic: redundant), not "SMW is more cognitively System-2." Abstract the DATA to match:
slow channel = re-abstracted objective (a)+(b); fast channel = the raw action chunks (already there);
bridge gradient gated by (c). Both split on timescale + frame-determinacy — never on a semantic S1/S2 tag
the owner himself can't assign.

## Q5 — One cheap experiment: validate the filter on the owner's OWN A/B labels (4-genre gradient).

The repo now has narration for FOUR genres — Sonic (type-B), SMW (mixed), Fire-Emblem (turn-based ≈ pure
type-A), Zelda (exploration) — and the seed already hand-sorted ~10 SMW lines into A vs B. One run, reusing
base_dit_perdim machinery on each demo.npz + narration.json:
- For each narration.json entry: `ρ_e = ‖base-DiT null chunk (_sample_chunk(frame, "", null=True)) −
  streamer real action over [frame, frame_end)‖`. High ρ_e = frame-underdetermined = type-A.
- **Metric 1 (method validation):** AUC of ρ_e separating the owner's hand-labeled type-A lines from
  type-B lines. **CONFIRM if AUC ≥ 0.75** (the filter recovers the owner's split it was never shown);
  **REFUTE if ≈0.5** → attribution is too lossy → fall back to (d), narration as eval/judge only.
- **Metric 2 (unifying hypothesis):** per-game f_A = frac(ρ_e > τ). Predict
  f_A(Fire-Emblem) > f_A(SMW) ≫ f_A(Sonic); Spearman(f_A, Δ_plan) > 0 vs the discriminator (SMW +0.20,
  Sonic −0.18). **CONFIRM if SMW f_A ≥ 1.3× Sonic f_A** and Fire-Emblem tops the ranking.

The AUC number decides whether the frame-counterfactual filter is a valid automated attributor; the f_A
ordering decides whether type-A fraction predicts plan benefit. Both honor the constraints: ONE generalist
(emergent per-game weighting, no heads), frozen-VLM-first (text_only prefill, no LoRA), null-invariance
preserved (residual mask zeros plan gradient exactly where the frame already explains the action).
