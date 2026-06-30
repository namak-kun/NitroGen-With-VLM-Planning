# ARCH WARROOM — Round 5 seed: the S1/S2 ATTRIBUTION problem in human demo narration

NEW topic (data↔architecture seam), raised by the project owner @namak-kun after writing real narration for his
own SMW demos (`demo_explanations.md`). This is the question that motivated his original "dual-channel /
how-much-is-S1-vs-S2" architecture instinct. Bring fresh thinking; do NOT just restate R1–R4.

## The owner's exact framing (paraphrased + quoted)
He hand-narrated his SMW playthroughs at ~1-second granularity. His words:
> "My explanations are NOT raw System-2 outputs. Since I'm working on seconds, **sub-second decision making is
> written into the plans** — so you'll see sentences where the **reaction** is also included. How much of this
> is System-1 and how much is System-2? **I can't know — I didn't think very actively, not in a monologue.**
> It's slightly ad-hoc, based on memories and logical guesses about why I might do things. Though **most of it
> is objective — it literally describes what I did**, like 'jump right to go on the platform.' A direct
> explanation. Hand-annotating everything cleanly is infeasible."

So the narration is a **conflated, partly-confabulated, mostly-descriptive** trace — not a clean System-2 plan.

## The real data — 4 categories the narration actually contains (verbatim lines from demo_explanations.md)
- **(A) System-2 — objective/tactical, frame-UNDERdetermined (the markov DiT can't infer this from one frame):**
  - "move right to intercept"
  - "decide not worth it and go right" / "decide chase not worth it"
  - "go left for a sec to see if the pipe can be entered" → "Confirmed no, continue right"
  - "go left to get space to do a run jump right onto the flagpole"
  - "see rex, take a step back" → "wait and spin jump to instantly kill" (deliberate timing/setup)
- **(B) System-1 — reflex/reaction, frame-DETERMINED (the DiT should produce this from the frame itself):**
  - "jump to dodge the koopa", "duck to dodge the bullet bill", "crouch in the ditch to dodge the bullet bill"
  - "spin jump on the rex to instantly kill", "veer left to land on the rex again, too late and run into it and die"
  - "jump too early, come back"
- **(C) Confabulation / post-hoc / no real plan (owner flags these himself):**
  - "Go left for no real reason, and then switch to right"
  - "No real reason to go left, maybe a vain attempt to jump on the bill"
- **(D) Aesthetic / meta (neither task-S1 nor task-S2):**
  - "a spin jump for flair", "spin jump onto the checkpoint for flair"

## Why this is architecturally load-bearing (the anchor — do not ignore)
The project ALREADY measured (DISCRIMINATOR_RESULTS.md, this session) a clean GAME-DEPENDENT dissociation:
- **SMW = PLAN-OOD** (oracle plan Δ=+0.20; lora-only actor-adapt ≈0) — and SMW narration is DENSE with type-(A)
  S2 decisions (enter-pipe?, chase-or-not, pit routing, flagpole setup).
- **Sonic = ACTOR-OOD** (actor-adapt Δ=+1.9; plan redundant/slightly hurts) — and Sonic is "run right fast":
  almost ALL type-(B) reflex, with little extractable (A).
HYPOTHESIS to evaluate: **the fraction of type-(A) extractable-S2 content in a game's demos predicts whether
plan-conditioning helps that game.** If true, it unifies the dissociation with the data-curation question and
tells us what the VLM is actually FOR (supply the (A) the frame underdetermines; stay out of (B) the DiT owns).
Also recall the verified "near-zero plan gradient on factual data" finding: if narration leaks type-(B) reflex
into the plan, the plan tokens carry info the DiT ALREADY has from the frame → wasted/!harmful conditioning.

## The questions for Round 5 (take a STANCE on each)

**Q1 — Is the S1/S2 split even recoverable from this narration, and HOW (automatably)?** Hand-annotation is
infeasible (owner). Options to argue for/against, or propose better:
  (a) **VLM re-abstraction:** feed (frames + the raw human narration) to the FROZEN Qwen and have it emit ONLY
      the slow objective (strip reaction) — narration as a GROUNDING HINT, not the plan itself. Decoded or
      decode-free? How do you stop it from just parroting type-(B)?
  (b) **Temporal abstraction:** S2 = the slowest-varying component over an A-chunk window; S1 = what changes
      per chunk. Aggregate/deduplicate narration across seconds to extract the persistent intent.
  (c) **Frame-counterfactual filter:** a line is type-(B) iff the FROZEN base DiT already predicts that action
      from the frame alone (we can MEASURE this — base_dit_perdim / the CFG v_c≈v_u test); keep only lines the
      DiT does NOT already produce (≈ the type-(A) residual). This operationalizes "frame-underdetermined."
  (d) **Don't split — use narration only as EVAL/JUDGE + checkpoint labels** (required vs bonus, level
      boundaries), never as plan-conditioning input. Argue if attribution is the wrong goal entirely.

**Q2 — What is the right SUPERVISION TARGET for the frozen-VLM→bridge→DiT, given conflated narration?** If you
train plan tokens on the whole narration you contaminate with (B)+(C)+(D). Spell out the concrete target +
loss that keeps only the (A) signal. Does this change with the dissociation (SMW vs Sonic)?

**Q3 — The confabulation (C) + aesthetic (D) noise.** The owner is explicit that parts are post-hoc guesses /
flair, not plans. How robust is each pipeline to this? (Tie to the earlier session flags: the trained VLM saw
some SYNTHETIC data, and popular games risk VLM-prior contamination.) What's the guardrail?

**Q4 — Architecture implication (the owner's original instinct).** If S1 and S2 are ENTANGLED in human
cognition at these timescales (he literally cannot separate them in retrospect), should the ARCHITECTURE hard-
split them, or split along TIMESCALE instead (fast frame-fresh latent every chunk + slow objective every A
chunks, à la Helix), with the DATA abstracted to match? Does the narration evidence argue for or against the
hard System-1/System-2 modular split this project assumes? Be willing to challenge the project's framing.

**Q5 — ONE cheap experiment** on the EXISTING repo that tests the unifying hypothesis (type-(A) fraction
predicts plan benefit) OR validates one attribution method. Use what exists: demo_explanations.md (now parseable
via planner_poc/demo_narration.py → frame-aligned narration.json with frame/frame_end/level per note), btn_s600,
the base-DiT counterfactual test (does the DiT already produce the narrated action from the frame?), the
discriminator harness (rwbc_actor_adapt.py), and the frozen VLM (planner.py encode_multimodal/generate_plan).
Name the metric + the number that would confirm/refute.

## Project facts (don't contradict)
- frozen Qwen3.5-2B (S2) → PlanResampler (K=8 continuous-latent query tokens) → PlanAdapter → PlanHead →
  injected into a frozen ~500M flow-matching DiT (S1, GR00T-N1.5 lineage, cross-attn V/L + AdaLN), markov
  (current frame → 18-step action chunk @60fps; demos are 60fps, action[i]↔obs[i]).
- Verified: base DiT reproduces streamer action from frame ALONE (buttons AUC 0.82–0.96, sticks 0.4–0.76) →
  factual plan-conditioning is near-zero-gradient; "we can't escape envs" for real counterfactual capability.
- Dissociation (this session): SMW plan-OOD (+0.20), Sonic actor-OOD (+1.9); SMW actor-adapt is seed-variance-
  dominated (retracted single-seed claims) — treat SMW numbers as noisy, Sonic actor-win as robust.
- Constraints: ONE generalist (no per-game models); frozen-VLM-FIRST (VLM-LoRA allowed but cost-flagged);
  preserve EXACT null-invariance; cite PRIMARY sources.

## Mechanics
GPT-5.5 → `ARCH_WARROOM_gpt55_r5.md`; Opus-4.8 → `ARCH_WARROOM_opus_r5.md` (absolute paths in each agent's
instructions). Then the orchestrator referees + runs a short cross-rebuttal. Bring NEW, concrete thinking on
the attribution problem — this is the owner's actual blocker for using his hand-written narration.
