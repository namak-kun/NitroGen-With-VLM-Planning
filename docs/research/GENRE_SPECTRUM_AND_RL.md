# Training discussion — the System-1 ↔ System-2 genre spectrum (2026-06-26)

@namak-kun's framing of where the plan-conditioned system works today:

```
SYSTEM-1 HEAVY  <-------------------- SWEET SPOT --------------------> SYSTEM-2 HEAVY
racing, football, shmups        platformers (works okay)         puzzles (sokoban, etc.)
   |                                    |                                   |
 VLM is a DISTRACTION             best regime                    doesn't work well either
 (fast reactive control;          (some planning + some           (needs exploration/creativity;
  plan token only adds noise)      reactive control both help)     VLM not "curious" enough to
                                                                    try multiple strategies)
```

Key claims:
- **System-1-heavy (racing/football/shmups):** reactive markov control dominates; the frozen DiT
  already does the job; the VLM plan mostly DISTRACTS. (Consistent with the L/R override ceiling +
  "base DiT already reproduces the action" findings.) RL on the plan here = low/negative value.
- **Middle (platformers):** the SWEET SPOT — both a bit of planning and reactive control matter, and
  the system "works okay". This is where plan-conditioning earns its keep.
- **System-2-heavy (puzzles/sokoban):** doesn't work well. Suspected cause (not confirmed): lack of
  EXPLORATION / creativity — the VLM isn't curious enough to try multiple strategies. NOT (yet)
  attributed to a control/grounding gap. Needs verification.

## Implications for the RL/data plan
- Target the MIDDLE/right of the spectrum (platformers → puzzles), NOT system-1-heavy games — that's
  where System-2 RL can actually move the needle. System-1-heavy games are a poor RL testbed for the
  PLAN policy (plan ~irrelevant).
- The system-2 failure is an EXPLORATION problem, which points at RL objectives that REWARD trying
  varied strategies (novelty/curiosity/diversity bonuses), not just terminal success — and at
  sampling diverse plans (temperature, GRPO-style K diverse rollouts) rather than greedy decode.

## NOTE FOR LATER DISCUSSION (user flagged explicitly)
**Provide older plans / prior experience in the prompt.** To get exploration + non-repetition, the
planner prompt should include a memory of PRIOR PLANS / past attempts (what was tried, what happened)
so it can (a) avoid repeating a failed strategy and (b) build on progress. This is a prompt/context
design change (an experience buffer in-context), related to:
  - generate_plan already has a `prev_plan` arg (single previous plan) — extend to a history.
  - s2_reaction_plans.py shows prior-plan + "what the player did after" context.
  - Ties to the plan-stability bottleneck (~50-60% flip-rate) in AGENTS.md §4.4.
  - The new generate_grounded_plan.py interleaved format is a natural place to inject a
    "previous attempts / outcomes" block before the OUTPUT INSTRUCTIONS.

## Refinements (2026-06-26, cont.) — three levels of System-2 + DEFERRAL
The role of System-2 has three levels (user):
  1. **In-level goal planning** — matters middle/right of spectrum; ~deferral on System-1-heavy.
  2. **Corrective planning given failure** — needed EVERYWHERE; the CREATIVITY CORE; hardest in puzzles.
  3. **Experiential learning ACROSS A (single) GAME** — the pipe dream (NOT cross-game; user corrected).
     (A bounded in-context memory of recent attempts is the tractable slice = iterated level 2.)
Corrective planning (2) is the center of gravity: needed across the whole spectrum, most creativity.

### DEFERRAL — via OUTPUT, not a model-controlled CFG weight (user decision)
- Model-controlled continuous plan STRENGTH (a learned CFG weight) is REJECTED: hard to train, "we
  lack signal" (no clean target for the right weight).
- Preferred: deferral **through the OUTPUT** — a special token or a NONCOMMITTAL plan, e.g.
  "This doesn't require guidance" / `<NO_GUIDANCE>`. Involves BOTH prompt changes (offer the option in
  OUTPUT INSTRUCTIONS) AND training.
- Mechanism (decouples DECISION from EFFECT): the VLM emits either a real plan OR a deferral output;
  downstream a deferral output routes to the NULL plan (existing masked-null → base DiT EXACT). So the
  DECISION lives in token space (trainable) and the EFFECT reuses the exact base behavior.
- KEY TRAINING-SIGNAL ARGUMENT (why output-deferral is trainable but weight isn't): because deferral
  is a discrete OUTPUT ACTION, the EPISODE REWARD is the signal. On a System-1-heavy game, sampling
  "defer" → base System-1 runs → good outcome → reward → reinforce deferral. On System-2-heavy,
  defer → no progress → low reward → reinforce planning. The policy learns WHEN to defer for free from
  the same RL reward (a continuous weight has no such target). Also reduces plan flip-rate (defer =
  abstain instead of emitting a fresh random plan each replan).
- BOOTSTRAP CAVEAT: RL can only reinforce deferral if the policy SAMPLES it. Seed the deferral token
  into the output distribution first (a little SFT / prompt-offer it as an allowed option) so RL has
  something to reinforce. Concrete hook: add the deferral option to generate_grounded_plan.py's
  OUTPUT_INSTRUCTIONS.

### RL stance (agreed)
RL (GRPO or similar) IS still needed, but CAREFULLY ENGINEERED. GRPO/RLVR SHARPENS toward existing
modes (Yue et al.: better pass@1, base matches at high pass@k; entropy collapse) → it consolidates,
does NOT invent corrective creativity. So: inject exploration FIRST (diverse plan sampling +
in-context experience memory + maybe novelty/diversity bonus), use RL to consolidate, guard against
entropy collapse (KL control / diversity terms).

## NOTE (user, 2026-06-26 pre-sleep): cookies.txt added for YouTube
- cookies.txt is at repo root (gitignored) for yt_farm.py video access (deno+ejs recipe in docs/SETUP_YOUTUBE.md).
- IDEA (IF NEEDED, not now): use YouTube videos to PERFECT THE JUDGE (train/calibrate the VLM-judge on
  real gameplay), separate from the harder IDM/state-creation use. Low priority; only if judge quality
  proves to be the bottleneck. User: "obviously an if needed thing" — do NOT get distracted by it.
