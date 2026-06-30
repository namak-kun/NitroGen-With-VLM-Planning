# ARCH WARROOM — Round 8 seed: does the validated recipe GENERALIZE to new games? (+ scaling to 3D)

The owner is away for the night, restarted the mandate, and added a LOT more data. He'll bring more 2D AND 3D
games in the morning. This round keeps both agents in the loop on: (1) does the night's validated recipe
generalize to the new games, (2) what's the right eval/training as data scales, (3) prep for 3D games.

## What's VALIDATED so far (this session — build on it, don't re-litigate)
- **The dissociation:** SMW=plan-OOD (oracle plan helps; RWBC actor-adapt collapses/variance-flips),
  Sonic=actor-OOD (plain local RWBC robustly +1.44, 3/3 seeds; plan redundant).
- **The WORKING routed recipe (one model, two MODES):**
  - plan-OOD games -> SUPERVISED demo-fit of the PLAN-HEAD ONLY (LoRA+base frozen), conditioned on the correct
    plan, no env reward. SMW: Δ_plan +35.7->+56.0 (4/5 seeds, bootstrap CI [+2.0,+38.7]>0, NO collapse,
    null-invariant 0.0).
  - actor-OOD games -> plain local RWBC (full-adapt). Sonic +1.44.
  - MERGE = disjoint params (plan-OOD writes plan_head, actor-OOD writes lora) -> combined keeps BOTH gains,
    positive cross-transfer, no interference (capstone).
- **What FAILED (refuted multi-seed):** rtg+advnorm and the residual-mask-as-loss-gate (deepened the collapse);
  RWBC for plan-OOD games in every variant.
- **Eval that's trustworthy:** Δ_plan = (plan-advance − null-advance) screen_x from FIXED demo save-states
  (low variance). Raw post-adapt env reward is variance-dominated.

## NEW DATA (integrated tonight)
- **MegaManX-Snes** (2 demos, eval-ready: experimental integration xpos/health/lives). A NEW side-scroller —
  plan-OOD-like (gaps, enemies, decisions). Supervised demo-fit RUNNING NOW (3 seeds) -> results appended.
- **SuperMarioAllStars-Snes / SMB1** (30 demos, ~7700 chunks through 7-4). A NEW obstacle/plan-OOD platformer.
  Training-ready; eval integration (x-progress RAM addr) is PROVING DIFFICULT (SMB1 x is screen-relative
  page+offset; single-byte scan gave a false 0 candidate). Training works without it.
- Death-tails trimmed (docs/demos/demo_trim.json): SMB #3/#5/#30, MMX #1, SMW 105913/110517 (owner-flagged;
  npz has no death signal so this is a hand-maintained list).

## The questions for Round 8 (be concrete; the orchestrator runs your suggestions tonight)

**Q1 — Generalization predictions.** Given MMX and SMB1 are both right-running obstacle platformers (plan-OOD-
like, similar to SMW), predict: does the SUPERVISED plan-head demo-fit recipe widen Δ_plan for them too? What
would FALSIFY "the recipe generalizes" (e.g. MMX Δ_plan flat/negative across seeds)? State the threshold.

**Q2 — SMB1 eval without a clean x-address.** The x-progress RAM address is screen-relative/elusive. Options:
(a) find the page+offset pair (SMB1 world-x = 256*page + x_in_page) via a 2-byte monotone scan; (b) use the
demo's OWN frames as the eval (compare policy rollout from a save-state to the demo's continuation by frame/
screen-scroll optical flow); (c) use score/coins/time RAM (monotone but confounded); (d) skip SMB1 eval, use
it for TRAINING only and eval the recipe on MMX (which has a clean integration). Recommend ONE for tonight +
the fallback.

**Q3 — The generalist as games scale.** We now have ~5 games (SMW, Sonic, MMX, SMB1, +Minish/FE). The merge so
far is 2 games (SMW plan-head + Sonic lora). As games scale: do plan-head deltas from MULTIPLE plan-OOD games
(SMW+MMX+SMB1) merge cleanly (all write plan_head — do they CONFLICT, unlike the disjoint SMW/Sonic case)?
Propose how to combine multiple same-mode deltas (average? sequential fit on pooled demos? one demo-fit on ALL
plan-OOD demos at once?). Pick the one to test tonight.

**Q4 — Prep for 3D games (morning).** 3D games (Mario64/OoT-style) break assumptions: no "move right = progress",
camera-relative control, 3D navigation. What in the current recipe TRANSFERS (the supervised plan-head demo-fit
is reward-free and game-agnostic — it just fits the plan-head to human actions) vs BREAKS (the screen_x eval,
the "correct plan" being directional, the markov frame assumption)? What's the minimal adaptation? Keep it
short — this is forward-planning, not tonight's run.

**Q5 — The ONE highest-value experiment tonight** beyond MMX (already running). Given the validated recipe +
the new data, what most advances "the setup works AND generalizes"? (Candidate: a SINGLE pooled demo-fit on ALL
plan-OOD demos (SMW+MMX+SMB1 trimmed) -> one plan-head delta -> eval Δ_plan on SMW & MMX; tests whether one
supervised objective over pooled obstacle-game demos beats per-game fits = the true generalist. Name the metric
+ pass number.)

## Constraints
ONE model (training modes ok, not per-game weights); frozen-VLM-first; exact null-invariance; recoverable
(additive flags/new files); NO commits; cite primary sources. Be decisive — the orchestrator implements tonight.

## Mechanics
GPT-5.5 -> ARCH_WARROOM_gpt55_r8.md; Opus-4.8 -> ARCH_WARROOM_opus_r8.md (absolute paths in each agent's
instructions). Orchestrator injects MMX results, referees, runs the agreed experiment, iterates.
