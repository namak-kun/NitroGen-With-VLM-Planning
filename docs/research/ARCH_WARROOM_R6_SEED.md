# ARCH WARROOM — Round 6 seed: converge on THE training recipe that makes plan-conditioning work

The owner is away for the night and asked us to "get the setup to WORK as best as you can, recoverable."
This round is CONSTRUCTIVE + CONVERGENT: pin down the MINIMAL, RUNNABLE training recipe (one generalist) that
robustly makes plan-conditioning help where it should, and validate it multi-seed. Then react to a fresh
weak experiment result. Latency stays OFF the table.

## Everything established this session (don't re-litigate; build on it)
- **R3:** subtle drift on a trained game = DiT-actor OOD (BC covariate shift). Plan-OOD is secondary/intermittent.
- **R4 discriminator (DATA):** GAME-DEPENDENT DISSOCIATION. SMW = PLAN-OOD (oracle plan Δ_plan=+0.20, lora-only
  actor-adapt ≈0). Sonic = ACTOR-OOD (Δ_actor=+1.9, plan redundant/slightly hurts). So NO global --lora-only.
- **RTG / reward:** local per-chunk screen_x reward collapses obstacle games (death penalty lands only on the
  death chunk). Added --reward-mode rtg (return-to-go, gamma 0.95). BUT multi-seed showed SMW actor-adaptation
  (either reward mode) is SEED-VARIANCE-DOMINATED: RTG {+0.36,-0.21,-0.25,-0.18,-0.04} mean -0.06; local
  {-0.13,+0.58,+0.70,+0.27,+0.69} mean +0.42. The single-seed "RTG fixes it" was RETRACTED. NOTE: the low-
  variance signal is the NO-ADAPT Δ_plan (R_B oracle plan std ~0.06); the POST-ADAPT eval is what's noisy.
- **R5 attribution CONSENSUS (you both):** attribute ACTIONS not WORDS; frame-counterfactual residual
  (‖a_real − base_DiT_null‖) = the operational type-A definition AND the training mask AND the contamination
  detector; residual-weighted flow-BC respects the verified near-zero-plan-gradient; timescale (not semantic)
  split. The residual MASK = the type-A fraction → auto-reproduces SMW-vs-Sonic with no per-game knob.
- **R5 EXPERIMENT (fresh, WEAK):** narration_residual.py with a DIRECTION-ONLY residual gave SMW f_A 0.268 vs
  Sonic 0.247 (1.09×, below the 1.3× confirm bar); mean_rho_dir 1.24×. CONFOUND: the base DiT's left/right is
  KNOWN-WEAK (jlx~0.5 neutral/dithers, corr 0.49), so direction-disagreement is universal, not SMW-specific.
  Per-line type-B (reflex) end is CLEAN; type-A end polluted by multi-intent lines + the owner's ±1-2s timing
  noise. => need the FULL-ACTION residual (per-dim 1−AUC / buttons+sticks+jump) or CFG ‖v_c−v_u‖ authority.
  (A full-action re-run is being done in parallel; its result will be appended before you finalize — design
  your recipe to NOT depend on its sign, but say what each outcome implies.)

## Repo levers you can assume exist (verified)
- `rwbc_actor_adapt.py`: on-policy RWBC. Flags: --use-correct-plan, --lora-only (freeze plan-head),
  --reward-mode {local,rtg}, --gamma, --seed-offset, --anchor (L2-to-init KL proxy), --explore-sigma,
  --plan-temp, --top-frac, --save-delta. collect()/eval_reward() reuse one env (segfault-safe). Metric =
  native env reward (screen_x + death/stuck shaping; SMW life_var="lives", death_penalty 5.0).
- base_dit_perdim.py: per-dim base-DiT fidelity (buttons AUC, sticks corr) from the frame alone.
- eval_policy.py `_sample_chunk(frame, plan, w, null=?, noise_sigma=?, noise_seed=?)`: CFG v_c/v_u available;
  SDE sampling for K-disagreement.
- planner.py: encode_multimodal(text_only=True) decode-free objective; generate_plan; distill_weight
  (InfoNCE/CLIP to de-collinearize K plan tokens); masked-null (exact null-invariance).
- emulator save/load (snes_env, mgba_env) for on-policy/save-state data.
- demos: SMW(8)+Sonic(4)+FireEmblem(2)+Minish(1), 60fps, demo.npz actions+obs, frame-aligned narration.json.
- reach_eval.py: survival-weighted, scripted-RIGHT-ceiling metric (LESS exploitable than raw screen_x).

## The questions for Round 6 (CONVERGE — this should end in a single agreed recipe + ablation list)

**Q1 — THE RECIPE.** Write the single minimal training recipe (one generalist, frozen-VLM-first) that should
make plan-conditioning robustly help SMW (plan-OOD) without collapse AND not hurt Sonic (actor-OOD). Specify
EXACTLY, in terms of the existing flags + any small additions:
  - which params train (plan-head+adapter+LoRA? when),
  - reward (rtg? multi-component? reach_eval survival-weighted? per-game advantage-norm/GRPO standardization?),
  - the RESIDUAL-WEIGHTED plan loss (how to compute the mask cheaply in-loop; where it multiplies),
  - data (fixed correct plan vs VLM-generated vs frame-counterfactual-filtered demo spans),
  - seeds/eval (how many seeds, which metric, what counts as "works").

**Q2 — Tame the seed variance** (the actual blocker). SMW post-adapt eval Δ has std ~0.4. What concretely
reduces it: more eval episodes? paired seeds + bootstrap CIs? lower LR / --anchor KL? eval on fixed save-state
starts (snes_env) instead of relaunch? freezing the plan path and only measuring Δ_plan (low-variance)? Give
the variance-reduction plan so tonight's numbers are TRUSTWORTHY.

**Q3 — The minimal ABLATION that proves the recipe** (must run on existing harness tonight, multi-seed). e.g.
{no-plan baseline} vs {fixed-correct-plan, no-adapt} vs {plan + residual-weighted adapt} vs {plan + naive
adapt} — on SMW and Sonic, >=5 seeds, reach_eval survival-weighted metric. Name the PASS criterion (the number
that means "the setup works").

**Q4 — FE / Minish: do we even need the owner's plans?** He offered PLAN-LEVEL (not per-action) objectives for
Fire Emblem + Minish Cap but isn't sure they're needed. Given the consensus (VLM re-abstraction generates
grounded objectives for free, esp. on games in Qwen's prior), state whether to (a) ask him to write them,
(b) generate them from the frozen VLM and validate via the residual/reward, or (c) skip FE/Minish tonight.
Recommend, with the cheap check that decides it.

**Q5 — Use of data tonight** (cookies updated → YouTube available; emulator available). Is any NEW data worth
pulling tonight to make the setup work, or is the existing demo+emulator substrate enough? If YouTube/IDM is
worth it, give the smallest slice that helps; if not, say skip and why.

## Constraints
ONE generalist (no per-game/per-genre models); frozen-VLM-FIRST (VLM-LoRA allowed, cost-flagged); EXACT
null-invariance preserved; NO git commits; recoverable (additive flags / new files only); cite primary sources.

## Mechanics
GPT-5.5 → `ARCH_WARROOM_gpt55_r6.md`; Opus-4.8 → `ARCH_WARROOM_opus_r6.md` (absolute paths in each agent's
instructions). The orchestrator will inject the full-action residual result, then run R6b convergence + start
implementing the agreed recipe. CONVERGE on ONE recipe — the orchestrator will build and run it tonight.
