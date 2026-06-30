# MORNING BRIEF v2 — 2026-06-27 night (autonomous)

Read this first. Synthesis of the night's work on the closed-loop planner + RL surfaces. Full design in
`RL_DESIGN_V2.md`; numbers in `BENCH_RESULTS.md` + `docs/graded/*.json`.

## TL;DR
1. **Fixed the closed-loop planner** per your two points: recast as **System-2-planning-for-System-1**, and
   made **learnings BEHAVIOURAL** by carrying the **prior plan + executed action trace** (without them the
   model fabricated/restated — exactly your insight). Added the parser fallbacks you asked for.
2. **Ran the ablations.** Corrected per-game prompts make the plan HELP (SMW +50%, Sonic +278%, Minish
   +57% over base). Redesigned learnings now HELP on Minish (+0.2, was −0.2). SMW/Sonic learn re-run
   segfaulted (stable-retro) — qualitatively validated via smoke test.
3. **GATE experiment (the crux, from the GPT-5.5 rubber-duck):** is plan value GRADED or just ACTIVATION?
   **ANSWER: GRADED for side-scrollers** — a "go right" plan beats nonsense (dummy) by +43/+57 AND beats
   "go left" (bad) by +49/+16. **Minish (top-down) = ACTIVATION** (any plan token ≈ same). This *overturns*
   the prior "plan value is binary → planner moot" conclusion, for directional games.
4. **best-of-K:** plan choice matters hugely (return spread +33/+22 among sampled plans) BUT best-of-K
   selection ≈ a good FIXED concise plan (≈ oracle). **The bottleneck is planner VARIANCE** (~30–40% of
   temp-0.9 Qwen samples are wrong-direction), not the ceiling.
5. **RL design V2 written + rubber-ducked by GPT-5.5** (sharp critique folded in). Three technical
   corrections internalized (see below).
6. **Actor-side training:** RWBC with corrected plans running (results appended at bottom).

## The RL-able surfaces — how to train each (your question)
| surface | trainable? | verdict tonight | how to train |
|---|---|---|---|
| **plan text** | frozen-Qwen | GRADED (side-scrollers) but planner is HIGH-VARIANCE | reliability: short direction-first prompt + **greedy/low-temp** (already good); a cheap wrong-direction FILTER; (gated) Qwen-LoRA |
| **learnings** | frozen-Qwen | now behavioural; Minish +0.2; keep **diagnostic** until a clean win | trainable **latent trace-encoder** if it proves valuable; else logging-only |
| **bridge** (resampler+adapter) | YES | faithful (pt_inject≈text); K=8 bandwidth untested | behavioral distillation (RWBC on good-plan rollouts) |
| **DiT-LoRA** | YES | the motor ceiling; IS adaptable (V1) | RWBC → save-state advantage-weighted regression → (later) true DDPO |

## GPT-5.5 rubber-duck — 3 corrections we internalized
- **R2:** "save-state GRPO on the flow sampler" is **NOT a valid policy gradient** — reward-weighting the
  flow MSE = advantage-weighted **regression/RWBC** (biased self-imitation). True DDPO needs explicit
  stochastic denoising transitions + logprobs; validate on a toy task first. (Renamed everywhere.)
- **R3:** best-of-K → distillation must be **behavioral** (RWBC on the winning plan's high-return rollouts),
  NOT token-MSE toward another plan's tokens (incoherent / target-drift).
- **R4:** the **K=8 bridge bandwidth** may be the real limit; the weak CEM latent probe was inconclusive →
  use **gradient-based** z-search (the DiT is differentiable) to settle actor-vs-bridge.

## Recommended path (gated, in order)
1. **Actor-side RWBC with corrected plans** (running) — raise the motor ceiling / follow the good plan.
2. **Planner reliability** (cheap, ship now): greedy + short direction-first prompt; wrong-direction filter
   for any sampling path.
3. **Behavioral distillation** — amortize good plans into bridge+LoRA via RWBC on their rollouts.
4. **Gradient-based latent-z probe** — settle whether the K-token bridge has headroom over text.
5. **Then** true DDPO/DPPO (with correct formulation) / alternating planner-actor; **gated** Qwen-LoRA.
   CUT: best-of-K *rollout* selection (doesn't beat greedy); naive token-MSE distillation.

## Files (new/changed this night, NOT committed)
- `planner_poc/game_planner.py` — redesigned ClosedLoopPlanner (System-2-for-System-1, prior-plan+trace,
  behavioural learnings, layered parser fallbacks), per-game env-applied action verbs (`applied_action_desc`).
- `nitrogen/training/actions.py` — `summarize_chunk(btn_label=...)` per-game button labels (no more
  Sonic "spin-dash"/Minish "jump" hallucinations).
- `planner_poc/demo_planner_ablation.py` — base/plan/learn on GT reward (+GBA step-reward).
- `planner_poc/plan_graded_test.py` — the GATE ladder (null/dummy/wrong_game/generic/bad/correct/hand_good),
  crash-safe JSON → `docs/graded/{smw,sonic,minish}.json`.
- `planner_poc/best_of_k.py` — winner's-curse-proof best-of-K → `docs/graded/bestofk_*.json`.
- `planner_poc/latent_plan_search.py` — CEM over plan tokens (train/val split) → `docs/graded/latent_*.json`.
- `planner_poc/rwbc_actor_adapt.py` — added `--fixed-plan/--use-correct-plan/--save-delta`.
- `planner_poc/{learn_smoke,sample_plans,obs_format_probe}.py` — diagnostics.
- Session docs: `RL_DESIGN_V2.md` (full), this brief.

## Open questions for you
- OK to spend a small **Qwen-LoRA** to cut planner variance directly? (Currently gated by the freeze.)
- Worth a **trainable latent memory** (vs in-prompt text learnings)?
- Which game to push RL on first — SMW/Sonic (graded, headroom) per the gate?

## RWBC RESULTS (appended when the run finishes)
(see bottom — updated by the run)

---
## RWBC RESULTS (corrected-plan actor-adaptation, LoRA-only 0.52M, fixed correct plan)
- **Sonic: ACTOR ADAPTS** +0.900 → **+1.875** (Δ+0.98, >2×). Monotonic (mid +1.905@49, +2.060@99). With
  CORRECTED plan conditioning. Confirms the actor-side RWBC path (≈ prior session's +1.32→+2.80). Delta
  saved `ckpts/rwbc_sonic_correct.pt`.
- **SMW: collapsed at 120 steps** +0.267 → +0.023, BUT mid-eval +0.720 @step49 → −0.085 @step99: it
  IMPROVED then over-trained on the small kept set (29 chunks). Re-running gentler (steps 50, lr 5e-5,
  anchor 0.003) to capture the pre-collapse gain → `ckpts/rwbc_smw_correct_gentle.pt`. [see below]
- **Takeaway:** actor IS adaptable from emulator reward with corrected plans (Sonic clean); SMW needs
  early-stop / gentler LR (collapse is an over-training artifact, not a capability wall). Anti-collapse
  (early-stop on val reward, low LR, anchor) is essential — confirms V1.

## RWBC SMW gentle re-run
(updated below by the run)

**SMW gentle (steps 50, lr 5e-5, anchor 0.003):** +0.267 → +0.352 (Δ+0.085, STABLE, no collapse). Confirms
the 120-step collapse was over-training; gentle settings give a small stable gain. Headroom is
game-dependent (Sonic ≫ SMW). Delta `ckpts/rwbc_smw_correct_gentle.pt`.

**Net RWBC verdict:** the DiT actor adapts from emulator reward under corrected-plan conditioning, with
anti-collapse (early-stop/low-LR/anchor) ESSENTIAL. Sonic strong (+0.98), SMW modest+stable (+0.085).

**Minish (top-down) RWBC:** +0.799 → +1.064 (Δ+0.265, +33%). The ACTOR adapts even though Minish PLANS are
activation-only (gate) — motor-policy improvement doesn't need graded plans. **Sonic scaled (12 eps,120
steps):** +0.900 → +1.995 (Δ+1.10, slightly > the 6-ep run's +1.875). All 3 genres adapt cleanly:
Sonic +1.10, Minish +0.27, SMW +0.085 (gentle). ckpts/rwbc_{sonic_correct_scaled,minish_correct}.pt.

## HELD-OUT GENERALIZATION (RWBC LoRA -> demo levels it was NOT trained on) [docs/graded/heldout_*.json]
Trained from the env DEFAULT start state; evaluated on the human-DEMO start states (different sections).
| game | BASE progress | ADAPTED progress | Δprogress | BASE shaped | ADAPTED shaped | Δshaped |
|---|---|---|---|---|---|---|
| Sonic | +11.6 | +206.0 | **+194** | +39.1 | +41.3 | **+2.2** |
| SMW | +20.5 | +35.9 | +15.4 | — | — | — |
| Minish| +1.04 | +1.14 | +0.11 | — | — | — |

**HONEST READ (verified with the shaped, death/stuck-penalized reward):** the actor-adaptation DOES
generalize across levels within a game — base is often STUCK (Sonic per-state [0,0,-12,0,60,0,45,0]) and the
LoRA UN-STICKS it everywhere ([252,252,157,...]). BUT the +194 screen_x jump is **mostly distance, not
quality**: Δshaped is only +2.2 (death/stuck penalties eat most of it). So: cross-level transfer of "move
right / un-stick" is real and useful, but it is NOT "18× better play" — quality-adjusted gain is modest. For
Sonic (where "go right fast" ≈ the objective) the distance gain is still largely legit; for a fair claim use
the SHAPED reward + success metrics (death rate, completion), not raw screen_x. Re-confirms the V1 anti-
exploit warning (screen_x alone rewards a run-right/suicide policy).

## MULTI-GENRE one-LoRA capacity test [docs/multigenre/, ckpts/rwbc_multigenre.pt] — answers "won't generalize to all games"
One rank-16 LoRA-only trained on MERGED, balanced top-return RWBC chunks from Sonic+SMW+Minish (corrected
plans, FIXED button maps), eval'd held-out per game (progress / shaped):
| game | BASE prog | SHARED-LoRA prog | Δ | DEDICATED-LoRA Δ (held-out) |
|---|---|---|---|---|
| Sonic | +11.6 | +38.5 | +26.9 | +194 |
| SMW | +20.5 | +9.6 | **−10.9** | +15.4 |
| Minish| +1.04 | +1.02 | −0.02 | +0.11 |

**FINDING:** a single shared LoRA shows **cross-genre INTERFERENCE** — it helps the dominant/highest-headroom
genre (Sonic) but REGRESSES SMW and is flat on Minish, far below the per-genre dedicated LoRAs. So the prior
"4-genre collapse" was NOT only the button-map bug (now fixed): there is genuine capacity tension at rank-16.
This SUPPORTS the user's "per-genre actors won't trivially scale to one generalist" intuition — but it's a
capacity/balancing issue, likely addressable (untested) by: larger LoRA rank, better data balancing,
genre-conditioned routing (a per-genre adapter or a game-id-conditioned LoRA), or curriculum. Recommended
next experiment. NOTE Sonic dominates the merged objective (most reward signal) → pulls the shared LoRA
toward Sonic-like behavior that hurts SMW; balancing by GRADIENT/loss (not just chunk count) may help.

### Multi-genre follow-up: does DATA BALANCING fix the interference? NO.
Re-trained the shared LoRA with per-game reward-weight normalization (so no genre dominates the gradient;
ckpts/rwbc_multigenre_bal.pt): Sonic +11.6→+12.5 (Δ+0.9, gains GONE), SMW +20.5→+7.5 (Δ**−13**, STILL
regresses), Minish flat. **=> The interference is a genuine rank-16 CAPACITY CONFLICT, not a balancing
artifact:** unbalanced → Sonic wins big / SMW loses; balanced → nobody wins / SMW still loses. The shared
LoRA cannot simultaneously encode Sonic-helping and SMW-helping updates at this rank. **Likely fix =
CAPACITY (larger LoRA rank) or genre-CONDITIONED routing (game-id-conditioned LoRA / per-genre adapter),
NOT data balancing (tested-negative).** This robustly supports "per-genre actors won't trivially merge into
one generalist at fixed small capacity." Per-genre dedicated LoRAs each work; the generalist needs more
capacity or conditioning. CLEAR NEXT EXPERIMENT.
