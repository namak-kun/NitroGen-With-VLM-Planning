# ARCH WARROOM R8 — Opus-4.8: the recipe generalizes (predicted), and it scales as ONE pooled objective

**Frame.** The validated R7 recipe is `plan-OOD → supervised plan-head-only demo-fit (correct plan, no
reward)` (SMW Δ_plan +35.7→+56.0, +57%, 4/5 seeds, bootstrap CI [+2.0,+38.7]>0, null-invariant 0.0) ╳
`actor-OOD → plain local RWBC` (Sonic +1.44) ╳ `merge = disjoint params` (capstone: SMW Δ_plan +30→+71,
Sonic +119→+333, positive cross-transfer). R8 asks: does it hold on MMX/SMB1, and how do same-mode deltas
scale. My headline: **the supervised fit is reward-free and game-agnostic, so it generalizes; and because
the plan-head is plan-*conditioned*, the right scaler is ONE pooled fit, not averaging or sequential.**

| Q | Decision |
|---|---|
| Q1 | MMX/SMB1 widen Δ_plan (same plan-OOD structure as SMW). Falsify if mean change ≤0 **or** bootstrap CI∋0. |
| Q2 | **(a)** adjacent 2-byte `<u2` monotone scan, CV'd across ≥3 demos → `smbas_progress.json`. Fallback **(d)**. |
| Q3 | **One pooled demo-fit on all plan-OOD demos** (plan text is the disambiguator). Avg = fallback; sequential = rejected. |
| Q4 | Objective TRANSFERS wholesale; screen_x eval + directional plan + markov frame BREAK. Swap to goal-relative. |
| Q5 | Pooled SMW+MMX+SMB1 plan-head fit → eval Δ_plan on SMW & MMX. Pass = pooled ≥0.8× per-game, CI>0, null 0.0. |

## Q1 — Generalization predictions + falsification threshold

**Prediction: YES, both widen.** MMX and SMB1 are right-running obstacle platformers with gaps, enemies,
and timing decisions — the *same* structure that makes SMW plan-OOD (the frame underdetermines the
expert's obstacle-timing action; the plan supplies it). The recipe fits the plan-head to a **trusted
human target** with no rollout and no reward, so none of the variance sources that voided RWBC apply
(Efron-bootstrap caught those as single-seed artifacts). Nothing in the objective is SMW-specific.

**Caveats that shape the threshold.** (i) MMX has only **2 demos** → eval runs from ~2 start states →
wide CIs; a "weak" verdict may be *under-power*, not failure. (ii) base DiT already reproduces ~81–83% of
the human semantic action (`narration_residual2`) ⇒ the type-A residual is small ⇒ expect a *modest*
absolute gain. (iii) **Absolute Δ_plan is NOT comparable across games** (MMX `xpos` units ≠ SMW
`screen_x`); judge by **sign + relative % + bootstrap CI**, never raw magnitude.

**Falsification threshold (per game, ≥3 seeds, change = POST−PRE Δ_plan):**
- **GENERALIZES:** widens in ≥⅔ seeds, mean relative change **≥ +20%** (SMW was +57%), bootstrap 95% CI on
  the change **> 0**, null path |change| ≈ 0 (plan-head-only ⇒ exact 0.0; any null drift = harness bug),
  no collapse seed.
- **FALSIFIES "recipe generalizes":** mean change **≤ 0**, OR widens in ≤⅓ seeds, OR **CI includes 0**
  (indistinguishable from noise ⇒ SMW-specific). Borderline +0–20%/CI-touches-0 ⇒ run 5 seeds before a call.

## Q2 — SMB1 eval without a clean x-address

**Pick (a): adjacent 2-byte page+offset monotone scan, cross-validated.** It is the only option that yields
the *same* trustworthy `screen_x`-style Δ_plan the whole session is built on, and we have **30 SMB demos**
= rich RAM traces to scan. Procedure (deterministic, ~minutes, one new file `smbas_find_progress2.py`):
replay each demo's `demo.npz` actions from `initial.state` dumping full RAM per row; for every **adjacent
(lo,hi)** byte pair compute `v = 256*hi + lo` and rank by *monotone-non-decreasing fraction* **within a
single level segment** (segment on large drops to tolerate pipe/level resets — this is what killed the
single-byte "false 0"); require the **same** pair to win across ≥3 demos. Adjacency matters: a winning
low-byte address `A` with little-endian `<u2` reads `256*hi+lo` for free, so the existing hook works with
**zero env code change** — write `tmp/retro_data/smbas_progress.json = {"prog_addr": A, "prog_type":"<u2"}`
(read by `make_env('smbas')` → `make_smbas`).

**Fallback (d): SMB1 train-only tonight.** If no adjacent pair survives CV, pool SMB1 into the fit (its
30 demos are the richest *training* signal) and **eval the generalization claim on MMX** (clean `xpos`).
This de-risks the night and dovetails with Q5 — the SMB1 x-address blocker never gates the headline.

**Reject (b)** optical-flow (a new, noisy, un-auditable metric inconsistent with screen_x Δ_plan) and
**(c)** score/coins/time (confounded — coins/score don't track x; time counts *down*; red-team bait).

## Q3 — Scaling same-mode deltas: ONE pooled fit

SMW+MMX+SMB1 all write `plan_head.*`, so independently-fit deltas can collide in weight space. Three
combiners: **(1) average** (model-soup), **(2) sequential** continual-fit, **(3) one pooled fit on all
plan-OOD demos.** **Pick (3).**

**Why pooled is the principled generalist (my distinctive angle).** The plan-head is **plan-conditioned**,
not game-indexed. SMW "jump over pits and enemies", MMX "jump gaps and shoot enemies", SMB1 "jump over
pits and enemies" select **overlapping** action distributions via **overlapping plan text**. A single
ERM/BC minimizer over the pooled `(frame, plan, action)` triples (the DAgger/BC objective, 1011.0686)
learns **one** plan→action function the three games *share in mechanism and select by text* — the plan
string is the disambiguator, so there is no multi-task head conflict to resolve. This is the residual-mask
intuition **resurrected at the dataset level**: frame-determined chunks common to all three (sprint-right
straightaways) agree across games and **self-attenuate** (small per-dim residual / mutual agreement),
while the game-specific obstacle-timing dims are exactly where each correct plan carries gradient.
**Averaging (1)** linearly combines deltas that were never trained together (destructive-interference
risk) → keep as the **zero-cost fallback** if pooled shows conflict. **Sequential (2)** is rejected:
catastrophic forgetting (SMB1's 7700 chunks overwrite SMW's 8 demos). Tonight run **pooled PLAIN**
demo-fit (residual-as-weight precompute is too slow over thousands of chunks — cap it later); the plain
fit is the proven-working headline.

## Q4 — 3D prep (Mario64/OoT) — brief

**TRANSFERS (no change):** the supervised plan-head-only demo-fit — it is reward-free and only needs
`(frame, plan, human-action)` + the frozen VLM/DiT bridge; **null-invariance** (architectural); the
disjoint-merge and the measure-Δ_plan-vs-Δ_actor routing methodology. Camera-relative sticks are already
encoded in the human demo and learned frame-conditioned — no remap.
**BREAKS:** (i) `screen_x` eval — "right = progress" is false; swap `reward_var` to a **3D progress
scalar** (Euclidean/geodesic distance-to-waypoint, or star/heart/room count from RAM) on the same
save-state Δ_plan substrate. (ii) **Directional** plans → **goal/landmark-relative** ("reach the castle
door") — this *plays to* the frozen VLM (it names landmarks); content shifts, generator unchanged. (iii)
**Markov frame** — 3D is more partially-observable, so the plan channel becomes *more* load-bearing (more
plan-OOD = good for the thesis) but stresses the markov actor; let the plan tokens carry the non-markov
goal/memory. **Minimal adaptation:** new in-process N64 env (frame-exact save/load + a 3D progress var) on
the `emulator_env.py` base; everything else is identical.

## Q5 — The ONE experiment tonight (beyond MMX)

**Run the pooled generalist.** One plain plan-head demo-fit on **SMW+MMX+SMB1 trimmed demos**, each
conditioned on its own `BATTERY[g]['correct']`, LoRA+base frozen, null masked → **one** plan-head delta;
eval Δ_plan on **SMW** (`screen_x`) and **MMX** (`xpos`). This single run answers R8's whole question:
generalization (new games) **and** scaling (Q3 same-mode merge) **and** ships the artifact (one generalist
delta), reward-free (no RWBC variance), reusing existing harness. SMB1 is **train-only** (Q2 fallback) so
its address blocker doesn't gate it.

**Metric.** Δ_plan = (plan-advance − null-advance) from FIXED demo start states (low-variance), per game,
each on its correct plan; + null-invariance check (plan-head-only ⇒ null path bit-exact, max|diff| = 0.0).

**PASS (decisive, ≥3 seeds, bootstrap CIs — Efron 1979):**
1. **Retention:** pooled Δ_plan(SMW) ≥ **0.8 ×** per-game SMW (≈ **≥ +45**, and > PRE +35.7) **AND**
   pooled Δ_plan(MMX) ≥ 0.8 × per-game MMX (and > MMX PRE), each with bootstrap 95% CI on the change
   **> 0**. ⇒ *one objective generalizes across obstacle games.*
2. **Invariant:** null path max|diff| = **0.0** exact; delta additive/recoverable; VLM frozen throughout.
3. **No collapse:** no seed with POST Δ_plan below base.
**Secondary win to watch:** if MMX's *per-game* fit is weak (2 demos), pooling that still lifts MMX
Δ_plan > PRE is **positive cross-transfer** (SMW+SMB1 → MMX) — the capstone effect at the dataset level,
arguably a *stronger* generalist result than per-game parity. Partial outcome (SMW retained, MMX lost)
cleanly diagnoses "scaling needs averaging/plan-routing" → fall back to (1).

## Runnable

```bash
RUN='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=/home/t-nagupta/NitroGen-With-VLM-Planning:/home/t-nagupta/NitroGen-With-VLM-Planning/planner_poc QWEN=Qwen/Qwen3.5-2B'
# Q1 MMX per-game (if not already): make_env('mmx') xpos eval is clean
for s in 0 1 2; do $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python planner_poc/demo_bc.py \
  --game mmx --train plan_head --use-correct-plan --lr 2e-5 --steps 600 \
  --seed-offset $s --save-delta runs/r8/mmx_planfit_s$s.pt; done
# Q5 pooled generalist — minimal ADDITIVE new file pool_demo_fit.py:
#   for g in [smw,mmx,smbas]: load_demo_chunks(GAME_CFG[g]['demo_glob'], plan=BATTERY[g]['correct'])
#   concat+shuffle; requires_grad only plan_head.*; AdamW lr2e-5 steps~900 (more data); --save-delta one delta
for s in 0 1 2; do $RUN CUDA_VISIBLE_DEVICES=0 .venv/bin/python planner_poc/pool_demo_fit.py \
  --games smw mmx smbas --train plan_head --use-correct-plan --lr 2e-5 --steps 900 \
  --seed-offset $s --save-delta runs/r8/pool_planfit_s$s.pt; done
# Q5 eval: per-game eval_from_states on its correct plan (or extend combine_eval with an mmx cell)
$RUN .venv/bin/python planner_poc/combine_eval.py --smw-planhead runs/r8/pool_planfit_s0.pt \
  --sonic-lora runs/r8/pool_planfit_s0.pt   # repurpose as pooled-vs-base; add mmx cell
# Q2 fallback path needs nothing; (a) primary:
$RUN .venv/bin/python planner_poc/smbas_find_progress2.py   # writes tmp/retro_data/smbas_progress.json
```

## Citations
BC/DAgger covariate-shift & the no-regret online-IL upgrade (future env-in-loop expert relabel): Ross,
Gordon & Bagnell 2011, **arXiv:1011.0686**. Why we keep this mode *reward-free* (the KL-to-base trust
region these enforce is what RWBC lacked; reserve them for the Sonic actor-OOD mode + save-state RL):
AWR — Peng et al. 2019, **arXiv:1910.00177**; GRPO — Shao et al. 2024, **arXiv:2402.03300**. Significance
of every Δ_plan change (the test that caught the single-seed RTG artifact): bootstrap — **Efron 1979**.
Repo grounding: `demo_bc.py` (`load_demo_chunks`/`eval_from_states`/`--train plan_head --use-correct-plan`,
`_trim_seconds`), `new_demo_envs.py` (`make_mmx` xpos / `make_smbas` `<u2` hook), `combine_eval.py`
(disjoint-merge eval), `plan_graded_test.py` `BATTERY` (smw/mmx/smbas correct plans),
`rwbc_actor_adapt.build_batch` (residual-as-weight path:127–152). Constraints honored: ONE model (modes,
not per-game weights), frozen-VLM-first, exact null-invariance (plan-head-only), recoverable (additive
files/flags), no commits.
