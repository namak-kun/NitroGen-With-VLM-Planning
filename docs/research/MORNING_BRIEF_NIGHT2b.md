# MORNING BRIEF — night-2 (2026-06-28, new games + generalization)

You added MMX + SMB All-Stars demos and restarted the night mandate. Here's what happened.

## TL;DR — the setup WORKS, GENERALIZES to new games, and MERGES into ONE model
| result | number |
|--------|--------|
| **MMX** (NEW game) supervised plan-head demo-fit | Δ_plan +34→+74, **3/3 seeds, +117%**, CI [+23,+54] |
| **POOLED generalist** (1 plan-head fit on SMW+MMX+SMB1) | SMW +26 (1.3× per-game), MMX +31, **3/3 seeds**, CIs>0, null 0.0 |
| **FULL one-generalist** (pooled plan-head + Sonic lora, 4 games) | SMW +29→+52, MMX +40→+70, Sonic +72→+321, **all retained, no interference** |

The R7 recipe (supervised plan-head demo-fit for plan-OOD games) **generalizes out of the box to a new game
(MMX)** and **scales via one pooled fit** across SMW+MMX+SMB1. Merged with Sonic's RWBC lora = one model that
improves all 4 games. Null-invariance exact throughout. This is the "setup works AND generalizes" deliverable.

## New games integrated (additive, recoverable)
- **MegaManX-Snes** (2 demos): eval-ready via the shipped experimental integration (xpos/health). Wired into
  make_env/GAME_CFG/BATTERY. The recipe works on it.
- **SuperMarioAllStars-Snes / SMB1** (30 demos, ~7700 chunks through 7-4): TRAINING-ready (pooled in). Eval
  integration BLOCKED — couldn't cleanly locate SMB1 All-Stars world-x RAM address (1-byte and 2-byte scans
  from demo states failed; needs a reference RAM map). Per both war-room models, SMB1 is **train-only** and we
  eval generalization on MMX (clean) — which passed. SMB1 eval is the one open item.
- **Death-tail trims** (docs/demos/demo_trim.json, YOUR list — the auto-heuristic was unreliable): SMB #3/#5/#30,
  MMX #1, SMW 105913/110517. Applied before training.

## War-room R8 (both agents, in the loop)
Both GPT-5.5 + Opus predicted generalization + "one pooled fit is the right same-mode scaler" — and both ran
true. They also gave the **3D-game plan** for your morning demos: the supervised demo-fit transfers wholesale
(it's reward-free, game-agnostic); only the EVAL changes (screen_x → 3D distance-to-waypoint / star-room count)
and the plan goes landmark-relative ("reach the door"). New N64 env on the emulator_env base; everything else
identical. Full debate in ARCH_WARROOM.md (R8 section) + ARCH_WARROOM_{gpt55,opus}_r8.md.


## CORRECTION: SMB1 x-address is NOT a blocker
It blocks nothing critical: SMB1 trains fine (uses demo.npz) and is already pooled into the generalist; generalization is validated on MMX (clean integration). The x-addr only gates SMB1 as an EXTRA eval game, and even that has alternatives (frame-scroll/optical-flow, or eval on SMW+MMX). De-prioritized.

## Open items (priority for morning)
1. **SMB1 x-address** — the only blocker. Options: a reference RAM map for SMB1 All-Stars, or probe in-game
   with a known landmark. Until then SMB1 is train-only (still contributes to the pooled generalist).
2. **3D demos** — harness is ready per the R8 plan; bring them and I'll wire a 3D progress var + landmark plans.
3. **Multi-seed the full merge** — the 4-game merge used 1 delta-seed each (n_starts=6, coarse). Average more
   seeds + (once SMB1 eval works) add SMB1 as an eval game.
4. **Reach_eval cross-check** — Δ_plan from fixed starts is low-variance but raw screen_x; cross-check the
   pooled generalist on the survival-weighted metric.

## Recoverability (nothing committed)
- New files: planner_poc/{pooled_demofit.py, full_merge_eval.py, combine_eval.py, narration_residual*.py,
  demo_narration.py, smbas_find_progress{,2,3}.py}, nitrogen/eval/envs/new_demo_envs.py.
- Edited (additive): planner_poc/demo_bc.py (new games + trim + Δ_plan eval + residual flags),
  rwbc_actor_adapt.py (mmx/smbas make_env + reward/seed/residual flags), plan_graded_test.py (mmx/smbas BATTERY).
- Config: docs/demos/demo_trim.json (death tails). Deltas: files/{pooled_planfit_s*, n2_mmx_planfit_s*, r7*}.pt.
- Drivers/logs: /tmp/run_n2_mmx.sh, /tmp/n2/*. Results: files/DISCRIMINATOR_RESULTS.md, files/full_merge_eval.json.
- GPUs idle, no stray procs. No git commits.

## Memories stored (night-2)
- New demo dataset (MMX + SMB All-Stars) + eval-readiness.
- Death-tail trim list + that npz has no death signal (hand-maintained).
- The recipe generalizes + scales via one pooled plan-head fit (the generalist training recipe).
