# `docs/research/` — research log index

The full research record for the plan-conditioning project, persisted out of the (ephemeral) agent session dir.
**Start with the synthesis docs**, then dig into the war-room rounds and lit reviews as needed.

## ⭐ Read first — synthesis & results
| file | what |
|---|---|
| **`DISCRIMINATOR_RESULTS.md`** | THE running results log: plan-OOD vs actor-OOD dissociation, demo-fit pivot, pooled generalist, KL-anchor (R9), staleness (R11), maneuver-router. Multi-seed + CIs. |
| **`FORWARD_S2_TO_S1.md`** | the System-2 → System-1 knowledge-transfer forward report (R13): evocation-vs-addition, "persistent evocation → instinct," the consolidation mechanism, CONSOLIDATE-DUCK experiment. |
| `VERIFY_SMW.md`, `VERIFY_CROSSGAME.md`, `ORCHESTRATOR_VERIFY.md` | RAM-free VLM-verified demo results (how far each model got vs base; RAM-vs-VLM disagreement findings). |
| `MORNING_BRIEF*.md` | end-of-session briefs (chronological). |

## 🏛️ War-room (architecture debate, GPT-5.5 ⨉ Opus-4.8)
- `ARCH_WARROOM.md` — the consolidated log, **R1–R13** (read this for the narrative).
- `ARCH_WARROOM_gpt55_r{1..13}.md`, `ARCH_WARROOM_opus_r{1..13}.md` — per-round positions from each model.
- `ARCH_WARROOM_R{3..13}_SEED.md` — the framing seed for each round.
- Round map: R3 actor-OOD vs plan-OOD · R4 dissociation · R6 recipe ablation · R7 supervised pivot · R8
  generalization/new games · R9 ducking-collapse + KL-anchor · R10 short/long path · R11 staleness/TTT ·
  R12 gold-action plans · **R13 S2→S1 knowledge transfer** (the latest).

## 📚 Literature reviews (arXiv-verified)
| file | topic |
|---|---|
| `LIT_R13_S2TOS1.md` | options/HRL, VOYAGER, fast-weights, RT-H, **skill-token codebooks (LISA/PRISE)**, ExIt — for the recursion. |
| `LIT_R11_ttt.md`, `LIT_R11_distill.md`, `LIT_R11_fastweights.md` | test-time training, distillation, fast-weights. |
| `ARCH_RESEARCH.md` | dual-rate VLA survey (Helix, GR00T, π0). |

## 🧠 Design & RL
`RL_DESIGN.md`, `RL_DESIGN_V2.md`, `DDPO_DESIGN.md`, `RL_PROMPT_FORMAT.md`, `GROUNDED_PROMPT_FORMAT.md`,
`GENRE_SPECTRUM_AND_RL.md`, `FEASIBILITY_JOINT_AND_JUDGE.md`.

## 🔎 Findings & eval methodology
`IDM_GATE_FINDINGS.md` (IDM feasibility — failed gate), `YT_IDM_FINDINGS.md` (YouTube pseudo-labeling),
`EVAL_PLAN.md`, `EVAL_CONTEXT.md`, `BENCH_RESULTS.md`, `ARCH_EXPERIMENTS.md`.

---
*These were working research notes; numbers are superseded where a later doc says so (e.g. RTG "fix" was
retracted — see DISCRIMINATOR_RESULTS). The authoritative current state is the root `README.md` +
`docs/TRAINING_RECIPE.md` + `docs/HANDOFF_2026-06-30.md`.*
