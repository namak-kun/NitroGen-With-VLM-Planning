# ATLAS — navigate the entire project graph

One map of everything: the research arc, every architecture decision, the full experiment graph
(EXP-000..054 + war-room R1–R13 + Stage-3 demo-fit), the checkpoints, and where each discussion lives. Use this
as the index; follow the links to the primary docs.

**The 4 anchor docs** (read in this order to catch up):
1. [`../README.md`](../README.md) — architecture + training stages + status (the overview).
2. [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) — exact recipes, text plans, **losses (§9)**, **data structures (§10)**.
3. [`PAPER.md`](PAPER.md) — the full academic writeup (abstract → method → results → discussion → future).
4. [`HANDOFF_2026-06-30.md`](HANDOFF_2026-06-30.md) — infra catches + next steps.

Plus the navigators: [`INDEX.md`](INDEX.md) (capability→experiment→checkpoint), [`CHECKPOINTS.md`](CHECKPOINTS.md)
(per-checkpoint recipes), [`EXPERIMENTS.md`](EXPERIMENTS.md) (EXP-000..054 log),
[`research/README.md`](research/README.md) (war-room + lit index).

---

## 1. The research arc (the spine of the whole project)

```
 Q: can a frozen VLM (System-2) steer a frozen flow-matching DiT (System-1) to play games?
   │
 Stage-1  ENV-FREE SYNTHETIC ALIGNMENT  (EXP-000..032)
   │  prove the bridge can steer: direction, null-invariance, ordering, cross-chunk, nested
   │  KEY: contrastive loss de-collinearizes plan tokens (EXP-007 root cause → EXP-009/016 fix)
   ▼
 Stage-2  REAL PLANS, 2B BACKBONE  (EXP-033..054)
   │  transcripts are chit-chat (EXP-033) → VLM-on-frames plans (EXP-035) → but plans don't predict
   │  the action (EXP-036/037) → ACTION-CONDITIONED plans fix it (EXP-038) → alignment gap closed by
   │  outcome-contrastive (EXP-043) → distillation (EXP-045) → 2B redesign (EXP-050) → button steer (EXP-052)
   │  NEGATIVE: env-free counterfactual OVERRIDE needs envs (EXP-047/049/054)
   ▼
 Stage-3  HUMAN-DEMO FIT + the war-room  (R1..R13, DISCRIMINATOR_RESULTS)
   │  plan-OOD demo-fit works + pools + merges (R7/R8) ; ducking collapse = data, KL-anchor fixes it (R9)
   │  staleness is text-bound (R11) ; evocation vs addition (R11/R13) ; consolidation→instinct (R13)
   ▼
 NOW  verified demos (RAM-free) + the forward report (S2→S1 knowledge transfer)
        next: persistent "learning tokens", consolidate-duck, save-state RL
```

---

## 2. Architecture map (component → code → the decision that shaped it)

| Component | Code | Shaped by |
|---|---|---|
| Flow-matching DiT actor (S1) | `nitrogen/flow_matching_transformer/nitrogen.py` | upstream NitroGen; action layout fixed in **EXP-000** |
| VLM planner (S2) | `nitrogen/planner.py:PlanEncoder` | 0.8B→2B redesign **EXP-050**; frozen (Qwen-LoRA now allowed) |
| Resampler (K=8 queries) | `planner.py:PlanResampler` | Perceiver vs Q-former A/B **EXP-032** (Perceiver kept) |
| Adapter (2048→1024) | `planner.py:PlanAdapter` | moved AFTER resampler in 2B redesign **EXP-050**; the left/right culprit **EXP-051** |
| Plan tokens injection | `nitrogen.py:prepare_input_embs` (`_PLAN_TOKEN=7`) | — |
| **Masked-null** (CFG anchor) | `nitrogen.py:apply_null_mask` | learned-vs-masked A/B **EXP-004** |
| Contrastive loss (de-collinearize) | `nitrogen.py:_plan_contrastive_loss` | root cause **EXP-007**, fix **EXP-009/016/043** |
| Plan-adaLN (global FiLM) | `planner.py:PlanHead.adaln_cond` | **EXP-048** (alone dampens) |
| DiT LoRA (fine routing) | `nitrogen/flow_matching_transformer/lora.py` | buttons/timing need capacity **EXP-021/024** |
| Game-id token (always-on) | `nitrogen.py` (`_GAME_ID_TOKEN=6`) | the R13 consolidation surface |
| Cross-chunk cursor (K·A blocks) | `planner.py:PlanHead.forward` | **EXP-027..031** |

Architecture rationale + lit: [`DESIGN.md`](DESIGN.md), [`MULTICHUNK_DESIGN.md`](MULTICHUNK_DESIGN.md),
[`LITERATURE.md`](LITERATURE.md), [`research/ARCH_RESEARCH.md`](research/ARCH_RESEARCH.md) (dual-rate VLA survey).

---

## 3. The experiment graph (EXP-000..054) — by theme

Full log: [`EXPERIMENTS.md`](EXPERIMENTS.md). Capability→ckpt: [`INDEX.md`](INDEX.md).

- **Action layout & null-invariance:** EXP-000 (action layout, CRITICAL), EXP-004 (masked-null).
- **The stick-collapse saga (the Stage-1 crux):** EXP-006 (right fails = stick collapses), **EXP-007 (root
  cause = collinear conditioning)**, EXP-009/010 (contrastive fix → direction steering works).
- **Temporal ordering (SEQ):** EXP-011 (fails), EXP-014 (order lost at the VLM), **EXP-016 (solved via
  order-aware contrastive)**, EXP-017/018 (SEQ3/SEQ4 zero-shot generalization).
- **Heterogeneous + duration (needs capacity):** EXP-019..024 (button/timing routing needs DiT LoRA).
- **Cross-chunk / nested / post-hoc:** EXP-027 (cursor mechanism), EXP-029 (nested), **EXP-031 (R0 post-hoc on
  real action targets)**.
- **No world model (the env negative):** **EXP-034 (no latent world model in the DiT)** → counterfactuals need envs.
- **Real plans (Stage-2):** EXP-033 (transcripts = chit-chat), EXP-035 (VLM-on-frames plans), EXP-036/037
  (plans don't predict action), **EXP-038 (action-conditioned plans = the fix)**, EXP-039 (Gemma-4-12B best).
- **Alignment gap closed:** EXP-041/042 (gap is representational), **EXP-043 (outcome-contrastive closes it)**,
  EXP-044/045 (privileged teacher → distillation), EXP-046 (counterfactual generalization emerges under CFG).
- **The env-free override NEGATIVE (robust):** EXP-047/047b (CFG override training fails), EXP-048/048b (adaLN
  dampens), **EXP-049/049b (data scale-up makes it WORSE)**, EXP-054 (override needs gold-token/envs).
- **2B era → shipped base:** EXP-050 (2B redesign), **EXP-051 (clean labels recover left/right; adapter was the
  culprit)**, **EXP-052 (button steering → `btn_s600`)**, EXP-053 (LoRA balance).

---

## 4. The war-room graph (R1..R13) — architecture debates (GPT-5.5 ⨉ Opus-4.8)

Consolidated log: [`research/ARCH_WARROOM.md`](research/ARCH_WARROOM.md). Per-round positions:
`research/ARCH_WARROOM_{gpt55,opus}_r{N}.md`. Seeds: `research/ARCH_WARROOM_R{N}_SEED.md`.

| Round | Question | Verdict | Primary doc |
|---|---|---|---|
| R3 | actor-OOD vs plan-OOD? | game-dependent dissociation | DISCRIMINATOR_RESULTS |
| R4 | the dissociation, with data | SMW plan-OOD, Sonic actor-OOD | DISCRIMINATOR_RESULTS |
| R6 | the training recipe | RWBC-for-plan-OOD collapses | DISCRIMINATOR_RESULTS |
| **R7** | the pivot | **supervised demo-fit works for plan-OOD** | DISCRIMINATOR_RESULTS |
| **R8** | generalize + pool | **pooled generalist + new games** | DISCRIMINATOR_RESULTS |
| **R9** | ducking collapse | **KL-anchor wins (add w/o forget)** | DISCRIMINATOR_RESULTS |
| R10 | short vs long path | evocation vs addition | ARCH_WARROOM |
| **R11** | staleness / TTT | **staleness is text-bound** | DISCRIMINATOR_RESULTS |
| R12 | gold-action plans | auto-narrate maneuvers | ARCH_WARROOM |
| **R13** | S2→S1 knowledge transfer | **consolidation→instinct (the forward report)** | [`research/FORWARD_S2_TO_S1.md`](research/FORWARD_S2_TO_S1.md) |

Literature grounding: [`research/LIT_R13_S2TOS1.md`](research/LIT_R13_S2TOS1.md) (options/HRL, VOYAGER,
fast-weights, skill-token codebooks, ExIt), `research/LIT_R11_{ttt,distill,fastweights}.md`.

---

## 5. The results & verification graph

| What | Doc | Headline |
|---|---|---|
| **THE results log** | [`research/DISCRIMINATOR_RESULTS.md`](research/DISCRIMINATOR_RESULTS.md) | demo-fit pivot, pooled generalist, KL-anchor, staleness, router — multi-seed + CIs |
| Demo verification (SMW) | [`research/VERIFY_SMW.md`](research/VERIFY_SMW.md) | RAM-free VLM-judged; honest modest-gains verdict |
| Demo verification (cross-game) | [`research/VERIFY_CROSSGAME.md`](research/VERIFY_CROSSGAME.md) | best demos + RAM-vs-VLM disagreements |
| Orchestrator direct-vision | [`research/ORCHESTRATOR_VERIFY.md`](research/ORCHESTRATOR_VERIFY.md) | the VLM-false-positive-death finding |
| Forward report (S2→S1) | [`research/FORWARD_S2_TO_S1.md`](research/FORWARD_S2_TO_S1.md) | evocation→consolidation→instinct |
| Eval methodology | [`research/EVAL_PLAN.md`](research/EVAL_PLAN.md), `reach_eval.py`, `ordinal_sanity.py` | survival-weighted reach + falsifier controls |

---

## 6. The checkpoint & artifact graph

| Artifact | Where | Built by |
|---|---|---|
| base DiT (frozen) | `ckpts/nitrogen/ng.pt` | NVIDIA (`hf download nvidia/NitroGen`) |
| **base eval model** `btn_s600` | `ckpts/btn_s600_full.pt` (full) / HF `base/btn_s600.pt` (slim) | EXP-052; rebuild via `merge_slim_to_full.py` |
| 5 Stage-2 research ckpts | [`CHECKPOINTS.md`](CHECKPOINTS.md) (clean/btn/override/dir) | EXP-050..053 |
| **9 Stage-3 deltas** (pooled/kl/situ) | HF `deltas/` | `demo_bc.py` / `pooled_demofit.py` (recipe: TRAINING_RECIPE §4) |
| actor-OOD rwbc deltas | HF `rwbc/` | `rwbc_actor_adapt.py` |
| 56 verified demos + 223 extras | HF `main_demos/` + `demos_extras/` | `furthest_rollout.py` |
| **everything, one command** | `scripts/recover_checkpoints.sh` | — |

HF repo (private): **`nmk-kun/nitrogen-vlm-planner-handoff`**. Recover all: `bash scripts/recover_checkpoints.sh`.

---

## 7. The code graph (entry points)

- Train: `scripts/train_planner.py` (Stage-1/2), `planner_poc/demo_bc.py` + `pooled_demofit.py` (Stage-3),
  `planner_poc/rwbc_actor_adapt.py` (actor-OOD). Merge: `combine_eval.py`.
- Eval/probe: `eval_policy.py` (core), `reach_eval.py`/`ordinal_sanity.py` (survival reach),
  `maneuver_router.py` (evocation/addition), `plan_authority_map.py` (plan vs DiT-adherence),
  `staleness_probe.py`, `expressiveness_battery.py`.
- Demos/verify: `furthest_rollout.py` (+`--log-plans`), `vlm_video_judge.py`.
- Data/env: `record_play_server.py`, `run_poc.py`, `idm_gen_emulator_data.py`, `yt_farm.py`.
- Full categorized index: [`../planner_poc/README.md`](../planner_poc/README.md). Archived one-offs: `planner_poc/attic/`.

---

## 8. Open threads (the frontier)

1. **Persistent "learning tokens"** — per-game prefix bank for owned capabilities (HANDOFF §3.1, PAPER §8).
2. **Consolidate-duck** — the minimal evocation→instinct experiment (FORWARD_S2_TO_S1 §4).
3. **Save-state RL** — survival-weighted reward on the emulator substrate (`research/RL_DESIGN*.md`).
4. **Plan-quality vs DiT-adherence** — diagnose failures via `--log-plans` + `plan_authority_map.py` (PAPER §7).
5. **Verification hardening** — HUD-OCR death detector (neither RAM nor VLM is reliable alone; HANDOFF §1.2).
