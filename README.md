# NitroGen + VLM Planner

A research fork that adds a **VLM planner (System 2)** on top of NVIDIA's
[NitroGen](https://github.com/MineDojo/NitroGen) — a ~500M flow-matching DiT that maps the *current frame* to a
chunk of gamepad actions (a fast-reacting *System 1*).

> Upstream NitroGen is **markov** (sees only the last frame) so it cannot plan. This fork adds a frozen
> **Qwen3.5 VLM** that reads a text plan + recent frames; a learned bridge distills that into **K=8 "plan
> tokens"** injected into NitroGen's cross-attention to steer the actions. A **masked-null** mode makes a null
> plan reproduce the base model *exactly*, so CFG-style plan guidance works.

The original upstream README is preserved at [`README_UPSTREAM.md`](README_UPSTREAM.md). This is a research fork,
**not** an official NVIDIA/upstream product.

```
plan text + frames ─▶ frozen Qwen3.5 ─▶ resampler (K=8 queries) ─▶ adapter ─▶ plan_head ─▶ K plan tokens
 frame ───────────────────────────────────────────────────────────────────────────────────▼ inject
                  NitroGen DiT (frozen, optional LoRA): actions cross-attend [ image tokens | plan tokens ]
                                                                            ─▶ 18-row action chunk
```

**👉 If you are catching up on the project, read in this order:**
1. This README (architecture + training + status + future).
2. [`docs/TRAINING_RECIPE.md`](docs/TRAINING_RECIPE.md) — exact recipes for the shipped models + every text plan + losses + data structures.
3. [`docs/HANDOFF_2026-06-30.md`](docs/HANDOFF_2026-06-30.md) — infra catches + next steps.
4. [`AGENTS.md`](AGENTS.md) — full task memory / run commands. [`docs/research/`](docs/research/) — war-room + results.

**📚 For the whole picture:** [`docs/PAPER.md`](docs/PAPER.md) is a full academic-style writeup
(abstract → method → results → limitations → future); [`docs/ATLAS.md`](docs/ATLAS.md) is a navigable map of the
*entire* graph — research arc, architecture decisions, every experiment (EXP-000..054 + war-room R1–R13),
checkpoints, and discussions, with links.

---

## 1. Architecture (precise)

| Component | What | State |
|---|---|---|
| **DiT actor (System 1)** | NitroGen flow-matching DiT (`ckpts/nitrogen/ng.pt`): 8-layer, 16 heads × 64, output 1024, ada-norm. Frame → **18-row × 25-dim** action chunk. Markov. | **frozen** (optional LoRA on cross-attn) |
| **VLM planner (System 2)** | Qwen3.5 (0.8B early, **2B** current) reads plan text + recent frames → hidden states. | **frozen** (Qwen-LoRA now allowed) |
| **Resampler** | Perceiver/Q-former, **K=8** learned queries cross-attend over plan text + frames → 8 frame-conditioned latents. | trainable |
| **Adapter** | Linear map from backbone hidden dim → DiT vision space (2048→1024). | trainable |
| **PlanHead** | resampler + adapter together; produces the **K=8 plan tokens** injected at the `_PLAN_TOKEN` positions. | trainable (the main lever) |
| **Game-id token** | `_GAME_ID_TOKEN`: an always-on per-game embedding (`padding_idx=0` = unconditional), **never masked**. | trainable table |

**Masked-null (key property):** under a null plan, `apply_null_mask` zeroes the K `_PLAN_TOKEN` positions → the
DiT cross-attention is **bit-identical to the bare NitroGen**. So plan-guidance is CFG: `v = v_null + w·(v_plan −
v_null)`. The game-id token is *not* masked, which matters for the "learning tokens" direction (§5).

**Action dims that matter** (env down-projects 25-dim → console buttons via `action_row_to_buttons`): `dim1` =
D-pad DOWN, `dim18` = jump/south, `dim21` = left/right analog (<0.5 left / >0.5 right), `dim22` = DOWN analog
stick. Human DOWN maps to the **stick** (dim22); the base model ducks via the **D-pad** (dim1).

Code: [`nitrogen/planner.py`](nitrogen/planner.py) (PlanEncoder/resampler/adapter/PlanHead/`generate_plan`),
[`nitrogen/flow_matching_transformer/nitrogen.py`](nitrogen/flow_matching_transformer/nitrogen.py) (injection,
masked-null, game-id token), `lora.py` (DiT LoRA). Tensor path: [`docs/DATAFLOW.md`](docs/DATAFLOW.md).

---

## 2. How it's trained (the stages)

Training is **staged**; each stage reuses the previous checkpoint and trains a small set of parameters. Full
detail + exact commands in [`docs/TRAINING_RECIPE.md`](docs/TRAINING_RECIPE.md) and
[`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md).

### Stage 1 — synthetic-plan alignment (frozen DiT, env-free)
Teach the bridge to steer the DiT from **synthetic** plans (counterfactual action chunks), no game execution.
Objective = per-token contrastive loss that de-collinearizes plan tokens along the action axis. Trains
PlanHead (+ optional LoRA). `scripts/train_planner.py`. This stage established: direction steering, exact
null-invariance, within-chunk ordering (SEQ), heterogeneous/uneven-duration, cross-chunk (cursor, A chunks),
nested, R0 post-hoc. See the capabilities table below.

### Stage 2 — 2B backbone + button steering → **`btn_s600`** (the base eval checkpoint)
Re-grounded resampler to Qwen3.5-2B's native dim; adapter moved to *after* the resampler. Stage-2 mm caches
(boundary-frame conditioning so the planner perceives motion). Result: **direction + 5/5 button steering**.
`btn_s600` (`ckpts/btn_s600_full.pt`) is the model everything below fine-tunes on, and the `base` in the demos.

### Stage 3 — demo-fit on human gameplay (this session's work; the plan-OOD recipe)
Supervised fit of the **PlanHead only** (DiT/LoRA/VLM frozen) on human demo chunks, each conditioned on a plan.
No env reward. This is what widened inference-time plan-following on obstacle games. Three variants ship (the
demo-video tags):

| model | recipe | one-liner |
|---|---|---|
| **pooled** | one PlanHead fit on **pooled SMW+MMX+SMB1** demos, each chunk on its game's plan (`pooled_demofit.py`, 900 steps, lr 5e-5) | the plan-OOD **generalist**; +26/+31 Δ_plan, transfers to new games |
| **kl** | **SMW-only + KL-anchor** (λ=0.3) pulling plan-tokens toward a frozen reference over a broad plan set (`demo_bc.py --kl-anchor`) | **R9 winner**: adds the skill *without* forgetting the rest of the action vocabulary (battery R=0.83) |
| **situ** | SMW-only, each chunk's plan **derived from its own action** (`demo_bc.py --situational-plans`) | recovers duck-on-command where actions are labelled; loses unlabelled actions |

### Stage 3′ — actor adaptation for open games (Sonic)
**Sonic** is *actor-OOD* (open/locomotion), not plan-OOD. There the lever is **plain local RWBC** (LoRA
adaptation with an env reward), `rwbc_actor_adapt.py`. The plan-OOD plan_head delta and the actor-OOD lora delta
write **disjoint params**, so they **merge into one generalist** with no interference (`combine_eval.py`).

> **One-generalist principle (owner mandate):** no per-game or per-genre models. Route each game to its mode
> (plan-OOD → plan-head demo-fit; actor-OOD → RWBC lora), train, then merge the disjoint param groups. The
> intra-genre interference problem is to be **solved**, not routed around.

---

## 3. Where we are (status, honest)

**What works (multi-seed, bootstrap CIs, null-invariance exact — see [`docs/research/DISCRIMINATOR_RESULTS.md`](docs/research/DISCRIMINATOR_RESULTS.md)):**
- **Plan-OOD demo-fit generalizes + pools + merges.** One pooled PlanHead fit widens Δ_plan on SMW/MMX/SMB1
  (3/3 seeds); transfers to a new game out of the box (MMX +117%); merges with the Sonic LoRA into one model.
- **Evocation works:** the VLM can *summon* a primitive the DiT already owns — e.g. SMW duck-on-command goes
  1.2% → 10.8% as the plan names it more explicitly.
- **KL-anchor adds a skill without forgetting** the rest of the action space (the consolidation primitive).
- **Staleness is text-bound:** the K plan tokens are remarkably stable (drift ~0.001); the lever is plan *text*
  quality, not token-TTA or actor-weight-TTT. Live re-planning can even *hurt*.

**What does NOT (yet):**
- **Precise outcomes the DiT lacks are "additions," not evocations** — e.g. spin-jump-*kills*-Rex: the jump
  button is 7× evocable but the kill outcome stays at chance. Those must be *learned* (demo-fit/LoRA on the
  maneuver), not just named.
- **No model beats a level.** On RAM-free VLM-verified rollouts (`docs/research/VERIFY_*.md`), trained models go
  **further** than base (MMX 3×, Sonic ~10× by rings, SMW kl 4×) but the far runs often **die**. The standout
  demos are the ones that go furthest *and survive* (Sonic `pooled_state0`: 40 rings + survives 90 s vs base 4
  rings + death at 29 s; SMW `kl_state7`: survives a lava fortress where base dies at 13.6 s).
- Real long-horizon counterfactual play is expected to need **environments/RL** — no reusable world model was
  found in the DiT internals, so counterfactual futures can't be manufactured purely offline.

**Capabilities demonstrated (Stage-1, frozen base):**

| Capability | Result |
|---|---|
| Direction steering + exact null-invariance | cos(L,R) 0.999→0.10, null-inv 0.000 |
| Within-chunk ordering (SEQ "left then right") | SEQ2/3 solved; SEQ4 zero-shot |
| Heterogeneous / uneven-duration sequencing | localized presses; content-driven timing |
| Cross-chunk (one plan spans A chunks) | 16/16, scales to A=4 |
| R0 post-hoc (real action targets) | 16/16 causal on real episodes |

Full log: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) (EXP-000..049b); 2B checkpoints in
[`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md).

---

## 4. Quickstart

**Recover all checkpoints on a fresh box (one command):**
```bash
hf auth login                              # the handoff weights repo is private
bash scripts/recover_checkpoints.sh        # -> ng.pt, Qwen, full base ckpt, 9 deltas, rwbc, 56 demos
```
This pulls the NVIDIA base DiT + Qwen backbone + our private handoff repo
(`nmk-kun/nitrogen-vlm-planner-handoff`) and **reconstructs the full eval checkpoint**
`ckpts/btn_s600_full.pt` from `ng.pt` + the slim base (verified bit-faithful). Idempotent; skips what's
already there. Then:

```bash
ENVP='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=.:planner_poc QWEN=Qwen/Qwen3.5-2B'
PY=.venv/bin/python
```

```bash
# longest closed-loop rollout until death, base vs a trained delta (writes mp4 + frames + RAM json)
$ENVP $PY planner_poc/furthest_rollout.py --game smw --tag base   --state 7 --seconds 90 --out out/
$ENVP $PY planner_poc/furthest_rollout.py --game smw --tag kl --mode plan \
    --delta ckpts/deltas/r9_smw_kl_s0.pt --state 7 --seconds 90 --log-plans --out out/

# RAM-free verification: a VLM watches the frames and judges progress/death
QWEN=google/gemma-4-12B-it $ENVP $PY planner_poc/vlm_video_judge.py \
    describe --frames out/kl__state7.frames.npz --game smw

# train a plan-OOD demo-fit (the KL-anchor winner)
$ENVP $PY planner_poc/demo_bc.py --game smw --train plan_head --use-correct-plan \
    --kl-anchor 0.3 --duck-probe --steps 600 --save-delta files/r9_smw_kl_s0.pt
```

Eval harness runs **38 games** under Xvfb (30 boot cleanly; `run_poc.list_envs()`), including **in-process
emulator envs** (mGBA, stable-retro) with **frame-exact save/load** — the substrate for save-state RL. A
**browser play-and-record server** (`record_play_server.py`) collects human gold trajectories on a headless box.
See [`docs/SETUP_EVAL.md`](docs/SETUP_EVAL.md).

---

## 5. Future plans

1. **Persistent "learning tokens" (the headline next direction).** A dedicated bank of N (32/64/128) per-game
   tokens that are *always present* and modulated to hold the **capabilities learned for that game** — separating
   **persistent capability** (learning tokens) from **transient intent** (plan tokens). This is per-game
   prefix-tuning in the DiT KV; it preserves exact plan-CFG null-invariance (plan tokens still mask cleanly on
   top), and an un-learned game stays base-exact via a zero learning-id. It directly attacks the **token-dilution
   wall** (S2 stops naming primitives in prose). See [`docs/HANDOFF_2026-06-30.md`](docs/HANDOFF_2026-06-30.md) §3.1.
2. **The recursion — "discovery becomes instinct."** Evoke a skill via text → verify in the emulator save-state →
   **consolidate** it (KL-anchored) into the persistent channel so the null policy owns it → S2 stops naming it
   and composes at a higher level. Minimal first experiment: **CONSOLIDATE-DUCK**. Full report:
   [`docs/research/FORWARD_S2_TO_S1.md`](docs/research/FORWARD_S2_TO_S1.md).
3. **Save-state RL** (verl / prime-rl): actor = plan-head (+ LoRA / learning tokens), reward must be
   survival-weighted (raw progress = suicide-sprint exploit). Needs sprite-status / level-complete detectors.
4. **Plan-quality vs DiT-adherence attribution (open diagnostic).** When a rollout fails (e.g. Sonic's
   spring-launch section, where you must go *back* to a spring to gain the speed to go up), is it because the
   **plan was wrong** or the **DiT ignored a correct plan**? Tools to answer it: log the live plans
   (`furthest_rollout.py --log-plans`) to read what S2 said, and `plan_authority_map.py` (CFG ‖v_plan − v_null‖)
   to measure how much the DiT actually *moves* under the plan. High authority + wrong move ⇒ plan problem; low
   authority ⇒ DiT ignored it.
5. **Verification hardening:** a HUD-OCR death detector (TIME-reset / lives / rings) — because *neither* RAM nor
   the VLM judge is individually reliable (RAM misses deaths + inflates magnitude; the VLM false-positives
   deaths). See HANDOFF §1.2.

---

## 6. Repository layout

```
nitrogen/planner.py                       # PlanEncoder / resampler / adapter / PlanHead / generate_plan
nitrogen/flow_matching_transformer/       # nitrogen.py (injection, masked-null, game-id), lora.py
nitrogen/eval/envs/                        # 38 game envs (proc/Xvfb + in-process emulators w/ save/load)
scripts/train_planner.py                  # Stage-1 alignment trainer
planner_poc/                               # evals, probes, demo-fit, data tooling, record server
    README.md                             #   ── index of the scripts (start here) ──
    demo_bc.py  pooled_demofit.py         #   plan-OOD demo-fit (kl / situ / pooled)
    rwbc_actor_adapt.py  combine_eval.py  #   actor-OOD RWBC + the one-generalist merge
    furthest_rollout.py  vlm_video_judge.py  # rollout demos + RAM-free VLM verification
    eval_policy.py  eval_common.py  plan_graded_test.py  # core eval primitives
docs/
    TRAINING_RECIPE.md  HANDOFF_2026-06-30.md            # ← read these to catch up
    INDEX.md  CHECKPOINTS.md  EXPERIMENTS.md  DATAFLOW.md DESIGN.md MULTICHUNK_DESIGN.md LITERATURE.md
    research/                              # war-room R1-R13, lit reviews, results, FORWARD/VERIFY reports
AGENTS.md                                 # start-here task memory / run commands
```

## Citation & disclaimer
Builds directly on NitroGen — please cite the upstream paper (bibtex in
[`README_UPSTREAM.md`](README_UPSTREAM.md)). Research project, strictly for research purposes; not an official
NVIDIA or upstream product. Repo contains **free/FOSS/homebrew games only**; commercial ROMs, video, and
`cookies.txt` are gitignored and never committed.
