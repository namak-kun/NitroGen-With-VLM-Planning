# Training recipe — base / pooled / kl / situ (excruciating detail)

This documents EXACTLY what the four models in the demo videos (`docs/furthest/<game>/<tag>__state<N>.mp4`,
tags `base|pooled|kl|situ`) are, the **exact text plans** they were trained on and run with, the
hyperparameters (read from the checkpoint metadata, not inferred), the data, and how to reload them. Written
2026-06-30 for handoff.

> The demo videos overlay only `MODE` + the RAM progress var, NOT the plan text — hence this doc. The plan the
> actor SEES at rollout time is described in §6 (it is the **live VLM plan**, not the training plan).

---

## 0. TL;DR — what the four tags are

| tag | what it is | trained on | delta (vs base) | file |
|---|---|---|---|---|
| **base** | the pretrained plan-conditioned model, **no fine-tune**. Reference baseline. | (pretraining only) | none | `ckpts/btn_s600_full.pt` |
| **pooled** | ONE plan-head fit on **pooled SMW+MMX+SMB1** human demos (the generalist for plan-OOD games) | 3 games' demos, each chunk on its own game's "correct" plan | 38 `plan_head.*` tensors | `files/pooled_planfit_s0.pt` |
| **kl** | SMW-only plan-head fit with a **KL-anchor** that preserves the whole action vocabulary (the R9 winner) | SMW demos, one terse "correct" plan + KL-anchor pool | 38 `plan_head.*` tensors | `files/r9_smw_kl_s0.pt` |
| **situ** | SMW-only plan-head fit with **per-chunk situational plans** (each chunk's plan derived from its own action) | SMW demos, per-chunk action-labeled plans | 38 `plan_head.*` tensors | `files/r9_smw_situ_s0.pt` |

All three trained deltas are **plan-head-only** (the DiT, its LoRA, the VLM, the resampler and adapter are ALL
frozen). A "delta" is just the 38 `plan_head.*` weight tensors; you load the base model, then overlay the delta
with `load_state_dict(..., strict=False)`. `files/` = the (ephemeral) session dir
`~/.copilot/session-state/dddebd2a-.../files/`; **copy these `.pt` off the box** (see HANDOFF doc).

---

## 1. The architecture the recipe sits in

```
plan text + recent frames ─▶ frozen VLM (Qwen3.5-2B) ─▶ resampler (K=8 queries, FROZEN)
                                                       ─▶ adapter (FROZEN) ─▶ plan_head (TRAINS) ─▶ K=8 plan tokens
 current frame ───────────────────────────────────────────────────────────────────────────▼ inject at _PLAN_TOKEN
              frozen NitroGen DiT (btn_s600): action chunk cross-attends [ image tokens | K plan tokens ]
                                                                         ─▶ 18-row gamepad action chunk
```

- **base weights** = `ckpts/btn_s600_full.pt` ("btn_s600"): the pretrained plan-conditioned NitroGen. It already
  does direction steering + 5/5 button steering with exact **masked-null** (a null plan zeroes the K plan-token
  positions → reproduces the bare DiT). This is the thing all three deltas fine-tune ON TOP of, and the `base`
  demo tag is this model with a **null** plan.
- **what trains** in all three recipes: only `plan_head.*` (38 tensors). Frozen: DiT, DiT-LoRA, the VLM,
  resampler, adapter. So the deltas only change how the K plan tokens are produced — never the actor's weights.
- **action space**: each chunk is 18 rows × 25 dims. The env's `action_row_to_buttons` down-projects the 25-dim
  row to console buttons. Load-bearing dims (used by `demo_bc.map_action`, console-button → 25-dim): `dim1` =
  D-pad DOWN, `dim18` = jump/south, `dim21` = left/right analog (JLX, <0.5 left / >0.5 right), `dim22` = DOWN
  analog stick (JLY) — **human DOWN maps to the STICK dim22, the base model ducks via the DPAD dim1.**

---

## 2. The data (human demos)

Human gameplay demos recorded via the browser record-server, stored as
`docs/demos/demos/<Game>/<timestamp>/demo.npz` (+ `initial.state`, frames). `demo_bc.load_demo_chunks` slices
them into 18-row chunks (`chunk_stride=18`, i.e. non-overlapping).

| game key | demo glob | demos | notes |
|---|---|---|---|
| `smw` | `docs/demos/demos/SuperMarioWorld-Snes/*` | several | SNES 12-button; the main eval game |
| `mmx` | `docs/demos/demos/MegaManX-Snes/*` | 2 (~6 min, level 1) | stable-retro experimental integration (xpos/health) |
| `smbas` | `docs/demos/demos/SuperMarioAllStars-Snes/*` | 30 (~39 min, SMB1 → 7-4) | x-addr = `0x7E0042`; SMB1 levels |
| `sonic` | `docs/demos/demos/SonicTheHedgehog2-Genesis/*` | 4 | Genesis 6-button; actor-OOD game (NOT in pooled) |

- **console → 25-dim** conversion: `demo_bc.map_action` (SNES order `B,Y,SELECT,START,UP,DOWN,LEFT,RIGHT,A,X,L,R`;
  Genesis order `B,A,MODE,START,UP,DOWN,LEFT,RIGHT,C,Y,X,Z`). Same letter ≠ same function across games — see §5.
- **death-tail trim**: some demos END IN DEATH (anti-expert). `docs/demos/demo_trim.json` lists per-demo tail
  seconds to drop (SMB #3/#5/#30, MMX #1). `load_demo_chunks` honours it. `demo.npz` rewards/dones are all-zero
  (the record server didn't log them) so death is NOT auto-detectable — the trim config is the source of truth.

---

## 3. The exact text plans

### 3a. Per-game "correct" plan — `plan_graded_test.BATTERY[game]['correct']`
The terse directional plan each game's demos are conditioned on in `--use-correct-plan` / pooled mode:

| game | `correct` plan text |
|---|---|
| smw | `Move right, run, and jump over pits and enemies to advance through the level.` |
| smbas | `Move right, run, and jump over pits and enemies to advance through the level.` |
| mmx | `Move right through the stage, jumping over gaps and shooting or dodging enemies to advance.` |
| sonic | `Run right at full speed, jumping over gaps and enemies to reach the end of the act.` |
| minish | `Move right toward the next area and attack anything blocking the path.` |
| fireemblem | `Move the cursor to select a unit and advance it toward the enemies.` |

(`BATTERY` also has `hand_good` = a wordier good plan, and `bad` = a reversed "go left" plan used as a negative
control in the graded-plan eval. The deltas were trained on `correct`.)

### 3b. `GENERIC_PLAN` (demo_bc default when `--use-correct-plan` is OFF)
`move right, run and jump over obstacles to advance`

### 3c. Situational plans — `demo_bc.SIT_PLANS` (used by the `situ` model)
Each chunk's action is labelled by `demo_bc.label_chunk` (priority duck > retreat > wait > jump > advance) and
mapped to:

| label | plan text |
|---|---|
| duck | `Press down to duck under the enemy or hazard, then keep moving right.` |
| retreat | `Turn around and move left, back away from the hazard.` |
| wait | `Wait and hold still, let the hazard pass before advancing.` |
| jump | (jump-flavored advance) |
| advance | falls back to the game's `correct` plan |

`label_chunk` thresholds (on the 25-dim chunk): DOWN if `dim1`-down ≥ 3/18 rows; retreat if left>right & left>0.30
(`dim21<0.5`); etc.

### 3d. Duck-probe plans — `demo_bc.DUCK_PLANS` (the expressiveness test + part of the KL-anchor pool)
`[("terse", None=correct plan), ("dodge", "duck to dodge the bullet ahead"), ("press_down", "press down to duck under it")]`
— used to measure DOWN-press rate vs plan explicitness (base: 1.2% → 3.4% → 10.8%).

---

## 4. The three training recipes (exact commands + hyperparameters)

Common prefix (uv venv, isolate from any outer virtualenv):
```bash
ENVP='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=/home/t-nagupta/NitroGen-With-VLM-Planning:/home/t-nagupta/NitroGen-With-VLM-Planning/planner_poc QWEN=Qwen/Qwen3.5-2B'
PY=.venv/bin/python
BASE=ckpts/btn_s600_full.pt
```

### 4a. `pooled` — the plan-OOD generalist (`pooled_demofit.py`)
**What:** ONE plan-head fit on the POOLED demo chunks of SMW+MMX+SMB1, **each chunk conditioned on ITS game's
`correct` plan** (§3a). LoRA + base frozen, `plan_head.*` trains. No reward (pure supervised BC of the human
action chunk given the plan). Naturally no-ops where the frame alone determines the action.
```bash
$ENVP CUDA_VISIBLE_DEVICES=0 $PY planner_poc/pooled_demofit.py \
  --train-games smw,mmx,smbas --eval-games smw,mmx --ckpt $BASE \
  --steps 900 --lr 5e-5 --bs 6 --cfg 8.0 --A 2 --eval-chunks 16 \
  --max-per-game 2500 --seed-offset 0 --save-delta files/pooled_planfit_s0.pt
```
Hyperparameters (from the script defaults, which produced `pooled_planfit_s0.pt`): **steps 900, lr 5e-5, bs 6,
cfg 8.0, A=2, ≤2500 chunks/game, plan-head-only.** Checkpoint metadata: `{train_games:'smw,mmx,smbas',
trainable: 38 plan_head tensors}`. Result: Δ_plan widens on SMW +26 / MMX +31 (3/3 seeds, CIs>0, null 0.0);
SMW benefits 1.3× from pooling (positive cross-game transfer). 3 seeds = `pooled_planfit_s{0,1,2}.pt`.

### 4b. `kl` — the R9 winner (KL-anchored, preserves expressiveness) (`demo_bc.py`)
**What:** SMW-only plan-head fit on one terse `correct` plan, PLUS a **KL-anchor**: an MSE penalty pulling the
K plan tokens toward a FROZEN pre-fit reference over a broad plan distribution (the anchor pool = `SIT_PLANS`
values + `DUCK_PLANS` texts + the `correct` plan; 32 frame×plan pairs). The anchor preserves the bridge's
responsiveness to ALL plans (duck/retreat/up), so demo-fit doesn't collapse the action space onto right+jump.
```bash
$ENVP CUDA_VISIBLE_DEVICES=0 $PY planner_poc/demo_bc.py --game smw \
  --train plan_head --use-correct-plan --kl-anchor 0.3 --kl-anchor-n 32 \
  --duck-probe --steps 600 --lr 2e-5 --bs 4 --cfg 8.0 --eval-chunks 16 --A 2 \
  --save-delta files/r9_smw_kl_s0.pt
```
Hyperparameters: **steps 600, lr 2e-5, bs 4, cfg 8.0, plan-head-only, kl-anchor λ=0.3 over 32 pairs.**
Checkpoint metadata: `{game:'smw', plan:'Move right, run, and jump over pits and enemies to advance through the
level.', train:'plan_head', residual:False}`. Anchor loss stays ~0.001 (non-distorting). Result: Δ_plan +45.1
(3/3 seeds, biggest+tightest), expressiveness battery R=0.83 (the only PASS), duck-on-command restored monotone.
The KL-anchor loss is `loss = bc_loss + 0.3 * MSE(current_plan_tokens, frozen_reference_plan_tokens)`
(demo_bc.py:353-362). 3 seeds = `r9_smw_kl_s{0,1,2}.pt`.

### 4c. `situ` — situational/action-labeled plans (`demo_bc.py`)
**What:** SMW-only plan-head fit where **each chunk's plan is DERIVED FROM ITS OWN ACTION** (`label_chunk` →
`SIT_PLANS`, §3c). No KL-anchor. Recovers duck-on-command where the action is labelled (duck/jump/wait restored,
duck STICK press_down up to 98%) but LOSES under-labelled actions (retreat, up) — a taxonomy treadmill.
```bash
$ENVP CUDA_VISIBLE_DEVICES=0 $PY planner_poc/demo_bc.py --game smw \
  --train plan_head --situational-plans --steps 600 --lr 2e-5 --bs 4 \
  --cfg 8.0 --eval-chunks 16 --A 2 --save-delta files/r9_smw_situ_s0.pt
```
Hyperparameters: **steps 600, lr 2e-5, bs 4, plan-head-only, per-chunk situational plans, no anchor.** Result:
Δ_plan +32.8 (3/3), expressiveness R=0.53 (a clear 2nd to KL; keeps jump+wait, loses retreat+up). 3 seeds =
`r9_smw_situ_s{0,1,2}.pt`.

### 4d. The CONTROL we DON'T ship (for context)
`P0A0` = terse `correct` plan, plain BC, NO anchor, NO situational labels. This is the **collapse** recipe:
Δ_plan goes slightly NEGATIVE and the whole action space dies (battery R=0.165, duck 0%). Proves the ducking
collapse is a DATA artifact (one constant plan → plain BC ignores the plan → drops rare actions). `kl` and
`situ` are the two fixes; `kl` wins.

---

## 5. Per-game button semantics (verified vs manuals + demo usage)

Same letter ≠ same function across consoles/games — critical for reading the videos and for `map_action`:
- **SMW:** B=Jump, A=Spin-jump, X/Y=Run/grab, Down=Duck, Up=fence/rope/door-grab.
- **SMB All-Stars (SMB1):** A/B=Jump/Swim, X/Y=Run/Fireball, Down=crouch/pipe, **NO spin-jump**.
- **MMX:** Y=Jump, B=Shoot, A=Dash, X=Special, L/R=cycle weapon.
- **Super Metroid:** A=Jump, X=Fire, B=Dash, Y=cancel, Up=aim-up, Down=kneel/morph, L=aim-down-angle, R=aim-up-angle.
- **Sonic 2:** A/B/C=Jump, Down+jump=Spin-dash, Down-while-running=Roll.

---

## 6. What plan the DEMO VIDEOS actually used at rollout time (IMPORTANT)

`furthest_rollout.py` runs CLOSED-LOOP. The plan the actor sees is **NOT** the training plan:
- **`base` tag** → `mode=base` → `null=True`: NO plan (masked-null) — the bare DiT. This is the honest "no
  System-2" baseline.
- **`pooled`/`kl`/`situ` tags** → `mode=plan` → every **A=2 chunks** the **LIVE VLM (Qwen3.5-2B)** regenerates a
  plan from the last 4 frames via `generate_plan`, and the **first sentence** is used (fallback `"move right"`).
  Prompts:
  - `INSTR = "In one sentence say what to do next using concrete directions (left,right,up,down,jump)."`
  - `SYS["thextech"]` (SMW/SMB/MMX) `= "You are the planner for a Mario-style platformer. Goal: advance RIGHT,
    jump platforms, avoid hazards. Output ONE short imperative plan (max 10 words)."`
  - `SYS["sonic"]` `= "You are the planner for Sonic, a 2D platformer. Goal: move RIGHT, jump gaps/enemies.
    Output ONE short plan (max 10 words)."`

So the demo comparison is **"bare DiT (null)" vs "trained delta + live Qwen-2B plan."** It conflates two things
on purpose (it's the deployment comparison): the trained bridge AND the presence of a live plan. The plan text
is regenerated live and not logged per-frame in the mp4 — to see it, re-run `furthest_rollout.py` with a print,
or use the fixed-plan Δ_plan eval (`demo_bc.eval_from_states`, which uses the `correct` plan).

---

## 7. How to reload a delta for eval

```python
import torch
from eval_policy import NitroGenPolicy
pol = NitroGenPolicy("ckpts/btn_s600_full.pt", qwen="Qwen/Qwen3.5-2B", default_cfg=8.0)
slim = torch.load("files/r9_smw_kl_s0.pt", map_location="cpu", weights_only=False)
pol.m.load_state_dict(slim["trainable"], strict=False)   # overlay 38 plan_head tensors
```
Or via the tools that already do it: `furthest_rollout.py --delta <pt> --mode plan`,
`demo_bc.eval_from_states`, `pooled_demofit.py`, `combine_eval.py`.

## 8. Eval metric (how Δ_plan / "reach" is measured)

- **Δ_plan** = (advance under the plan) − (advance under null), in the game's RAM progress var (SMW/Sonic
  `screen_x`, MMX/SMB `xpos`), from a set of FIXED demo save-states over 16 chunks. It isolates "does the plan
  help" while the frame-determined part cancels.
- **survival-weighted reach** (`eval_common.survival_advance`) = running-max progress BEFORE death (death =
  lives-decrement 1–3, or progress reset > 40). The headline must be death-aware (a suicide-sprint scores 0).
- **Caveat (see HANDOFF §infra):** RAM `screen_x` MAGNITUDE is not cross-level comparable (warp/underground
  scaling) and misses some deaths; the VLM video-judge is the RAM-free cross-check (but it FALSE-POSITIVES
  deaths). Trust a game HUD counter (TIME/lives/rings) as the final arbiter.
