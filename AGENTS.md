# AGENTS.md — start here

Entry point for any agent (or human) continuing this work. Read this first, then
[`docs/INDEX.md`](docs/INDEX.md) (capability→checkpoint map) and [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)
(full log). This file is the **task memory**: what this project is, what works, what's next, and how to
run everything.

> **2026-06-30 HANDOFF:** start with [`docs/HANDOFF_2026-06-30.md`](docs/HANDOFF_2026-06-30.md) (infra catches +
> next steps incl. the "learning tokens" direction), [`docs/TRAINING_RECIPE.md`](docs/TRAINING_RECIPE.md) (what
> base/pooled/kl/situ are + every text plan), and [`docs/research/`](docs/research/) (the full war-room R1–R13,
> lit reviews, results logs, and the S2→S1 `FORWARD_S2_TO_S1.md` + demo `VERIFY_*.md` reports — persisted out of
> the ephemeral session dir). All root content `.md` were moved into `docs/`.

---

## 1. What this project is

A research fork adding a **VLM planner (System 2)** on top of NVIDIA's **NitroGen** — a ~500M
flow-matching DiT that maps the *current frame* → a chunk of gamepad actions (a fast-reacting
*System 1*). NitroGen is markov (sees only the last frame) so it can't plan. This fork adds a **frozen
Qwen3.5 VLM** that reads frames + a plan, a learned **Perceiver resampler** distills the plan into K
tokens, an **adapter** maps them into NitroGen's vision space, and they're injected into the DiT's
cross-attention to steer the actions. A **masked-null** mode makes a null plan reproduce the base model
exactly, so CFG-style plan guidance works.

```
plan text + frames ─▶ frozen Qwen3.5 ─▶ resampler (K queries) ─▶ adapter ─▶ K plan tokens
 frame ──────────────────────────────────────────────────────────────────────▼ inject
                  NitroGen DiT (frozen, optional LoRA): actions cross-attend [image | plan tokens]
                                                       ─▶ 18-step action chunk
```

**Hard constraints (carry these forward):**
- The **VLM planner stays FROZEN.** Only the resampler/adapter/plan-head (+ optional DiT LoRA) train.
- **GPU is scarce** (single A100 80GB on the dev box). Be frugy.
- **Free/FOSS/homebrew games only** in the repo. Commercial ROMs (`Game data/`), YouTube video, and
  `cookies.txt` are **gitignored — never commit them** (copyrighted).
- Kill processes by **numeric PID only** (no pkill/killall).

---

## 2. Current status (2026-06-25)

**Stage-1 (synthetic-plan alignment) WORKS** on a frozen DiT across: direction steering, exact
null-invariance, within-chunk ordering (SEQ), heterogeneous (left-then-jump), uneven-duration,
cross-chunk (A=4), nested, and R0 post-hoc (real action targets). See `docs/INDEX.md` for the
capability→checkpoint→result map. Button steering added: `stage2_2b_btn/s600` = 5/5 buttons selective
while retaining directions (the **main eval checkpoint**).

**Key research finding (see memories + EXPERIMENTS):** env-free left/right is *recoverable* but fragile.
The loss localizes to the **PlanAdapter** (projection-after-resampler); cleaning contrastive label noise
(training only on direction-consistent chunks) lifts DiT-space left/right token separation 0.50→0.688.
Up/down works easily (whole-scene correlated). The broad conclusion the user reached: **real
long-horizon counterfactual capability needs environments/RL** ("it is clear we cant escape envs").

**This session's additions (committed, on branch `namak-kun/plan-conditioning`):**
1. **Record/play server** for collecting human gold trajectories on a headless box.
2. **Eval-env harness**: 30/38 envs boot; emulator (in-process, frame-exact save/load) envs added.
3. **Data-bootstrap pipeline** (popular ROMs + YouTube + VLM objectives + emulator IDM data).

**Checkpoints:** slim (plan_head+LoRA, EMA, fp16) zips in `ckpts/handoff_zips/` (gitignored; user scp's
them off). Full checkpoints are in `runs/<exp>/plan_stage1_*.pt` on the dev box (217 GB, not portable).
Extract more with `planner_poc/extract_plan_head.py <full.pt> <slim.pt>`. **See
[`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md) for extreme-detail per-checkpoint training recipes, data
lineage, and results.** Quick guide: `btn_s600` = default (direction + 5/5 buttons); `clean_lora_s1200`
= best balanced 4-way; `clean_s2000` = left/right recovery research; `override_s2000` = counterfactual
override; `dir_s2500` = NEGATIVE control (collapsed left/right), comparison only.

---

## 3. The three workstreams

### A. Plan-conditioning (the core research) — `nitrogen/`, `scripts/train_planner.py`, `planner_poc/`
Mature. Train with `scripts/train_planner.py` (see §5). Behavioral evals + probes in `planner_poc/`
(`eval_buttons.py`, `eval_balance.py`, `probe_*.py`, `cf_*` counterfactual scorers). Read `docs/INDEX.md`
+ `docs/EXPERIMENTS.md`.

### B. Eval environments + record server — `nitrogen/eval/envs/`, `planner_poc/run_poc.py`,
`planner_poc/record_play_server.py`
- 38 game envs (`run_poc.list_envs()`); **30 boot cleanly**, 5 hidden (lost build artifacts:
  daemon_vs_demon, megaman_maverick, openmw, theseeker, zelda_classic — see `run_poc.BROKEN_ENVS`).
- **Emulator envs** (`emulator_env.py` base, `mgba_env.py` GB/GBC/GBA, `snes_env.py`) are in-process with
  **frame-exact save/load** — the RL/counterfactual substrate.
- **Record server** (`record_play_server.py`): browser play-and-record for gold actions. Lag-free by
  design (synchronous stepping locks game-time to delivered frames). See §5 to launch.
- Health-check all envs: `planner_poc/env_healthcheck.py`.

### C. Data bootstrapping (newest; the path forward) — `planner_poc/{yt_farm,objective_label_demo,idm_gen_emulator_data}.py`
Strategy doc: session `files/BOOTSTRAP_DATA_STRATEGY.md` (also summarized in plan.md). The plan:
- **Objectives (System-2):** popular games are in the VLM's pretraining → it emits correct grounded
  objectives **for free** (demonstrated, `objective_label_demo.py`). No human labels.
- **Actions (System-1):** the DiT is from-scratch → needs pixel→button grounding. Get ground-truth from
  the **emulator** (`idm_gen_emulator_data.py`, we drive it so actions are exact), train an **IDM
  (VPT-style)**, then pseudo-label **YouTube** frames (`yt_farm.py`) at scale.
- **Verify counterfactuals** via emulator save-states (dense checkable reward → RL/GRPO).
- **Eval on held-out OBSCURE/custom games** (homebrew + custom levels) to avoid VLM-memorization
  contamination.

---

## 4. What's next (priority order)

1. **IDM (VPT) model + training** — the missing System-1 action source. `idm_gen_emulator_data.py`
   produces ground-truth (frame, action). NEXT: (a) the cold-start sampler gets stuck/dies in SMW
   (motion ~flat, frames go black on death) — seed from mid-game save-states or use a smarter macro
   sampler so actions cause visible motion; (b) train a non-causal IDM (window→action); (c) **domain-bridge**
   emulator→YouTube (downscale/recompress augmentation) since YouTube is 360p/compressed.
2. **TAS-as-IDM clarification (user):** do NOT use TASVideos movie files (too glitchy/precise to learn).
   "TAS" here means *infer* actions from ordinary YouTube videos via the IDM. Drop the `retro.Movie`
   replay path.
3. **RL loop** (user mentioned verl / prime-rl) on the emulator save-state substrate. Actor = plan-head
   + DiT-LoRA. Either a critic (long credit path frame→plan→18 actions→sparse reward) OR save-state GRPO
   (sample K plans from a save-state, rank by reward, no critic net).
4. **Stabilize plans** (user's stated bottleneck): ~50-60% plan flip-rate between re-plans. Implement
   `--replan-every N` + hysteresis; consider freezing resampler+adapter so the text plan is the only var.
5. **(DONE this session) De-fork state export** — TheXTech reads `/proc/pid/mem` (stock RelWithDebInfo
   build), Solarus auto-overlays the quest at boot. See §6. No source/quest fork remains.

---

## 5. Run commands (verified)

Common env prefix (the repo uses a uv `.venv`; isolate from any outer virtualenv):
```bash
ENVP='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=/home/t-nagupta/NitroGen-With-VLM-Planning:/home/t-nagupta/NitroGen-With-VLM-Planning/planner_poc QWEN=Qwen/Qwen3.5-2B'
# (repo path is now overridable via the NITROGEN_REPO env var; scripts default to the path above)
PY=.venv/bin/python      # uv-managed; install pkgs with `uv pip install <x> --python .venv/bin/python`
```

**Setup (base weights):**
```bash
pip install -e .                                  # or: uv pip install -e .
hf download nvidia/NitroGen ng.pt                 # base DiT -> ckpts/nitrogen/ng.pt
hf download Qwen/Qwen3.5-2B                        # planner backbone (0.8B also supported)
```

**Train Stage-1 (frozen DiT, direction + ordering):**
```bash
$PY scripts/train_planner.py --ng-ckpt ckpts/nitrogen/ng.pt --qwen ckpts/qwen35-0.8b \
  --shard-root <chunks_dir> --frames-dir <frames_dir> \
  --null-mode masked --freeze-dit --seq-heavy \
  --contrastive-weight 1.0 --contrastive-mode pertoken --plan-ratio 0.6 --lr-plan 3e-4
# add: --num-chunks A --cross-chunk (long horizon), --lora-dit 16 (fine routing)
```

**Reload a slim checkpoint for eval:**
```python
import torch
slim = torch.load("ckpts/handoff_zips/btn_s600.pt", map_location="cpu", weights_only=False)
model.load_state_dict(slim["trainable_ema"], strict=False)   # after loading base ng.pt
```

**Record/play server (collect gold actions in a browser):**
```bash
$ENVP nohup .venv/bin/python planner_poc/record_play_server.py --env thextech_get_flower --port 8123 \
  > /tmp/record_server.log 2>&1 &
# forward port 8123 to your laptop (VS Code PORTS tab, or ssh -L 9123:localhost:8123) and open it.
# STEP mode (default): hold a key + tap Tab = one deterministic move (over-input-proof). Toggle
# auto-step / REALTIME 0.06-2x for smoother play. STOP with `kill <pid>` (graceful), NOT kill -9.
```

**Env health-check / list:**
```bash
$ENVP .venv/bin/python planner_poc/env_healthcheck.py            # boot-test all envs
$ENVP .venv/bin/python -c "from run_poc import list_envs; print(list_envs())"
```

**Data bootstrap:**
```bash
# YouTube frames (needs cookies.txt + deno; see docs/SETUP_YOUTUBE.md):
$ENVP .venv/bin/python planner_poc/yt_farm.py search "Super Metroid longplay no commentary" -n 5
$ENVP .venv/bin/python planner_poc/yt_farm.py grab "<url>" --sections 600-660 1200-1260 --fps 4
# VLM objective labels on farmed frames (free System-2 labels):
$ENVP .venv/bin/python planner_poc/objective_label_demo.py --frames "docs/yt_farm/<id>/frames/*.png"
# Ground-truth (frame, action) from an emulator for IDM training:
$ENVP .venv/bin/python planner_poc/idm_gen_emulator_data.py --rom "Game data/Super Mario World.sfc" \
  --system snes --steps 4000 --out docs/idm_data/smw
```

---

## 6. Gotchas & known issues

- **YouTube n-challenge:** bare yt-dlp returns only storyboards. WORKING recipe (verified this session):
  `python -m yt_dlp --cookies cookies.txt --js-runtimes deno:/home/t-nagupta/.deno/bin/deno
  --remote-components ejs:npm`. (deno is NOT on PATH; node alone didn't work.) `yt_farm.py` wires this.
  Note: `docs/SETUP_YOUTUBE.md` documents an alternate bgutil-PO-token + ejs:github recipe — both paths
  exist; deno+ejs:npm is the one confirmed on 2026-06-25.
- **Record server stop:** graceful `kill <pid>` cleans up the child game+Xvfb. `kill -9` orphans them
  (leaks Xvfb/game procs). The server traps SIGTERM.
- **mGBA "blank" frames:** mgba doesn't repaint until stepped; a grab *before* stepping looks black.
  notebook_adventure/blind_jump render fine once stepped (env_healthcheck flags them BLANK — false neg).
- **Genesis has no in-process backend** (Mednafen subprocess only) → no frame-exact save/load for `.md`.
- **State export is now DE-FORKED (implemented this session — no source/quest fork needed):**
  - *TheXTech*: reads player state directly from process memory (`thextech_memread.TheXTechMemReader`).
    Build STOCK TheXTech with debug symbols (`cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo`, keeps symbols;
    Release strips them). The env resolves global addresses via `nm` (binary is **non-PIE** → fixed
    addresses) and reads `/proc/<pid>/mem` from the parent env process (yama ptrace_scope=1 allows
    parent→child). Verified: x/y/vx/lives/menu match the old patch exactly; movement + reset tracked.
    Layout (in the reader): `Player[1].Location` = `&Player + 1*488 + 248`; fields X@0 Y@8 SpeedX@32
    SpeedY@40; `num_t` is **32.32 fixed-point** (real = int64 / 2**32, per `lib/fixed_point.h`);
    `Player_t.Dead`@388; scalars Lives/GameMenu/LevelSelect/GamePaused/EndLevel/LevelBeatCode/numPlayers
    resolved by symbol. Re-derive offsets if TheXTech updates: `gdb -batch -ex 'print sizeof(Player_t)'
    -ex 'print/d &((Player_t*)0)->Location' <binary>`. If the binary is stripped, the reader raises at
    construct and the env falls back to `{}` (graceful).
  - *Solarus*: the ENGINE is stock; the committed repo + ZSDX source are stock. The env applies the Lua
    state-export hook (`nitrogen_state_export.lua`) PROGRAMMATICALLY at boot — an idempotent,
    marker-guarded append to the LOCAL (gitignored `tmp/zsdx`) quest's `data/main.lua`
    (`solarus_zelda.py._ensure_state_overlay`). A fresh stock ZSDX build "just works" with no manual
    patch. (Can't use the engine's `-s=` flag: ZSDX's main.lua defines `function sol.main:on_update`
    which clobbers anything `-s` registers before main.lua — verified.) Reads `hero:get_position()`,
    `game:get_life()`, current map → `$SOLARUS_STATE_EXPORT`. Verified: after reset, real in-game state
    (x=1008 y=613 life=12 map='3').
  - The old reference patches remain in `docs/env_candidates/{thextech,solarus}_state_export.patch`
    (no longer used by the envs).
- **Emulator envs need NO fork** — mgba/snes read RAM natively.

---

## 7. Map of important files

| Path | What |
|---|---|
| `docs/INDEX.md` | capability → experiment → checkpoint map (read after this) |
| `docs/CHECKPOINTS.md` | **extreme-detail per-checkpoint training recipes, data lineage, results** |
| `docs/EXPERIMENTS.md` | full experiment log (EXP-000..049b; predates the 2B/clean/btn era — see CHECKPOINTS.md for that) |
| `docs/DESIGN.md`, `docs/MULTICHUNK_DESIGN.md`, `docs/LITERATURE.md` | architecture + lit |
| `docs/TRAINING_RECIPE.md` | **base/pooled/kl/situ recipes + every text plan + reload (2026-06-30)** |
| `docs/HANDOFF_2026-06-30.md` | **infra catches + next steps (learning tokens, consolidate-duck, RL)** |
| `docs/research/` | persisted war-room R1–R13, lit, results, FORWARD_S2_TO_S1.md, VERIFY_*.md |
| `nitrogen/planner.py` | PlanEncoder / resampler / adapter / PlanHead / `generate_plan` |
| `nitrogen/flow_matching_transformer/nitrogen.py`, `lora.py` | plan injection, masked-null, DiT LoRA |
| `scripts/train_planner.py` | Stage-1 trainer |
| `planner_poc/run_poc.py` | env factory + `list_envs()` + `BROKEN_ENVS` |
| `planner_poc/record_play_server.py` | browser play-and-record server |
| `planner_poc/env_healthcheck.py` | boot-test all envs |
| `planner_poc/yt_farm.py` | YouTube → frames (+chapters) |
| `planner_poc/objective_label_demo.py` | frozen VLM → grounded objectives on frames |
| `planner_poc/idm_gen_emulator_data.py` | emulator → ground-truth (frame, action) for IDM |
| `planner_poc/extract_plan_head.py` | full ckpt → slim trainable delta |
| `nitrogen/eval/envs/{emulator_env,mgba_env,snes_env}.py` | in-process emulator envs (save/load) |
| `nitrogen/eval/envs/proc_game_env.py` | base proc env (`reset_by_relaunch`, speedhack, grab) |
| `ckpts/handoff_zips/` | slim checkpoint zips (gitignored; scp off the box) |

Session research notes (not in git): `~/.copilot/session-state/<id>/plan.md` and
`files/BOOTSTRAP_DATA_STRATEGY.md`.
