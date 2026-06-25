# NitroGen-With-VLM-Planning

A research fork adding a **VLM planner (System 2)** on top of NVIDIA's
[NitroGen](https://github.com/MineDojo/NitroGen) — a 500M-param flow-matching DiT that
maps the current frame to gamepad actions (a fast-reacting *System 1*).

> Upstream NitroGen sees only the **last frame**, so it cannot plan over long horizons.
> This fork adds a frozen **Qwen3.5-0.8B planner** that reads a plan and steers NitroGen's
> actions via injected **plan tokens** — a System-2 → System-1 hierarchy.

The original upstream README is preserved at
[`README_UPSTREAM.md`](README_UPSTREAM.md). This is a research fork, **not** an official
NVIDIA or upstream product.

---

## What this fork adds

A frozen VLM reads a plan (Stage-1: synthetic; Stage-2: transcript-derived) and produces
hidden states. A learned **Perceiver resampler** distills them into **K plan tokens**, a
per-token **adapter** maps them into NitroGen's vision space, and they are injected into
the DiT's vision–language cross-attention. A **masked-null** mode makes a null plan
reproduce the base model exactly (so CFG-style plan guidance works).

Stage-1 alignment is trained from synthetic plans (counterfactual action chunks) with no
game execution; the DiT can stay frozen for coarse control, with optional LoRA for
fine-grained routing — a convenient property of the synthetic stage, not a design
constraint (DiT fine-tuning is expected for later stages).

```
plan text + frames ─▶ frozen Qwen3.5-0.8B ─▶ resampler (K queries) ─▶ adapter ─▶ K plan tokens
                                                                                      │ inject
 frame ─────────────────────────────────────────────────────────────────────────────▼
                      NitroGen DiT (frozen) :  action tokens cross-attend to [image | plan tokens]
                                            ─▶ 18-step action chunk
```

See [`docs/DATAFLOW.md`](docs/DATAFLOW.md) for the exact module/tensor path,
[`docs/INDEX.md`](docs/INDEX.md) for a capability → experiment → checkpoint map, and
[`AGENTS.md`](AGENTS.md) for the **start-here handoff** (task state, run commands, what's next).
Per-checkpoint training detail is in [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md).

## Capabilities demonstrated (Stage-1, frozen base)

| Capability | Result | Where |
|---|---|---|
| **Direction steering** ("go left") + exact null-invariance | cos(L,R) 0.999→0.10, null-inv 0.000 | EXP-009/010 |
| **Within-chunk ordering** (SEQ: "left then right") | SEQ2/SEQ3 solved; SEQ4 generalizes zero-shot | EXP-016/017/018 |
| **Heterogeneous** sequencing ("left then jump") | strong localized button presses (via LoRA) | EXP-021/022 |
| **Uneven-duration** plans ("briefly left, then right") | content-driven transition timing | EXP-023/024 |
| **Cross-chunk** (one plan spans A chunks, cursor-selected) | 16/16, scales to A=4 | EXP-027/028/030 |
| **Nested** (cross-chunk + within-chunk ordering) | 24/24 + holds retained | EXP-029 |
| **R0 post-hoc** (real action targets; plan overrides the frame) | 16/16 causal on real episodes | EXP-031 |

**Finding:** direction + ordering already work on a **frozen DiT** (the representation and
contrastive loss are the lever); *fine* cross-modal/temporal routing (a specific button in
a specific half, exact transition timing) is where DiT capacity via **LoRA** helps.

Full experiment log: [`EXPERIMENTS.md`](EXPERIMENTS.md) (EXP-000..034).

## Repository layout (fork additions)

```
nitrogen/planner.py                         # PlanEncoder / PlanResampler / PlanAdapter / PlanHead
nitrogen/flow_matching_transformer/
    nitrogen.py                             # plan injection, masked-null, order-aware contrastive
    lora.py                                 # LoRA on the DiT cross-attention
nitrogen/training/{plans,dataset,actions,video}.py   # synthetic + cross-chunk plans, data pipeline
scripts/train_planner.py                    # Stage-1 alignment trainer
scripts/{extract_cc_frames,download_more_videos}.py  # frame/data tooling
planner_poc/                                # behavioral evals + probes
docs/{DATAFLOW,INDEX}.md                    # architecture + navigation
DESIGN.md  MULTICHUNK_DESIGN.md  LITERATURE.md  EXPERIMENTS.md
```

## Setup

Same base install as upstream, plus the planner backbone:

```bash
pip install -e .
hf download nvidia/NitroGen ng.pt              # base DiT  -> ckpts/nitrogen/ng.pt
hf download Qwen/Qwen3.5-0.8B                   # planner   -> ckpts/qwen35-0.8b
```

The `nvidia/NitroGen` dataset ships **action labels only**; frames are fetched from the
source videos (see [`docs/SETUP_YOUTUBE.md`](docs/SETUP_YOUTUBE.md) for the cookies +
PO-token + Deno ingestion recipe).

## Train (Stage-1 alignment)

Example — direction steering + within-chunk ordering, frozen DiT:

```bash
python scripts/train_planner.py \
  --ng-ckpt ckpts/nitrogen/ng.pt --qwen ckpts/qwen35-0.8b \
  --shard-root <chunks_dir> --frames-dir <frames_dir> \
  --null-mode masked --freeze-dit --seq-heavy \
  --contrastive-weight 1.0 --contrastive-mode pertoken \
  --plan-ratio 0.6 --lr-plan 3e-4
```

Useful flags: `--num-chunks A --cross-chunk` (long-horizon cursor), `--lora-dit 16`
(fine button/timing routing), `--resampler-self-attn` (Q-former toggle),
`--cc-pool {hold,nested,hold4}`, `--cc-posthoc` (real-action-target R0).

## Status & next directions

- Stage-1 (synthetic-plan alignment) is **working** across the capabilities above.
- A query self-attention (true Q-former) toggle exists but gave no measurable benefit on
  synthetic tasks (EXP-032); its intended test is Stage-2.
- **Stage-2 transcripts** are available (captions for most videos) but commentary-heavy —
  the plan here is to LLM-**relabel** them into terse intents rather than condition on raw
  text (EXP-033).
- Long-horizon **counterfactual** play (R2) is expected to need a game environment: no
  reusable world model was found in NitroGen's DiT internals (EXP-034), so counterfactual
  futures can't be manufactured offline.
- Cross-game **abstraction** (a shared plan/skill space over NitroGen's many games) is the
  main remaining direction that stays env-free.

> Sidenote: Stage-1 happens to be fully **env-free** (synthetic plans, no game rollouts)
> and **parameter-efficient** (frozen base, optional LoRA). These are conveniences of the
> synthetic alignment stage, not goals of the method — later stages are expected to fine-
> tune the DiT and may require a live environment.

## Citation

This fork builds directly on NitroGen — please cite the upstream paper (bibtex in
[`README_UPSTREAM.md`](README_UPSTREAM.md)).

**Disclaimer:** Research project, strictly for research purposes. Not an official NVIDIA
or upstream product.
