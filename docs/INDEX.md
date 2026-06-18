# Capability → Experiment → Checkpoint Index

A navigation map for the plan-conditioned NitroGen work. Each capability lists the
**experiment(s)** that established it, the **best checkpoint**, and the one-line result.
Read EXPERIMENTS.md for full configs; this is the map. Current as of EXP-032.

Status legend: ✅ solved/working · ⚠️ partial/caveated · 🔬 diagnostic only · ❌ open issue

---

## 1. Direction steering ("go left/right/up/down")  ✅
- **What:** a plan steers the stick/d-pad in a specific cardinal direction.
- **Experiments:** EXP-006/007 (diagnosed stick-collapse = collinear conditioning),
  EXP-008/009 (FIX: direction-heavy + SupCon contrastive), EXP-010 (confirmed @225 chunks).
- **Best ckpt:** `runs/best_large/` — cos(L,R) 0.999→0.10, frozen DiT.
- **Recipe:** `--null-mode masked --freeze-dit --direction-heavy --contrastive-weight 1.0
  --contrastive-mode mean --plan-ratio 0.6 --lr-plan 3e-4`.

## 2. Exact null-invariance (CFG unconditional == base)  ✅
- **What:** a null/dropped plan reproduces the base model EXACTLY (so CFG works).
- **Experiments:** EXP-004 (learned vs masked A/B), EXP-008/010.
- **Mechanism:** `null_mode="masked"` zeros the K plan key positions in both attention
  masks; with a frozen DiT → null-invariance = 0.000. LoRA breaks exactness (~0.02).

## 3. Within-chunk temporal ordering (SEQ: "left THEN right")  ✅
- **What:** within one 18-step chunk, route segment-i meaning to chunk-region-i.
- **Experiments:** EXP-011 (failed first), EXP-014 (diagnosis: order lost at VLM +
  mean-pooled contrastive is order-blind), EXP-015 (cross-model probe: universal),
  **EXP-016 (FIX: order-aware contrastive flatten/pertoken)**, EXP-017 (SEQ3 thirds),
  EXP-018 (SEQ4 quarters generalize ZERO-SHOT).
- **Best ckpt:** `runs/seq_pertoken/` — SEQ2/3 all dirs, SEQ4 11/12 zero-shot, frozen DiT.
- **Key idea:** the LOSS was order-blind (mean), not the architecture. flatten=concat K,
  pertoken=per-position SupCon. Action-position embeddings do the routing.
- ❌ **See §11 (known regression):** this no longer trains from scratch; the ckpt is fine.

## 4. Heterogeneous sequencing (mix modalities: "left THEN jump")  ⚠️→✅(LoRA)
- **What:** sequence a stick DIRECTION and a discrete BUTTON across a chunk.
- **Experiments:** EXP-019 (zero-shot FAILS — buttons were never trained, grounded_only=True),
  EXP-020 (frozen training installs WEAK presses; + sampler-eval methodology note),
  **EXP-021 (LoRA → STRONG localized presses)**, EXP-022 (balance fixes y-axis: 11/12).
- **Best ckpt:** `runs/het_lora_bal/` — 14/16 on 8-combo eval, sharp buttons, null-inv 0.02.
- **Key idea:** routing a button to a temporal half needs DiT CAPACITY (LoRA); direction
  doesn't. Use the SAMPLER eval for buttons (velocity proxy lies on binary actions).

## 5. Uneven-duration plans ("briefly left, then right a long time")  ⚠️
- **What:** the plan TEXT sets the transition point (25/50/75%), not a fixed even split.
- **Experiments:** EXP-023 (zero-shot = fixed even prior; frozen training → COARSE
  content-driven), EXP-024 (LoRA → SHARPER, 3/4 pairs near-exact).
- **Best ckpt:** `runs/dur_lora/` (LoRA) / `runs/dur_pertoken/` (frozen, coarse).

## 6. Cross-chunk: one plan spans A chunks (cursor)  ✅
- **What:** K×A plan tokens; a per-chunk cursor selects block a → different per-chunk behavior.
- **Experiments:** **EXP-027 (mechanism, 16/16 shared frame)**, EXP-028 (REAL per-chunk
  frames, 16/16), EXP-030 (scales to A=4, 16/16).
- **Best ckpts:** `runs/cc_realframes/` (R0 real frames), `runs/cc_a4/` (A=4 horizon).
- **Key idea:** cursor selects a resampler block; routing is a property of the BLOCK, not
  the frame. Backward-compatible (A=1 unchanged).

## 7. Nested control (cross-chunk + within-chunk SEQ)  ✅
- **What:** "chunk0: left→right; chunk1: up→down" — cursor across chunks AND SEQ within each.
- **Experiment:** EXP-029 (24/24 nested + 16/16 hold retained, real frames).
- **Best ckpt:** `runs/cc_nested/`.
- **Key idea:** the two routing mechanisms (cursor ⊥ action-positions) COMPOSE cleanly.

## 8. R0 post-hoc: follow REAL multi-chunk plans  ✅
- **What:** trained on REAL action-chunk targets; plan = post-hoc dominant dirs; the cursor
  causally OVERRIDES the frame when the plan disagrees.
- **Experiment:** EXP-031 (16/16 causal on distinct-dir real episodes + 16/16 synth retained).
- **Best ckpt:** `runs/cc_posthoc/` — the most "real-data" result so far.
- **Boundary:** the frame is still the REAL frame; "override the streamer across the FUTURE
  the plan implies" (R2) needs counterfactual frames (world model / env). See §10.

## 9. Query self-attention (Q-former) toggle  ⚠️ (no synthetic benefit)
- **What:** optional `self-attn(queries)` in the resampler (true BLIP-2 Q-former).
- **Experiment:** EXP-032 (A/B). Direction 4/4→3/4, cross-chunk A=4 13/16→8/16 — ON is
  == or WORSE on synthetic (homogenizes queries vs needing distinct blocks).
- **Verdict:** keep OFF for Stage 1. **Untested + main hypothesis:** may help on Stage-2
  REAL transcripts (long/variable/semantic). Toggle `--resampler-self-attn` ready.

## 10. World model / R2 (counterfactual multi-chunk)  🔬 NOT STARTED
- **What:** override the streamer across REAL future frames the plan implies.
- **Plan:** Probe H1 — decode next-frame SigLIP features from DiT internals on real
  consecutive chunks (we HAVE the targets). If it predicts → free 1-step world model → R2.
- **Status:** designed (MULTICHUNK_DESIGN.md §5), not yet run.

## 11. ❌ KNOWN REGRESSION — order-contrastive won't learn from scratch
- **What:** a fresh seq_pertoken-recipe run fails to separate opposite orderings
  (cos(LR,RL) → ~0.9–1.0 vs original 0.16). DIRECTION (content) still learns fine.
- **Where:** EXP-032 addendum. Verified NOT the loss fn / labels / targets / mode / data.
- **Impact:** only blocks reproducing SEQ FROM SCRATCH; all existing ckpts (seq_pertoken,
  cc_*) are unaffected (they inherited ordering). Under investigation (variance vs
  too-few-batch-positives vs genuine bug).

---

## Data & infra facts (measured)
- **Direction scarcity (EXP-026):** strong-up = 5.1% of frames vs left/right 17%; y-axis
  used 1.63× less than x. UP is the consistent weak link — a DATA limit, not a recipe one.
- **Button↔down coupling:** pressing a button correlates with stick-down (+0.19/+0.23).
- **Dataset:** `nvidia/NitroGen` HF (actions only); SHARD_0000 extracted at /tmp/ds_real
  (166 videos, 48k chunks). We use 225 chunks/18 videos (/tmp/stage1_big, /tmp/frames_pre)
  + 1350 per-chunk cross-chunk frames (/tmp/frames_cc) + 417 expansion frames (/tmp/frames_more).
- **Train gotchas:** num_workers MUST be 0 (collate uses CUDA encoder); cache pixel_values;
  launch with python -u. ~7-9 it/s after cache warms.

## Doc map
- **EXPERIMENTS.md** — full configs/results, EXP-000..032 (authoritative).
- **DATAFLOW.md** — architecture / tensor path (System-2 → System-1), updated EXP-032.
- **MULTICHUNK_DESIGN.md** — long-horizon design (R0/R1/R2, K×A, world-model probe).
- **DESIGN.md** — original architecture + the "no pixels" dataset finding.
- **LITERATURE.md** — related work (GR00T/Helix/pi0/Genie/Hi Robot).
- **INDEX.md** — this file.
