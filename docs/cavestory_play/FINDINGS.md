# EXP-050 / EXP-051: Frame-conditioned planning — findings

_Session 2026-06-21. All checkpoints in `runs/`, data caches in `/tmp`, artifacts here in `docs/cavestory_play/`._

## TL;DR
- **EXP-050 (frame-conditioning) worked for what it targets:** distill loss 1.50 → **0.35**, plan-token
  collinearity **0.350 → 0.223** across frames. The planner now produces frame-specific tokens, and in-game
  it's more active than the text student.
- **But the DiT does NOT obey a counterfactual plan at ANY CFG weight.** Asking "move left" on a
  rightward-prior frame never flips the stick left — for any variant (mm, mm+contrastive, mm+LoRA,
  mm+LoRA+counterfactual).
- **Root cause = the drowning effect.** In `encode_multimodal`, the plan text (~5 tokens) is drowned by
  ~300 image tokens, so opposite plan texts on the *same* frame produce near-identical plan tokens
  (cosine **0.9996**). Frame-conditioning improved specificity *across* frames but *weakened*
  text-controllability *on a fixed* frame — the regime override needs.

## The variants (all warm-started from the EXP-050a mm student, frozen DiT)
| ckpt | recipe |
|---|---|
| `runs/stage2_student_mm` | EXP-050a: frame-conditioned distillation |
| `runs/stage2_student_mm_con` | EXP-050b: + outcome-contrastive |
| `runs/stage2_student_mm_lora` | EXP-051a: + LoRA(16) on DiT cross-attn |
| `runs/stage2_student_mm_lora_cf` | EXP-051b: LoRA + counterfactual override |
| `runs/stage2_student` | text-only student (EXP-045), reference |

## Decisive eval — `planner_poc/eval_cfg_override.py`
Sweep CFG, opposing plans (move left vs right) × 3 real frames. Metric = mean direction-separation
`stick_x[right] − stick_x[left]` (>0 ⇒ plan controls direction). See `exp051_override_summary.png`.

| variant | mean_sep | frac separating | note |
|---|---|---|---|
| text_student (text-only) | **+0.077** | 0.27 | strongest text authority |
| mm (050a) | +0.002 | 0.13 | ~0: no direction control |
| mm_con (050b) | +0.005 | 0.20 | ~0 |
| mm_lora (051a) | −0.032 | 0.07 | LoRA alone didn't help |
| mm_lora_cf (051b) | +0.017 | **0.33** | best mm; counterfactual nudged it |

**stick_x stays positive (rightward) in ~every cell, even with "move left."** None achieve reliable
override. Consistent with the prior EXP-047/049 "data negative is robust" finding.

## Root-cause probe (same frame, "move left" vs "move right" pooled plan tokens)
- mm (frame+text): cosine **0.9996** (indistinct)
- text-only: cosine **0.9989** (slightly more distinct) — and 4× the override separation.

## The tension (the real takeaway)
Frame-conditioning and counterfactual override want **opposite regimes**:
- **Factual frame-grounded assistance** ("what should I do here?") → frame should dominate → **mm wins**.
- **Counterfactual override** ("do X regardless of the scene") → plan text should dominate → **text wins**.

A bigger encoder won't fix this (the image:text token ratio is unchanged). Candidate fixes:
inject the plan text as a **separate, undrowned token stream** (or upsample/weight text tokens in the
resampler), or a **hybrid** (mm hidden for grounding + dedicated text-plan tokens), or simply route by
intent (text-encode for override, mm-encode for assistance).

## Closed-loop System-2 (`cavestory_closed_loop.py`, `closed_loop.mp4`)
The planner side **works**: a capable VLM (9B), re-invoked when Quote is stuck, correctly says
"move left and jump onto the ledge". The **0.8B planner says the wrong thing ("go right")**; 2B and 9B
get it right. But the frozen DiT ignores the corrective plan → Quote stays cornered. Bottleneck is DiT
authority, not the planner.

## Artifacts in this folder
- `exp051_override_summary.png` — the bar chart above.
- `compare_text_vs_mm.png`, `null_vs_mm.png` — in-game frame strips.
- `annotated_mm.mp4` + `actions_mm.txt` — per-action playthrough (action drawn on each frame).
- `closed_loop.mp4` + `actions_closed_loop.txt` — 9B replanning loop (plan updates; DiT ignores).
- `run_text.mp4`, `run_mm.mp4` — text vs mm playthroughs.

---

## UPDATE — EXP-052 + GOLD TOKENS: counterfactual override WORKS in-game (env-free)

### The chain that cracked it
1. **EXP-052 (text-token-only encoding):** modified `encode_multimodal(text_only=True)` to return only
   the ~20 frame-contextualized text tokens (drop the ~600 image tokens). This **un-drowns the plan**:
   override separation rose from mm's +0.002 to mm_txt's +0.056 (toward text-only's +0.077). Necessary
   but not sufficient — fixed-frame stick still stayed rightward (strong prior).
2. **Diagnostic — gold-token injection:** injecting a TRAINING teacher token whose chunk went LEFT vs one
   that went RIGHT gave stick_x +0.285 vs +0.944 on the Cave Story start frame — **separation 0.66**.
   => the frozen DiT DOES respond strongly to plan tokens; the bottleneck was Cave Story plan-text
   producing weak/washed tokens, NOT the DiT being deaf.
3. **The fix (env-free, your "gold plans" idea at the token level):** use the single best gold-LEFT
   teacher token + gold-token CFG (w=16) + multi-seed selection (pick the most-left of N seeds). On the
   start frame this hits stick_x −0.68 (seed0), 50% of seeds < −0.3.

### In-game POC result (`cavestory_poc_gold.py`, `poc_gold.mp4`, `poc_gold_trajectory.png`)
From the SAME First Cave start (x=160):
- **plan TEXT "go left"** -> drifts RIGHT to the corner (x=211, +51 tiles) — the prior wins.
- **GOLD-LEFT token**     -> goes LEFT (x=160 -> 125, **−35 tiles**) — **counterfactual override works.**

Quote did NOT enter the door (it fell to a lower-left ledge; stick-only gold tokens + crude jump-assist
couldn't climb the final platforms back up to the upper-left door). But the HARD, NOVEL part — moving
LEFT against NitroGen's strong rightward prior, with NO env training — is demonstrated.

### What this means
- Env-free counterfactual control **is achievable** on a strong-prior OOD frame, but it needs STRONG plan
  tokens, not plan text. The Cave Story plan-text path (even text-only) gives tokens too weak to flip the
  prior; a training-derived "gold" direction token does it.
- Path to full plans: distill the planner so its OUTPUT tokens match the strong gold-token geometry
  (i.e. make "go left" plan text produce a token like the gold-left token), or build a small gold-token
  library keyed to intent and have the System-2 VLM select among them.
- Remaining gap for the door: vertical navigation (jump/climb) — the gold tokens are stick-only; a
  "jump"/south-button gold token (or a jump sub-policy) is needed to actually enter the door.

### New files
- `nitrogen/planner.py` encode_multimodal(text_only=...); `eval_policy.py` sample_chunk_token (gold-token CFG).
- `planner_poc/build_gold_tokens.py` (-> /tmp/gold_tokens.pt), `cavestory_poc_gold.py`, `eval_cfg_override.py`,
  `ingame_dir_test.py`, `run_exp052.sh`.
- Checkpoints: runs/stage2_student_mm_txt{,_lora_cf}/plan_stage1_2500.pt
