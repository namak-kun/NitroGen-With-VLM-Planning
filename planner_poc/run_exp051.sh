#!/usr/bin/env bash
# EXP-051 A/B: frame-conditioned student + LoRA on DiT cross-attention, to give the frozen DiT
# the capacity to actually ATTEND to the plan tokens (the EXP-050 CFG-sweep showed the DiT obeys
# at NO weight -> the plan is structurally outvoted). Warm-start from the EXP-050a mm student.
#   051a: LoRA + factual distillation (capacity amplifies the plan where it agrees w/ the frame).
#   051b: LoRA + counterfactual override (s2_cf_ratio: plan contradicts the frame -> teach OVERRIDE).
set -e
NITROGEN_REPO="${NITROGEN_REPO:-/home/t-nagupta/NitroGen-With-VLM-Planning}"
cd "$NITROGEN_REPO"
export PYTHONPATH="$NITROGEN_REPO"
PY=.venv/bin/python
LORA_RANK=${LORA_RANK:-16}

COMMON=(
  --ng-ckpt ckpts/nitrogen/ng.pt --qwen ckpts/qwen35-0.8b
  --shard-root /tmp/stage1_big --shard-root /tmp/stage1_more
  --frames-dir /tmp/frames_cc
  --vlm-plan-lookup /tmp/stage2_plan_lookup_mm.json
  --mm-plan-hidden-lookup /tmp/stage2_mm_hidden.pt
  --teacher-token-lookup /tmp/stage2_teacher_tokens_mm.pt
  --distill-weight 1.0 --gameplay-only --freeze-dit
  --lora-dit "$LORA_RANK"
  --init-from runs/stage2_student_mm/plan_stage1_2500.pt
  --plan-ratio 0.5 --batch-size 8 --steps 2500
  --save-every 2500 --log-every 25
)

echo "===== EXP-051a: LoRA(rank=$LORA_RANK) + factual distillation ====="
env -u VIRTUAL_ENV $PY -u scripts/train_planner.py "${COMMON[@]}" \
  --out-dir runs/stage2_student_mm_lora

echo "===== EXP-051a DONE ====="
