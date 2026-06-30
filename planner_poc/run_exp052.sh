#!/usr/bin/env bash
# EXP-052: TEXT-TOKEN-ONLY frame-conditioned student. encode_multimodal returns only the
# frame-contextualized text tokens (~20) instead of the full ~600 image + ~20 text sequence,
# so the plan text is NOT drowned -> should restore counterfactual/directional authority while
# keeping frame grounding (text tokens attended to the frame). Warm-start from the mm student.
#   052a: text-only + factual distillation (clean grounding baseline).
#   052b: text-only + LoRA + counterfactual override (the override test).
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
  --mm-plan-hidden-lookup /tmp/stage2_mm_hidden_txt.pt
  --teacher-token-lookup /tmp/stage2_teacher_tokens_mm.pt
  --distill-weight 1.0 --gameplay-only --freeze-dit
  --init-from runs/stage2_student_mm/plan_stage1_2500.pt
  --plan-ratio 0.5 --batch-size 8 --steps 2500
  --save-every 2500 --log-every 25
)

echo "===== EXP-052a: text-only + factual distillation ====="
env -u VIRTUAL_ENV $PY -u scripts/train_planner.py "${COMMON[@]}" \
  --out-dir runs/stage2_student_mm_txt

echo "===== EXP-052b: text-only + LoRA + counterfactual override ====="
env -u VIRTUAL_ENV $PY -u scripts/train_planner.py "${COMMON[@]}" \
  --lora-dit "$LORA_RANK" \
  --mm-cf-lookup /tmp/stage2_mm_cf_txt.pt --s2-cf-ratio 0.5 \
  --out-dir runs/stage2_student_mm_txt_lora_cf

echo "===== EXP-052 BOTH DONE ====="
