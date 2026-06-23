#!/usr/bin/env bash
# EXP-050 A/B: frame-conditioned plan distillation, with and without outcome-contrastive.
# Run A: pure distillation (matches EXP-045 recipe, frame input). Run B: + outcome-contrastive
# (EXP-043/046 de-collinearization on top). Frozen DiT, warm-started from the text student.
set -e
cd /home/t-nagupta/NitroGen
export PYTHONPATH=/home/t-nagupta/NitroGen
PY=.venv/bin/python

COMMON=(
  --ng-ckpt ckpts/nitrogen/ng.pt --qwen ckpts/qwen35-0.8b
  --shard-root /tmp/stage1_big --shard-root /tmp/stage1_more
  --frames-dir /tmp/frames_cc
  --vlm-plan-lookup /tmp/stage2_plan_lookup_mm.json
  --mm-plan-hidden-lookup /tmp/stage2_mm_hidden.pt
  --teacher-token-lookup /tmp/stage2_teacher_tokens_mm.pt
  --distill-weight 1.0 --gameplay-only --freeze-dit
  --init-from runs/stage2_student/plan_stage1_2500.pt
  --plan-ratio 0.5 --batch-size 8 --steps 2500
  --save-every 2500 --log-every 25
)

echo "===== EXP-050a: pure distillation (frame-conditioned) ====="
env -u VIRTUAL_ENV $PY -u scripts/train_planner.py "${COMMON[@]}" \
  --out-dir runs/stage2_student_mm

echo "===== EXP-050b: distillation + outcome-contrastive ====="
env -u VIRTUAL_ENV $PY -u scripts/train_planner.py "${COMMON[@]}" \
  --s2-outcome-contrastive --contrastive-weight 1.0 --contrastive-mode pertoken \
  --out-dir runs/stage2_student_mm_con

echo "===== BOTH DONE ====="
