#!/usr/bin/env bash
# LoRA balance/robustness sweep. Warm-starts the clean plan-head model + LoRA on DiT cross-attn + the
# counterfactual override objective (the validated recipe), varying the knobs that plausibly control the
# 600-balanced / 1200-left-lean drift: DiT learning rate, cf-ratio (override pressure), and LoRA rank.
# Frequent checkpoints (--save-every 250) so we can map the balance trajectory per config.
set -u
NITROGEN_REPO="${NITROGEN_REPO:-/home/t-nagupta/NitroGen-With-VLM-Planning}"
cd "$NITROGEN_REPO"
export PYTHONPATH="$NITROGEN_REPO:$NITROGEN_REPO/planner_poc"

COMMON=(scripts/train_planner.py
  --ng-ckpt ckpts/nitrogen/ng.pt --qwen Qwen/Qwen3.5-2B
  --shard-root /tmp/stage1_more --frames-dir /tmp/frames_cc
  --init-from runs/stage2_2b_clean/plan_stage1_2000.pt
  --vlm-plan-lookup /tmp/stage2_plan_lookup_mm_clean.json
  --mm-plan-hidden-lookup /tmp/stage2_mm_hidden_2b_clean.pt
  --mm-cf-lookup /tmp/stage2_mm_cf_2b.pt
  --gameplay-only --s2-outcome-contrastive --contrastive-weight 1.0 --contrastive-mode pertoken
  --num-plan-tokens 8 --num-chunks 1 --null-mode masked --freeze-dit
  --plan-ratio 0.6 --lr-plan 1e-4 --batch-size 32 --num-workers 4 --save-every 250 --log-every 50)

run () {  # name  lora_rank  cf_ratio  lr_dit  steps
  local name=$1 rank=$2 cf=$3 lrdit=$4 steps=$5
  local out=runs/sweep_${name}
  echo "=== [$(date -u +%H:%M:%S)] START $name (rank=$rank cf=$cf lr_dit=$lrdit steps=$steps) ==="
  .venv/bin/python "${COMMON[@]}" \
    --out-dir "$out" --lora-dit "$rank" --s2-cf-ratio "$cf" --lr-dit "$lrdit" --steps "$steps" \
    > /tmp/sweep_${name}.log 2>&1
  echo "=== [$(date -u +%H:%M:%S)] DONE $name -> $out ($(ls $out/*.pt 2>/dev/null | wc -l) ckpts) ==="
}

# config            name        rank  cf    lr_dit  steps
run lowlr           16    0.5   5e-5    2000
run lowcf           16    0.35  2e-4    2000
run rank32          32    0.5   2e-4    1500
echo "=== [$(date -u +%H:%M:%S)] SWEEP COMPLETE ==="
