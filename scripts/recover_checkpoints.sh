#!/usr/bin/env bash
# recover_checkpoints.sh — reconstruct ALL relevant checkpoints on a fresh box.
#
# Pulls everything needed to run the plan-conditioning models from scratch:
#   1. NVIDIA NitroGen base DiT            (hf: nvidia/NitroGen ng.pt)        -> ckpts/nitrogen/ng.pt
#   2. Qwen planner backbone               (hf: Qwen/Qwen3.5-2B)              -> HF cache
#   3. our handoff repo (slim base + 9 plan_head deltas + rwbc + 56 demos)
#                                          (hf: nmk-kun/nitrogen-vlm-planner-handoff, PRIVATE)
#                                                                             -> ckpts/handoff_hf/
#   4. reconstruct the FULL base eval ckpt from ng.pt + the slim base
#                                          (merge_slim_to_full.py)            -> ckpts/btn_s600_full.pt
#   5. expose the deltas at a stable path                                     -> ckpts/deltas/, ckpts/rwbc/
#
# Idempotent: skips anything already present. Re-run safe.
#
# Prereqs: the repo's uv venv (.venv) and `hf auth login` (the handoff repo is PRIVATE).
#
# Usage:
#   bash scripts/recover_checkpoints.sh
#   QWEN=Qwen/Qwen3.5-0.8B bash scripts/recover_checkpoints.sh   # 0.8B backbone instead
set -euo pipefail

REPO_ROOT="${NITROGEN_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"
PY="${PY:-.venv/bin/python}"
HF="${HF:-.venv/bin/hf}"
QWEN="${QWEN:-Qwen/Qwen3.5-2B}"
HANDOFF_REPO="${HANDOFF_REPO:-nmk-kun/nitrogen-vlm-planner-handoff}"

say() { printf "\n\033[1;36m== %s\033[0m\n" "$*"; }
have() { [ -s "$1" ]; }

[ -x "$PY" ] || { echo "ERROR: $PY not found. Create the uv venv first (uv venv .venv; uv pip install -e .)"; exit 1; }

say "0/5  auth check (handoff repo is PRIVATE)"
if ! "$HF" auth whoami >/dev/null 2>&1; then
  echo "ERROR: not logged in to Hugging Face. Run:  $HF auth login   (paste a token with read access)"; exit 1
fi
"$HF" auth whoami 2>/dev/null | head -2 || true

say "1/5  NVIDIA NitroGen base DiT -> ckpts/nitrogen/ng.pt"
mkdir -p ckpts/nitrogen
if have ckpts/nitrogen/ng.pt; then
  echo "  already present, skip"
else
  "$HF" download nvidia/NitroGen ng.pt --local-dir ckpts/nitrogen
  # some HF layouts nest it; normalize
  [ -s ckpts/nitrogen/ng.pt ] || find ckpts/nitrogen -name ng.pt -exec cp {} ckpts/nitrogen/ng.pt \;
fi
have ckpts/nitrogen/ng.pt && echo "  OK ng.pt ($(du -h ckpts/nitrogen/ng.pt | cut -f1))"

say "2/5  Qwen planner backbone -> HF cache ($QWEN)"
"$HF" download "$QWEN" >/dev/null && echo "  OK $QWEN cached"

say "3/5  handoff repo (slim base + deltas + rwbc + demos) -> ckpts/handoff_hf/"
mkdir -p ckpts/handoff_hf
"$HF" download "$HANDOFF_REPO" --repo-type model --local-dir ckpts/handoff_hf >/dev/null
echo "  OK ($(find ckpts/handoff_hf -type f | wc -l) files, $(du -sh ckpts/handoff_hf | cut -f1))"

say "4/5  reconstruct FULL base eval ckpt -> ckpts/btn_s600_full.pt"
if have ckpts/btn_s600_full.pt; then
  echo "  already present, skip"
else
  env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH="$REPO_ROOT:$REPO_ROOT/planner_poc" QWEN="$QWEN" \
    "$PY" planner_poc/merge_slim_to_full.py ckpts/handoff_hf/base/btn_s600.pt ckpts/btn_s600_full.pt
fi
have ckpts/btn_s600_full.pt && echo "  OK btn_s600_full.pt ($(du -h ckpts/btn_s600_full.pt | cut -f1))"

say "5/5  expose deltas at stable paths -> ckpts/deltas/, ckpts/rwbc/"
mkdir -p ckpts/deltas ckpts/rwbc
cp -f ckpts/handoff_hf/deltas/*.pt ckpts/deltas/ 2>/dev/null || true
cp -f ckpts/handoff_hf/rwbc/*.pt   ckpts/rwbc/   2>/dev/null || true
echo "  deltas: $(ls ckpts/deltas/*.pt 2>/dev/null | wc -l) | rwbc: $(ls ckpts/rwbc/*.pt 2>/dev/null | wc -l)"

say "DONE — recovered checkpoints"
cat <<EOF
  base (frozen DiT)        ckpts/nitrogen/ng.pt
  base eval ckpt (full)    ckpts/btn_s600_full.pt        <- load this in NitroGenPolicy
  plan_head deltas         ckpts/deltas/{pooled_planfit,r9_smw_kl,r9_smw_situ}_s{0,1,2}.pt
  actor-OOD rwbc deltas    ckpts/rwbc/*.pt
  demos                    ckpts/handoff_hf/demos/furthest/<game>/<tag>__state<N>.mp4

  Quick test (SMW, KL-anchor delta, base vs trained):
    ENVP='env -u VIRTUAL_ENV -u PYTHONPATH PYTHONPATH=$REPO_ROOT:$REPO_ROOT/planner_poc QWEN=$QWEN'
    \$ENVP $PY planner_poc/furthest_rollout.py --game smw --tag kl --mode plan \\
        --delta ckpts/deltas/r9_smw_kl_s0.pt --state 7 --seconds 90 --log-plans --out out/

  Recipe + how the deltas load on top of the base: docs/TRAINING_RECIPE.md
EOF
