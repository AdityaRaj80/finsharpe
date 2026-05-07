#!/usr/bin/env bash
# Fan eval_v2.py over every Stage-1 checkpoint. Runs locally on whichever
# host has GPU access (HPC login or workstation) -- a single-cell eval is
# fast (~30-90s on H100) so we don't need SLURM fan-out for this stage.
#
# Usage:
#   bash scripts/eval_campaign.sh [stage]
#
# stage: stage1 = F4 only, stage2 = F1/F2/F3, stage3 = F1..F4 (everything)
#
# For each (model, horizon, fold, arm) combination we:
#   1. Check that the checkpoint exists; skip if not.
#   2. Skip if results/eval_v2/summary_<cell>.json already exists (resume).
#   3. Run eval_v2.py.

set -euo pipefail

STAGE="${1:-stage1}"

case "$STAGE" in
    stage1)  FOLDS=("F4") ;;
    stage2)  FOLDS=("F1" "F2" "F3") ;;
    stage3)  FOLDS=("F1" "F2" "F3" "F4") ;;
    *)       echo "Unknown stage: $STAGE"; exit 1 ;;
esac

MODELS=("PatchTST" "TFT" "GCFormer" "DLinear" "LSTM" "RNN" "CNN")
HORIZONS=(5 20 60 120 240)
ARMS=("mse" "riskhead")

REPO=${REPO:-$HOME/finsharpe}
CKPT_DIR="$REPO/checkpoints"
OUT_DIR="$REPO/results/eval_v2"
mkdir -p "$OUT_DIR"

n_total=$((${#MODELS[@]} * ${#HORIZONS[@]} * ${#FOLDS[@]} * ${#ARMS[@]}))
n_evaluated=0
n_skipped=0
n_missing=0

for MODEL in "${MODELS[@]}"; do
  for H in "${HORIZONS[@]}"; do
    for FOLD in "${FOLDS[@]}"; do
      for ARM in "${ARMS[@]}"; do
        ckpt="$CKPT_DIR/${MODEL}_global_H${H}_${FOLD}_${ARM}.pth"
        summary="$OUT_DIR/summary_${MODEL}_H${H}_${FOLD}_${ARM}.json"

        if [ ! -f "$ckpt" ]; then
          n_missing=$((n_missing + 1))
          continue
        fi
        if [ -f "$summary" ] && [ -z "${FORCE:-}" ]; then
          n_skipped=$((n_skipped + 1))
          continue
        fi
        echo "[eval] $MODEL H=$H $FOLD $ARM"
        cd "$REPO"
        python smoke/eval_v2.py \
            --model "$MODEL" --horizon "$H" --fold "$FOLD" --arm "$ARM" \
            --batch_size 512 \
            2>&1 | tail -10 || echo "  [skip-on-error]"
        n_evaluated=$((n_evaluated + 1))
      done
    done
  done
done

echo
echo "============================================================"
echo "eval_campaign $STAGE: evaluated=$n_evaluated  skipped(cached)=$n_skipped  missing(no_ckpt)=$n_missing  total=$n_total"
echo "Output dir: $OUT_DIR"
