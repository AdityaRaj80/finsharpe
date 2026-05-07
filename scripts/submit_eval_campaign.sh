#!/usr/bin/env bash
# Fan eval_v2 over all checkpoints via SLURM (each cell on its own GPU).
# Resume-safe: skips cells whose summary JSON already exists (FORCE=1 to override).
#
# Usage:  bash scripts/submit_eval_campaign.sh stage1
set -euo pipefail
STAGE="${1:-stage1}"
case "$STAGE" in
    stage1)  FOLDS=("F4") ;;
    stage2)  FOLDS=("F1" "F2" "F3") ;;
    stage3)  FOLDS=("F1" "F2" "F3" "F4") ;;
    *)       echo "Unknown stage"; exit 1 ;;
esac

MODELS=("PatchTST" "TFT" "GCFormer" "DLinear" "LSTM" "RNN" "CNN")
HORIZONS=(5 20 60 120 240)
ARMS=("mse" "riskhead")

PARTITIONS=("gpu_h100_4" "gpu_h200_8" "gpu_a100_8")
QOSES=("qos_gpu_h100" "qos_gpu_h200" "qos_gpu_a100")

REPO=$HOME/finsharpe
CKPT_DIR=$REPO/checkpoints
OUT_DIR=$REPO/results/eval_v2
mkdir -p "$OUT_DIR" "$HOME/logs"

n_submit=0; n_skip=0; n_miss=0; rr=0
for MODEL in "${MODELS[@]}"; do
  for H in "${HORIZONS[@]}"; do
    for FOLD in "${FOLDS[@]}"; do
      for ARM in "${ARMS[@]}"; do
        ckpt="$CKPT_DIR/${MODEL}_global_H${H}_${FOLD}_${ARM}.pth"
        summary="$OUT_DIR/summary_${MODEL}_H${H}_${FOLD}_${ARM}.json"
        if [ ! -f "$ckpt" ]; then n_miss=$((n_miss+1)); continue; fi
        if [ -f "$summary" ] && [ -z "${FORCE:-}" ]; then n_skip=$((n_skip+1)); continue; fi
        p_idx=$((rr % ${#PARTITIONS[@]}))
        sbatch --partition="${PARTITIONS[$p_idx]}" --qos="${QOSES[$p_idx]}" \
            --job-name="ev_${MODEL}_H${H}_${FOLD}_${ARM}" \
            --export=ALL,MODEL="$MODEL",HORIZON="$H",FOLD="$FOLD",ARM="$ARM" \
            "$REPO/scripts/eval_v2_slurm.sbatch" > /dev/null
        n_submit=$((n_submit+1))
        rr=$((rr+1))
      done
    done
  done
done
echo "submitted=$n_submit  skipped=$n_skip  missing=$n_miss"
