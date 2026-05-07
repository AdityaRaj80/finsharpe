#!/usr/bin/env bash
# Fan-out the v2 training campaign across ALL available GPU partitions
# (H100, H200, A100) to maximise concurrent throughput.
#
# Usage:
#   ./scripts/submit_campaign_v2.sh [stage] [models] [horizons] [folds] [arms]
#
# Defaults (Stage 1 = F4-headline only):
#   models   = PatchTST,TFT,GCFormer,DLinear,LSTM,RNN,CNN
#   horizons = 5,20,60,120,240
#   folds    = F4
#   arms     = mse,riskhead
#   -> 7*5*1*2 = 70 jobs, round-robin'd across H100 / H200 / A100
#
# Each cell job is single-GPU; concurrency comes from spreading across
# partitions (each partition has its own QOS limit). This avoids the
# QOSMaxCpuPerUserLimit pile-up that earlier blocked stragglers.
#
# Stage 2 (walk-forward F1-F3): pass folds=F1,F2,F3 -> 210 more jobs
# Stage 3 (full grid all-folds): pass folds=F1,F2,F3,F4

set -euo pipefail

STAGE="${1:-stage1}"
MODELS_CSV="${2:-PatchTST,TFT,GCFormer,DLinear,LSTM,RNN,CNN}"
HORIZONS_CSV="${3:-5,20,60,120,240}"
FOLDS_CSV="${4:-F4}"
ARMS_CSV="${5:-mse,riskhead}"

case "$STAGE" in
    stage1)  FOLDS_CSV="F4" ;;
    stage2)  FOLDS_CSV="F1,F2,F3" ;;
    stage3)  FOLDS_CSV="F1,F2,F3,F4" ;;
    smoke)   ;;
    *)       echo "Unknown stage: $STAGE" >&2; exit 1 ;;
esac

IFS=',' read -ra MODELS <<<"$MODELS_CSV"
IFS=',' read -ra HORIZONS <<<"$HORIZONS_CSV"
IFS=',' read -ra FOLDS <<<"$FOLDS_CSV"
IFS=',' read -ra ARMS <<<"$ARMS_CSV"

# Round-robin partition pool (paired with QOS).
# Order matters: index 0 is the first one assigned, index 1 second, etc.
# Larger VRAM (H200 141GB > H100 80GB > A100 80GB) gets bigger batch.
PARTITIONS=("gpu_h100_4"     "gpu_h200_8"     "gpu_a100_8")
QOSES=(      "qos_gpu_h100"  "qos_gpu_h200"   "qos_gpu_a100")
BATCH_SIZES=("512"           "768"            "512")

n_jobs=$((${#MODELS[@]} * ${#HORIZONS[@]} * ${#FOLDS[@]} * ${#ARMS[@]}))

echo "============================================================"
echo "Campaign v2 -- $STAGE"
echo "============================================================"
echo "models   : ${MODELS[*]}"
echo "horizons : ${HORIZONS[*]}"
echo "folds    : ${FOLDS[*]}"
echo "arms     : ${ARMS[*]}"
echo "n_jobs   : $n_jobs"
echo "partitions (round-robin): ${PARTITIONS[*]}"
echo

read -p "Submit $n_jobs jobs? [y/N] " -r REPLY
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

mkdir -p "$HOME/logs"

submitted=0
rr_idx=0
for MODEL in "${MODELS[@]}"; do
    for H in "${HORIZONS[@]}"; do
        for FOLD in "${FOLDS[@]}"; do
            for ARM in "${ARMS[@]}"; do
                p_idx=$((rr_idx % ${#PARTITIONS[@]}))
                PART="${PARTITIONS[$p_idx]}"
                QOS="${QOSES[$p_idx]}"
                BS="${BATCH_SIZES[$p_idx]}"
                jname="${MODEL}_H${H}_${FOLD}_${ARM}"

                jid=$(sbatch --parsable \
                    --partition="$PART" \
                    --qos="$QOS" \
                    --job-name="$jname" \
                    --export=ALL,MODEL="$MODEL",HORIZON="$H",FOLD="$FOLD",ARM="$ARM",BATCH_SIZE="$BS" \
                    scripts/train_campaign.sbatch)
                submitted=$((submitted + 1))
                echo "[$submitted/$n_jobs] $jname  partition=$PART  bs=$BS  jobid=$jid"
                rr_idx=$((rr_idx + 1))
            done
        done
    done
done

echo
echo "Submitted $submitted jobs across ${#PARTITIONS[@]} partitions."
echo "Monitor with:"
echo "  squeue -u \$USER -o '%.10i %.30j %.18P %.8T %.10M %R'"
