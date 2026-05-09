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
HORIZONS_CSV="${3:-5,20,60}"   # H=240 + H=120 dropped (n_val=0 in 1yr folds)
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
# Jury 2 fix C1: BATCH_SIZE held constant across partitions for cross-cell
# reproducibility. Larger H200 VRAM is left as headroom (we could use it
# for more concurrent jobs but NOT for varying bs within the same campaign).
PARTITIONS=("gpu_h100_4"     "gpu_h200_8"     "gpu_a100_8")
QOSES=(      "qos_gpu_h100"  "qos_gpu_h200"   "qos_gpu_a100")
BATCH_SIZE_FIXED="${BATCH_SIZE_FIXED:-512}"  # default 512; can be overridden via env

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

# Auto-confirm path for non-interactive launches: pass YES=1 env var.
if [ "${YES:-}" = "1" ]; then
    echo "[YES=1] auto-confirming submission of $n_jobs jobs"
else
    read -p "Submit $n_jobs jobs? [y/N] " -r REPLY
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
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
                jname="${MODEL}_H${H}_${FOLD}_${ARM}"

                # Jury 2 fix J1: skip if checkpoint already exists (resume safety)
                ckpt="$HOME/finsharpe/checkpoints/${MODEL}_global_H${H}_${FOLD}_${ARM}.pth"
                if [ -f "$ckpt" ] && [ -z "${FORCE:-}" ]; then
                    echo "  [skip] $jname  (checkpoint exists; FORCE=1 to override)"
                    rr_idx=$((rr_idx + 1))
                    continue
                fi

                jid=$(sbatch --parsable \
                    --partition="$PART" \
                    --qos="$QOS" \
                    --job-name="$jname" \
                    --export=ALL,MODEL="$MODEL",HORIZON="$H",FOLD="$FOLD",ARM="$ARM",BATCH_SIZE="$BATCH_SIZE_FIXED" \
                    scripts/train_campaign.sbatch)
                submitted=$((submitted + 1))
                echo "[$submitted/$n_jobs] $jname  partition=$PART  bs=$BATCH_SIZE_FIXED  jobid=$jid"
                rr_idx=$((rr_idx + 1))
            done
        done
    done
done

echo
echo "Submitted $submitted jobs across ${#PARTITIONS[@]} partitions."
echo "Monitor with:"
echo "  squeue -u \$USER -o '%.10i %.30j %.18P %.8T %.10M %R'"
