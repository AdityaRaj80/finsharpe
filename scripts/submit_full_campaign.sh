#!/bin/bash
# finsharpe — Full submission script.
#
# Two-arm two-stage campaign on the SENTIMENT-CORRECTED data:
#   Stage A.  MSE-baselines from scratch (35 jobs, ~5 days HPC at 4 in flight)
#   Stage B.  Track B fine-tunes via --init_from on Stage-A MSE checkpoints
#             (35 jobs, ~3 days HPC)
#
# The NO-SENTIMENT arm already exists (v1 fan-out 167702-167731 on the
# predecessor data; harvested results into finsharpe/results/no_sentiment/).
#
# Usage:
#   bash scripts/submit_full_campaign.sh mse           ALL    # Stage A, all 7 models
#   bash scripts/submit_full_campaign.sh riskhead      ALL    # Stage B, all 7 models
#   bash scripts/submit_full_campaign.sh mse           DLinear iTransformer
#   bash scripts/submit_full_campaign.sh riskhead      GCFormer

set -e
cd /scratch/goyalpoonam/finsharpe

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <mse|riskhead> <MODEL>... | ALL"
    exit 1
fi

STAGE="$1"; shift
case "$STAGE" in
    mse|riskhead) ;;
    *) echo "ERROR: stage must be 'mse' or 'riskhead' (got '$STAGE')" >&2; exit 1 ;;
esac

if [[ "$1" == "ALL" ]]; then
    MODELS=(DLinear iTransformer GCFormer PatchTST AdaPatch TFT VanillaTransformer)
else
    MODELS=("$@")
fi

HORIZONS=(5 20 60 120 240)
PARTITIONS=(gpu_h100_4 gpu_h200_8 gpu_h100_4 gpu_h200_8 gpu_h100_4)

JOB_NUM=0
for MODEL in "${MODELS[@]}"; do
    if [[ "$STAGE" == "riskhead" ]]; then
        # Pre-flight: every horizon must have a sentiment-MSE checkpoint to init from.
        OK=true
        for H in "${HORIZONS[@]}"; do
            CKPT="checkpoints/${MODEL}_global_H${H}.pth"
            if [[ ! -f "$CKPT" ]]; then
                echo "[SKIP $MODEL H$H] missing init checkpoint at $CKPT — train Stage A first."
                OK=false
            fi
        done
        if [[ "$OK" == "false" ]]; then
            echo "[ERROR] $MODEL not ready for Stage B."
            continue
        fi
    fi
    for i in "${!HORIZONS[@]}"; do
        H="${HORIZONS[$i]}"
        PART="${PARTITIONS[$i]}"
        if [[ "$STAGE" == "mse" ]]; then
            JOB_NAME="${MODEL}_MSE_H${H}"
            SBATCH_FILE="scripts/train_mse_baseline.sbatch"
        else
            JOB_NAME="${MODEL}_RH_H${H}"
            SBATCH_FILE="scripts/train_riskhead.sbatch"
        fi
        echo "[submit] $JOB_NAME on $PART"
        sbatch --partition="$PART" --job-name="$JOB_NAME" \
               --export=ALL,MODEL="$MODEL",H="$H" \
               "$SBATCH_FILE"
        JOB_NUM=$((JOB_NUM + 1))
        sleep 1
    done
done

echo
echo "Submitted $JOB_NUM job(s). Monitor with:"
echo "  squeue -u \$USER --format='%.10i %.30j %.10P %.10M %.8T'"
