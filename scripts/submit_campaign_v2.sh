#!/usr/bin/env bash
# Fan-out the v2 training campaign over the requested grid.
#
# Usage:
#   ./scripts/submit_campaign_v2.sh [stage] [models] [horizons] [folds] [arms]
#
# Defaults (Stage 1 = F4-headline only):
#   models   = PatchTST,TFT,GCFormer,DLinear,LSTM,RNN,CNN
#   horizons = 5,20,60,120,240
#   folds    = F4
#   arms     = mse,riskhead
#   -> 7*5*1*2 = 70 jobs
#
# Stage 2 (walk-forward F1-F3): pass folds=F1,F2,F3 -> 210 more jobs
# Stage 3 (full grid all-folds): pass folds=F1,F2,F3,F4 (do NOT re-run F4)
#
# Examples:
#   ./scripts/submit_campaign_v2.sh stage1                # F4 only, all models, all H, both arms
#   ./scripts/submit_campaign_v2.sh stage2                # F1-F3 only, rest of grid
#   ./scripts/submit_campaign_v2.sh smoke PatchTST 5 F4 mse   # 1-job smoke test
#
# Each job's stdout/stderr go to:
#   $HOME/logs/fs_<MODEL>_<H>_<FOLD>_<ARM>_<JOBID>.{out,err}

set -euo pipefail

STAGE="${1:-stage1}"
MODELS_CSV="${2:-PatchTST,TFT,GCFormer,DLinear,LSTM,RNN,CNN}"
HORIZONS_CSV="${3:-5,20,60,120,240}"
FOLDS_CSV="${4:-F4}"
ARMS_CSV="${5:-mse,riskhead}"

# Stage-specific overrides
case "$STAGE" in
    stage1)  FOLDS_CSV="F4" ;;
    stage2)  FOLDS_CSV="F1,F2,F3" ;;
    stage3)  FOLDS_CSV="F1,F2,F3,F4" ;;
    smoke)   ;;   # honour user-passed args
    *)       echo "Unknown stage: $STAGE (expected stage1|stage2|stage3|smoke)" >&2; exit 1 ;;
esac

IFS=',' read -ra MODELS <<<"$MODELS_CSV"
IFS=',' read -ra HORIZONS <<<"$HORIZONS_CSV"
IFS=',' read -ra FOLDS <<<"$FOLDS_CSV"
IFS=',' read -ra ARMS <<<"$ARMS_CSV"

n_jobs=$((${#MODELS[@]} * ${#HORIZONS[@]} * ${#FOLDS[@]} * ${#ARMS[@]}))

echo "============================================================"
echo "Campaign v2 — $STAGE"
echo "============================================================"
echo "models   : ${MODELS[*]}"
echo "horizons : ${HORIZONS[*]}"
echo "folds    : ${FOLDS[*]}"
echo "arms     : ${ARMS[*]}"
echo "n_jobs   : $n_jobs"
echo

read -p "Submit $n_jobs jobs? [y/N] " -r REPLY
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

mkdir -p "$HOME/logs"

submitted=0
for MODEL in "${MODELS[@]}"; do
    for H in "${HORIZONS[@]}"; do
        for FOLD in "${FOLDS[@]}"; do
            for ARM in "${ARMS[@]}"; do
                jname="${MODEL}_H${H}_${FOLD}_${ARM}"
                jid=$(sbatch --parsable \
                    --job-name="$jname" \
                    --export=ALL,MODEL="$MODEL",HORIZON="$H",FOLD="$FOLD",ARM="$ARM" \
                    scripts/train_campaign.sbatch)
                submitted=$((submitted + 1))
                echo "[$submitted/$n_jobs] $jname  jobid=$jid"
            done
        done
    done
done

echo
echo "Submitted $submitted jobs."
echo "Monitor with:"
echo "  squeue -u \$USER -o '%.10i %.30j %.10P %.8T %.10M %R'"
