#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=gpunode5
#SBATCH --output=/scratch/goyalpoonam/logs/%x_%j.out
#SBATCH --error=/scratch/goyalpoonam/logs/%x_%j.err
# MSE-baseline FROM-SCRATCH training on the sentiment-corrected dataset.
# This is the "with-sentiment" arm of the headline ablation:
#   no-sentiment baseline  = the v1 fan-out already complete on HPC
#   with-sentiment baseline = THIS sbatch
#
# After both arms exist, we (a) compare them directly (sentiment-vs-no-sentiment
# table) and (b) Track-B retrain from these new with-sentiment MSE checkpoints.
#
# Required env vars:
#   MODEL : DLinear | iTransformer | GCFormer | PatchTST | AdaPatch | TFT | VanillaTransformer
#   H     : 5 | 20 | 60 | 120 | 240
#
# Optional (with defaults):
#   EPOCHS   = 60       (from-scratch needs more epochs than fine-tune)
#   BATCH    = 512
#   LR       = 1e-4
#   PATIENCE = 10

set -e
echo "=== finsharpe MSE-baseline | $(hostname) | $SLURM_JOB_ID | $(date) ==="

: "${EPOCHS:=60}"
: "${BATCH:=512}"
: "${LR:=1e-4}"
: "${PATIENCE:=10}"

if [[ -z "$MODEL" || -z "$H" ]]; then
    echo "ERROR: MODEL and H must be set via sbatch --export." >&2
    exit 2
fi

echo "MODEL=$MODEL  H=$H  EPOCHS=$EPOCHS  BATCH=$BATCH  LR=$LR"

source /apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/anaconda3-2023.09-0-ku6kdpolxwbdydg5xyawasvogquyjj4b/etc/profile.d/conda.sh
conda activate sr_opt

cd /scratch/goyalpoonam/finsharpe

# Pre-flight: cache must exist + audit must pass. Catches the v0 incident.
python data_pipeline/audit.py cache 2>&1 | tail -5 || {
    echo "[FATAL] cache audit failed; aborting" >&2
    exit 3
}

# (Cache rebuild handled separately; this sbatch assumes data prep is done.)
stdbuf -oL -eL python -u train.py \
    --model "$MODEL" --method global --horizon "$H" \
    --device auto --batch_size "$BATCH" --epochs "$EPOCHS" \
    --lr "$LR" --lradj type3 --patience "$PATIENCE" --use_amp

echo "=== finsharpe MSE-baseline done: $MODEL H${H} | $(date) ==="
