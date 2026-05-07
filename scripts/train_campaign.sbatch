#!/bin/bash
#SBATCH --job-name=fs_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/home/goyalpoonam/logs/fs_%x_%j.out
#SBATCH --error=/home/goyalpoonam/logs/fs_%x_%j.err
# NOTE: --partition and --qos are passed by submit_campaign_v2.sh
# (round-robin across H100/H200/A100 to maximise concurrent throughput).
# 12h walltime + --requeue (Jury 2 fix J3: 6h was insufficient for
# large H + large backbone cells; SLURM auto-requeues on preempt).

# Generic training submission for the dual-arm benchmark campaign.
#
# Required env vars passed via --export:
#   MODEL    : one of {PatchTST, TFT, GCFormer, DLinear, LSTM, RNN, CNN}
#   HORIZON  : one of {5, 20, 60, 120, 240}
#   FOLD     : one of {F1, F2, F3, F4}    -- walk-forward CV fold
#   ARM      : one of {mse, riskhead}     -- baseline vs Track-B
#
# Submission is via scripts/submit_campaign_v2.sh (parses sweep grid +
# launches one sbatch per cell with proper --job-name).

set -euo pipefail

REPO_DIR="$HOME/finsharpe"
cd "$REPO_DIR"

# Activate sr_opt conda env (verified torch 2.11+cu130, transformers 5.8)
source /apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/anaconda3-2023.09-0-ku6kdpolxwbdydg5xyawasvogquyjj4b/etc/profile.d/conda.sh
conda activate sr_opt

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION"
echo "[$(date)] gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "[$(date)] cell: model=$MODEL horizon=$HORIZON fold=$FOLD arm=$ARM batch_size=${BATCH_SIZE:-128}"

# Build train.py invocation. For Track-B arm we add --use_risk_head.
ARM_FLAGS=""
if [ "$ARM" = "riskhead" ]; then
    ARM_FLAGS="--use_risk_head"
fi

# Training-recipe defaults updated 2026-05-08 (Jury 2 fixes F14-F17):
#   --epochs 80         (was 60; matches HIST CIKM'22 / closer to PatchTST 100)
#   --patience 20       (was 10; on val rank-IC instead of val MSE — IC is
#                       noisier, longer patience needed)
#   --lradj cosine      (was type3; warmup 5 epochs then cosine to lr/100)
#   --warmup_epochs 5
#   --lr 3e-4           (was 1e-4; with warmup + cosine, can afford a higher
#                       peak LR — matches PatchTST AdamW recipe.)
#   --lr_min 3e-6
python train.py \
    --model "$MODEL" \
    --horizon "$HORIZON" \
    --fold "$FOLD" \
    --method global \
    --batch_size "${BATCH_SIZE:-128}" \
    --epochs 80 \
    --lr 3e-4 \
    --lr_min 3e-6 \
    --lradj cosine \
    --warmup_epochs 5 \
    --patience 20 \
    --use_amp \
    $ARM_FLAGS

echo "[$(date)] cell done."
