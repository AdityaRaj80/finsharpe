#!/bin/bash
#SBATCH --job-name=fs_train
#SBATCH --partition=gpu_h100_4
#SBATCH --qos=qos_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/goyalpoonam/logs/fs_%x_%j.out
#SBATCH --error=/home/goyalpoonam/logs/fs_%x_%j.err

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

echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID"
echo "[$(date)] gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[$(date)] cell: model=$MODEL horizon=$HORIZON fold=$FOLD arm=$ARM"

# Build train.py invocation. For Track-B arm we add --use_risk_head.
ARM_FLAGS=""
if [ "$ARM" = "riskhead" ]; then
    ARM_FLAGS="--use_risk_head"
fi

python train.py \
    --model "$MODEL" \
    --horizon "$HORIZON" \
    --fold "$FOLD" \
    --method global \
    --batch_size 128 \
    --epochs 60 \
    --lr 1e-4 \
    --patience 10 \
    --use_amp \
    $ARM_FLAGS

echo "[$(date)] cell done."
