#!/bin/bash
#SBATCH --job-name=fs_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/goyalpoonam/logs/eval_%x_%j.out
#SBATCH --error=/scratch/goyalpoonam/logs/eval_%x_%j.err

# Single eval cell on GPU. Reads MODEL, HORIZON, FOLD, ARM from --export.
set -euo pipefail
cd /scratch/goyalpoonam/finsharpe
source /apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/anaconda3-2023.09-0-ku6kdpolxwbdydg5xyawasvogquyjj4b/etc/profile.d/conda.sh
conda activate sr_opt

echo "[$(date)] eval cell: $MODEL H=$HORIZON $FOLD $ARM"
python smoke/eval_v2.py \
    --model "$MODEL" --horizon "$HORIZON" \
    --fold "$FOLD" --arm "$ARM" \
    --batch_size 512
echo "[$(date)] done"
