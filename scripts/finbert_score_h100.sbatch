#!/bin/bash
#SBATCH --job-name=finbert_score
#SBATCH --partition=gpu_h100_4
#SBATCH --qos=qos_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/goyalpoonam/logs/finbert_score_%j.out
#SBATCH --error=/home/goyalpoonam/logs/finbert_score_%j.err

# FinBERT inference on the 300-stock filtered news subset (~1.26M articles).
# Expected wall-clock on H100 80GB: 10-20 minutes.

set -euo pipefail

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
echo "[$(date)] node=$(hostname) job=$SLURM_JOB_ID"
echo "[$(date)] gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Paths (HPC layout)
REPO_DIR="$HOME/finsharpe"
NEWS_FILE="$HOME/data/filtered_news_universe_main.csv"
OUT_DIR="$HOME/data/finbert_scores_v2"
HF_CACHE="$HOME/.cache/huggingface"

cd "$REPO_DIR"

# Activate env (assumes conda env 'finsharpe' has torch+transformers; adjust if not)
if [ -d "$HOME/miniconda3" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate finsharpe || conda activate base
fi

# Pre-fetch the model so the worker node doesn't hit the network mid-job
python -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
cache = os.path.expanduser('$HF_CACHE')
AutoTokenizer.from_pretrained('ProsusAI/finbert', cache_dir=cache)
AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert', cache_dir=cache)
print('FinBERT cached at', cache)
"

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
mkdir -p "$OUT_DIR"
python data_pipeline/finbert_score_hpc.py \
    --news_file "$NEWS_FILE" \
    --out_dir "$OUT_DIR" \
    --batch_size 256 \
    --hf_cache_dir "$HF_CACHE"

echo "[$(date)] done. Output -> $OUT_DIR"
ls -la "$OUT_DIR" | head -10
echo "n_files: $(ls "$OUT_DIR" | wc -l)"
echo "total_size: $(du -sh "$OUT_DIR")"
