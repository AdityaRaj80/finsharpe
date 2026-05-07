import os
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths (PLAN_v2 — see reports/PLAN_v2_2026_05_07.md)
# ─────────────────────────────────────────────────────────────────────────────
# DATA_DIR points to the v2 merged dataset produced by:
#   1. data_pipeline/finbert_score_hpc.py     (FinBERT on the 23 GB FNSPID file)
#   2. data_pipeline/aggregate_daily_sentiment.py
#   3. data_pipeline/rebuild_merged_v2.py
#
# Local + HPC fallback paths.
if os.path.exists(r"D:\Study\CIKM\fin-sent-optimized\data\merged_v3"):
    DATA_DIR = r"D:\Study\CIKM\fin-sent-optimized\data\merged_v3"
elif os.path.exists("/home/goyalpoonam/data/merged_v3"):
    DATA_DIR = "/home/goyalpoonam/data/merged_v3"
else:
    DATA_DIR = None    # Set later by pipelines that don't need merged data yet
                       # (e.g., the FinBERT scoring step itself).

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
# Pre-scaled per-stock .npy cache for the memory-mapped global loader.
# Contents are regenerable from raw CSVs, so this dir is in .gitignore.
CACHE_DIR = os.path.join(BASE_DIR, ".cache", "global_scaled")
VALTEST_CACHE_DIR = os.path.join(BASE_DIR, ".cache", "valtest_scaled")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(VALTEST_CACHE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data / Feature config
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = ["Open", "High", "Low", "Close", "Volume", "scaled_sentiment"]
CLOSE_IDX = 3  # Index of Close in FEATURES array

# Target reformulation (PLAN_v2): predict log-returns directly, NOT
# MinMax-scaled prices. This makes our MSE comparable to PatchTST,
# iTransformer, MASTER, and the 2024 cross-sectional ranking literature.
TARGET_MODE = "log_return"   # {"log_return", "scaled_price"}; default v2 = log_return

# Stock-split convention (PLAN_v2): use the SAME 300-stock universe across
# train/val/test, calendar-only split. Matches MASTER/DeepClair/Qlib.
STOCK_SPLIT_MODE = "calendar_only"   # {"calendar_only", "disjoint_stocks"}

# ─────────────────────────────────────────────────────────────────────────────
# Stock universe (PLAN_v2 — 300 stocks, curated 2026-05-07)
# ─────────────────────────────────────────────────────────────────────────────
UNIVERSE_FILE = os.path.join(BASE_DIR, "data", "universe_main.csv")
SEQVSGLOB_UNIVERSE_FILE = os.path.join(BASE_DIR, "data", "universe_seqvsglob.csv")


def load_universe(file_path: str = None) -> list[str]:
    """Return the list of UPPERCASE tickers in the universe."""
    p = file_path or UNIVERSE_FILE
    if not os.path.exists(p):
        raise FileNotFoundError(f"Universe file not found: {p}")
    return list(pd.read_csv(p)["ticker"].astype(str).str.upper())


# ─────────────────────────────────────────────────────────────────────────────
# Common training config
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN = 504           # 2 trading years look-back (ratio >= 2x for all horizons)
HORIZONS = [5, 20, 60]   # PLAN_v2: dropped {120, 240} -- bootstrap CIs invalid at H>=120

# ─────────────────────────────────────────────────────────────────────────────
# Calendar split (PLAN_v2 headline fold F4)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_END_DATE  = "2021-12-31"
VAL_START_DATE  = "2022-01-01"
VAL_END_DATE    = "2022-12-31"
TEST_START_DATE = "2023-01-01"
TEST_END_DATE   = "2023-12-29"   # FNSPID effective price-data end

# Walk-forward CV folds (PLAN_v2 §4)
WALK_FORWARD_FOLDS = [
    {"name": "F1", "train_end": "2018-12-31", "val_year": 2019, "test_year": 2020},
    {"name": "F2", "train_end": "2019-12-31", "val_year": 2020, "test_year": 2021},
    {"name": "F3", "train_end": "2020-12-31", "val_year": 2021, "test_year": 2022},
    {"name": "F4", "train_end": "2021-12-31", "val_year": 2022, "test_year": 2023},
]

# ─────────────────────────────────────────────────────────────────────────────
# Pre-defined Hyperparameters for each model
# ─────────────────────────────────────────────────────────────────────────────
# All models will output [B, pred_len] (univariate future Close)

PATCHTST_CONFIG = {
    "d_model": 256,
    "n_heads": 8,
    "e_layers": 6,
    "d_ff": 512,
    "patch_len": 16,
    "stride": 8,
    "dropout": 0.1,
    "fc_dropout": 0.05,
    "head_dropout": 0.0,
    "individual": 1,
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "padding_patch": "end"
}

TFT_CONFIG = {
    "d_model": 256,
    "n_heads": 4,
    "d_ff": 256,
    "lstm_layers": 2,
    "dropout": 0.1,
    "quantiles": [0.1, 0.5, 0.9]
}

ADAPATCH_CONFIG = {
    "slice_len": 8,
    "middle_len": 128,
    "hidden_len": 16,
    "slice_stride": 1,
    "encoder_dropout": 0.1,
    "d_ff": 2048,
    "alpha": 0.5  # Customizable via CLI
}

GCFORMER_CONFIG = {
    "d_model": 256,
    "n_heads": 8,
    "e_layers": 3,
    "global_layers": 1,
    "d_ff": 512,
    "patch_len": 16,
    "stride": 8,
    "dropout": 0.05,
    "fc_dropout": 0.05,
    "head_dropout": 0.0,
    "individual": 1,
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "padding_patch": "end",
    "global_model": "Gconv",
    "norm_type": "revin",
    "h_token": 512,
    "h_channel": 32,
    "local_bias": 0.5,
    "global_bias": 0.5,
    "atten_bias": 0.5,
    "TC_bias": 1,
    "decomposition": 0,
    "context_len": SEQ_LEN,
}

ITRANSFORMER_CONFIG = {
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_ff": 2048,
    "dropout": 0.1,
    "activation": "gelu"
}

VANILLA_TRANSFORMER_CONFIG = {
    "d_model": 256,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 2048,
    "dropout": 0.1,
    "activation": "gelu",
    "factor": 1,
}

TIMESNET_CONFIG = {
    "d_model": 512,
    "e_layers": 2,
    "d_ff": 2048,
    "dropout": 0.1,
    "top_k": 5,
    "num_kernels": 6
}

DLINEAR_CONFIG = {
    "moving_avg": 25,
    "individual": False,
    "enc_in": len(FEATURES)
}

# Simple classical baselines (added under PLAN_v2 for the simple-vs-complex
# axis of the benchmark contribution). Kept deliberately small so that any
# improvement the modern transformers show over these is the price-of-complexity.
LSTM_CONFIG = {
    "d_model": 128,
    "e_layers": 2,
    "dropout": 0.1,
}

RNN_CONFIG = {
    "d_model": 64,
    "e_layers": 1,
    "dropout": 0.1,
}

CNN_CONFIG = {
    "d_model": 64,
    "e_layers": 3,
    "kernel_size": 3,
    "dropout": 0.1,
}
