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
# Feature set (Alpha158-lite, 2026-05-09).
# The first 6 are the original raw features (OHLCV + scaled_sentiment) — the
# raw OHLC is Adj-Close-adjusted in the data loader and Volume is log1p'd.
# CLOSE_IDX = 3 points to "Close" so all downstream code (heads, eval_v2,
# trainer log_vol_target) keeps working without changes.
# The remaining 63 are Qlib Alpha158-lite features computed per-stock during
# data load (see data_pipeline/alpha_features.py for the full list and
# Yang et al. 2020 arXiv:2009.11189 for the canonical Alpha158).
from data_pipeline.alpha_features import ALL_FEATURE_NAMES as _ALL_FEAT
FEATURES = list(_ALL_FEAT)
CLOSE_IDX = 3  # Index of Close in FEATURES array (raw OHLC[Volume,sentiment] block first)

# Target convention (PLAN_v2):
#   "scaled_price": [pred_len] z-scored close window. The MSE arm uses
#                   this directly as the regression target -- same
#                   convention as PatchTST/iTransformer/MASTER (which
#                   train on z-score-normalised closes via RevIN). The
#                   "scaled" prefix here means z-score (NOT MinMax v1
#                   bug). Cross-sectional rank-IC and downstream
#                   trading-grade metrics (Sharpe etc.) are computed
#                   from the model output via cross_sectional_smoke.py
#                   which un-z-scores and converts to log-returns.
#   "log_return":   [scalar] H-step log-return. Track-B's RiskAwareHead
#                   computes mu_return_H from mu_close[:, -1] and
#                   last_close internally, so the model's output is
#                   still [pred_len] z-scored close. This mode is
#                   useful only for non-Track-B variants that want to
#                   train on log-return MSE directly (would require
#                   architectural change: model output -> [scalar]
#                   instead of [pred_len]).
# Default v2 = "scaled_price" so the MSE arm's standard nn.MSELoss
# matches model output shape [B, pred_len] -- target [B, pred_len]; and
# the Track-B arm's CompositeRiskLoss handles the log-return derivation
# internally. The cross_sectional_smoke evaluator converts predictions
# to log-returns for the headline trading metrics.
TARGET_MODE = "scaled_price"

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
# Horizons after Jury 2 / baseline-evidence (2026-05-08):
#   H=240 DROPPED -- non-overlap n_obs = 0 in every 1-year fold; even
#                    with 4-fold pooling the test panel was empty.
#                    Confirmed empirically by baseline run 169105:
#                    "test panel: (0, 0)" → all metrics NaN.
#   H=120 DROPPED from ML training campaign (2026-05-08): n_val=0 in
#                    most folds after embargo (504 lookback + 120 horizon
#                    consumes the entire 1-year val window). Empirically
#                    confirmed: 8/8 CNN H=120 jobs failed with empty val
#                    loader. Baselines still produced H=120 entries
#                    (parameter-free; no val needed) and those summaries
#                    are retained for descriptive reporting only.
HORIZONS = [5, 20, 60]

# Per-horizon CI validity tier (used by cross_sectional_smoke.py +
# bootstrap_paired.py to decide what to report):
#   "full"      -- single-fold bootstrap CIs valid + aggregate strong
#   "aggregate" -- single-fold CIs weak; aggregate-across-folds strong
#                  (HAC overlap fallback usable)
#   "point"     -- aggregate CI borderline; report point estimates,
#                  no CI, with explicit "n too small" footnote
HORIZON_CI_TIER = {
    5:   "full",
    20:  "full",
    60:  "aggregate",
    120: "point",
}

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

# Jury 2 fix F18 (2026-05-08): PatchTST shrunk to ICLR'23 published config.
# Was: d=256, L=6, h=8, drop=0.1 (over-deep for noisy 6-channel input).
# Now: d=128, L=3, h=16, drop=0.2, patch_len=24 (longer patch for 504-day
# lookback). Halves parameter count; raises dropout to combat overfit.
PATCHTST_CONFIG = {
    "d_model": 128,
    "n_heads": 16,
    "e_layers": 3,
    "d_ff": 256,
    "patch_len": 24,
    "stride": 12,
    "dropout": 0.2,
    "fc_dropout": 0.1,
    "head_dropout": 0.0,
    "individual": 1,
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "padding_patch": "end"
}

# Jury 2 fix F19: TFT dropout 0.1 → 0.3 (Lim 2021 finance-default).
# Width 256 is over Lim's 160; we keep 256 to ensure capacity for the
# variable-selection layer but raise dropout to compensate.
TFT_CONFIG = {
    "d_model": 256,
    "n_heads": 4,
    "d_ff": 256,
    "lstm_layers": 2,
    "dropout": 0.3,
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

# Jury 2 fix F20 (2026-05-08): GCFormer dropout raised 0.05 → 0.2 (was
# severely under-regularised on a noisy 6-feature input). Width kept at
# 256 — model is already at ~8M params so capacity is not the bottleneck.
GCFORMER_CONFIG = {
    "d_model": 256,
    "n_heads": 8,
    "e_layers": 3,
    "global_layers": 1,
    "d_ff": 512,
    "patch_len": 16,
    "stride": 8,
    "dropout": 0.2,
    "fc_dropout": 0.2,
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

# Jury 2 fix F21 (2026-05-08): CNN receptive field extended via 6 dilated
# blocks (dilations 1,2,4,8,16,32) so the model can see ~64 input
# timesteps. Previous 3-layer dilated TCN (dilations 1,2,4) had RF≈15
# which is wildly insufficient for SEQ_LEN=504. The CNN backbone respects
# `e_layers` and uses `2**i` dilation per block (see models/cnn.py).
CNN_CONFIG = {
    "d_model": 64,
    "e_layers": 6,
    "kernel_size": 3,
    "dropout": 0.1,
}
