"""Alpha158-lite feature engineering for FNSPID stock-prediction models.

Computes a Qlib-style cross-sectional feature set on top of OHLCV+sentiment
data. The feature list is a curated subset of Microsoft Qlib's Alpha158 (Yang
et al. 2020, arXiv:2009.11189) keeping the highest-impact features as
identified by HIST/MASTER/FactorVAE feature-importance studies. ~63 alpha
features are added on top of the 6 raw features (OHLCV + scaled_sentiment),
giving enc_in ≈ 69.

Citation lineage:
- Microsoft Qlib (Yang et al. 2020): the canonical 158-feature library used
  by HIST (CIKM'22), MASTER (AAAI'24), FactorVAE (AAAI'22), TFT-ASRO,
  GraphSAGE-KE, etc. Reproducing the full 158-set verbatim is not necessary
  — the literature consistently shows that ~50-80 of them carry most of the
  IC; we keep the high-IC set.
- Jegadeesh & Titman 1993: multi-lag momentum (RET_d, MOM_d).
- López de Prado 2018 Ch. 3: K-line features as bar-shape descriptors.

Design notes:
- All features are SCALE-INVARIANT (ratios, log-returns, normalised ranks)
  so they are comparable across stocks before per-stock z-score
  normalisation in the data loader.
- Features that need the previous day's close use shift(1); rolling
  features have NaN in the first `window` rows (data_loader drops them).
- Sentiment features assume `scaled_sentiment ∈ [0, 1]` with 0.5 = neutral
  (per the FNSPID rebuild_merged_v2 convention).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Standard Qlib lag windows (trading days).
WINDOWS = (5, 10, 20, 30, 60)
WINDOWS_VOLPRICE = (10, 20, 60)
WINDOWS_SENT = (5, 20)
WINDOWS_MOM = (5, 10, 20)


# -----------------------------------------------------------------------------
# The list of feature names this module emits, in canonical column order.
# Used by config.FEATURES so the data loader and downstream code stay in sync.
# -----------------------------------------------------------------------------
def _alpha_feature_names() -> list[str]:
    names = [
        # K-line shape (7)
        "KMID", "KLEN", "KMID2", "KUP", "KLOW", "KSFT", "KSFT2",
    ]
    names += [f"RET_{d}" for d in WINDOWS]                  # 5
    names += [f"MA_{d}" for d in WINDOWS]                   # 5
    names += [f"STD_{d}" for d in WINDOWS]                  # 5
    names += [f"MAX_{d}" for d in WINDOWS]                  # 5
    names += [f"MIN_{d}" for d in WINDOWS]                  # 5
    names += [f"RANK_{d}" for d in WINDOWS]                 # 5
    names += [f"HL_{d}" for d in WINDOWS]                   # 5
    names += [f"VMA_{d}" for d in WINDOWS]                  # 5
    names += [f"VSTD_{d}" for d in WINDOWS]                 # 5
    names += [f"CORR_{d}" for d in WINDOWS_VOLPRICE]        # 3
    names += [f"MOM_{d}" for d in WINDOWS_MOM]              # 3
    names += ["SENT_DELTA"]                                 # 1
    names += [f"SENT_MA_{d}" for d in WINDOWS_SENT]         # 2
    names += [f"SENT_STD_{d}" for d in WINDOWS_SENT]        # 2
    return names


ALPHA_FEATURE_NAMES = _alpha_feature_names()
N_ALPHA_FEATURES = len(ALPHA_FEATURE_NAMES)


# -----------------------------------------------------------------------------
# Computation
# -----------------------------------------------------------------------------
def compute_alpha_features(df: pd.DataFrame, eps: float = 1e-9) -> pd.DataFrame:
    """Compute the alpha feature set as new columns on the input frame.

    Args
    ----
        df : pd.DataFrame with at least the columns
             {"Open", "High", "Low", "Close", "Volume", "scaled_sentiment"}
             in REAL (Adj-Close-adjusted) price units. Volume in raw shares.
             scaled_sentiment ∈ [0, 1] with 0.5 = neutral.
             Rows must be in chronological order.

    Returns
    -------
        Same DataFrame with N_ALPHA_FEATURES new columns appended.
        Caller is responsible for dropping rows with NaN (typically the
        first max(WINDOWS) rows where rolling stats are undefined).
    """
    out = df.copy()
    O = df["Open"].astype(np.float64)
    H = df["High"].astype(np.float64)
    L = df["Low"].astype(np.float64)
    C = df["Close"].astype(np.float64)
    V = df["Volume"].astype(np.float64).clip(lower=1.0)   # avoid log(0)

    # Sentiment: default to neutral if column missing.
    if "scaled_sentiment" in df.columns:
        S = df["scaled_sentiment"].astype(np.float64)
    else:
        S = pd.Series(0.5, index=df.index, dtype=np.float64)

    # Per-day log return (used as ingredient for several rolling features).
    log_ret = np.log(C / C.shift(1).clip(lower=eps))
    log_v   = np.log(V)

    # === K-line shape features (7) ============================================
    # Scale-invariant by construction (all numerators and denominators in same
    # price units). KMID = bar drift; KLEN = bar range; KUP/KLOW = wick extents.
    out["KMID"]  = (C - O) / (O.clip(lower=eps))
    out["KLEN"]  = (H - L) / (O.clip(lower=eps))
    rng = (H - L).clip(lower=eps)
    out["KMID2"] = (C - O) / rng
    out["KUP"]   = (H - np.maximum(O, C)) / (O.clip(lower=eps))
    out["KLOW"]  = (np.minimum(O, C) - L) / (O.clip(lower=eps))
    out["KSFT"]  = (2.0 * C - H - L) / (O.clip(lower=eps))
    out["KSFT2"] = (2.0 * C - H - L) / rng

    # === Multi-lag log returns (5) — Jegadeesh-Titman canonical momentum =====
    for d in WINDOWS:
        out[f"RET_{d}"] = np.log(C / C.shift(d).clip(lower=eps))

    # === Rolling MA / Close ratio (5) ========================================
    # Tracks where current close sits relative to its rolling mean. > 0 →
    # above mean (recently strong); < 0 → mean reversion candidate.
    for d in WINDOWS:
        ma = C.rolling(d, min_periods=d).mean()
        out[f"MA_{d}"] = (C / (ma + eps)) - 1.0

    # === Rolling STD of log returns (5) — realised vol over multiple horizons
    for d in WINDOWS:
        out[f"STD_{d}"] = log_ret.rolling(d, min_periods=d).std(ddof=1)

    # === Rolling MAX / MIN ratio (5 + 5) — distance to recent extremes =======
    for d in WINDOWS:
        out[f"MAX_{d}"] = C.rolling(d, min_periods=d).max() / (C + eps) - 1.0
        out[f"MIN_{d}"] = C.rolling(d, min_periods=d).min() / (C + eps) - 1.0

    # === Rolling rank of close (5) — percentile in last d-day window ========
    for d in WINDOWS:
        out[f"RANK_{d}"] = C.rolling(d, min_periods=d).rank(pct=True)

    # === High-Low rolling range relative to close (5) =======================
    # Captures vol regime via the bar range, complementary to STD.
    for d in WINDOWS:
        out[f"HL_{d}"] = (
            H.rolling(d, min_periods=d).max() - L.rolling(d, min_periods=d).min()
        ) / (C + eps)

    # === Volume features (5 + 5) ============================================
    for d in WINDOWS:
        v_ma = V.rolling(d, min_periods=d).mean()
        out[f"VMA_{d}"] = (V / (v_ma + eps)) - 1.0
        out[f"VSTD_{d}"] = log_v.rolling(d, min_periods=d).std(ddof=1)

    # === Volume-price log-return correlation (3) ============================
    log_v_diff = log_v.diff()
    for d in WINDOWS_VOLPRICE:
        out[f"CORR_{d}"] = log_ret.rolling(d, min_periods=d).corr(log_v_diff)

    # === Momentum (3) — simple-return momentum at short lags ================
    for d in WINDOWS_MOM:
        out[f"MOM_{d}"] = (C - C.shift(d)) / (C.shift(d) + eps)

    # === Sentiment features (1 + 2 + 2) =====================================
    # SENT_DELTA: deviation from neutral (positive = bullish news).
    out["SENT_DELTA"] = S - 0.5
    for d in WINDOWS_SENT:
        out[f"SENT_MA_{d}"] = S.rolling(d, min_periods=d).mean() - 0.5
        out[f"SENT_STD_{d}"] = S.rolling(d, min_periods=d).std(ddof=1)

    # Replace inf (from divide-by-zero edge cases) with NaN so the
    # downstream NaN drop catches them.
    new_cols = [c for c in out.columns if c not in df.columns]
    out[new_cols] = out[new_cols].replace([np.inf, -np.inf], np.nan)

    return out


# -----------------------------------------------------------------------------
# Convenience: list of feature names INCLUDING the 6 raw features in canonical
# order. Use this as `config.FEATURES` so CLOSE_IDX=3 still points to "Close".
# -----------------------------------------------------------------------------
RAW_FEATURE_NAMES = ["Open", "High", "Low", "Close", "Volume", "scaled_sentiment"]

ALL_FEATURE_NAMES = RAW_FEATURE_NAMES + ALPHA_FEATURE_NAMES
N_ALL_FEATURES = len(ALL_FEATURE_NAMES)
