"""Cross-sectional rank-IC + Newey-West/Fama-MacBeth-style inference.

Background
----------
The cross_sectional_smoke.py headline reports portfolio Sharpe — a
joint test of (signal, sizing, costs). Reviewers familiar with quant
finance will additionally ask for a *signal-only* validity test:

    "Does the predicted return rank the cross-section?"

This is the standard rank-IC literature (Qlib, Alpha158, Fama-MacBeth):

    1. At each rebalance t, compute Spearman correlation
       rho_t = corr( prediction_t , realised_return_t )
       across the cross-section of 49 hold-out stocks.

    2. Average rho_t over t and report:
        * IC_mean = mean(rho_t)
        * IC_std  = std(rho_t)
        * ICIR    = IC_mean / IC_std * sqrt(252 / horizon)
        * Newey-West HAC-corrected t-stat with lag = horizon - 1
          (the standard Hansen-Hodrick / Newey-West choice for
          overlapping returns, automatic at horizon = 1).

This script reads the same CSV/JSON artefacts produced by
cross_sectional_smoke.py and produces a rank-IC table per checkpoint
that complements the portfolio Sharpe table.

References
----------
- Newey, W. and West, K. (1987). A Simple Positive Semi-Definite,
  Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.
  Econometrica 55(3):703-708.
- Fama, E. and MacBeth, J. (1973). Risk, Return, and Equilibrium:
  Empirical Tests. JPE 81(3):607-636.
- Qlib (Microsoft Research). Alpha158: Standard alpha factors and IC
  evaluation pipeline. https://github.com/microsoft/qlib

Usage
-----
    # Full forward pass on a checkpoint, then rank-IC
    python smoke/rank_ic.py --model GCFormer --horizon 5 --use_risk_head \
        --ckpt_suffix _riskhead

    # Or compute rank-IC from a pre-saved panel CSV
    python smoke/rank_ic.py --panel_csv panel.csv --horizon 5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

# Make the repo root importable when run as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))


def newey_west_se(x: np.ndarray, lag: int) -> float:
    """Newey-West HAC standard error of the *mean* of x.

    Computes the Bartlett-kernel HAC variance estimator at the specified
    lag truncation, then divides by sqrt(n) for the SE of the mean.

    Args
    ----
        x   : 1-D series.
        lag : truncation lag for the Bartlett kernel.

    Returns
    -------
        SE of the mean of x, accounting for autocorrelation up to `lag`.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    e = x - x.mean()
    gamma_0 = np.dot(e, e) / n
    s = gamma_0
    for k in range(1, min(lag, n - 1) + 1):
        gamma_k = np.dot(e[k:], e[:-k]) / n
        weight = 1.0 - k / (lag + 1.0)            # Bartlett
        s += 2.0 * weight * gamma_k
    if s <= 0:
        return float("nan")
    return float(math.sqrt(s / n))


def cross_sectional_spearman_series(
    pred_M: np.ndarray, actual_M: np.ndarray,
) -> np.ndarray:
    """Per-timestamp Spearman rank correlation across the cross-section.

    Args
    ----
        pred_M, actual_M : [T, N] panels (NaN-padded). Each row is a
                            cross-section at one rebalance timestamp.

    Returns
    -------
        rho_t : [T] series of Spearman correlations (NaN where < 5
                valid stocks at that t — too few to rank meaningfully).
    """
    T, N = pred_M.shape
    rho = np.full(T, np.nan)
    for t in range(T):
        valid = ~np.isnan(pred_M[t]) & ~np.isnan(actual_M[t])
        if valid.sum() < 5:
            continue
        r, _ = spearmanr(pred_M[t][valid], actual_M[t][valid])
        rho[t] = r
    return rho


def cross_sectional_kendall_series(
    pred_M: np.ndarray, actual_M: np.ndarray,
) -> np.ndarray:
    """Like spearman_series but Kendall's tau (more robust to outliers,
    tie-corrected). Slower but standard in the IC literature."""
    T, N = pred_M.shape
    tau = np.full(T, np.nan)
    for t in range(T):
        valid = ~np.isnan(pred_M[t]) & ~np.isnan(actual_M[t])
        if valid.sum() < 5:
            continue
        r, _ = kendalltau(pred_M[t][valid], actual_M[t][valid])
        tau[t] = r
    return tau


def rank_ic_summary(
    rho_series: np.ndarray, horizon: int, label: str = "spearman",
) -> dict:
    """Compute IC mean / std / ICIR / NW t-stat from a rho series.

    Newey-West lag = horizon - 1 (zero at H=1, which is just simple
    OLS SE; Hansen-Hodrick lag = H-1 is the standard choice for
    overlapping H-day returns).
    """
    rho = np.asarray(rho_series, dtype=np.float64)
    rho_clean = rho[~np.isnan(rho)]
    n = len(rho_clean)
    if n < 2:
        return {f"{label}_ic_mean": float("nan"),
                f"{label}_ic_std": float("nan"),
                f"{label}_icir": float("nan"),
                f"{label}_nw_se": float("nan"),
                f"{label}_nw_tstat": float("nan"),
                f"{label}_n_obs": int(n)}

    ic_mean = float(rho_clean.mean())
    ic_std  = float(rho_clean.std(ddof=1))
    icir = ic_mean / ic_std * math.sqrt(252.0 / horizon) if ic_std > 1e-12 else float("nan")

    nw_lag = max(0, horizon - 1)
    nw_se = newey_west_se(rho_clean, nw_lag)
    nw_t = ic_mean / nw_se if math.isfinite(nw_se) and nw_se > 1e-12 else float("nan")

    return {
        f"{label}_ic_mean": ic_mean,
        f"{label}_ic_std": ic_std,
        f"{label}_icir": icir,
        f"{label}_nw_se": nw_se,
        f"{label}_nw_tstat": nw_t,
        f"{label}_nw_lag": nw_lag,
        f"{label}_n_obs": n,
    }


def _hit_rate(rho_series: np.ndarray) -> float:
    """Fraction of rebalances with positive IC. Diagnostic only."""
    r = rho_series[~np.isnan(rho_series)]
    if len(r) == 0:
        return float("nan")
    return float((r > 0).mean())


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    sub = p.add_mutually_exclusive_group(required=True)
    sub.add_argument("--panel_csv", help="Pre-saved [T, 2N] panel CSV with "
                                          "alternating pred_<sid>, actual_<sid> "
                                          "columns.")
    sub.add_argument("--from_smoke", action="store_true",
                     help="Compute panel by re-running cross_sectional_smoke "
                          "predict() on a checkpoint.")
    p.add_argument("--model", default="PatchTST")
    p.add_argument("--method", default="global", choices=["global", "sequential"])
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--use_risk_head", action="store_true")
    p.add_argument("--ckpt_suffix", default="")
    p.add_argument("--out", default=None,
                   help="Optional output JSON path.")
    args = p.parse_args()

    if args.from_smoke:
        # Inline the imports — only if we actually need to forward-pass
        import torch
        from config import SEQ_LEN, CLOSE_IDX, MODEL_SAVE_DIR, FEATURES
        from data_loader import UnifiedDataLoader
        from models import model_dict
        from train import get_config_for_model
        from engine.heads import RiskAwareHead
        from smoke.cross_sectional_smoke import (
            predict, predict_with_risk, to_returns, build_panel)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loader_obj = UnifiedDataLoader(
            seq_len=SEQ_LEN, horizon=args.horizon, batch_size=512)
        _, test_loader = loader_obj.get_val_test_loaders_mmap()

        configs = get_config_for_model(args.model, args.horizon)
        backbone = model_dict[args.model](configs)
        if args.use_risk_head:
            model = RiskAwareHead(backbone=backbone, n_features=len(FEATURES),
                                  pred_len=args.horizon, close_idx=CLOSE_IDX,
                                  lookback_for_aux=20, d_hidden=64).to(device)
        else:
            model = backbone.to(device)
        ckpt = os.path.join(MODEL_SAVE_DIR,
                            f"{args.model}_{args.method}_H{args.horizon}{args.ckpt_suffix}.pth")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded {ckpt}", flush=True)

        if args.use_risk_head:
            tp, _, _, _, tt, tlc = predict_with_risk(model, test_loader, device)
        else:
            tp, tt, tlc = predict(model, test_loader, device)
        cmin = loader_obj.test_close_min.astype(np.float64)
        cmax = loader_obj.test_close_max.astype(np.float64)
        test_pr, test_ar = to_returns(tp, tt, tlc, cmin, cmax)
        pred_M, actual_M, _ = build_panel(test_pr, test_ar, cmax)
    else:
        # Pre-saved panel CSV
        df = pd.read_csv(args.panel_csv)
        pred_cols = [c for c in df.columns if c.startswith("pred_")]
        act_cols  = [c for c in df.columns if c.startswith("actual_")]
        # Match by suffix
        pred_keys = [c[len("pred_"):] for c in pred_cols]
        act_keys  = [c[len("actual_"):] for c in act_cols]
        common = [k for k in pred_keys if k in set(act_keys)]
        if not common:
            raise SystemExit(f"No matching pred_/actual_ pairs in {args.panel_csv}")
        pred_M   = df[[f"pred_{k}"   for k in common]].to_numpy()
        actual_M = df[[f"actual_{k}" for k in common]].to_numpy()

    print(f"Panel: {pred_M.shape[0]} timestamps x {pred_M.shape[1]} stocks", flush=True)

    # ── Compute rank-IC series ──
    rho = cross_sectional_spearman_series(pred_M, actual_M)
    tau = cross_sectional_kendall_series(pred_M, actual_M)

    # Subsample to non-overlapping rebalances to avoid double-counting
    rho_nover = rho[::args.horizon]
    tau_nover = tau[::args.horizon]

    summary = {
        "model": args.model,
        "horizon": args.horizon,
        "n_panel_timestamps": int(pred_M.shape[0]),
        "n_stocks": int(pred_M.shape[1]),
        **rank_ic_summary(rho_nover, args.horizon, label="spearman"),
        **rank_ic_summary(tau_nover, args.horizon, label="kendall"),
        "spearman_hit_rate": _hit_rate(rho_nover),
        "kendall_hit_rate": _hit_rate(tau_nover),
    }

    print("\n========== RANK-IC SUMMARY ==========", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    out = args.out or os.path.join(
        os.path.dirname(_HERE), "results",
        f"rank_ic_{args.model}_{args.method}_H{args.horizon}{args.ckpt_suffix}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nWrote: {out}", flush=True)


if __name__ == "__main__":
    main()
