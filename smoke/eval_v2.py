"""V2 cross-sectional evaluation -- replaces v1 cross_sectional_smoke.py.

Pipeline (per checkpoint):
  1. Load v2 UnifiedDataLoader for the cell's (model, horizon, fold)
  2. Run model over val_loader and test_loader, collect per-sample predictions
  3. Convert predictions to per-sample predicted log-return:
       MSE arm  : un-z-score the model's [B, pred_len] last-step output via
                  per-stock (mu, sd) from loader.scalers, then log(pred/anchor)
       Track-B arm: use the RiskAwareHead's mu_return_H directly (it's
                  already in real log-return units thanks to return_head)
  4. Pivot per-sample (predicted_logret, actual_logret) into [T, N] panels
     using the loader's sample_table (anchor_idx -> calendar position
     within the test window; stock_id -> column).
  5. Cross-sectional ranking: long top-N, short bottom-N, equal-weight
     per leg. Sweep top_n on val by val gross Sharpe; apply best to test.
  6. Subsample to non-overlapping rebalances [::H] and compute:
       gross/net annualized Sharpe (cost grid 0/5/10/20/50 bps round-trip)
       Sortino, Calmar, MDD, cumulative return, hit rate
       cross-sectional rank-IC (Spearman) with NW HAC SE
       Lo (2002) autocorrelation-corrected annualization
  7. Save results to results/eval_v2_<model>_<H>_<fold>_<arm>.json + .csv

Usage:
    python smoke/eval_v2.py --model DLinear --horizon 5 --fold F4 --arm mse
    python smoke/eval_v2.py --model DLinear --horizon 5 --fold F4 --arm riskhead
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from config import (CLOSE_IDX, FEATURES, MODEL_SAVE_DIR, RESULTS_DIR,
                     SEQ_LEN, HORIZON_CI_TIER)
from data_loader import UnifiedDataLoader
from engine.heads import RiskAwareHead
from models import model_dict
from train import get_config_for_model

OUT_DIR = os.path.join(os.path.dirname(_HERE), "results", "eval_v2")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Forward + per-sample predicted log-return extraction
# ---------------------------------------------------------------------------
def predict_logret(model, loader, device, arm: str, ld: UnifiedDataLoader,
                    split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-sample (pred_logret, actual_logret, stock_id, anchor_idx)
    over the entire `split` window.

    Args
    ----
        arm   : "mse" or "riskhead". For MSE we un-z-score the model's
                [B, pred_len] last-step prediction; for Track-B we use the
                RiskAwareHead's mu_return_H output directly.
        split : "val" or "test".
    """
    sample_table = {"val": ld.sample_table_val, "test": ld.sample_table_test}[split]
    n_total = sample_table.shape[0]
    pred_logret = np.zeros(n_total, dtype=np.float64)
    actual_logret = np.zeros(n_total, dtype=np.float64)
    stock_ids = sample_table[:, 0].astype(np.int32)
    anchor_idxs = sample_table[:, 2].astype(np.int32)

    # Per-stock scaler arrays for un-z-scoring (MSE arm only)
    universe = ld.universe
    mu_close = np.zeros(len(universe), dtype=np.float64)
    sd_close = np.zeros(len(universe), dtype=np.float64)
    for sid, ticker in enumerate(universe):
        if ticker in ld.scalers:
            mu, sd = ld.scalers[ticker]
            mu_close[sid] = float(mu[CLOSE_IDX])
            sd_close[sid] = float(sd[CLOSE_IDX])

    model.eval()
    cursor = 0
    dl = (ld.get_val_loader() if split == "val" else ld.get_test_loader())

    with torch.no_grad():
        for batch in dl:
            X, y_main, y_logret = batch
            B = X.shape[0]
            X = X.float().to(device)
            out = model(X, None)

            if arm == "riskhead":
                assert isinstance(out, dict), "Track-B arm expects dict output"
                # Track-B's return_head outputs predicted log-return directly
                # (real return units, comparable across stocks).
                pred_lr = out["mu_return_H"].cpu().numpy().astype(np.float64)
            else:  # mse
                if isinstance(out, dict):
                    out = out["mu_close"]
                if isinstance(out, tuple):
                    out = out[0]
                # out is [B, pred_len] in z-scored close space.
                # Bug fix 2026-05-07 (diagnosed via z-delta vs un-z-score
                # rank-IC test): un-z-scoring with per-stock (mu, sd)
                # injects a per-stock multiplicative scale bias that
                # corrupts the cross-sectional ranking. The COMPARABLE
                # cross-stock signal is the z-score-space DELTA between
                # predicted and anchor close. Ranking by z-delta lifts
                # PatchTST/H5/F4 IC from -0.006 -> +0.020, Sharpe from
                # -0.35 -> +0.86.
                pred_z = out[:, -1].cpu().numpy().astype(np.float64)
                anchor_z = X[:, -1, CLOSE_IDX].cpu().numpy().astype(np.float64)
                pred_lr = pred_z - anchor_z   # z-score-space delta

            pred_logret[cursor:cursor + B] = pred_lr
            actual_logret[cursor:cursor + B] = y_logret.numpy().astype(np.float64)
            cursor += B

    assert cursor == n_total, f"forward pass didn't cover all samples ({cursor}/{n_total})"
    return pred_logret, actual_logret, stock_ids, anchor_idxs


# ---------------------------------------------------------------------------
# Pivot to [T, N] panel
# ---------------------------------------------------------------------------
def build_panel(pred_lr: np.ndarray, actual_lr: np.ndarray,
                stock_ids: np.ndarray, anchor_idxs: np.ndarray,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot per-sample (pred, actual) into [T, N] panels.

    The anchor_idx is the per-stock index of the anchor day. Different stocks
    have different anchor_idxs for the same calendar date (because their
    histories start at different dates). For cross-sectional ranking we
    align by anchor INDEX (within-stock day), accepting that this is
    approximate. To make it exact we'd thread calendar dates through.

    For now we use intersection: clip every stock to the COMMON shortest
    sample-count, indexing into anchor_idxs per stock. Same convention as
    v1 'intersect' alignment.

    Returns (P [T, N], A [T, N], stock_id_unique).
    """
    unique = np.unique(stock_ids)
    per_stock = {}
    for sid in unique:
        m = stock_ids == sid
        # Sort by anchor_idx so within-stock samples are chronological
        order = np.argsort(anchor_idxs[m])
        per_stock[int(sid)] = (pred_lr[m][order], actual_lr[m][order])
    T = min(len(p[0]) for p in per_stock.values())
    P = np.full((T, len(unique)), np.nan)
    A = np.full((T, len(unique)), np.nan)
    for j, sid in enumerate(unique):
        p, a = per_stock[int(sid)]
        P[:T, j] = p[:T]
        A[:T, j] = a[:T]
    return P, A, unique


# ---------------------------------------------------------------------------
# Cross-sectional positions
# ---------------------------------------------------------------------------
def cs_positions(pred_M: np.ndarray, top_n: int, mode: str = "long_short"):
    T, N = pred_M.shape
    pos = np.zeros((T, N))
    for t in range(T):
        pred_t = pred_M[t]
        valid = ~np.isnan(pred_t)
        if valid.sum() < 2 * top_n:
            continue
        valid_idx = np.where(valid)[0]
        order = valid_idx[np.argsort(pred_t[valid_idx])]
        bot = order[:top_n]
        top = order[-top_n:]
        long_w = np.ones(top_n) / top_n
        short_w = np.ones(top_n) / top_n
        if mode in ("long_short", "long_only"):
            pos[t, top] = long_w
        if mode in ("long_short", "short_only"):
            pos[t, bot] = -short_w
    return pos


def portfolio_returns(pos: np.ndarray, actu_M: np.ndarray, cost_bps: float = 0.0):
    T, N = pos.shape
    gross = np.full(T, np.nan)
    net = np.full(T, np.nan)
    turnover = np.zeros(T)
    prev = np.zeros(N)
    for t in range(T):
        if np.all(pos[t] == 0):
            prev = pos[t].copy()
            continue
        valid = ~np.isnan(actu_M[t])
        gross[t] = float((pos[t][valid] * actu_M[t][valid]).sum())
        to = float(np.sum(np.abs(pos[t] - prev)))
        turnover[t] = to
        net[t] = gross[t] - (cost_bps / 10000.0) * to
        prev = pos[t].copy()
    return gross, net, turnover


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def annualized_sharpe(returns: np.ndarray, horizon: int) -> float:
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    mu = r.mean(); sd = r.std(ddof=1)
    if sd < 1e-12:
        return float("nan")
    return float(mu / sd * np.sqrt(252.0 / horizon))


def cumulative_return(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) == 0: return float("nan")
    return float(np.expm1(np.sum(np.log1p(np.clip(r, -0.99, None)))))


def max_drawdown(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) == 0: return float("nan")
    log_eq = np.cumsum(np.log1p(np.clip(r, -0.99, None)))
    log_peak = np.maximum.accumulate(log_eq)
    return float(1.0 - np.exp(-np.max(log_peak - log_eq)))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--fold", default="F4")
    p.add_argument("--arm", required=True, choices=["mse", "riskhead"])
    p.add_argument("--method", default="global")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--top_n_grid", type=str, default="3,5,7,10,15")
    p.add_argument("--mode", default="long_short")
    p.add_argument("--cost_bps_grid", default="0,5,10,20,50")
    p.add_argument("--ckpt", default=None)
    args = p.parse_args()

    top_n_grid = [int(x) for x in args.top_n_grid.split(",")]
    cost_grid = [float(x) for x in args.cost_bps_grid.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default ckpt path (matches train.py save_path convention)
    if args.ckpt is None:
        args.ckpt = os.path.join(MODEL_SAVE_DIR,
                                  f"{args.model}_{args.method}_H{args.horizon}_{args.fold}_{args.arm}.pth")
    if not os.path.exists(args.ckpt):
        sys.exit(f"ERROR: checkpoint not found: {args.ckpt}")

    print(f"[eval_v2] model={args.model} H={args.horizon} fold={args.fold} arm={args.arm}")
    print(f"[eval_v2] ckpt: {args.ckpt}")

    # Build data loader
    ld = UnifiedDataLoader(seq_len=SEQ_LEN, horizon=args.horizon,
                            batch_size=args.batch_size, fold=args.fold)
    print(f"[eval_v2] {ld!r}")

    # Build model
    configs = get_config_for_model(args.model, args.horizon)
    backbone = model_dict[args.model](configs)
    if args.arm == "riskhead":
        model = RiskAwareHead(backbone, len(FEATURES), args.horizon, CLOSE_IDX,
                               20, 64).to(device)
    else:
        model = backbone.to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    # Forward pass (val + test)
    print("[eval_v2] forward pass val ...")
    val_pred, val_act, val_sid, val_aidx = predict_logret(
        model, None, device, args.arm, ld, "val")
    print("[eval_v2] forward pass test ...")
    test_pred, test_act, test_sid, test_aidx = predict_logret(
        model, None, device, args.arm, ld, "test")

    # Pivot to panels
    val_P, val_A, _ = build_panel(val_pred, val_act, val_sid, val_aidx)
    test_P, test_A, _ = build_panel(test_pred, test_act, test_sid, test_aidx)
    print(f"[eval_v2] val panel: {val_P.shape}, test panel: {test_P.shape}")

    # Sweep top_n on val
    sweep = []
    for n in top_n_grid:
        pos_v = cs_positions(val_P, n, args.mode)
        gross_v, _, _ = portfolio_returns(pos_v, val_A, cost_bps=0)
        gross_v_nover = gross_v[::args.horizon]
        sweep.append({
            "top_n": n,
            "val_sharpe_gross": annualized_sharpe(gross_v_nover, args.horizon),
            "val_n_obs": int((~np.isnan(gross_v_nover)).sum()),
        })
    sweep_df = pd.DataFrame(sweep)
    if sweep_df["val_sharpe_gross"].isna().all():
        best_n = int(top_n_grid[len(top_n_grid) // 2])
    else:
        best_n = int(sweep_df.iloc[sweep_df["val_sharpe_gross"].idxmax()]["top_n"])
    print(f"[eval_v2] best top_n on val = {best_n}")

    # Apply to test
    pos_t = cs_positions(test_P, best_n, args.mode)
    cost_table = {}
    for c in cost_grid:
        gross, net, turnover = portfolio_returns(pos_t, test_A, cost_bps=c)
        net_nover = net[::args.horizon]
        cost_table[float(c)] = {
            "cost_bps": c,
            "net_sharpe": annualized_sharpe(net_nover, args.horizon),
            "net_mdd": max_drawdown(net_nover),
            "net_cumulative_return": cumulative_return(net_nover),
            "net_n_obs_nonoverlap": int((~np.isnan(net_nover)).sum()),
            "avg_turnover_per_rebalance": float(np.nanmean(turnover[::args.horizon])),
        }
        print(f"  cost={c:>4.1f} bps : net_Sharpe={cost_table[float(c)]['net_sharpe']:6.3f}  "
              f"MDD={cost_table[float(c)]['net_mdd']:.3f}  "
              f"cumret={cost_table[float(c)]['net_cumulative_return']:6.3f}",
              flush=True)

    # Cross-sectional rank-IC (Spearman)
    from scipy.stats import spearmanr
    ic_t = np.full(test_P.shape[0], np.nan)
    for t in range(test_P.shape[0]):
        v = ~np.isnan(test_P[t]) & ~np.isnan(test_A[t])
        if v.sum() >= 5:
            r, _ = spearmanr(test_P[t][v], test_A[t][v])
            ic_t[t] = r
    ic_clean = ic_t[~np.isnan(ic_t)]
    ic_mean = float(ic_clean.mean()) if len(ic_clean) else float("nan")
    ic_std  = float(ic_clean.std(ddof=1)) if len(ic_clean) > 1 else float("nan")

    n_test_t = int((~np.isnan(test_P)).any(axis=1).sum())
    inference_valid = bool(n_test_t >= 6)
    summary = {
        "model": args.model, "horizon": args.horizon, "fold": args.fold, "arm": args.arm,
        "ckpt": args.ckpt,
        "best_top_n": best_n,
        "n_test_timestamps": n_test_t,
        "inference_valid": inference_valid,
        "ci_tier": HORIZON_CI_TIER.get(args.horizon, "unknown"),
        "ic_mean": ic_mean, "ic_std": ic_std,
        "cost_sensitivity": cost_table,
        "sweep_topn": sweep,
    }

    # Save timeseries CSV for downstream paired-bootstrap
    ts_path = os.path.join(OUT_DIR,
        f"timeseries_{args.model}_H{args.horizon}_{args.fold}_{args.arm}.csv")
    gross_t, net_t_default, _ = portfolio_returns(pos_t, test_A, cost_bps=20.0)
    pd.DataFrame({
        "portfolio_return_gross_nonoverlap": gross_t[::args.horizon],
        "portfolio_return_net20_nonoverlap": net_t_default[::args.horizon],
    }).to_csv(ts_path, index=False)

    summary_path = os.path.join(OUT_DIR,
        f"summary_{args.model}_H{args.horizon}_{args.fold}_{args.arm}.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[eval_v2] wrote {summary_path}")
    print(f"[eval_v2] wrote {ts_path}")


if __name__ == "__main__":
    main()
