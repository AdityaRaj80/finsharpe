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
from engine.heads import RiskAwareHead, MSEReturnHead
from models import model_dict
from train import get_config_for_model

OUT_DIR = os.path.join(os.path.dirname(_HERE), "results", "eval_v2")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Forward + per-sample predicted log-return extraction
# ---------------------------------------------------------------------------
def predict_logret(model, loader, device, arm: str, ld: UnifiedDataLoader,
                    split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-sample (pred_logret, actual_logret, stock_id, anchor_idx,
    anchor_date) over the entire `split` window.

    Apples-to-apples (Jury 2 fix B1+B2, 2026-05-08): both arms produce a
    real-log-return prediction:

        * MSE arm        — model is `MSEReturnHead(backbone)`; its
                            forward returns `[B]` predicted log-return.
        * Track-B arm    — model is `RiskAwareHead(backbone)`; we read
                            `out["mu_return_H"]`, also a `[B]` predicted
                            log-return.

    The previous z-delta hack injected an inverse-volatility bias in the
    MSE arm and made the two arms incomparable; both paths are removed.

    Args
    ----
        arm   : "mse" or "riskhead". Determines which output channel to read.
        split : "val" or "test".

    Returns
    -------
        pred_logret    : [N] predicted H-step log-return per sample
        actual_logret  : [N] true H-step log-return per sample (from loader)
        stock_ids      : [N] int stock identifier
        anchor_idxs    : [N] within-stock anchor index (legacy, for diagnostics)
        anchor_dates   : [N] int64 anchor calendar date (ns since epoch);
                         used by `build_panel` for calendar-aligned pivot.
    """
    sample_table = {"val": ld.sample_table_val, "test": ld.sample_table_test}[split]
    anchor_dates_split = (ld.val_anchor_date if split == "val"
                            else ld.test_anchor_date)
    n_total = sample_table.shape[0]
    pred_logret = np.zeros(n_total, dtype=np.float64)
    actual_logret = np.zeros(n_total, dtype=np.float64)
    stock_ids = sample_table[:, 0].astype(np.int32)
    anchor_idxs = sample_table[:, 2].astype(np.int32)
    anchor_dates = anchor_dates_split.astype(np.int64)

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
                pred_lr = out["mu_return_H"].cpu().numpy().astype(np.float64)
            else:  # mse — MSEReturnHead returns [B] log-return scalar
                if isinstance(out, dict):
                    # Defensive: legacy checkpoint that was wrapped in
                    # RiskAwareHead even for the MSE arm. Read mu_return_H.
                    pred_lr = out["mu_return_H"].cpu().numpy().astype(np.float64)
                elif isinstance(out, tuple):
                    # AdaPatch legacy path or backbone returning a tuple.
                    arr = out[0].cpu().numpy().astype(np.float64)
                    if arr.ndim == 2 and arr.shape[1] > 1:
                        # Legacy [B, pred_len] z-scored close — refuse to rank
                        # on this (it injects bias). Caller must retrain with
                        # MSEReturnHead.
                        raise RuntimeError(
                            "MSE arm checkpoint produces [B, pred_len] z-scored "
                            "close, but this version of eval_v2 ranks on real "
                            "log-return only (Jury 2 fix B1+B2). Retrain with "
                            "MSEReturnHead-wrapped backbone.")
                    pred_lr = arr.squeeze(-1)
                else:
                    arr = out.cpu().numpy().astype(np.float64)
                    if arr.ndim == 2 and arr.shape[1] > 1:
                        raise RuntimeError(
                            "MSE arm checkpoint produces [B, pred_len] z-scored "
                            "close, but this version of eval_v2 ranks on real "
                            "log-return only. Retrain with MSEReturnHead.")
                    pred_lr = arr.squeeze(-1) if arr.ndim == 2 else arr

            pred_logret[cursor:cursor + B] = pred_lr
            actual_logret[cursor:cursor + B] = y_logret.numpy().astype(np.float64)
            cursor += B

    assert cursor == n_total, f"forward pass didn't cover all samples ({cursor}/{n_total})"
    return pred_logret, actual_logret, stock_ids, anchor_idxs, anchor_dates


# ---------------------------------------------------------------------------
# Pivot to [T, N] panel BY CALENDAR DATE
# ---------------------------------------------------------------------------
def build_panel(pred_lr: np.ndarray, actual_lr: np.ndarray,
                stock_ids: np.ndarray, anchor_dates: np.ndarray,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pivot per-sample (pred, actual) into [T, N] panels aligned BY DATE.

    Jury 2 fix B3 (2026-05-08): row T of the panel now corresponds to a
    SHARED calendar date across all stocks. Stocks not trading on date T
    (or with no sample anchored on that date) get NaN in row T.
    Previously the pivot used per-stock anchor INDEX, which paired
    different calendar dates for different stocks in the same panel row —
    destroying the cross-sectional Sharpe / IC interpretation entirely.

    Args
    ----
        pred_lr       : [N] predicted log-return per sample
        actual_lr     : [N] true log-return per sample
        stock_ids     : [N] int stock identifier
        anchor_dates  : [N] int64 anchor calendar date (ns since epoch)

    Returns
    -------
        P : [T, N_stocks] predicted log-return panel (NaN where missing)
        A : [T, N_stocks] actual    log-return panel (NaN where missing)
        unique_stocks : [N_stocks] sorted unique stock IDs
        unique_dates  : [T]        sorted unique anchor dates (int64 ns)
    """
    unique_dates  = np.unique(anchor_dates)
    unique_stocks = np.unique(stock_ids)
    T = len(unique_dates)
    M = len(unique_stocks)
    # Map each row sample to (date_idx, stock_idx) via searchsorted.
    date_idx  = np.searchsorted(unique_dates,  anchor_dates)
    stock_idx = np.searchsorted(unique_stocks, stock_ids)
    P = np.full((T, M), np.nan, dtype=np.float64)
    A = np.full((T, M), np.nan, dtype=np.float64)
    P[date_idx, stock_idx] = pred_lr
    A[date_idx, stock_idx] = actual_lr
    return P, A, unique_stocks, unique_dates


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
    """Compute gross/net portfolio returns and turnover per row.

    `cost_bps` is interpreted as ONE-SIDE cost in basis points (e.g. 20 bps
    one-side ≈ 40 bps round-trip). Turnover is the L1 distance between
    consecutive position vectors, which already counts BOTH legs of every
    trade (entering a long + exiting a short on the same day contributes
    |w_long_new| + |w_short_old| to turnover). Therefore the cost charge
    is `cost_bps × turnover / 10_000` — already a round-trip equivalent.

    Jury 2 fix N9 (2026-05-08): turnover at the FIRST trading row is now
    correctly computed against the zero starting position. The previous
    bug skipped rows entirely when `pos[t]==0`, leaving `prev` un-updated
    and double-counting turnover on the next non-zero row.
    """
    T, N = pos.shape
    gross = np.full(T, np.nan)
    net = np.full(T, np.nan)
    turnover = np.zeros(T)
    prev = np.zeros(N)
    for t in range(T):
        # Always compute turnover (even on zero-position rows — they pay
        # the cost of UNWINDING the prior position when relevant).
        to = float(np.sum(np.abs(pos[t] - prev)))
        turnover[t] = to
        if np.all(pos[t] == 0) and to == 0.0:
            # Truly idle row (no position, no trade); skip return calc.
            prev = pos[t].copy()
            continue
        valid = ~np.isnan(actu_M[t])
        gross[t] = float((pos[t][valid] * actu_M[t][valid]).sum())
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

    # Build model — wrap per arm. Jury 2 fix B1+B2: MSE arm now uses
    # MSEReturnHead so its output is real-log-return scalar (apples-to-
    # apples with Track-B's mu_return_H).
    configs = get_config_for_model(args.model, args.horizon)
    backbone = model_dict[args.model](configs)
    if args.arm == "riskhead":
        model = RiskAwareHead(backbone, len(FEATURES), args.horizon, CLOSE_IDX,
                               20, 64).to(device)
    elif args.model == "AdaPatch":
        model = backbone.to(device)
    else:
        model = MSEReturnHead(backbone, args.horizon).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    # Forward pass (val + test)
    print("[eval_v2] forward pass val ...")
    val_pred, val_act, val_sid, val_aidx, val_adate = predict_logret(
        model, None, device, args.arm, ld, "val")
    print("[eval_v2] forward pass test ...")
    test_pred, test_act, test_sid, test_aidx, test_adate = predict_logret(
        model, None, device, args.arm, ld, "test")

    # Pivot to panels (BY CALENDAR DATE — Jury 2 fix B3, 2026-05-08).
    val_P, val_A, _, val_dates_unique  = build_panel(val_pred,  val_act,  val_sid,  val_adate)
    test_P, test_A, _, test_dates_unique = build_panel(test_pred, test_act, test_sid, test_adate)
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

    # Apply to test (HEADLINE protocol: long-short at best_top_n).
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

    # ─── Long-only protocol (peer-comparable to MASTER/HIST/FactorVAE).
    # Computes Sharpe of the long leg ALONE (top-N equal-weight) so our
    # numbers can be directly compared to peer papers that report
    # long-only top-K Sharpe (gross).
    # We do this at TWO N values:
    #   * best_top_n          — the same N selected by val sweep above (apples
    #                            to apples with our long-short headline).
    #   * peer_top_n = 30      — top 10% of universe; matches MASTER's Top-30
    #                            convention on CSI300 and HIST's Top-50/CSI300
    #                            (~17%). 10% is the standard quintile-decile
    #                            split in the factor-investing literature.
    longonly_table = {}
    for n_label, n in (("best_n", best_n), ("peer30", 30)):
        pos_lo = cs_positions(test_P, n, "long_only")
        for c in cost_grid:
            gross, net, _ = portfolio_returns(pos_lo, test_A, cost_bps=c)
            net_nover = net[::args.horizon]
            longonly_table[f"{n_label}_c{int(c)}"] = {
                "n_label": n_label, "top_n": int(n), "cost_bps": c,
                "longonly_net_sharpe": annualized_sharpe(net_nover, args.horizon),
                "longonly_net_mdd": max_drawdown(net_nover),
                "longonly_net_cumulative_return": cumulative_return(net_nover),
            }
    # Print one summary row (peer N=30 at 20bps) for visibility.
    p30 = longonly_table.get("peer30_c20", {})
    if p30:
        print(f"  [long-only N=30 @ 20bps]  Sharpe={p30.get('longonly_net_sharpe', float('nan')):6.3f}  "
              f"MDD={p30.get('longonly_net_mdd', float('nan')):.3f}  "
              f"cumret={p30.get('longonly_net_cumulative_return', float('nan')):6.3f}",
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
        "longonly_appendix": longonly_table,
        "sweep_topn": sweep,
    }

    # Save timeseries CSV for downstream paired-bootstrap.
    # Jury 2 fix B8 / F24 (2026-05-08): write BOTH non-overlapping (legacy)
    # and OVERLAPPING DAILY return series. The aggregator uses the daily
    # series with a Newey-White HAC block when the non-overlapping series
    # has n<6 (typical at H=60 on a 1-year fold) — restoring statistical
    # power at long horizons that were previously untestable.
    # Long-only series ALSO written for peer-comparable Sharpe (2026-05-08).
    ts_path = os.path.join(OUT_DIR,
        f"timeseries_{args.model}_H{args.horizon}_{args.fold}_{args.arm}.csv")
    gross_t, net_t_default, _ = portfolio_returns(pos_t, test_A, cost_bps=20.0)
    # Long-only at peer N=30 — gross + net20.
    pos_lo30 = cs_positions(test_P, 30, "long_only")
    lo_gross_t, lo_net_t, _ = portfolio_returns(pos_lo30, test_A, cost_bps=20.0)
    # Non-overlapping series: subsampled every H rows.
    nover_gross = gross_t[::args.horizon]
    nover_net   = net_t_default[::args.horizon]
    lo_nover_g  = lo_gross_t[::args.horizon]
    lo_nover_n  = lo_net_t[::args.horizon]
    # Daily-overlap series: full-length, one entry per anchor day.
    n_full = len(gross_t)
    df_ts = {
        "portfolio_return_gross_nonoverlap": np.concatenate(
            [nover_gross, np.full(n_full - len(nover_gross), np.nan)]),
        "portfolio_return_net20_nonoverlap": np.concatenate(
            [nover_net, np.full(n_full - len(nover_net), np.nan)]),
        "portfolio_return_gross_daily": gross_t,
        "portfolio_return_net20_daily": net_t_default,
        # Long-only N=30 columns (peer-comparable to MASTER/HIST/FactorVAE).
        "longonly_n30_gross_nonoverlap": np.concatenate(
            [lo_nover_g, np.full(n_full - len(lo_nover_g), np.nan)]),
        "longonly_n30_net20_nonoverlap": np.concatenate(
            [lo_nover_n, np.full(n_full - len(lo_nover_n), np.nan)]),
        "longonly_n30_gross_daily": lo_gross_t,
        "longonly_n30_net20_daily": lo_net_t,
    }
    pd.DataFrame(df_ts).to_csv(ts_path, index=False)

    summary_path = os.path.join(OUT_DIR,
        f"summary_{args.model}_H{args.horizon}_{args.fold}_{args.arm}.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[eval_v2] wrote {summary_path}")
    print(f"[eval_v2] wrote {ts_path}")


if __name__ == "__main__":
    main()
