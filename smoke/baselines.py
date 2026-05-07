"""No-ML cross-sectional baselines for the finsharpe benchmark.

Implements three classical, parameter-free strategies that share the
SAME evaluation pipeline as the deep-learning arms (`smoke/eval_v2.py`):

  * `xs_momentum`  — Jegadeesh & Titman (1993). Each day, rank stocks by
                     their trailing 12-month return EXCLUDING the most
                     recent 1 month (the canonical "12-1" momentum
                     filter that strips short-term reversal noise).
                     Long top-N, short bottom-N, equal-weight.
  * `xs_reversal`  — Lehmann (1990). Each day, rank stocks by their
                     trailing 5-day return; long the LOSERS, short the
                     winners (short-term reversal). Counter-momentum.
  * `ridge_alpha`  — Cheap parametric baseline. Fits a per-day cross-
                     sectional ridge (closed form) on a small bag of
                     simple features and uses the predicted alpha to
                     rank. No deep learning, no time-dependent training;
                     a sanity check on whether ML can lift over a
                     1-line linear regression.

All three reuse `smoke.eval_v2.cs_positions`, `portfolio_returns`,
`annualized_sharpe`, etc. so the produced JSON / CSV are
indistinguishable from the deep-learning cells and aggregate cleanly
through `smoke.aggregate_eval_v2`.

Citations (canonical):
    - Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners and
      Selling Losers: Implications for Stock Market Efficiency."
      J. Finance 48(1):65-91.
    - Lehmann, B. (1990). "Fads, Martingales, and Market Efficiency."
      QJE 105(1):1-28.
    - De Bondt, W. & Thaler, R. (1985). "Does the Stock Market
      Overreact?" J. Finance 40(3):793-805. (Long-run reversal; we
      implement the short-run Lehmann variant.)

Usage:
    python smoke/baselines.py --strategy xs_momentum --horizon 5  --fold F4
    python smoke/baselines.py --strategy xs_reversal --horizon 20 --fold F4
    python smoke/baselines.py --strategy ridge_alpha --horizon 5  --fold F4

Output:
    results/eval_v2/summary_BASELINE-<strategy>_H<H>_<fold>_baseline.json
    results/eval_v2/timeseries_BASELINE-<strategy>_H<H>_<fold>_baseline.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from config import (CLOSE_IDX, FEATURES, SEQ_LEN, HORIZON_CI_TIER)
from data_loader import UnifiedDataLoader
from smoke.eval_v2 import (annualized_sharpe, build_panel, cs_positions,
                            cumulative_return, max_drawdown,
                            portfolio_returns)

OUT_DIR = os.path.join(os.path.dirname(_HERE), "results", "eval_v2")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-sample baseline signal computation
# ---------------------------------------------------------------------------
def baseline_signal(strategy: str, X_window: np.ndarray) -> float:
    """Compute a single scalar signal per sample.

    Args
    ----
        strategy : "xs_momentum" | "xs_reversal" | "ridge_alpha"
        X_window : [seq_len, n_features] z-scored feature window for ONE
                   sample. We use the close column (CLOSE_IDX) — z-scores
                   preserve relative differences within a stock, which is
                   all the strategy needs (it ranks across stocks per day).

    Returns
    -------
        signal : float. Higher = "long this stock"; lower = "short this
                 stock". Cross-sectional ranking only — absolute scale
                 doesn't matter.
    """
    closes_z = X_window[:, CLOSE_IDX]                    # [seq_len]
    L = len(closes_z)
    if strategy == "xs_momentum":
        # 12-1 momentum: change from ~252 days ago to ~21 days ago.
        # We don't have raw prices here (data is z-scored), but z-score
        # CHANGE is monotone in real-price log-return WITHIN a stock, so
        # using the z-delta is fine for ranking purposes.
        idx_far  = max(0, L - 252)            # ~12 months back (or earliest)
        idx_near = max(0, L - 21)             # ~1 month back
        return float(closes_z[idx_near] - closes_z[idx_far])
    elif strategy == "xs_reversal":
        # 5-day reversal: SHORT recent winners → signal = -last_5d_change.
        idx_5 = max(0, L - 5)
        return float(-(closes_z[-1] - closes_z[idx_5]))
    elif strategy == "ridge_alpha":
        # Cheap 4-feature alpha:
        #   f1 = 12-1 momentum (long-short)
        #   f2 = 1-day change (short-term reversal proxy)
        #   f3 = 60-day change (medium-term momentum)
        #   f4 = sentiment last value (z-scored sentiment)
        # Ridge weights set BY HAND (not fit per-fold to avoid leakage):
        #   1.0, -0.3, 0.5, 0.2
        # so f1+f3 dominate (momentum), f2 contributes negatively (reversal),
        # sentiment as a small tilt.
        idx_far  = max(0, L - 252); idx_near = max(0, L - 21)
        idx_60   = max(0, L - 60)
        f1 = closes_z[idx_near] - closes_z[idx_far]
        # 1-day reversal proxy (last - prior); guard L<2.
        f2 = (closes_z[-1] - closes_z[max(0, L - 2)]) if L >= 2 else 0.0
        f3 = closes_z[-1] - closes_z[idx_60]
        # Sentiment column is the LAST of FEATURES list ("scaled_sentiment").
        f4 = float(X_window[-1, len(FEATURES) - 1])
        return float(1.0 * f1 - 0.3 * f2 + 0.5 * f3 + 0.2 * f4)
    else:
        raise ValueError(f"Unknown baseline strategy: {strategy!r}")


def collect_signals(ld: UnifiedDataLoader, strategy: str,
                     split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (signal, actual_logret, stock_id, anchor_date) for every
    sample in `split` using the no-ML strategy `strategy`.

    Iterates per-sample directly off the loader's sample_table (no model
    forward pass needed — these strategies are parameter-free).
    """
    sample_table = (ld.sample_table_val if split == "val"
                    else ld.sample_table_test)
    anchor_dates_split = (ld.val_anchor_date if split == "val"
                          else ld.test_anchor_date)
    n = sample_table.shape[0]
    pred = np.zeros(n, dtype=np.float64)
    actu = np.zeros(n, dtype=np.float64)
    stock_ids = sample_table[:, 0].astype(np.int32)
    anchor_dates = anchor_dates_split.astype(np.int64)

    seq_len = ld.seq_len
    horizon = ld.horizon
    for k in range(n):
        sid = int(sample_table[k, 0])
        i   = int(sample_table[k, 1])
        X = ld.raw_normed_list[sid][i: i + seq_len]
        pred[k] = baseline_signal(strategy, X)
        close = ld.close_raw_list[sid]
        ac = close[i + seq_len - 1]
        tc = close[i + seq_len + horizon - 1]
        actu[k] = float(np.log(tc / ac))
    return pred, actu, stock_ids, anchor_dates


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", required=True,
                   choices=["xs_momentum", "xs_reversal", "ridge_alpha"])
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--fold", default="F4")
    p.add_argument("--top_n_grid", type=str, default="3,5,7,10,15")
    p.add_argument("--cost_bps_grid", default="0,5,10,20,50")
    p.add_argument("--mode", default="long_short")
    args = p.parse_args()

    top_n_grid = [int(x) for x in args.top_n_grid.split(",")]
    cost_grid  = [float(x) for x in args.cost_bps_grid.split(",")]

    print(f"[baseline] strategy={args.strategy} H={args.horizon} fold={args.fold}")

    ld = UnifiedDataLoader(seq_len=SEQ_LEN, horizon=args.horizon,
                            fold=args.fold, batch_size=512)
    print(f"[baseline] {ld!r}")

    val_pred, val_act, val_sid, val_adate = collect_signals(ld, args.strategy, "val")
    test_pred, test_act, test_sid, test_adate = collect_signals(ld, args.strategy, "test")

    val_P, val_A, _, _   = build_panel(val_pred, val_act, val_sid, val_adate)
    test_P, test_A, _, _ = build_panel(test_pred, test_act, test_sid, test_adate)
    print(f"[baseline] val panel: {val_P.shape}  test panel: {test_P.shape}")

    # Sweep top_n on val (gross Sharpe, like the ML cells).
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
    print(f"[baseline] best top_n on val = {best_n}")

    # Apply to test (HEADLINE: long-short).
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

    # ─── Long-only appendix (peer-comparable to MASTER/HIST/FactorVAE).
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
    p30 = longonly_table.get("peer30_c20", {})
    if p30:
        print(f"  [long-only N=30 @ 20bps]  Sharpe={p30.get('longonly_net_sharpe', float('nan')):6.3f}  "
              f"MDD={p30.get('longonly_net_mdd', float('nan')):.3f}  "
              f"cumret={p30.get('longonly_net_cumulative_return', float('nan')):6.3f}",
              flush=True)

    # Cross-sectional rank-IC.
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
    summary = {
        "model": f"BASELINE-{args.strategy}",
        "horizon": args.horizon, "fold": args.fold, "arm": "baseline",
        "best_top_n": best_n,
        "n_test_timestamps": n_test_t,
        "inference_valid": bool(n_test_t >= 6),
        "ci_tier": HORIZON_CI_TIER.get(args.horizon, "unknown"),
        "ic_mean": ic_mean, "ic_std": ic_std,
        "cost_sensitivity": cost_table,
        "longonly_appendix": longonly_table,
        "sweep_topn": sweep,
    }

    ts_path = os.path.join(OUT_DIR,
        f"timeseries_BASELINE-{args.strategy}_H{args.horizon}_{args.fold}_baseline.csv")
    gross_t, net_t_default, _ = portfolio_returns(pos_t, test_A, cost_bps=20.0)
    pos_lo30 = cs_positions(test_P, 30, "long_only")
    lo_gross_t, lo_net_t, _ = portfolio_returns(pos_lo30, test_A, cost_bps=20.0)
    nover_g = gross_t[::args.horizon]; nover_n = net_t_default[::args.horizon]
    lo_nover_g = lo_gross_t[::args.horizon]; lo_nover_n = lo_net_t[::args.horizon]
    n_full = len(gross_t)
    pd.DataFrame({
        "portfolio_return_gross_nonoverlap": np.concatenate(
            [nover_g, np.full(n_full - len(nover_g), np.nan)]),
        "portfolio_return_net20_nonoverlap": np.concatenate(
            [nover_n, np.full(n_full - len(nover_n), np.nan)]),
        "portfolio_return_gross_daily": gross_t,
        "portfolio_return_net20_daily": net_t_default,
        "longonly_n30_gross_nonoverlap": np.concatenate(
            [lo_nover_g, np.full(n_full - len(lo_nover_g), np.nan)]),
        "longonly_n30_net20_nonoverlap": np.concatenate(
            [lo_nover_n, np.full(n_full - len(lo_nover_n), np.nan)]),
        "longonly_n30_gross_daily": lo_gross_t,
        "longonly_n30_net20_daily": lo_net_t,
    }).to_csv(ts_path, index=False)

    summary_path = os.path.join(OUT_DIR,
        f"summary_BASELINE-{args.strategy}_H{args.horizon}_{args.fold}_baseline.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[baseline] wrote {summary_path}")
    print(f"[baseline] wrote {ts_path}")


if __name__ == "__main__":
    main()
