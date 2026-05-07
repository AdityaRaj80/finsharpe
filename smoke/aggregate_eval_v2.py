"""Aggregate Stage-1 eval_v2 outputs into headline tables.

Reads every results/eval_v2/summary_<cell>.json and produces:
  1. results/eval_v2/headline_table.csv  -- one row per (model, horizon,
     fold, arm) with key metrics flattened (best_top_n, ic_mean, and
     net Sharpe / MDD / cumret at every cost level).
  2. results/eval_v2/paired_bootstrap_F4.csv  -- per (model, horizon)
     paired stationary-bootstrap CI on (Sharpe_riskhead - Sharpe_mse).
  3. results/eval_v2/dsr_F4.json  -- Deflated Sharpe Ratio (Bailey-LdP
     2014) on the headline cell using the full sweep as N_trials.
  4. Markdown table for paste-into-paper.

Usage:
    python smoke/aggregate_eval_v2.py --fold F4
    python smoke/aggregate_eval_v2.py --fold F4 --cost_bps 20
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from smoke.bootstrap_paired import (politis_white_block_length,
                                      stationary_bootstrap_indices,
                                      annualized_sharpe)
from smoke.deflated_sharpe import compute_dsr_from_returns

OUT_DIR = os.path.join(os.path.dirname(_HERE), "results", "eval_v2")


def load_all_summaries(fold: str) -> pd.DataFrame:
    """Load every summary JSON for the given fold; return one row per cell."""
    pat = os.path.join(OUT_DIR, f"summary_*_{fold}_*.json")
    rows = []
    for path in sorted(glob.glob(pat)):
        try:
            with open(path) as fh:
                s = json.load(fh)
        except Exception as e:
            print(f"  [warn] {path}: {e}", file=sys.stderr)
            continue
        flat = {
            "model": s.get("model"),
            "horizon": s.get("horizon"),
            "fold": s.get("fold"),
            "arm": s.get("arm"),
            "best_top_n": s.get("best_top_n"),
            "ic_mean": s.get("ic_mean"),
            "ic_std": s.get("ic_std"),
            "n_test_t": s.get("n_test_timestamps"),
            "ci_tier": s.get("ci_tier"),
        }
        cost_table = s.get("cost_sensitivity", {})
        for cost_str, m in cost_table.items():
            try:
                c = float(cost_str)
            except ValueError:
                c = float(m.get("cost_bps", 0))
            tag = f"c{int(c)}"
            flat[f"{tag}_net_sharpe"] = m.get("net_sharpe")
            flat[f"{tag}_net_mdd"] = m.get("net_mdd")
            flat[f"{tag}_net_cumret"] = m.get("net_cumulative_return")
            flat[f"{tag}_avg_turnover"] = m.get("avg_turnover_per_rebalance")
        rows.append(flat)
    return pd.DataFrame(rows)


def pivot_arms(df: pd.DataFrame, cost_bps: int = 20) -> pd.DataFrame:
    """Pivot MSE vs Track-B arm side-by-side per (model, horizon)."""
    sharpe_col = f"c{cost_bps}_net_sharpe"
    sub = df[["model", "horizon", "arm", "best_top_n", "ic_mean",
              sharpe_col, f"c{cost_bps}_net_mdd",
              f"c{cost_bps}_net_cumret", f"c{cost_bps}_avg_turnover"]].copy()
    return sub.pivot_table(
        index=["model", "horizon"],
        columns="arm",
        values=[sharpe_col, "ic_mean", "best_top_n",
                f"c{cost_bps}_net_mdd", f"c{cost_bps}_net_cumret"],
        aggfunc="first",
    )


def paired_bootstrap_per_cell(fold: str, cost_bps: int = 20,
                                n_boot: int = 2000) -> pd.DataFrame:
    """For every (model, horizon) load both arms' timeseries CSVs and
    run a paired stationary bootstrap on (SR_riskhead - SR_mse)."""
    rows = []
    pat_mse = os.path.join(OUT_DIR, f"timeseries_*_H*_{fold}_mse.csv")
    for ts_mse_path in sorted(glob.glob(pat_mse)):
        base = os.path.basename(ts_mse_path).replace("timeseries_", "").replace("_mse.csv", "")
        ts_rh_path = os.path.join(OUT_DIR, f"timeseries_{base}_riskhead.csv")
        if not os.path.exists(ts_rh_path):
            continue
        try:
            mse_ts = pd.read_csv(ts_mse_path)["portfolio_return_net20_nonoverlap"].dropna().values
            rh_ts  = pd.read_csv(ts_rh_path)["portfolio_return_net20_nonoverlap"].dropna().values
        except Exception as e:
            continue
        n = min(len(mse_ts), len(rh_ts))
        if n < 6:
            continue
        a, b = rh_ts[:n], mse_ts[:n]    # A = riskhead, B = mse
        # Parse "<model>_H<horizon>_<fold>"
        parts = base.split("_H")
        model = parts[0]
        # parts[1] looks like "20_F4" -- take the leading digits
        horizon_str = parts[1].split("_")[0]
        horizon = int(horizon_str)

        # Block length
        eb = max(2.0, (politis_white_block_length(a) + politis_white_block_length(b)) / 2)
        rng = np.random.default_rng(2026)
        boot_d = np.zeros(n_boot)
        for k in range(n_boot):
            idx = stationary_bootstrap_indices(n, eb, rng)
            sa = annualized_sharpe(a[idx], horizon)
            sb = annualized_sharpe(b[idx], horizon)
            boot_d[k] = sa - sb
        boot_d = boot_d[~np.isnan(boot_d)]
        if len(boot_d) < 100:
            continue
        point_a = annualized_sharpe(a, horizon)
        point_b = annualized_sharpe(b, horizon)
        d_lo, d_md, d_hi = np.percentile(boot_d, [2.5, 50, 97.5])
        p_one = float((boot_d <= 0).mean())
        rows.append({
            "model": model, "horizon": horizon, "n": n,
            "block_length": eb,
            "sharpe_riskhead": point_a, "sharpe_mse": point_b,
            "delta_sharpe": point_a - point_b,
            "ci95_diff_lo": d_lo, "ci95_diff_md": d_md, "ci95_diff_hi": d_hi,
            "p_one_sided_RH_le_MSE": p_one,
            "significant_at_5pct": bool(p_one < 0.05),
        })
    return pd.DataFrame(rows)


def headline_dsr(fold: str, cost_bps: int = 20) -> dict:
    """Compute DSR for the BEST (highest net Sharpe at cost_bps) cell of
    the entire campaign."""
    df = load_all_summaries(fold)
    sharpe_col = f"c{cost_bps}_net_sharpe"
    if df.empty or sharpe_col not in df.columns:
        return {"error": "no summaries available"}

    best = df.loc[df[sharpe_col].idxmax()]
    base = f"{best['model']}_H{int(best['horizon'])}_{fold}_{best['arm']}"
    ts_path = os.path.join(OUT_DIR, f"timeseries_{base}.csv")
    if not os.path.exists(ts_path):
        return {"error": f"timeseries missing: {ts_path}"}

    winning = pd.read_csv(ts_path)["portfolio_return_net20_nonoverlap"].dropna().values
    trial_sharpes = df[sharpe_col].dropna().values
    res = compute_dsr_from_returns(winning, trial_sharpes, int(best["horizon"]))
    res["winning_cell"] = base
    res["winning_arm"] = best["arm"]
    res["winning_sharpe"] = float(best[sharpe_col])
    res["n_trials_total"] = int(len(trial_sharpes))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", default="F4")
    p.add_argument("--cost_bps", type=int, default=20,
                   help="Cost level for headline metrics (default 20 bps).")
    p.add_argument("--n_boot", type=int, default=2000)
    args = p.parse_args()

    print(f"[aggregate_eval_v2] fold={args.fold}  headline_cost={args.cost_bps}bps")
    print(f"[aggregate_eval_v2] reading from {OUT_DIR}")

    # 1. Headline table (all cells)
    df = load_all_summaries(args.fold)
    if df.empty:
        sys.exit(f"No summaries found in {OUT_DIR} for fold {args.fold}")
    out_path = os.path.join(OUT_DIR, f"headline_table_{args.fold}.csv")
    df.to_csv(out_path, index=False)
    print(f"[aggregate_eval_v2] {len(df)} cells -> {out_path}")

    # 2. Pivot view
    pv = pivot_arms(df, args.cost_bps)
    pv_path = os.path.join(OUT_DIR, f"pivot_arms_{args.fold}.csv")
    pv.to_csv(pv_path)
    print(f"[aggregate_eval_v2] pivot_arms -> {pv_path}")

    # 3. Paired bootstrap per cell
    print(f"[aggregate_eval_v2] paired bootstrap (RH vs MSE)...")
    pb = paired_bootstrap_per_cell(args.fold, args.cost_bps, args.n_boot)
    pb_path = os.path.join(OUT_DIR, f"paired_bootstrap_{args.fold}.csv")
    pb.to_csv(pb_path, index=False)
    print(f"[aggregate_eval_v2] {len(pb)} (model, horizon) pairs -> {pb_path}")
    if not pb.empty:
        sig = int(pb["significant_at_5pct"].sum())
        print(f"  -> significant at p<0.05: {sig}/{len(pb)}")

    # 4. Headline DSR
    print(f"[aggregate_eval_v2] headline DSR (Bailey-LdP 2014)...")
    dsr = headline_dsr(args.fold, args.cost_bps)
    dsr_path = os.path.join(OUT_DIR, f"dsr_{args.fold}.json")
    with open(dsr_path, "w") as fh:
        json.dump(dsr, fh, indent=2)
    print(f"[aggregate_eval_v2] {dsr_path}")
    if "error" not in dsr:
        print(f"  winning: {dsr['winning_cell']}  Sharpe={dsr['winning_sharpe']:.3f}")
        print(f"  N_trials={dsr['n_trials_total']}  T={dsr['T_periods']}")
        print(f"  DSR={dsr.get('dsr', float('nan')):.4f}  "
              f"({'SIGNIFICANT' if dsr.get('is_significant_at_95') else 'not significant'} at 95%)")

    # 5. Markdown headline table for paste
    md_path = os.path.join(OUT_DIR, f"headline_md_{args.fold}.md")
    sharpe_col = f"c{args.cost_bps}_net_sharpe"
    md_rows = ["# Headline net-Sharpe table (cost = " + str(args.cost_bps) + " bps)\n",
               "| Model | H | Arm | Sharpe | IC | top_n | n_obs | tier |",
               "|-------|--:|-----|-------:|---:|------:|------:|------|"]
    for _, r in df.sort_values(["model", "horizon", "arm"]).iterrows():
        md_rows.append(
            f"| {r['model']} | {r['horizon']} | {r['arm']} | "
            f"{r.get(sharpe_col, float('nan')):.3f} | "
            f"{r.get('ic_mean', float('nan')):.3f} | "
            f"{int(r.get('best_top_n', 0))} | "
            f"{int(r.get('n_test_t', 0))} | {r.get('ci_tier', '?')} |"
        )
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md_rows))
    print(f"[aggregate_eval_v2] markdown headline -> {md_path}")


if __name__ == "__main__":
    main()
