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
    run a paired stationary bootstrap on (SR_riskhead - SR_mse).

    Long-horizon handling (Jury 2 fix B8 / F24, 2026-05-08): when the
    non-overlapping series has n<6 (typical for H=60 on a 1-year fold),
    we instead run the bootstrap on the OVERLAPPING daily series (with
    Newey-White HAC block length) and report `n_overlap` along with a
    flag `used_overlap_HAC=True`. This restores statistical validity
    at long horizons (Lo 2002), at the cost of correlated rebalances —
    the block length absorbs the daily autocorrelation so the bootstrap
    variance is approximately correct under H0.
    """
    rows = []
    pat_mse = os.path.join(OUT_DIR, f"timeseries_*_H*_{fold}_mse.csv")
    NONOVER_COL = "portfolio_return_net20_nonoverlap"
    OVER_COL    = "portfolio_return_net20_daily"   # may not exist in legacy CSVs
    for ts_mse_path in sorted(glob.glob(pat_mse)):
        base = os.path.basename(ts_mse_path).replace("timeseries_", "").replace("_mse.csv", "")
        ts_rh_path = os.path.join(OUT_DIR, f"timeseries_{base}_riskhead.csv")
        if not os.path.exists(ts_rh_path):
            continue
        try:
            mse_df = pd.read_csv(ts_mse_path)
            rh_df  = pd.read_csv(ts_rh_path)
        except Exception:
            continue
        # Parse "<model>_H<horizon>_<fold>"
        parts = base.split("_H")
        model = parts[0]
        horizon_str = parts[1].split("_")[0]
        horizon = int(horizon_str)

        # First try non-overlapping; if too short, fall back to daily HAC.
        used_overlap = False
        if NONOVER_COL in mse_df.columns and NONOVER_COL in rh_df.columns:
            mse_ts = mse_df[NONOVER_COL].dropna().values
            rh_ts  = rh_df[NONOVER_COL].dropna().values
            n = min(len(mse_ts), len(rh_ts))
        else:
            n = 0
        if n < 6 and OVER_COL in mse_df.columns and OVER_COL in rh_df.columns:
            mse_ts = mse_df[OVER_COL].dropna().values
            rh_ts  = rh_df[OVER_COL].dropna().values
            n = min(len(mse_ts), len(rh_ts))
            used_overlap = True
        if n < 6:
            continue
        a, b = rh_ts[:n], mse_ts[:n]    # A = riskhead, B = mse

        # Block length. For overlapping daily returns at horizon H, the
        # autocorrelation has support ~H lags, so floor block length at H.
        pw_a = politis_white_block_length(a)
        pw_b = politis_white_block_length(b)
        eb = max(2.0, (pw_a + pw_b) / 2)
        if used_overlap:
            eb = max(eb, float(horizon))   # Newey-White HAC bandwidth ≥ H

        rng = np.random.default_rng(2026)
        boot_d = np.zeros(n_boot)
        # When using overlapping returns the annualisation is sqrt(252)
        # (per-day Sharpe times sqrt(252)), not sqrt(252/H). We compute
        # both the point and the bootstrap on a per-PERIOD basis and only
        # annualise at the end.
        eff_horizon = 1 if used_overlap else horizon
        for k in range(n_boot):
            idx = stationary_bootstrap_indices(n, eb, rng)
            sa = annualized_sharpe(a[idx], eff_horizon)
            sb = annualized_sharpe(b[idx], eff_horizon)
            boot_d[k] = sa - sb
        boot_d = boot_d[~np.isnan(boot_d)]
        if len(boot_d) < 100:
            continue
        point_a = annualized_sharpe(a, eff_horizon)
        point_b = annualized_sharpe(b, eff_horizon)
        d_lo, d_md, d_hi = np.percentile(boot_d, [2.5, 50, 97.5])
        p_one = float((boot_d <= 0).mean())
        rows.append({
            "model": model, "horizon": horizon, "n": n,
            "block_length": eb,
            "used_overlap_HAC": used_overlap,
            "sharpe_riskhead": point_a, "sharpe_mse": point_b,
            "delta_sharpe": point_a - point_b,
            "ci95_diff_lo": d_lo, "ci95_diff_md": d_md, "ci95_diff_hi": d_hi,
            "p_one_sided_RH_le_MSE": p_one,
            "significant_at_5pct": bool(p_one < 0.05),
        })
    return pd.DataFrame(rows)


def _collect_swept_trial_sharpes(fold: str, arm: str) -> np.ndarray:
    """Read the val-sweep Sharpes from every summary JSON for the given
    (fold, arm) and return them as a flat array.

    Each summary JSON has a `sweep_topn` list with one entry per top_n
    grid value, each containing `val_sharpe_gross`. These are the ACTUAL
    Sharpes the analyst could have selected as headline (one per
    (model, horizon, top_n) tuple), so they are the correct trial pool
    for Bailey-LdP DSR variance estimation (Jury 2 fix IMP3).

    Falls back to the cell-level argmax Sharpe when sweep_topn is missing.
    """
    pat = os.path.join(OUT_DIR, f"summary_*_{fold}_{arm}.json")
    out = []
    for path in sorted(glob.glob(pat)):
        try:
            with open(path) as fh:
                s = json.load(fh)
        except Exception:
            continue
        sweep = s.get("sweep_topn") or []
        if sweep:
            for entry in sweep:
                v = entry.get("val_sharpe_gross")
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    out.append(float(v))
        else:
            # Fallback: legacy summaries that lack `sweep_topn`. ic_mean
            # and Sharpe are in different units, so this is a known
            # approximation. Print once so it's visible.
            ic = s.get("ic_mean")
            if ic is not None:
                out.append(float(ic))
                print(f"[aggregate_eval_v2] WARN: {os.path.basename(path)} has "
                      f"no sweep_topn; using ic_mean={ic:.4f} as a Sharpe proxy "
                      f"(legacy summary; rerun eval to populate sweep_topn).",
                      file=sys.stderr)
    return np.asarray(out, dtype=np.float64)


def headline_dsr(fold: str, cost_bps: int = 20) -> dict:
    """Compute DSR for the BEST (highest net Sharpe at cost_bps) cell.

    Jury 2 fix IMP3 (2026-05-08): the trial-Sharpe distribution used for
    `V[SR̂_n]` is now the ACTUAL swept Sharpes across (model, horizon,
    top_n) configs (read from each summary's `sweep_topn` field) —
    instead of the previous `np.tile` of cell argmax Sharpes which
    artificially inflated N without simulating the variance spread.

    Trials are also restricted to the SAME ARM as the headline winner
    (Jury 2 fix B7+N6) — pooling MSE and Track-B trials over-counts N
    because they share the same forward pass on the same panel.
    """
    df = load_all_summaries(fold)
    sharpe_col = f"c{cost_bps}_net_sharpe"
    if df.empty or sharpe_col not in df.columns:
        return {"error": "no summaries available"}

    best = df.loc[df[sharpe_col].idxmax()]
    arm_of_winner = best["arm"]
    base = f"{best['model']}_H{int(best['horizon'])}_{fold}_{arm_of_winner}"
    ts_path = os.path.join(OUT_DIR, f"timeseries_{base}.csv")
    if not os.path.exists(ts_path):
        return {"error": f"timeseries missing: {ts_path}"}

    winning = pd.read_csv(ts_path)["portfolio_return_net20_nonoverlap"].dropna().values

    # Pull the actual swept val-Sharpes for this arm (across model ×
    # horizon × top_n). This is the principled Bailey-LdP trial pool.
    trial_sharpes = _collect_swept_trial_sharpes(fold, arm_of_winner)
    if len(trial_sharpes) < 2:
        # Fall back to cell-level argmax sharpes if no sweep data.
        same_arm = df[df["arm"] == arm_of_winner]
        trial_sharpes = same_arm[sharpe_col].dropna().values

    res = compute_dsr_from_returns(winning, trial_sharpes, int(best["horizon"]))
    res["winning_cell"] = base
    res["winning_arm"] = arm_of_winner
    res["winning_sharpe"] = float(best[sharpe_col])
    res["n_trials_total"] = int(len(trial_sharpes))
    res["trial_pool_source"] = "sweep_topn (actual swept Sharpes)" \
        if len(trial_sharpes) >= 2 else "cell-argmax fallback"
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

    # 3. Paired bootstrap per cell + multiple-testing correction.
    # Jury 2 fix IMP4 (2026-05-08): add Holm-Bonferroni correction to the
    # raw p-values across the K (model, horizon) pairs reported. Also
    # add a Benjamini-Hochberg FDR column for exploratory tables.
    print(f"[aggregate_eval_v2] paired bootstrap (RH vs MSE)...")
    pb = paired_bootstrap_per_cell(args.fold, args.cost_bps, args.n_boot)
    if not pb.empty:
        p_raw = pb["p_one_sided_RH_le_MSE"].to_numpy()
        K = len(p_raw)
        # Holm-Bonferroni: sort ascending, multiply each by (K - rank), enforce
        # monotonicity, clip at 1.
        order = np.argsort(p_raw)
        p_sorted = p_raw[order]
        adj_sorted = np.minimum.accumulate(
            np.minimum(p_sorted * (K - np.arange(K)), 1.0)[::-1])[::-1]
        # Above is monotone-decreasing-from-the-right; for Holm we want
        # cumulative MAX as we move down the sorted list:
        running = 0.0
        adj = np.zeros(K)
        for k in range(K):
            v = min(p_sorted[k] * (K - k), 1.0)
            running = max(running, v)
            adj[k] = running
        p_holm = np.empty(K)
        p_holm[order] = adj
        # Benjamini-Hochberg FDR.
        adj_bh = np.empty(K)
        running_bh = 1.0
        for k in range(K - 1, -1, -1):
            v = min(p_sorted[k] * K / max(1, (k + 1)), 1.0)
            running_bh = min(running_bh, v)
            adj_bh[k] = running_bh
        p_bh = np.empty(K)
        p_bh[order] = adj_bh
        pb["p_holm"] = p_holm
        pb["p_bh_fdr"] = p_bh
        pb["significant_holm_5pct"] = pb["p_holm"] < 0.05
        pb["significant_bh_5pct"] = pb["p_bh_fdr"] < 0.05
    pb_path = os.path.join(OUT_DIR, f"paired_bootstrap_{args.fold}.csv")
    pb.to_csv(pb_path, index=False)
    print(f"[aggregate_eval_v2] {len(pb)} (model, horizon) pairs -> {pb_path}")
    if not pb.empty:
        sig_raw  = int(pb["significant_at_5pct"].sum())
        sig_holm = int(pb["significant_holm_5pct"].sum())
        sig_bh   = int(pb["significant_bh_5pct"].sum())
        print(f"  -> significant raw p<0.05: {sig_raw}/{len(pb)}")
        print(f"  -> significant Holm p<0.05: {sig_holm}/{len(pb)}")
        print(f"  -> significant BH-FDR <0.05: {sig_bh}/{len(pb)}")

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
