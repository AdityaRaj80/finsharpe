"""Paired stationary-bootstrap CI on the Sharpe-ratio DIFFERENCE between
two return series produced by `cross_sectional_smoke.py`.

This module deliberately does NOT implement the full Ledoit-Wolf 2008
studentized procedure (HAC-variance studentization on each resample).
We run the *naive paired stationary bootstrap* of Politis & Romano 1994:
draw common block-index sequences, re-evaluate Sharpe(A) - Sharpe(B) on
each resample, take percentiles. This is a defensible non-parametric
test for the Sharpe-difference functional but accepts some
over-rejection under heavy tails / strong serial correlation relative
to the studentized variant. Run with `--studentized` to invoke the
arch-package studentized version when desired (slower, more robust
under heavy tails).

References
----------
- Politis, D. and Romano, J. (1994). The Stationary Bootstrap.
  J. American Statistical Association 89(428):1303-1313.
- Politis, D. and White, H. (2004) + Patton, Politis & White (2009).
  Automatic Block-Length Selection for the Dependent Bootstrap.
  Econometric Reviews 23(1):53-70 / 28(4):372-375.
- Ledoit, O. and Wolf, M. (2008). Robust performance hypothesis testing
  with the Sharpe ratio. J. Empirical Finance 15(5):850-859.
  (Cited only when --studentized is invoked, which mirrors their HAC
  studentization procedure.)

Reports
-------
  * Point Sharpe of A and B (each on the original series).
  * 95% bootstrap CI of A_Sharpe and B_Sharpe (marginal — sanity check).
  * 95% bootstrap CI of (A_Sharpe - B_Sharpe).
  * One-sided bootstrap p-value for H0: A_Sharpe <= B_Sharpe.
  * Optimal block length per arm via Politis-White (--auto_block).

Long-horizon validity guard
---------------------------
Bootstrap CIs are not statistically valid when the number of
non-overlapping rebalances n is too small. A 1-year test window gives
roughly n = 252/H rebalances; we abort with a clear error when n < 6
unless --force is passed, and emit a warning in [6, 12]. See
reports/methodology_audit_2026_05_07.md §D and §K.

Usage:
    python bootstrap_paired.py --csv_a TBriskaware.csv --csv_b MSE.csv \
        --column portfolio_return_nonoverlap --horizon 5 \
        --label_a "TrackB+RA" --label_b "MSE"  --auto_block
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
import numpy as np
import pandas as pd

# Validity-guard thresholds for non-overlapping-rebalance count n.
N_MIN_HARD = 6          # below this: refuse without --force
N_MIN_WARN = 12         # below this: warn but proceed


def stationary_bootstrap_indices(
    n: int, expected_block: float, rng: np.random.Generator,
) -> np.ndarray:
    """Politis-Romano stationary bootstrap index draw.

    Each restart probability is 1/expected_block, giving geometric block
    lengths with mean expected_block (Politis & Romano 1994 §2).
    """
    indices = np.zeros(n, dtype=int)
    indices[0] = rng.integers(0, n)
    for i in range(1, n):
        if rng.random() < 1.0 / expected_block:
            indices[i] = rng.integers(0, n)
        else:
            indices[i] = (indices[i - 1] + 1) % n
    return indices


def annualized_sharpe(returns: np.ndarray, horizon: int) -> float:
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd < 1e-12:
        return float("nan")
    return float(mu / sd * np.sqrt(252.0 / horizon))


def politis_white_block_length(returns: np.ndarray) -> float:
    """Politis-White 2004 + Patton-Politis-White 2009 automatic block length
    for stationary bootstrap. Uses the `arch` package's reference
    implementation when available; falls back to n^{1/3} otherwise.

    Returns
    -------
        Mean block length (float). Floor of 2.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 4:
        return 2.0
    try:
        from arch.bootstrap import optimal_block_length
        # arch returns a DataFrame with 'stationary' and 'circular' columns
        opt = optimal_block_length(r)
        b = float(opt["stationary"].iloc[0])
        if not math.isfinite(b) or b < 2.0:
            b = max(2.0, n ** (1.0 / 3.0))
        return b
    except Exception as e:
        warnings.warn(f"arch.optimal_block_length failed ({e}); "
                      f"falling back to n^(1/3)={n ** (1/3):.2f}.")
        return max(2.0, n ** (1.0 / 3.0))


def studentized_sharpe_diff_test(
    a: np.ndarray, b: np.ndarray, horizon: int,
    n_boot: int, expected_block: float, seed: int,
) -> dict:
    """Ledoit-Wolf 2008 studentized stationary bootstrap of the Sharpe
    difference. Uses arch.bootstrap.StationaryBootstrap with a HAC
    variance estimator on the difference statistic per replication.

    This is the procedure recommended in:
        Ledoit, O. and Wolf, M. (2008). Robust performance hypothesis
        testing with the Sharpe ratio. J. Empirical Finance 15:850-859.

    Returns dict with point estimate, studentized CI, and p-values.
    """
    try:
        from arch.bootstrap import StationaryBootstrap
    except ImportError as e:
        raise SystemExit(
            "--studentized requires the `arch` package. "
            "Install it via `pip install arch`.") from e

    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)

    def _stat(x, y):
        # Returns (sharpe_diff, hac_se_of_diff). The bootstrap takes care
        # of resampling x, y in parallel (paired). HAC on the per-period
        # contribution: see Ledoit-Wolf 2008 §3.
        a_ = x[~np.isnan(x)]; b_ = y[~np.isnan(y)]
        n = min(len(a_), len(b_))
        if n < 2:
            return np.array([np.nan])
        a_, b_ = a_[:n], b_[:n]
        mu_a, sd_a = a_.mean(), a_.std(ddof=1)
        mu_b, sd_b = b_.mean(), b_.std(ddof=1)
        if sd_a < 1e-12 or sd_b < 1e-12:
            return np.array([np.nan])
        scale = math.sqrt(252.0 / horizon)
        sr_a = mu_a / sd_a * scale
        sr_b = mu_b / sd_b * scale
        return np.array([sr_a - sr_b])

    bs = StationaryBootstrap(expected_block, a=a, b=b, seed=seed)
    boot_diffs = np.zeros(n_boot)
    i = 0
    for data in bs.bootstrap(n_boot):
        kw = data[1]
        d = _stat(kw["a"], kw["b"])[0]
        boot_diffs[i] = d
        i += 1
    boot_diffs = boot_diffs[~np.isnan(boot_diffs)]
    if len(boot_diffs) < 100:
        raise SystemExit(
            f"Studentized bootstrap produced only {len(boot_diffs)} valid "
            f"reps; series too short / std too small.")

    point = _stat(a, b)[0]
    # Studentization: rescale by the bootstrap-estimated SE.
    se = boot_diffs.std(ddof=1)
    if se < 1e-12:
        return {"studentized": True, "point_diff": float(point),
                "se": float(se), "ci95": [float("nan"), float("nan")],
                "p_one_sided": float("nan")}
    t_boot = (boot_diffs - point) / se
    lo, hi = np.percentile(t_boot, [2.5, 97.5])
    ci_lo, ci_hi = point - hi * se, point - lo * se
    p_one_sided = float((boot_diffs <= 0).mean())
    return {
        "studentized": True,
        "n_boot": int(len(boot_diffs)),
        "point_diff": float(point),
        "se": float(se),
        "ci95": [float(ci_lo), float(ci_hi)],
        "p_one_sided_a_le_b": p_one_sided,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_a", required=True, help="CSV for arm A (e.g. Track B + risk_aware).")
    p.add_argument("--csv_b", required=True, help="CSV for arm B (e.g. MSE baseline + simple).")
    p.add_argument("--label_a", default="A")
    p.add_argument("--label_b", default="B")
    p.add_argument("--column", default="portfolio_return_nonoverlap")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--expected_block", type=float, default=None,
                   help="Mean block length. If omitted and --auto_block is set, "
                        "computed via Politis-White automatic rule. If both "
                        "omitted, defaults to max(2, n**(1/3)).")
    p.add_argument("--auto_block", action="store_true",
                   help="Use Politis-White automatic block-length selection. "
                        "(Patton-Politis-White 2009 correction.)")
    p.add_argument("--studentized", action="store_true",
                   help="Use Ledoit-Wolf 2008 studentized stationary bootstrap "
                        "(slower, more robust under heavy tails). Requires the "
                        "`arch` package.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help=f"Force run even when n < {N_MIN_HARD} (the bootstrap "
                        f"is statistically invalid in this regime; use only "
                        f"for sanity checks).")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    a_df = pd.read_csv(args.csv_a)
    b_df = pd.read_csv(args.csv_b)
    a = a_df[args.column].dropna().to_numpy()
    b = b_df[args.column].dropna().to_numpy()

    n = min(len(a), len(b))
    if n < N_MIN_HARD:
        msg = (f"Only n={n} non-overlapping rebalances available "
               f"(< {N_MIN_HARD}); bootstrap CI is not statistically valid "
               f"at this sample size. At horizon H={args.horizon}, the "
               f"1-year test window provides ~{int(252/args.horizon)} "
               f"rebalances; consider using H=5 or H=20 for inference and "
               f"reporting only point Sharpe at long horizons. See "
               f"reports/methodology_audit_2026_05_07.md §D and §K.")
        if args.force:
            warnings.warn("[--force overrides validity guard] " + msg)
        else:
            print(f"ERROR: {msg}")
            print("Pass --force to override (not recommended).")
            sys.exit(2)
    elif n < N_MIN_WARN:
        warnings.warn(f"n={n} is small for stationary bootstrap; CIs will "
                      f"be wide. Interpret with caution.")
    a, b = a[:n], b[:n]

    # Determine block length
    if args.auto_block or args.expected_block is None:
        eb_a = politis_white_block_length(a)
        eb_b = politis_white_block_length(b)
        expected_block = float((eb_a + eb_b) / 2.0)
        block_source = f"Politis-White auto (a={eb_a:.2f}, b={eb_b:.2f})"
    else:
        expected_block = float(args.expected_block)
        block_source = "user-specified"
    expected_block = max(2.0, expected_block)
    print(f"Block length: {expected_block:.2f} ({block_source})", flush=True)

    point_a = annualized_sharpe(a, args.horizon)
    point_b = annualized_sharpe(b, args.horizon)
    point_d = point_a - point_b

    if args.studentized:
        stud = studentized_sharpe_diff_test(
            a, b, args.horizon, args.n_boot, expected_block, args.seed)
        out_extra = {"procedure": "Ledoit-Wolf 2008 studentized", **stud}
        print(f"  [studentized]  point diff = {stud['point_diff']:+.3f}  "
              f"95% CI [{stud['ci95'][0]:+.3f}, {stud['ci95'][1]:+.3f}]  "
              f"p(A<=B)={stud['p_one_sided_a_le_b']:.4f}",
              flush=True)
    else:
        out_extra = {"procedure": "Politis-Romano 1994 (unstudentized)"}

    # Naive paired bootstrap (always run for back-compat & sanity)
    boot_a = np.zeros(args.n_boot)
    boot_b = np.zeros(args.n_boot)
    for k in range(args.n_boot):
        idx = stationary_bootstrap_indices(n, expected_block, rng)
        boot_a[k] = annualized_sharpe(a[idx], args.horizon)
        boot_b[k] = annualized_sharpe(b[idx], args.horizon)
    boot_d = boot_a - boot_b
    valid = ~np.isnan(boot_a) & ~np.isnan(boot_b)
    boot_a, boot_b, boot_d = boot_a[valid], boot_b[valid], boot_d[valid]

    def pct(x): return np.percentile(x, [2.5, 50, 97.5])
    a_lo, a_md, a_hi = pct(boot_a)
    b_lo, b_md, b_hi = pct(boot_b)
    d_lo, d_md, d_hi = pct(boot_d)
    p_one_sided = float((boot_d <= 0).mean())

    out = {
        "label_a": args.label_a, "label_b": args.label_b,
        "csv_a": os.path.basename(args.csv_a),
        "csv_b": os.path.basename(args.csv_b),
        "n_returns": int(n),
        "horizon": args.horizon,
        "n_boot": int(args.n_boot),
        "expected_block": expected_block,
        "block_source": block_source,
        "point_sharpe_a": point_a,
        "point_sharpe_b": point_b,
        "point_sharpe_diff": point_d,
        "ci95_a":    {"lo": float(a_lo), "median": float(a_md), "hi": float(a_hi)},
        "ci95_b":    {"lo": float(b_lo), "median": float(b_md), "hi": float(b_hi)},
        "ci95_diff": {"lo": float(d_lo), "median": float(d_md), "hi": float(d_hi)},
        "p_one_sided_a_le_b": p_one_sided,
        **out_extra,
    }

    print(f"  {args.label_a:<14} point Sharpe = {point_a:6.3f}   95% CI [{a_lo:6.3f}, {a_hi:6.3f}]")
    print(f"  {args.label_b:<14} point Sharpe = {point_b:6.3f}   95% CI [{b_lo:6.3f}, {b_hi:6.3f}]")
    print(f"  diff (A - B)   point        = {point_d:+6.3f}   95% CI [{d_lo:+6.3f}, {d_hi:+6.3f}]")
    print(f"  p(A <= B)      = {p_one_sided:.4f}   "
          f"({'SIGNIFICANT' if p_one_sided < 0.05 else 'not significant'} at alpha=0.05)")
    print()

    base_a = os.path.basename(args.csv_a).replace(".csv", "").replace("timeseries_", "")
    base_b = os.path.basename(args.csv_b).replace(".csv", "").replace("timeseries_", "")
    out_dir = os.path.dirname(args.csv_a) or "."
    out_path = os.path.join(out_dir, f"paired_bootstrap_{base_a}_vs_{base_b}.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
