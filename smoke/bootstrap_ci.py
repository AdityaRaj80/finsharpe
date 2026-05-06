"""Single-arm Politis-Romano stationary bootstrap CI on Sharpe ratio.

Uses Politis-Romano 1994 stationary bootstrap to handle serial
correlation in the per-period return series. Block length defaults to
the Politis-White automatic rule (Patton-Politis-White 2009 correction)
when --auto_block is set, falling back to max(2, n^(1/3)) otherwise.

References
----------
- Politis, D. and Romano, J. (1994). The Stationary Bootstrap.
  J. American Statistical Association 89(428):1303-1313.
- Politis, D. and White, H. (2004) + Patton, Politis & White (2009).
  Automatic Block-Length Selection for the Dependent Bootstrap.
  Econometric Reviews 23(1):53-70 / 28(4):372-375.

Long-horizon validity guard
---------------------------
Aborts when n < 6 unless --force is passed. Warns in [6, 12].
See reports/methodology_audit_2026_05_07.md §D and §K.

Usage:
    python bootstrap_ci.py --csv timeseries_xs_PatchTST_global_H5_long_short.csv \
        --column portfolio_return_nonoverlap --horizon 5 --auto_block
"""
from __future__ import annotations

import argparse
import math
import os
import json
import sys
import warnings
import numpy as np
import pandas as pd

# Validity guard
N_MIN_HARD = 6
N_MIN_WARN = 12


def stationary_bootstrap_indices(n, expected_block, rng):
    """Generate indices via Politis-Romano stationary bootstrap.
    Each block has geometrically-distributed length with mean expected_block."""
    indices = np.zeros(n, dtype=int)
    indices[0] = rng.integers(0, n)
    for i in range(1, n):
        if rng.random() < 1.0 / expected_block:
            indices[i] = rng.integers(0, n)
        else:
            indices[i] = (indices[i - 1] + 1) % n
    return indices


def annualized_sharpe(returns, horizon):
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd < 1e-12:
        return float("nan")
    return mu / sd * np.sqrt(252.0 / horizon)


def politis_white_block_length(returns):
    """Politis-White / Patton-Politis-White automatic block length.

    Returns mean block length (float, ≥ 2). Falls back to n^(1/3) if
    the arch package is unavailable.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 4:
        return 2.0
    try:
        from arch.bootstrap import optimal_block_length
        opt = optimal_block_length(r)
        b = float(opt["stationary"].iloc[0])
        if not math.isfinite(b) or b < 2.0:
            b = max(2.0, n ** (1.0 / 3.0))
        return b
    except Exception as e:
        warnings.warn(f"arch.optimal_block_length failed ({e}); "
                      f"falling back to n^(1/3)={n**(1/3):.2f}.")
        return max(2.0, n ** (1.0 / 3.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--column", default="portfolio_return_nonoverlap")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--expected_block", type=float, default=None,
                   help="Mean block length. Omit to use --auto_block or default.")
    p.add_argument("--auto_block", action="store_true",
                   help="Use Politis-White automatic block-length selection.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help=f"Force run when n < {N_MIN_HARD} (statistically invalid).")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.csv)
    if args.column not in df.columns:
        raise SystemExit(f"Column {args.column!r} not in {args.csv}. "
                          f"Available: {list(df.columns)}")
    r = df[args.column].dropna().to_numpy()
    n = len(r)
    if n < N_MIN_HARD:
        msg = (f"Only n={n} non-overlapping rebalances; bootstrap CI "
               f"is not statistically valid at this sample size. At "
               f"horizon H={args.horizon}, the 1-year test window "
               f"provides ~{int(252/args.horizon)} rebalances; consider "
               f"H=5 or H=20 for inference. See methodology audit §D, §K.")
        if args.force:
            warnings.warn("[--force overrides validity guard] " + msg)
        else:
            print(f"ERROR: {msg}\nPass --force to override (not recommended).")
            sys.exit(2)
    elif n < N_MIN_WARN:
        warnings.warn(f"n={n} is small for stationary bootstrap; CI will be wide.")

    print(f"Loaded {n} returns from {args.csv} / column {args.column}", flush=True)

    if args.auto_block or args.expected_block is None:
        expected_block = politis_white_block_length(r)
        block_source = "Politis-White auto"
    else:
        expected_block = float(args.expected_block)
        block_source = "user-specified"
    expected_block = max(2.0, expected_block)
    print(f"Block length: {expected_block:.2f} ({block_source})", flush=True)

    point = annualized_sharpe(r, args.horizon)
    print(f"Point Sharpe: {point:.4f}", flush=True)

    boot_sharpes = np.zeros(args.n_boot)
    for b in range(args.n_boot):
        idx = stationary_bootstrap_indices(len(r), expected_block, rng)
        boot_sharpes[b] = annualized_sharpe(r[idx], args.horizon)
    boot_sharpes = boot_sharpes[~np.isnan(boot_sharpes)]

    lo, mid, hi = np.percentile(boot_sharpes, [2.5, 50, 97.5])
    se = boot_sharpes.std(ddof=1)

    out = {
        "csv": args.csv,
        "column": args.column,
        "n_returns": int(n),
        "horizon": args.horizon,
        "n_boot": int(args.n_boot),
        "expected_block": float(expected_block),
        "block_source": block_source,
        "point_estimate": float(point),
        "boot_median": float(mid),
        "boot_se": float(se),
        "ci95_lower": float(lo),
        "ci95_upper": float(hi),
    }
    print(f"\n95% bootstrap CI on Sharpe: [{lo:.3f}, {hi:.3f}]  "
          f"(median {mid:.3f}, SE {se:.3f})", flush=True)
    print(f"Point estimate {point:.3f} sits at percentile "
          f"{(boot_sharpes < point).mean()*100:.1f}", flush=True)

    base = os.path.basename(args.csv).replace(".csv", "")
    out_path = os.path.join(os.path.dirname(args.csv) or ".", f"bootstrap_{base}.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote: {out_path}", flush=True)


if __name__ == "__main__":
    main()
