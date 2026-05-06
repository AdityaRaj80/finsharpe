"""Deflated Sharpe Ratio (DSR) — Bailey & Lopez de Prado 2014.

Reference
---------
Bailey, D. H. & Lopez de Prado, M. (2014). The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality.
Journal of Portfolio Management 40(5):94-107. SSRN:2460551.

Why this script exists
----------------------
Our backtest pipeline sweeps top_n on val ∈ {3,5,7,10,15} for every
(model × horizon × strategy) cell, which inflates the *trial count* well
beyond the headline number we report. With 7 models × 5 horizons × 2
strategies × 5 top_n = 350 backtests on a 1-year (T ≈ 50 non-overlapping
H=5 rebalances) test window, a point Sharpe of 1.5 can be statistically
indistinguishable from "lucky pick from random alphas" once selection
bias is accounted for.

The deflated Sharpe ratio reports the *probability* that the observed
SR̂ exceeds an SR₀ threshold determined by N (number of trials), T
(sample length in periods), and the higher moments γ̂₃ (skew) and γ̂₄
(kurtosis) of the per-period returns. A DSR ≥ 0.95 (5% selection-aware
type-I error) is the minimum bar for "this strategy is more than chance"
under Bailey-Lopez de Prado.

Key formulas (Bailey & Lopez de Prado 2014, eqs. 7-9):

    SR₀ = sqrt(V[SR̂_n]) · ((1 - γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)))

where γ ≈ 0.5772 (Euler-Mascheroni) and V[SR̂_n] is the variance of the
SR estimator across the N trials. We approximate V[SR̂_n] from the
empirical std across the available trial Sharpes (across the swept
configs the user supplies), which is the per-paper convention.

    DSR = Φ((SR̂* − SR₀) · sqrt(T - 1) /
              sqrt(1 − γ̂₃·SR̂* + ((γ̂₄ − 1)/4)·SR̂*²))

where SR̂* is the *non-annualized* per-period Sharpe (mean / std of
per-period returns, no √(252/H) factor — this matters: the deflation
formula's units are per-period). γ̂₃, γ̂₄ are skew and kurtosis (NOT
excess kurtosis — γ̂₄ for a normal is 3) of the per-period returns of
the WINNING strategy.

Returns
-------
A dict with the deflated Sharpe ratio (probability ∈ [0, 1]), the SR₀
threshold, and the inputs used. The reported DSR should be ≥ 0.95 for
the headline result to be statistically defensible after selection.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def per_period_sharpe(returns: np.ndarray) -> float:
    """Sharpe in per-period units (no annualization). Bailey-LdP eqs.

    Returns nan when std is zero or n<2 (degenerate).
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd < 1e-12:
        return float("nan")
    return float(mu / sd)


def trial_sr_variance(trial_sharpes: Sequence[float]) -> float:
    """V[SR̂_n] in eq. 7. Empirical variance across the N trial Sharpes
    (NOT across bootstrap reps of one trial — across the *configs* the
    user actually swept). ddof=1 because the trials are a sample.
    """
    v = np.asarray(list(trial_sharpes), dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return float("nan")
    return float(np.var(v, ddof=1))


def expected_max_sr(N: int, var_sr: float) -> float:
    """SR₀ = E[max of N i.i.d. SR̂'s under the null mean=0] (eq. 7).

    First-order extreme-value approximation in Bailey-LdP:
        SR₀ ≈ sqrt(V[SR̂_n]) · ((1 - γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)))

    where γ is the Euler-Mascheroni constant. This is the threshold the
    *winning* (max) Sharpe must exceed before deflation can be > 0.5.
    """
    if N < 2 or not math.isfinite(var_sr) or var_sr <= 0:
        return float("nan")
    a = (1.0 - EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / N)
    b = EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (N * math.e))
    return float(math.sqrt(var_sr) * (a + b))


def deflated_sharpe_ratio(
    sr_star: float,
    T: int,
    skew: float,
    kurt: float,
    sr_zero: float,
) -> float:
    """DSR — eq. 9 of Bailey & Lopez de Prado 2014.

    Args
    ----
        sr_star : per-period Sharpe of the winning strategy (NOT annualised).
        T       : number of per-period observations underlying sr_star.
        skew    : sample skew γ̂₃ of those T returns.
        kurt    : sample kurtosis γ̂₄ (4th standardised moment, normal = 3).
        sr_zero : threshold from expected_max_sr.

    Returns
    -------
        DSR ∈ [0, 1] : probability that the true Sharpe exceeds sr_zero
                      after selection bias and non-normality correction.
                      ≥ 0.95 → significant.
    """
    if T < 2 or not all(math.isfinite(x) for x in (sr_star, skew, kurt, sr_zero)):
        return float("nan")
    denom = 1.0 - skew * sr_star + ((kurt - 1.0) / 4.0) * sr_star ** 2
    if denom <= 0:
        # Heavy-tail / extreme skew renders the variance estimate
        # unidentified — DSR is undefined. Return nan rather than fudge.
        return float("nan")
    z = (sr_star - sr_zero) * math.sqrt(T - 1) / math.sqrt(denom)
    return float(norm.cdf(z))


def annualization_factor(horizon: int) -> float:
    """sqrt(252 / horizon) — same convention as smoke/cross_sectional_smoke.py."""
    return math.sqrt(252.0 / float(horizon))


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────
def compute_dsr_from_returns(
    winning_returns: np.ndarray,
    trial_sharpes_annualised: Sequence[float],
    horizon: int,
) -> dict:
    """Full DSR computation from raw artefacts.

    Args
    ----
        winning_returns          : per-period returns of the WINNING
                                    backtest (subsampled [::H], i.e.
                                    non-overlapping H-day rebalances).
        trial_sharpes_annualised : list of *annualised* Sharpes from
                                    every trial in the search (the more
                                    inclusive the better; under-counting
                                    trials inflates DSR).
        horizon                  : H, used to convert annualised → per-period.
    """
    r = np.asarray(winning_returns, dtype=np.float64)
    r = r[~np.isnan(r)]
    T = int(len(r))
    sr_star_per_period = per_period_sharpe(r)
    sr_star_annualised = sr_star_per_period * annualization_factor(horizon) \
        if math.isfinite(sr_star_per_period) else float("nan")

    # Convert annualised trial Sharpes → per-period for variance estimate.
    factor = annualization_factor(horizon)
    trial_pp = np.asarray(list(trial_sharpes_annualised), dtype=np.float64) / factor
    trial_pp = trial_pp[~np.isnan(trial_pp)]
    N = int(len(trial_pp))
    var_sr = trial_sr_variance(trial_pp)
    sr_zero_per_period = expected_max_sr(N, var_sr)
    sr_zero_annualised = sr_zero_per_period * factor if math.isfinite(sr_zero_per_period) else float("nan")

    # Sample skew + kurtosis (NOT excess kurtosis — γ̂₄ for normal = 3).
    if T >= 4 and r.std(ddof=1) > 1e-12:
        skew = float(((r - r.mean()) ** 3).mean() / r.std(ddof=1) ** 3)
        kurt = float(((r - r.mean()) ** 4).mean() / r.std(ddof=1) ** 4)
    else:
        skew = float("nan")
        kurt = float("nan")

    dsr = deflated_sharpe_ratio(sr_star_per_period, T, skew, kurt, sr_zero_per_period)

    return {
        "n_trials": N,
        "T_periods": T,
        "horizon": horizon,
        "sr_star_per_period": sr_star_per_period,
        "sr_star_annualised": sr_star_annualised,
        "trial_sr_variance_per_period": var_sr,
        "sr_zero_per_period": sr_zero_per_period,
        "sr_zero_annualised": sr_zero_annualised,
        "skew": skew,
        "kurt": kurt,
        "dsr": dsr,
        "is_significant_at_95": bool(dsr >= 0.95) if math.isfinite(dsr) else None,
    }


def main():
    p = argparse.ArgumentParser(
        description="Compute Deflated Sharpe Ratio (Bailey-Lopez de Prado 2014) "
                    "for the headline backtest given a winning return series and "
                    "a CSV listing every trial Sharpe in the sweep.")
    p.add_argument("--winning_csv", required=True,
                   help="CSV with winning strategy's non-overlap returns "
                        "(typically the timeseries_*.csv file from "
                        "cross_sectional_smoke.py for the headline cell).")
    p.add_argument("--column", default="portfolio_return_nonoverlap",
                   help="Column in winning_csv containing per-period returns.")
    p.add_argument("--trials_csv", required=True,
                   help="CSV with one row per trial, must contain a "
                        "'sharpe_annualised' column. Pass every config you "
                        "swept — under-counting inflates DSR.")
    p.add_argument("--trials_column", default="sharpe_annualised",
                   help="Column in trials_csv with annualised Sharpes.")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--out", default=None,
                   help="Optional output JSON path. Defaults to "
                        "dsr_<winning_csv_basename>.json next to winning_csv.")
    args = p.parse_args()

    win = pd.read_csv(args.winning_csv)
    if args.column not in win.columns:
        raise SystemExit(f"Column {args.column!r} not in {args.winning_csv}. "
                         f"Available: {list(win.columns)}")
    r = win[args.column].dropna().to_numpy()

    trials = pd.read_csv(args.trials_csv)
    if args.trials_column not in trials.columns:
        raise SystemExit(f"Column {args.trials_column!r} not in {args.trials_csv}. "
                         f"Available: {list(trials.columns)}")
    trial_srs = trials[args.trials_column].dropna().to_numpy()

    result = compute_dsr_from_returns(r, trial_srs, args.horizon)

    print(json.dumps(result, indent=2))
    print()
    n = result["n_trials"]; T = result["T_periods"]
    print(f"Trials swept (N) : {n}")
    print(f"Test periods (T) : {T}")
    print(f"Winning Sharpe   : {result['sr_star_annualised']:6.3f}  (annualised) "
          f"= {result['sr_star_per_period']:.4f}  (per-period)")
    print(f"SR_0 threshold   : {result['sr_zero_annualised']:6.3f}  (annualised) "
          f"= {result['sr_zero_per_period']:.4f}  (per-period)")
    print(f"Skew / Kurt      : γ₃={result['skew']:+.3f}  γ₄={result['kurt']:.3f}  (normal γ₄=3)")
    if math.isfinite(result["dsr"]):
        print(f"DSR              : {result['dsr']:.4f}    "
              f"({'SIGNIFICANT' if result['dsr'] >= 0.95 else 'NOT significant'} at 95%)")
    else:
        print("DSR              : undefined (denominator non-positive — heavy tails / extreme skew)")

    out_path = args.out
    if out_path is None:
        base = os.path.basename(args.winning_csv).replace(".csv", "")
        out_path = os.path.join(os.path.dirname(args.winning_csv) or ".",
                                 f"dsr_{base}.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
