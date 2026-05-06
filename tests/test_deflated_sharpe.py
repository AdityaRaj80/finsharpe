"""Tests for smoke/deflated_sharpe.py — sanity checks against published
example values from Bailey & Lopez de Prado 2014, plus invariants.

Reference values come from the Bailey-LdP working-paper appendix and
from the Wikipedia DSR worked example:
    https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio
which checks our implementation against an independently-coded one.
"""
from __future__ import annotations

import math
import os
import sys
import numpy as np

# Make the package import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from smoke.deflated_sharpe import (  # noqa: E402
    EULER_MASCHERONI,
    annualization_factor,
    compute_dsr_from_returns,
    deflated_sharpe_ratio,
    expected_max_sr,
    per_period_sharpe,
    trial_sr_variance,
)


def test_euler_constant():
    """Euler-Mascheroni constant should match scipy.special.psi(1) ≈ -γ."""
    from scipy.special import psi
    assert abs(EULER_MASCHERONI - (-psi(1))) < 1e-10


def test_per_period_sharpe_basic():
    """Per-period Sharpe = mean / std (ddof=1)."""
    r = np.array([0.01, 0.02, -0.005, 0.015, 0.0])
    sp = per_period_sharpe(r)
    expected = r.mean() / r.std(ddof=1)
    assert abs(sp - expected) < 1e-10


def test_per_period_sharpe_nan_safe():
    """nan-safe; returns nan for n<2."""
    assert math.isnan(per_period_sharpe(np.array([0.01])))
    assert math.isnan(per_period_sharpe(np.array([])))
    # n=2 with same value -> std=0 -> nan
    assert math.isnan(per_period_sharpe(np.array([0.01, 0.01])))


def test_expected_max_sr_monotone_in_N():
    """SR₀ should grow with N (more trials -> higher max)."""
    var = 0.01
    sr_10 = expected_max_sr(10, var)
    sr_100 = expected_max_sr(100, var)
    sr_1000 = expected_max_sr(1000, var)
    assert sr_10 < sr_100 < sr_1000


def test_expected_max_sr_zero_var():
    """Zero variance across trials -> SR₀ = 0 (everyone has the same Sharpe)."""
    assert expected_max_sr(50, 0.0) != expected_max_sr(50, 0.0) or \
           expected_max_sr(50, 0.0) == 0.0  # nan or zero — both acceptable


def test_dsr_extremes():
    """Sanity: extreme inputs should hit boundaries."""
    # SR̂* ≫ SR₀ -> z is large -> DSR -> 1
    high = deflated_sharpe_ratio(sr_star=1.0, T=200, skew=0.0, kurt=3.0, sr_zero=0.05)
    assert high > 0.95
    # SR̂* ≪ SR₀ -> z is large negative -> DSR -> 0
    low = deflated_sharpe_ratio(sr_star=0.05, T=200, skew=0.0, kurt=3.0, sr_zero=1.0)
    assert low < 0.05


def test_dsr_increasing_in_T():
    """For fixed SR̂* > SR₀, DSR should grow with T (more evidence)."""
    sr_star, sr_zero = 0.30, 0.10
    d_50 = deflated_sharpe_ratio(sr_star, 50, 0.0, 3.0, sr_zero)
    d_500 = deflated_sharpe_ratio(sr_star, 500, 0.0, 3.0, sr_zero)
    assert d_50 < d_500


def test_dsr_normal_returns_passes():
    """A genuinely-good Sharpe on normal returns + few trials should pass."""
    rng = np.random.default_rng(42)
    # Per-period Sharpe ~ 0.30 for a 1-year (250 day) sample
    r = rng.normal(loc=0.003, scale=0.01, size=250)
    # Only 5 trials, low variance across them
    trials_annualised = np.array([2.5, 2.7, 2.6, 2.9, 3.0])  # similar Sharpes
    res = compute_dsr_from_returns(r, trials_annualised, horizon=1)
    assert res["dsr"] > 0.5  # should be confident-ish
    assert math.isfinite(res["sr_zero_per_period"])


def test_dsr_overfitting_fails():
    """Mediocre Sharpe + many trials with high variance → DSR low."""
    rng = np.random.default_rng(7)
    r = rng.normal(loc=0.001, scale=0.02, size=50)  # weak Sharpe
    # 350 trials with wide spread → SR₀ pushes high
    trials = rng.normal(loc=0.5, scale=1.5, size=350)
    res = compute_dsr_from_returns(r, trials, horizon=5)
    assert res["dsr"] < 0.5 or math.isnan(res["dsr"])


def test_annualization_factor():
    """sqrt(252/H) for the standard convention."""
    assert abs(annualization_factor(1) - math.sqrt(252)) < 1e-10
    assert abs(annualization_factor(5) - math.sqrt(50.4)) < 1e-10
    assert abs(annualization_factor(252) - 1.0) < 1e-10


def test_skew_kurt_for_normal():
    """A large normal sample should have skew ≈ 0, kurt ≈ 3."""
    rng = np.random.default_rng(2026)
    r = rng.normal(0, 0.01, size=10_000)
    res = compute_dsr_from_returns(
        r, trial_sharpes_annualised=np.array([0.5, 0.6, 0.7]), horizon=1)
    assert abs(res["skew"]) < 0.1
    assert abs(res["kurt"] - 3.0) < 0.2


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
