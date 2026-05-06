"""Tests for smoke/rank_ic.py — Newey-West HAC SE + cross-sectional IC."""
from __future__ import annotations

import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from smoke.rank_ic import (  # noqa: E402
    cross_sectional_kendall_series,
    cross_sectional_spearman_series,
    newey_west_se,
    rank_ic_summary,
)


def test_newey_west_iid():
    """For i.i.d. series, NW SE at lag=0 should match standard SE = sd/sqrt(n)."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1000)
    nw = newey_west_se(x, lag=0)
    se_std = x.std(ddof=0) / math.sqrt(len(x))
    # NW lag-0 = sqrt(gamma_0 / n) = sd_pop / sqrt(n)
    assert abs(nw - se_std) < 1e-6


def test_newey_west_inflates_with_autocorr():
    """For positively autocorrelated series, NW SE should EXCEED iid SE."""
    rng = np.random.default_rng(1)
    n = 500
    eps = rng.normal(0, 1, n)
    x = np.zeros(n)
    x[0] = eps[0]
    for i in range(1, n):
        x[i] = 0.7 * x[i-1] + eps[i]   # AR(1), rho=0.7
    nw_lag5 = newey_west_se(x, lag=5)
    se_iid = x.std(ddof=1) / math.sqrt(n)
    assert nw_lag5 > 1.5 * se_iid   # at least 50% inflation


def test_spearman_series_perfect_signal():
    """When pred and actual are monotonically related, IC -> +1."""
    rng = np.random.default_rng(2)
    T, N = 30, 20
    actual = rng.normal(0, 1, (T, N))
    pred = actual + 0.01 * rng.normal(0, 1, (T, N))   # near-noise
    rho = cross_sectional_spearman_series(pred, actual)
    assert np.nanmean(rho) > 0.95


def test_spearman_series_zero_signal():
    """When pred and actual are independent, IC -> 0 in expectation."""
    rng = np.random.default_rng(3)
    T, N = 100, 30
    pred = rng.normal(0, 1, (T, N))
    actual = rng.normal(0, 1, (T, N))
    rho = cross_sectional_spearman_series(pred, actual)
    assert abs(np.nanmean(rho)) < 0.1


def test_kendall_runs():
    """Kendall variant returns a series of same length."""
    rng = np.random.default_rng(4)
    T, N = 20, 15
    pred = rng.normal(0, 1, (T, N))
    actual = rng.normal(0, 1, (T, N))
    tau = cross_sectional_kendall_series(pred, actual)
    assert tau.shape == (T,)


def test_summary_normal():
    """ICIR is finite for non-degenerate series and matches mean/std*sqrt(252/H)."""
    rng = np.random.default_rng(5)
    rho = rng.normal(0.05, 0.1, 50)   # weak positive IC
    s = rank_ic_summary(rho, horizon=5, label="spearman")
    expected_icir = s["spearman_ic_mean"] / s["spearman_ic_std"] * math.sqrt(252.0 / 5)
    assert abs(s["spearman_icir"] - expected_icir) < 1e-9
    assert math.isfinite(s["spearman_nw_se"])
    assert math.isfinite(s["spearman_nw_tstat"])


def test_summary_handles_few_obs():
    """n<2 should return nan stats but n_obs reported."""
    rho = np.array([0.1])
    s = rank_ic_summary(rho, horizon=5)
    assert math.isnan(s["spearman_ic_mean"])
    assert s["spearman_n_obs"] == 1


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
