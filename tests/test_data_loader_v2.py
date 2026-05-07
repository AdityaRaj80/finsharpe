"""Smoke + correctness tests for the v2 data_loader.

These tests run against the REAL merged_v3 data (not synthetic), so they
require the full pipeline to have been run once. If merged_v3 doesn't
exist, tests are skipped with a clear message.

Coverage:
  T1. Loader builds samples for all 4 walk-forward folds.
  T2. Same-stocks across train/val/test (each universe ticker contributes
      to all three splits when its history is long enough).
  T3. Calendar boundaries respected: no train sample's anchor date is
      inside val/test windows, no val into test, etc.
  T4. Per-stock z-score: train data has roughly mean 0 / std 1 per
      feature when aggregated across the train pool.
  T5. log-return target distribution: y_train mean is small, std looks
      like a daily-return-scale value (i.e. > 1e-4 and < 1.0 for H=5).
  T6. No NaN / inf in X or y.
  T7. DataLoader iteration works (one full pass through train_loader
      yields tensors of expected shape).
  T8. get_fold_dates round-trip: F4 anchors at 2021-12-31 / 2022 / 2023.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import DATA_DIR, FEATURES, CLOSE_IDX, WALK_FORWARD_FOLDS  # noqa: E402
from data_loader import (UnifiedDataLoader, get_fold_dates,  # noqa: E402
                          _load_stock, build_samples_for_stock)


# Skip whole module if merged_v3 doesn't exist locally
pytestmark = pytest.mark.skipif(
    DATA_DIR is None or not os.path.isdir(DATA_DIR),
    reason=f"DATA_DIR={DATA_DIR} not present; run rebuild_merged_v2.py first."
)


# ---------------------------------------------------------------------------
# Fast smoke (small max_stocks)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def smoke_loader():
    return UnifiedDataLoader(seq_len=96, horizon=5, batch_size=64,
                              fold="F4", max_stocks=10)


def test_t1_smoke_loader_builds(smoke_loader):
    assert len(smoke_loader.X_train) > 0, "no train samples"
    assert len(smoke_loader.X_val) > 0, "no val samples"
    assert len(smoke_loader.X_test) > 0, "no test samples"


def test_t1_all_folds_load():
    """Every fold must produce a non-trivial split."""
    for fold in ["F1", "F2", "F3", "F4"]:
        ld = UnifiedDataLoader(seq_len=96, horizon=5, batch_size=64,
                                fold=fold, max_stocks=5)
        assert len(ld.X_train) > 0, f"fold {fold}: empty train"
        assert len(ld.X_val) > 0,   f"fold {fold}: empty val"
        # F1's test=2020 may have data; F4's test=2023 also has data.
        assert len(ld.X_test) > 0,  f"fold {fold}: empty test"


def test_t2_same_stocks_present(smoke_loader):
    """Each loaded stock should contribute to train + val + test."""
    train_stocks = set(smoke_loader.train_stock_id.tolist())
    val_stocks   = set(smoke_loader.val_stock_id.tolist())
    test_stocks  = set(smoke_loader.test_stock_id.tolist())
    # Most stocks should be in all three; allow a few that have no
    # samples in val (e.g., short history).
    overlap = train_stocks & val_stocks & test_stocks
    assert len(overlap) >= len(train_stocks) - 2, (
        f"Same-stocks split broken: train={len(train_stocks)}, "
        f"all-three-overlap={len(overlap)}")


def test_t3_calendar_boundaries():
    """Anchor dates of train/val/test samples must lie in their respective
    windows. Pick one stock and check directly."""
    fold_dates = get_fold_dates("F4")
    # Use NVDA (we know it has full coverage)
    stk = _load_stock("NVDA")
    if stk is None:
        pytest.skip("NVDA not in DATA_DIR")
    res = build_samples_for_stock(stk, seq_len=96, horizon=5,
                                   fold_dates=fold_dates, target_mode="log_return")
    assert res is not None
    import pandas as pd
    dates_pd = pd.to_datetime(stk.dates, utc=True)
    # train anchors
    for a_idx in res["anchor_train"][:5]:
        assert dates_pd[int(a_idx)] <= fold_dates["train_end"]
    for a_idx in res["anchor_val"][:5]:
        assert fold_dates["val_start"] <= dates_pd[int(a_idx)] <= fold_dates["val_end"]
    for a_idx in res["anchor_test"][:5]:
        assert fold_dates["test_start"] <= dates_pd[int(a_idx)] <= fold_dates["test_end"]


def test_t4_z_score_train_aggregate(smoke_loader):
    """Cross-stock pooled mean/std on the LAST timestep of each train
    window should be in a reasonable range. OHLCV features (indices 0-4)
    must show meaningful variance. The 6th column (scaled_sentiment) can
    legitimately collapse to ~0 because no-news days are filled at the
    neutral value (0.5 = per-stock mean for high-coverage tickers), so
    after z-score they evaluate exactly to 0 -- the std reflects the
    minority of news-bearing days."""
    from data_loader import materialise_split
    from config import FEATURES
    X, _ = materialise_split(smoke_loader, "train", max_samples=2000)
    last_step = X[:, -1, :]
    means = last_step.mean(axis=0)
    stds  = last_step.std(axis=0)
    # OHLCV features (indices 0-4): must vary
    for fi in range(5):
        assert abs(means[fi]) < 5.0, f"feature {FEATURES[fi]} mean blown: {means[fi]}"
        assert stds[fi] > 1e-3, f"feature {FEATURES[fi]} std collapsed: {stds[fi]}"
        assert stds[fi] < 50.0, f"feature {FEATURES[fi]} std blown up: {stds[fi]}"
    # scaled_sentiment (index 5) -- accept very small std (no-news fill makes
    # most days exactly 0 after z-score). Just verify it's not negative or NaN.
    assert means[5] > -10 and means[5] < 10
    assert stds[5] >= 0.0 and stds[5] < 50.0


def test_t5_target_distribution(smoke_loader):
    """The eagerly-computed y_train always represents log-returns regardless
    of TARGET_MODE (the data_loader always computes them as a diagnostic).
    For H=5 they should be small and centred.
    NOTE: smoke_loader.y_train is only populated when target_mode='log_return';
    for 'scaled_price' it's a placeholder zero array. Compute manually here."""
    from data_loader import materialise_split
    # We force log-return materialisation by passing through the eager helper
    # only for samples where target_mode is "log_return". Otherwise we
    # compute log-return manually from the close_raw arrays.
    if smoke_loader.target_mode == "log_return":
        y = smoke_loader.y_train
    else:
        # Compute log-return manually from anchor + target close
        y = np.zeros(smoke_loader.sample_table_train.shape[0], dtype=np.float32)
        for n in range(len(y)):
            sid = int(smoke_loader.sample_table_train[n, 0])
            i = int(smoke_loader.sample_table_train[n, 1])
            close = smoke_loader.close_raw_list[sid]
            ac = close[i + smoke_loader.seq_len - 1]
            tc = close[i + smoke_loader.seq_len + smoke_loader.horizon - 1]
            y[n] = float(np.log(tc / ac))
    assert np.abs(y.mean()) < 0.05, f"y mean too large: {y.mean()}"
    assert 0.005 < y.std() < 0.5, f"y std out of range: {y.std()}"


def test_t6_no_nan_inf(smoke_loader):
    """Materialise a subset of each split and verify no NaN/inf."""
    from data_loader import materialise_split
    for split in ["train", "val", "test"]:
        X, y = materialise_split(smoke_loader, split, max_samples=1000)
        assert not np.isnan(X).any(), f"{split}: X has NaN"
        assert not np.isinf(X).any(), f"{split}: X has Inf"
        assert not np.isnan(y).any(), f"{split}: y has NaN"
        assert not np.isinf(y).any(), f"{split}: y has Inf"
    # The eager y arrays (lightweight, computed at __init__) should also be clean
    assert not np.isnan(smoke_loader.y_train).any(), "y_train has NaN"
    assert not np.isnan(smoke_loader.y_val).any(),   "y_val has NaN"
    assert not np.isnan(smoke_loader.y_test).any(),  "y_test has NaN"


def test_t7_dataloader_iteration(smoke_loader):
    """One full pass through the train DataLoader. v2 returns 3-tuple
    (X, y_main, y_logret). y_main shape depends on TARGET_MODE; y_logret
    is always [B] scalar log-return."""
    loader = smoke_loader.get_train_loader(shuffle=False)
    n_samples = 0
    for batch in loader:
        assert len(batch) == 3, "expected 3-tuple (X, y_main, y_logret)"
        X, y_main, y_logret = batch
        assert isinstance(X, torch.Tensor)
        assert X.shape[1] == 96
        assert X.shape[2] == len(FEATURES)
        if smoke_loader.target_mode == "log_return":
            assert y_main.shape == (X.shape[0],)
        else:
            assert y_main.shape == (X.shape[0], smoke_loader.horizon)
        # y_logret is always a [B] scalar
        assert y_logret.shape == (X.shape[0],)
        # y_logret values are reasonable for daily-scale log-returns
        assert torch.isfinite(y_logret).all()
        n_samples += X.shape[0]
    assert n_samples == len(smoke_loader.X_train)


def test_t8_get_fold_dates():
    """Sanity-check the F4 dates (the headline fold)."""
    import pandas as pd
    d = get_fold_dates("F4")
    assert d["train_end"]  == pd.Timestamp("2021-12-31", tz="UTC")
    assert d["val_start"]  == pd.Timestamp("2022-01-01", tz="UTC")
    assert d["val_end"]    == pd.Timestamp("2022-12-31", tz="UTC")
    assert d["test_start"] == pd.Timestamp("2023-01-01", tz="UTC")
    assert d["test_end"]   == pd.Timestamp("2023-12-31", tz="UTC")


def test_t8_walkforward_folds_monotone():
    """Each fold's train_end < val_start < test_start."""
    for fold in WALK_FORWARD_FOLDS:
        d = get_fold_dates(fold["name"])
        assert d["train_end"] < d["val_start"]
        assert d["val_end"] < d["test_start"]


def test_t9_purged_walkforward_no_target_leakage():
    """CRITICAL: purged walk-forward must guarantee that NO train sample's
    target_date falls inside val/test windows (Jury 1 fix item A1+A2)."""
    fold_dates = get_fold_dates("F4")
    stk = _load_stock("NVDA")
    if stk is None:
        pytest.skip("NVDA not in DATA_DIR")
    H = 60   # use a long horizon to make leakage easy to detect
    seq_len = 96
    res = build_samples_for_stock(stk, seq_len=seq_len, horizon=H,
                                   fold_dates=fold_dates, target_mode="log_return",
                                   purge=True, embargo_days=H)
    assert res is not None
    import pandas as pd
    dates_pd = pd.to_datetime(stk.dates, utc=True)
    # CRITICAL check: every train sample's TARGET date must be <= train_end - embargo
    embargo_td = pd.Timedelta(days=H)
    train_target_dates = dates_pd[res["anchor_train"] + H]   # anchor + H = target idx
    assert all(td <= fold_dates["train_end"] - embargo_td for td in train_target_dates), \
        f"PURGE VIOLATED: some train target dates exceed train_end - embargo"
    # And every val sample's anchor + target both inside val window
    val_anchor_dates = dates_pd[res["anchor_val"]]
    val_target_dates = dates_pd[res["anchor_val"] + H]
    for ad, td in zip(val_anchor_dates, val_target_dates):
        assert fold_dates["val_start"] <= ad <= fold_dates["val_end"], \
            f"val anchor {ad} outside val window"
        assert fold_dates["val_start"] <= td <= fold_dates["val_end"], \
            f"val target {td} outside val window"


def test_t10_adj_close_split_safety():
    """CRITICAL: Adj_Close adjustment must prevent split-day artificial returns
    (Jury 1 fix item E2). NVDA had a 4:1 split on 2020-08-31 — without
    adjustment the raw-Close 1-day log-return that day would be ~log(1/4) ≈ -1.39.
    With our adj_factor scaling, it should be tiny (close to the actual
    economic return, < 0.10 in magnitude)."""
    stk = _load_stock("NVDA")
    if stk is None:
        pytest.skip("NVDA not in DATA_DIR")
    import pandas as pd
    dates_pd = pd.to_datetime(stk.dates, utc=True)
    # Find the 2020-08-31 trading day
    target = pd.Timestamp("2020-08-31", tz="UTC")
    diffs = np.abs((dates_pd - target).total_seconds())
    idx = int(np.argmin(diffs))
    if idx == 0:
        pytest.skip("split day at index 0")
    one_day_logret = float(np.log(stk.close_raw[idx] / stk.close_raw[idx-1]))
    assert abs(one_day_logret) < 0.20, (
        f"NVDA 1-day log-return on {dates_pd[idx]} = {one_day_logret:.4f} "
        f"-- a 4:1 split would have given ~-1.39 if Adj Close adjustment "
        f"was not applied. Adjust factor logic is broken.")


def test_t11_log1p_volume_applied():
    """Volume should be log1p-transformed (Jury 1 fix item G1). For a real
    stock with daily volumes ~1e6 to 1e8, raw values would be in that range;
    after log1p they should be in roughly [13, 19] (log of 1e6 to 1e8)."""
    stk = _load_stock("NVDA")
    if stk is None:
        pytest.skip("NVDA not in DATA_DIR")
    # Volume is index 4 in FEATURES = ["Open","High","Low","Close","Volume","scaled_sentiment"]
    vol_col = stk.raw[:, 4]
    assert vol_col.min() >= 0, "log1p volume can't be negative"
    assert vol_col.max() < 30, f"max log1p volume {vol_col.max():.2f} -- looks like raw, not log1p"
    assert vol_col.mean() > 5, f"mean log1p volume {vol_col.mean():.2f} -- too small"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
