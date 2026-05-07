"""Data loader v2 — same-stocks calendar-only split, log-return targets,
walk-forward fold support, train-only per-stock z-score normalisation.

Conventions (PLAN_v2):
  * Universe: 300 stocks from `data/universe_main.csv`. SAME stocks across
    train/val/test (matches MASTER/DeepClair/Qlib/FactorVAE convention).
  * Features: ["Open","High","Low","Close","Volume","scaled_sentiment"]
    (6 features; CLOSE_IDX=3).
  * Target: H-step log-return  y = log(close[i+seq_len+H-1] / close[i+seq_len-1]).
  * Normalisation: per-stock z-score with mu, sd computed on the FOLD's
    TRAINING WINDOW ONLY (no leakage).
  * Walk-forward CV: 4 folds (F1..F4) per config.WALK_FORWARD_FOLDS.
  * Calendar boundary: a sample whose anchor (last input day) date falls
    in a split's window belongs to that split.

Memory model: per-stock raw_normed arrays cached in RAM (300 stocks *
~3,800 days * 6 features * 8 bytes = 55 MB total). Samples are produced
LAZILY in __getitem__ -- no 10 GB materialised tensor.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import (DATA_DIR, FEATURES, CLOSE_IDX, SEQ_LEN, TARGET_MODE,
                    WALK_FORWARD_FOLDS, load_universe)


# ---------------------------------------------------------------------------
# Per-stock raw container
# ---------------------------------------------------------------------------
@dataclass
class StockData:
    ticker: str
    dates: np.ndarray          # [N] np.datetime64[ns, UTC]
    raw: np.ndarray            # [N, n_features] float64
    close_raw: np.ndarray      # [N] float64 -- raw close for log-return target
    raw_normed: np.ndarray     # [N, n_features] float32 (z-scored on train window)
    mu: np.ndarray             # [n_features] train-window mean
    sd: np.ndarray             # [n_features] train-window std


def _load_stock(ticker: str, data_dir: str = None) -> StockData | None:
    """Load merged_v3/<TICKER>.csv with split/dividend adjustment + log1p volume.

    Critical correctness fixes (per Jury 1 review 2026-05-07):

    F-A (Adj Close + OHLC adjustment):
        merged_v3 has BOTH `Close` (raw) and `Adj Close` (split/dividend adj).
        We compute `adj_factor = Adj_Close / Close` and apply it to Open,
        High, Low, Close uniformly so the entire OHLC tuple is internally
        consistent with the adjusted close. This is essential because
        major splits in the universe (AAPL 2014/2020 4:1, NVDA 2021 4:1,
        TSLA 2020 5:1, GOOG 2022 20:1) would otherwise inject artificial
        ±50%+ one-day returns. close_raw (used for log-return target) is
        Adj Close.

    F-G (log1p Volume):
        Volume is right-skewed by 2-3 orders of magnitude. Raw z-score
        gives sigma dominated by a few high-volume days. We log1p before
        passing to z-score normalisation downstream.

    Returns None on missing column / insufficient rows.
    """
    data_dir = data_dir or DATA_DIR
    if data_dir is None:
        raise RuntimeError("DATA_DIR not set. Run rebuild_merged_v2.py first.")
    path = os.path.join(data_dir, f"{ticker}.csv")
    if not os.path.exists(path):
        alt = os.path.join(data_dir, f"{ticker.lower()}.csv")
        if os.path.exists(alt):
            path = alt
        else:
            return None
    df = pd.read_csv(path, low_memory=False)
    if "Date" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # ----- F-A: split/dividend adjustment via adj_factor -----
    # merged_v3 carries 'Adj Close' (FNSPID full_history adjusted close).
    # adj_factor scales raw OHLC backward to be split-consistent with the
    # adjusted close. Equal to 1.0 historically when no corporate actions
    # have occurred since the date; <1 prior to a split.
    if "Adj Close" not in df.columns:
        # Fallback: assume already adjusted (older merged files).
        df["Adj Close"] = df["Close"]
    raw_close = df["Close"].astype(np.float64).values
    adj_close = df["Adj Close"].astype(np.float64).values
    safe_raw = np.where(raw_close > 1e-9, raw_close, 1.0)
    adj_factor = adj_close / safe_raw
    # Apply adjustment to Open/High/Low/Close (Volume is share count, not price).
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = df[col].astype(np.float64).values * adj_factor

    # ----- F-G: log1p Volume -----
    if "Volume" in df.columns:
        df["Volume"] = np.log1p(df["Volume"].astype(np.float64).clip(lower=0).values)

    feats = []
    for col in FEATURES:
        if col in df.columns:
            feats.append(df[col].astype(np.float64).values)
        elif "sentiment" in col.lower():
            feats.append(np.full(len(df), 0.5))
        else:
            raise RuntimeError(f"{ticker}: missing required column '{col}'")
    raw = np.column_stack(feats)
    valid = ~np.any(np.isnan(raw), axis=1)
    raw = raw[valid]
    dates = df["Date"].values[valid]
    if len(raw) < 100:
        return None
    # close_raw: used for log-return target. Now this is the SPLIT-ADJUSTED
    # close (because we replaced raw['Close'] with adj_factor * raw_close
    # above). Splits no longer cause artificial one-day returns.
    return StockData(ticker=ticker, dates=dates, raw=raw,
                     close_raw=raw[:, CLOSE_IDX].copy(),
                     raw_normed=None, mu=None, sd=None)


# ---------------------------------------------------------------------------
# Fold lookup
# ---------------------------------------------------------------------------
def get_fold_dates(fold_name: str) -> dict:
    for fold in WALK_FORWARD_FOLDS:
        if fold["name"] == fold_name:
            ty = int(fold["test_year"]); vy = int(fold["val_year"])
            return {
                "train_end":  pd.Timestamp(fold["train_end"], tz="UTC"),
                "val_start":  pd.Timestamp(f"{vy}-01-01", tz="UTC"),
                "val_end":    pd.Timestamp(f"{vy}-12-31", tz="UTC"),
                "test_start": pd.Timestamp(f"{ty}-01-01", tz="UTC"),
                "test_end":   pd.Timestamp(f"{ty}-12-31", tz="UTC"),
            }
    raise KeyError(f"Unknown fold {fold_name!r}")


# ---------------------------------------------------------------------------
# Per-stock fit & sample-index build
# ---------------------------------------------------------------------------
def _fit_normalise_and_index(stk: StockData, seq_len: int, horizon: int,
                              fold_dates: dict, target_mode: str,
                              min_close: float = 1e-6,
                              purge: bool = True,
                              embargo_days: int = None):
    """Fit per-stock z-score on train rows, fill stk.raw_normed, and return
    per-split index arrays of valid sample START offsets.

    A sample starts at index i; its anchor date is dates[i+seq_len-1] and
    target is at dates[i+seq_len+horizon-1].

    PURGED WALK-FORWARD (per Jury 1 review 2026-05-07, fixing item A1+A2):
    Instead of assigning a sample to a split based on the ANCHOR date alone,
    we additionally require that the TARGET date is also within the same
    split's window. This prevents train labels from reaching into val/test
    windows when H>=5. Specifically:
        train assigned iff:  dates[target_idx] <= fold.train_end
        val assigned iff:    fold.val_start <= dates[target_idx] <= fold.val_end
                              AND fold.val_start <= dates[anchor_idx]
        test assigned iff:   fold.test_start <= dates[target_idx] <= fold.test_end
                              AND fold.test_start <= dates[anchor_idx]
    Samples whose target straddles a split boundary are dropped.

    EMBARGO (López de Prado 2018, Ch. 7.4): An additional `embargo_days`
    after each split end are excluded from the next split's training/eval
    windows. Default = horizon (Lopez de Prado's recommendation for
    overlapping returns: embargo = forecast horizon, so one full horizon's
    worth of "warm-up" is dropped).
    """
    if embargo_days is None:
        embargo_days = int(horizon)

    N = stk.raw.shape[0]
    if N < seq_len + horizon:
        return None

    dates_pd = pd.to_datetime(stk.dates, utc=True)
    train_mask_rows = dates_pd <= fold_dates["train_end"]

    train_rows = stk.raw[train_mask_rows]
    if train_rows.shape[0] < seq_len + horizon:
        return None
    mu = train_rows.mean(axis=0)
    sd = train_rows.std(axis=0)
    sd[sd < 1e-9] = 1.0
    stk.mu = mu; stk.sd = sd
    stk.raw_normed = ((stk.raw - mu) / sd).astype(np.float32)

    # Embargo bounds
    train_end = fold_dates["train_end"]
    val_start = fold_dates["val_start"]
    val_end   = fold_dates["val_end"]
    test_start = fold_dates["test_start"]
    test_end   = fold_dates["test_end"]
    embargo_td = pd.Timedelta(days=embargo_days)

    train_idx, val_idx, test_idx = [], [], []
    for i in range(0, N - seq_len - horizon + 1):
        anchor_idx = i + seq_len - 1
        target_idx = i + seq_len + horizon - 1
        anchor_date = dates_pd[anchor_idx]
        target_date = dates_pd[target_idx]

        # Filter on close-price validity for log-return target
        if target_mode == "log_return":
            ac = stk.close_raw[anchor_idx]; tc = stk.close_raw[target_idx]
            if ac <= min_close or tc <= min_close:
                continue

        if purge:
            # Purged + embargoed assignment:
            #   TRAIN: target_date <= train_end - embargo
            #   VAL:   target_date in [val_start, val_end] AND anchor_date in [val_start, val_end]
            #   TEST:  target_date in [test_start, test_end] AND anchor_date in [test_start, test_end]
            # Samples that straddle boundaries (target outside anchor's window
            # OR target lies inside the embargo zone) are dropped.
            if target_date <= train_end - embargo_td:
                train_idx.append(i)
            elif (val_start <= anchor_date <= val_end) and (val_start <= target_date <= val_end):
                val_idx.append(i)
            elif (test_start <= anchor_date <= test_end) and (test_start <= target_date <= test_end):
                test_idx.append(i)
            # else: dropped (boundary or gap-year sample)
        else:
            # Legacy anchor-only assignment (kept for backward compat / ablation)
            if anchor_date <= train_end:
                train_idx.append(i)
            elif val_start <= anchor_date <= val_end:
                val_idx.append(i)
            elif test_start <= anchor_date <= test_end:
                test_idx.append(i)

    return {"train": np.array(train_idx, dtype=np.int32),
            "val":   np.array(val_idx, dtype=np.int32),
            "test":  np.array(test_idx, dtype=np.int32)}


# ---------------------------------------------------------------------------
# Lazy on-the-fly Dataset
# ---------------------------------------------------------------------------
class LazyStockDataset(Dataset):
    """Memory-efficient sample-on-demand dataset.

    Backing storage: a tuple (raw_normed_per_stock, close_raw_per_stock,
    sample_table). sample_table is a [N, 3] int32 array of
    (stock_id, start_offset, anchor_offset) where:
        stock_id    -> index into the per-stock list
        start_offset = i (where the input window begins)
        anchor_offset = i + seq_len - 1 (last input day)
    """

    def __init__(self, raw_normed_list, close_raw_list, sample_table,
                 seq_len, horizon, target_mode):
        self.raw_normed_list = raw_normed_list  # list of [N_s, F] float32
        self.close_raw_list = close_raw_list    # list of [N_s] float64
        self.sample_table = sample_table        # [N, 3] int32
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.target_mode = target_mode

    def __len__(self):
        return self.sample_table.shape[0]

    def __getitem__(self, idx):
        stock_id = int(self.sample_table[idx, 0])
        i = int(self.sample_table[idx, 1])
        raw_normed = self.raw_normed_list[stock_id]
        X = raw_normed[i : i + self.seq_len]            # [seq_len, F]
        if self.target_mode == "log_return":
            close = self.close_raw_list[stock_id]
            anchor_idx = i + self.seq_len - 1
            target_idx = i + self.seq_len + self.horizon - 1
            y = np.float32(np.log(close[target_idx] / close[anchor_idx]))
        elif self.target_mode == "scaled_price":
            # Return the H-step normalised close window
            y = raw_normed[i + self.seq_len : i + self.seq_len + self.horizon, CLOSE_IDX].astype(np.float32)
        else:
            raise ValueError(f"Unknown target_mode {self.target_mode!r}")
        return X.copy(), y


# ---------------------------------------------------------------------------
# UnifiedDataLoader (v2 entrypoint, lazy)
# ---------------------------------------------------------------------------
class UnifiedDataLoader:
    """Build train/val/test PyTorch DataLoaders for one walk-forward fold.

    Stores per-stock raw_normed + close_raw arrays in RAM (~55 MB total
    for 300 stocks). Samples are produced lazily in __getitem__ -- no
    materialised sample tensor.
    """

    def __init__(self, seq_len: int = SEQ_LEN, horizon: int = 5,
                 batch_size: int = 128, fold: str = "F4",
                 max_stocks: int | None = None,
                 target_mode: str = None,
                 num_workers: int = 0,
                 universe_file: str = None,
                 purge: bool = True,
                 embargo_days: int = None):
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.batch_size = int(batch_size)
        self.fold = fold
        self.fold_dates = get_fold_dates(fold)
        self.target_mode = target_mode or TARGET_MODE
        self.num_workers = int(num_workers)
        self.purge = bool(purge)
        self.embargo_days = int(embargo_days) if embargo_days is not None else int(horizon)

        universe = load_universe(universe_file) if universe_file else load_universe()
        if max_stocks:
            universe = universe[:max_stocks]
        self.universe = universe

        self.raw_normed_list = []      # one [N_s, F] float32 per stock
        self.close_raw_list = []       # one [N_s] float64 per stock
        self.scalers = {}              # {ticker: (mu, sd)}
        self.skipped = []
        # Per-split sample tables: rows = (stock_id, start_offset, anchor_offset)
        train_rows, val_rows, test_rows = [], [], []
        train_stock_id, val_stock_id, test_stock_id = [], [], []
        train_anchor, val_anchor, test_anchor = [], [], []

        for ticker in universe:
            stk = _load_stock(ticker)
            if stk is None:
                self.skipped.append(ticker); continue
            res = _fit_normalise_and_index(stk, self.seq_len, self.horizon,
                                            self.fold_dates, self.target_mode,
                                            purge=self.purge,
                                            embargo_days=self.embargo_days)
            if res is None or len(res["train"]) == 0:
                self.skipped.append(ticker); continue
            sid = len(self.raw_normed_list)
            self.raw_normed_list.append(stk.raw_normed)
            self.close_raw_list.append(stk.close_raw)
            self.scalers[ticker] = (stk.mu, stk.sd)
            for i_arr, rows, sids, anchors in [
                (res["train"], train_rows, train_stock_id, train_anchor),
                (res["val"],   val_rows,   val_stock_id,   val_anchor),
                (res["test"],  test_rows,  test_stock_id,  test_anchor),
            ]:
                if len(i_arr) == 0:
                    continue
                rows.append(np.column_stack([
                    np.full(len(i_arr), sid, dtype=np.int32),
                    i_arr.astype(np.int32),
                    (i_arr + self.seq_len - 1).astype(np.int32),
                ]))
                sids.append(np.full(len(i_arr), sid, dtype=np.int32))
                anchors.append((i_arr + self.seq_len - 1).astype(np.int32))

        def _vstack(rows): return np.concatenate(rows, axis=0) if rows else np.zeros((0, 3), dtype=np.int32)
        def _vec(rows): return np.concatenate(rows, axis=0) if rows else np.zeros(0, dtype=np.int32)

        self.sample_table_train = _vstack(train_rows)
        self.sample_table_val   = _vstack(val_rows)
        self.sample_table_test  = _vstack(test_rows)
        self.train_stock_id = _vec(train_stock_id)
        self.val_stock_id   = _vec(val_stock_id)
        self.test_stock_id  = _vec(test_stock_id)
        self.train_anchor_idx = _vec(train_anchor)
        self.val_anchor_idx   = _vec(val_anchor)
        self.test_anchor_idx  = _vec(test_anchor)

        # Convenience attributes (eagerly compute targets for diagnostic
        # use; this is small -- one float per sample, so n_train * 4 bytes).
        self.y_train = self._compute_targets(self.sample_table_train)
        self.y_val   = self._compute_targets(self.sample_table_val)
        self.y_test  = self._compute_targets(self.sample_table_test)

    def _compute_targets(self, sample_table):
        """Eagerly compute the y vector for diagnostics. Lightweight (one
        float per sample). For target_mode='log_return' returns [N];
        for 'scaled_price' returns the empty array (computed in
        __getitem__)."""
        if self.target_mode != "log_return":
            return np.zeros(sample_table.shape[0], dtype=np.float32)
        if sample_table.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        out = np.zeros(sample_table.shape[0], dtype=np.float32)
        for n in range(sample_table.shape[0]):
            sid = int(sample_table[n, 0])
            i = int(sample_table[n, 1])
            close = self.close_raw_list[sid]
            ac = close[i + self.seq_len - 1]
            tc = close[i + self.seq_len + self.horizon - 1]
            out[n] = float(np.log(tc / ac))
        return out

    @property
    def X_train(self):
        # Compatibility alias for tests; LAZY -- accessing this returns a
        # virtual proxy with a length and shape but no materialised data.
        return _LazyShapeProxy(self.sample_table_train.shape[0],
                                (self.seq_len, len(FEATURES)))

    @property
    def X_val(self):
        return _LazyShapeProxy(self.sample_table_val.shape[0],
                                (self.seq_len, len(FEATURES)))

    @property
    def X_test(self):
        return _LazyShapeProxy(self.sample_table_test.shape[0],
                                (self.seq_len, len(FEATURES)))

    def __repr__(self):
        return (f"UnifiedDataLoader(fold={self.fold}, seq_len={self.seq_len}, "
                f"horizon={self.horizon}, n_stocks={len(self.universe)-len(self.skipped)}/"
                f"{len(self.universe)}, n_train={self.sample_table_train.shape[0]}, "
                f"n_val={self.sample_table_val.shape[0]}, "
                f"n_test={self.sample_table_test.shape[0]}, "
                f"target_mode={self.target_mode}, lazy=True)")

    def _make_loader(self, sample_table, shuffle):
        ds = LazyStockDataset(self.raw_normed_list, self.close_raw_list,
                               sample_table, self.seq_len, self.horizon,
                               self.target_mode)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                           num_workers=self.num_workers, pin_memory=True)

    def get_train_loader(self, shuffle: bool = True) -> DataLoader:
        return self._make_loader(self.sample_table_train, shuffle)

    def get_val_loader(self) -> DataLoader:
        return self._make_loader(self.sample_table_val, False)

    def get_test_loader(self) -> DataLoader:
        return self._make_loader(self.sample_table_test, False)

    def get_train_val_test_loaders(self):
        return (self.get_train_loader(), self.get_val_loader(), self.get_test_loader())


class _LazyShapeProxy:
    """Tiny proxy so existing tests can call `len(loader.X_train)` etc.
    without materialising the 10 GB tensor. Not iterable; use the
    DataLoader for actual iteration."""
    def __init__(self, n: int, sample_shape: tuple):
        self.n = int(n)
        self.shape = (self.n,) + tuple(sample_shape)
    def __len__(self):
        return self.n


# ---------------------------------------------------------------------------
# Eager helper (deprecated; kept for tests that need a small materialised
# slice). Returns (X_concat, y_concat) for the requested split.
# ---------------------------------------------------------------------------
def materialise_split(ld: UnifiedDataLoader, split: str, max_samples: int = None):
    """Materialise up to max_samples for the given split. Used in tests."""
    sample_table = {"train": ld.sample_table_train,
                    "val":   ld.sample_table_val,
                    "test":  ld.sample_table_test}[split]
    n = sample_table.shape[0] if max_samples is None else min(sample_table.shape[0], max_samples)
    X = np.zeros((n, ld.seq_len, len(FEATURES)), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    for k in range(n):
        sid = int(sample_table[k, 0])
        i = int(sample_table[k, 1])
        X[k] = ld.raw_normed_list[sid][i : i + ld.seq_len]
        if ld.target_mode == "log_return":
            close = ld.close_raw_list[sid]
            ac = close[i + ld.seq_len - 1]
            tc = close[i + ld.seq_len + ld.horizon - 1]
            y[k] = float(np.log(tc / ac))
    return X, y


# ---------------------------------------------------------------------------
# Backward-compat wrapper for older code paths that called
# build_samples_for_stock(...) -- delegate to the new fit/index function.
# ---------------------------------------------------------------------------
def build_samples_for_stock(stk: StockData, seq_len: int, horizon: int,
                             fold_dates: dict, target_mode: str = "log_return",
                             purge: bool = True, embargo_days: int = None):
    """Convenience: returns same dict as v1 (for tests)."""
    res = _fit_normalise_and_index(stk, seq_len, horizon, fold_dates, target_mode,
                                    purge=purge, embargo_days=embargo_days)
    if res is None:
        return None
    # Build small materialised samples (test-only, slow path)
    def _make(idx_arr):
        n = len(idx_arr)
        X = np.zeros((n, seq_len, len(FEATURES)), dtype=np.float32)
        y_logret = np.zeros(n, dtype=np.float32)
        anchors = np.zeros(n, dtype=np.int32)
        for k, i in enumerate(idx_arr):
            X[k] = stk.raw_normed[i : i + seq_len]
            ac = stk.close_raw[i + seq_len - 1]
            tc = stk.close_raw[i + seq_len + horizon - 1]
            y_logret[k] = float(np.log(tc / ac))
            anchors[k] = i + seq_len - 1
        return X, y_logret, anchors
    Xt, yt, at = _make(res["train"])
    Xv, yv, av = _make(res["val"])
    Xs, ys, as_ = _make(res["test"])
    return {
        "X_train": Xt, "y_train": yt, "anchor_train": at,
        "X_val":   Xv, "y_val":   yv, "anchor_val":   av,
        "X_test":  Xs, "y_test":  ys, "anchor_test":  as_,
        "mu": stk.mu, "sd": stk.sd,
        "n_train": int(len(Xt)),
        "n_val":   int(len(Xv)),
        "n_test":  int(len(Xs)),
    }
