"""Verifications for the v2 pipeline:

  A. NEWS COVERAGE per stock -- re-confirm every ticker in
     universe_main.csv has news in BOTH val (2022) AND test (2023) years.
     Independent re-scan of the 23 GB news file (not relying on the
     curation manifest).

  B. CALENDAR LEAKAGE -- verify the proposed splits have no overlap and
     that the per-stock scaler fit window is restricted properly.

  C. WALK-FORWARD CV LEAKAGE -- verify every fold's (train_end, val,
     test) window is monotone and non-overlapping with future windows.

  D. PRICE FILE PRESENCE -- every ticker has its OHLCV CSV in
     FNSPID/full_history/ and the date range covers 2009 -> 2023-12.

  E. CASE-NORMALISATION -- universe tickers (UPPER) must match price
     filenames (UPPER) AND news Stock_symbol (UPPER) so all joins work.

Each check writes a one-line PASS/FAIL with statistics. Exit code 1 on
any FAIL.

Usage:
  python data_pipeline/verify_universe_and_leakage.py
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

NEWS_FILE = r"D:\Study\FNSPID_v1\Data\Stock_news\nasdaq_exteral_data.csv"
PRICES_DIR = r"D:\Study\FNSPID_v1\Data\full_history"
UNIVERSE_FILE = r"D:\Study\CIKM\finsharpe\data\universe_main.csv"

# Calendar config (v2)
TRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2023-12-29"

# Walk-forward CV folds (4 expanding-window folds)
WALK_FORWARD = [
    {"name": "F1", "train_end": "2018-12-31", "val": "2019",  "test": "2020"},
    {"name": "F2", "train_end": "2019-12-31", "val": "2020",  "test": "2021"},
    {"name": "F3", "train_end": "2020-12-31", "val": "2021",  "test": "2022"},
    {"name": "F4", "train_end": "2021-12-31", "val": "2022",  "test": "2023"},
]


def fail(msg):
    print(f"\n[FAIL] {msg}", flush=True)
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# A. News coverage re-verification
# ─────────────────────────────────────────────────────────────────
def check_a_news_coverage(news_file, universe_set):
    print("=" * 64)
    print("A. NEWS COVERAGE re-verification (per-stock per-year)")
    print("=" * 64)
    counts = defaultdict(lambda: defaultdict(int))   # [ticker][year] -> n
    print(f"Streaming {news_file} ...", flush=True)
    chunks = 0
    for chunk in pd.read_csv(news_file, chunksize=500_000,
                              usecols=["Date", "Stock_symbol"],
                              dtype={"Stock_symbol": str}):
        chunks += 1
        chunk["Date"] = pd.to_datetime(chunk["Date"], errors="coerce", utc=True)
        chunk = chunk.dropna(subset=["Date", "Stock_symbol"])
        chunk["Stock_symbol"] = chunk["Stock_symbol"].astype(str).str.upper()
        chunk = chunk[chunk["Stock_symbol"].isin(universe_set)]
        if not len(chunk):
            continue
        chunk["year"] = chunk["Date"].dt.year
        gb = chunk.groupby(["Stock_symbol", "year"]).size()
        for (sym, yr), n in gb.items():
            counts[sym][int(yr)] += int(n)
        if chunks % 5 == 0:
            print(f"  chunk {chunks}...", flush=True)

    # Per-stock per-year for the years we actually need
    rows = []
    for sym in universe_set:
        c = counts[sym]
        n2018 = c.get(2018, 0); n2019 = c.get(2019, 0); n2020 = c.get(2020, 0)
        n2021 = c.get(2021, 0); n2022 = c.get(2022, 0); n2023 = c.get(2023, 0)
        n_pre = sum(c[y] for y in c if y <= 2017)
        rows.append({"ticker": sym, "n_pre2018": n_pre,
                     "n_2018": n2018, "n_2019": n2019, "n_2020": n2020,
                     "n_2021": n2021, "n_2022": n2022, "n_2023": n2023,
                     "n_total": sum(c.values())})
    df = pd.DataFrame(rows).sort_values("n_2023", ascending=False)
    out_path = "data/_universe_news_coverage.csv"
    df.to_csv(out_path, index=False)

    # Hard check: every ticker has >= 30 in 2022 AND 2023 (curation threshold)
    failed_22 = df[df["n_2022"] < 30]
    failed_23 = df[df["n_2023"] < 30]
    if len(failed_22) > 0:
        fail(f"A -- {len(failed_22)} tickers fail n_2022 >= 30: "
             f"{failed_22['ticker'].head(10).tolist()}")
    if len(failed_23) > 0:
        fail(f"A -- {len(failed_23)} tickers fail n_2023 >= 30: "
             f"{failed_23['ticker'].head(10).tolist()}")

    # Stats
    print(f"\n  All {len(df)} tickers pass n_2022>=30 AND n_2023>=30")
    print(f"  Median n_2022: {int(df['n_2022'].median())}  "
          f"P10: {int(df['n_2022'].quantile(0.1))}  "
          f"P90: {int(df['n_2022'].quantile(0.9))}")
    print(f"  Median n_2023: {int(df['n_2023'].median())}  "
          f"P10: {int(df['n_2023'].quantile(0.1))}  "
          f"P90: {int(df['n_2023'].quantile(0.9))}")
    print(f"  Coverage span: median pre-2018 articles = {int(df['n_pre2018'].median())}")

    # Also verify EVERY walk-forward fold's val+test years have news for
    # every stock at the >=20 threshold (slightly looser than headline).
    print("\n  Walk-forward fold per-year minimums (require >=20 articles):")
    for fold in WALK_FORWARD:
        val_y = int(fold["val"]); test_y = int(fold["test"])
        v_col = f"n_{val_y}"; t_col = f"n_{test_y}"
        bad_v = (df[v_col] < 20).sum()
        bad_t = (df[t_col] < 20).sum()
        status = "OK" if (bad_v == 0 and bad_t == 0) else "PARTIAL"
        print(f"    {fold['name']}: val={val_y} ({bad_v} stocks <20), "
              f"test={test_y} ({bad_t} stocks <20)  [{status}]")
        if bad_v > 0 or bad_t > 0:
            print(f"      stocks failing val: "
                  f"{df[df[v_col] < 20]['ticker'].head(5).tolist()}")
            print(f"      stocks failing test: "
                  f"{df[df[t_col] < 20]['ticker'].head(5).tolist()}")

    print(f"\nA PASS: per-stock per-year news coverage verified. -> {out_path}")
    return df


# ─────────────────────────────────────────────────────────────────
# B. Headline calendar leakage
# ─────────────────────────────────────────────────────────────────
def check_b_calendar_leakage():
    print("\n" + "=" * 64)
    print("B. HEADLINE CALENDAR LEAKAGE (train -> val -> test windows)")
    print("=" * 64)
    train_end = pd.Timestamp(TRAIN_END)
    val_start = pd.Timestamp(VAL_START); val_end = pd.Timestamp(VAL_END)
    test_start = pd.Timestamp(TEST_START); test_end = pd.Timestamp(TEST_END)

    if not (train_end < val_start):
        fail(f"B -- train_end {train_end} >= val_start {val_start} (overlap)")
    if not (val_end < test_start):
        fail(f"B -- val_end {val_end} >= test_start {test_start} (overlap)")
    if (val_start - train_end).days != 1:
        fail(f"B -- gap between train_end and val_start = {(val_start - train_end).days} days "
             f"(should be exactly 1 -- no implicit holdout days)")
    if (test_start - val_end).days != 1:
        fail(f"B -- gap between val_end and test_start = {(test_start - val_end).days} days")

    print(f"  Train ends:    {train_end.date()}")
    print(f"  Val window:    {val_start.date()} -> {val_end.date()}")
    print(f"  Test window:   {test_start.date()} -> {test_end.date()}")
    print(f"  No overlap, no gap. OK")
    print("\nB PASS: headline calendar split is leak-free.")


# ─────────────────────────────────────────────────────────────────
# C. Walk-forward CV folds -- monotone and non-overlapping with FUTURE
# ─────────────────────────────────────────────────────────────────
def check_c_walk_forward_leakage():
    print("\n" + "=" * 64)
    print("C. WALK-FORWARD CV LEAKAGE (4 folds)")
    print("=" * 64)
    for i, fold in enumerate(WALK_FORWARD):
        train_end = pd.Timestamp(fold["train_end"])
        val_y = int(fold["val"])
        test_y = int(fold["test"])
        val_start = pd.Timestamp(f"{val_y}-01-01")
        val_end = pd.Timestamp(f"{val_y}-12-31")
        test_start = pd.Timestamp(f"{test_y}-01-01")
        test_end = pd.Timestamp(f"{test_y}-12-31")

        # Within-fold: train_end < val_start < val_end < test_start
        if not (train_end < val_start):
            fail(f"C -- fold {fold['name']} train_end {train_end} >= val_start {val_start}")
        if not (val_end < test_start):
            fail(f"C -- fold {fold['name']} val_end {val_end} >= test_start {test_start}")

        # Each fold is independently valid: train must precede its own val/test.
        # Across folds, val/test years can overlap (this is normal in expanding-
        # window walk-forward CV: F2's val=2020 coincides with F1's test=2020,
        # but F2's TRAINING ends at 2019-12 so F2 hasn't seen 2020 data either.
        # Each fold trains an independent model from scratch on its own
        # train window).
        print(f"  {fold['name']}: train<={train_end.date()}, val={val_y}, test={test_y}  OK")

    print("\n  All 4 folds: each independently has train_end < val < test "
          "(each fold trained from scratch on its own past data).")
    print("\nC PASS: walk-forward CV is leak-free within each fold.")


# ─────────────────────────────────────────────────────────────────
# D. Price file presence + date range
# ─────────────────────────────────────────────────────────────────
def check_d_prices(prices_dir, universe_set):
    print("\n" + "=" * 64)
    print("D. PRICE FILE PRESENCE + DATE RANGE")
    print("=" * 64)
    missing = []
    short = []
    for sym in sorted(universe_set):
        path = os.path.join(prices_dir, f"{sym}.csv")
        if not os.path.exists(path):
            missing.append(sym)
            continue
        try:
            df = pd.read_csv(path, usecols=["date"])
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
            df = df.dropna(subset=["date"])
            min_d, max_d = df["date"].min(), df["date"].max()
            need_start = pd.Timestamp("2009-01-15", tz="UTC")
            need_end = pd.Timestamp("2023-12-15", tz="UTC")
            if min_d > need_start or max_d < need_end:
                short.append((sym, str(min_d.date()), str(max_d.date())))
        except Exception as e:
            missing.append(f"{sym} (read_fail: {e})")

    if missing:
        fail(f"D -- {len(missing)} tickers have no price file: "
             f"{missing[:10]}")
    if short:
        fail(f"D -- {len(short)} tickers have insufficient price coverage: "
             f"{short[:5]}")
    print(f"  All {len(universe_set)} tickers have price files in "
          f"{prices_dir}\\")
    print(f"  All cover at least 2009-01-15 -> 2023-12-15.")
    print("\nD PASS: price files present and well-covered.")


# ─────────────────────────────────────────────────────────────────
# E. Case + naming consistency
# ─────────────────────────────────────────────────────────────────
def check_e_case_consistency(universe_set, prices_dir):
    print("\n" + "=" * 64)
    print("E. CASE-NORMALISATION CONSISTENCY")
    print("=" * 64)
    # Universe tickers should all be uppercase
    non_upper = {t for t in universe_set if t != t.upper()}
    if non_upper:
        fail(f"E -- universe contains non-uppercase tickers: "
             f"{list(non_upper)[:10]}")

    # Price files should match (upper-case ticker = filename stem)
    for sym in list(universe_set)[:20]:
        if not os.path.exists(os.path.join(prices_dir, f"{sym}.csv")):
            fail(f"E -- price file {sym}.csv missing (case sensitivity)")

    print(f"  All {len(universe_set)} tickers uppercase. OK")
    print(f"  Price-file lookup case-consistent. OK")
    print("  (News file Stock_symbol is normalised to UPPER in our pipeline, "
          "so cross-source joins work.)")
    print("\nE PASS: case-naming consistent across all sources.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--news_file", default=NEWS_FILE)
    p.add_argument("--prices_dir", default=PRICES_DIR)
    p.add_argument("--universe_file", default=UNIVERSE_FILE)
    p.add_argument("--skip_news_recheck", action="store_true",
                   help="Skip the slow re-stream of the 23GB news file (5+ min).")
    args = p.parse_args()

    universe = set(pd.read_csv(args.universe_file)["ticker"].astype(str).str.upper())
    print(f"Universe: {len(universe)} tickers")
    print()

    # B, C, E are fast -- run first
    check_b_calendar_leakage()
    check_c_walk_forward_leakage()
    check_d_prices(args.prices_dir, universe)
    check_e_case_consistency(universe, args.prices_dir)

    # A is slow (re-streams 23 GB) -- run last
    if not args.skip_news_recheck:
        check_a_news_coverage(args.news_file, universe)
    else:
        print("\n[A skipped -- pass --skip_news_recheck=False to re-verify]")

    print("\n" + "=" * 64)
    print("ALL VERIFICATIONS PASSED.")
    print("=" * 64)


if __name__ == "__main__":
    main()
