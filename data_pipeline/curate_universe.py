"""Curate the stock universe for the finsharpe paper.

Selection criteria (all hard requirements):

  C1. Price coverage: complete daily OHLCV from PRICE_START to PRICE_END.
      We use FNSPID's `full_history/<TICKER>.csv` files (7,693 tickers
      with prices 2009-ish to 2023-12-28).

  C2. News coverage: at least N_NEWS_VAL articles in VAL year (default
      2022) AND at least N_NEWS_TEST articles in TEST year (default 2023).
      Computed from the 23 GB nasdaq_exteral_data.csv.

  C3. Liquidity: median daily dollar-volume in TEST year > MIN_DOLLAR_VOL
      (default $5M). Filters penny stocks and illiquid OTC issues.

  C4. Continuity: no gap > MAX_GAP_DAYS in price data inside the
      [PRICE_START, PRICE_END] window. Prevents survivor-bias artefacts
      from delisted-and-relisted tickers.

Output:
  data/universe_v1.csv with one row per selected ticker:
      ticker, sector, news_2022, news_2023, news_total,
      price_start, price_end, n_price_rows, median_dollar_vol_2023,
      tier  (STRONG=top quartile by news; MEDIUM=2nd-3rd quartile; WEAK=bottom)

Two universes are emitted:
  data/universe_main.csv      -- top ~300 stocks (the paper headline)
  data/universe_seqvsglob.csv -- 50-stock subset for the sequential-vs-global
                                  side study (small enough to run sequential
                                  fine-tuning per stock without exhausting time
                                  budget; same stocks as the main universe's
                                  STRONG tier for consistency).

Reproducibility: every random sampling step uses seed 2026.

Usage:
  python scripts/curate_universe.py
  python scripts/curate_universe.py --target_n 300 --seqvsglob_n 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from glob import glob

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
NEWS_FILE = r"D:\Study\FNSPID_v1\Data\Stock_news\nasdaq_exteral_data.csv"
PRICES_DIR = r"D:\Study\FNSPID_v1\Data\full_history"
OUT_DIR = r"D:\Study\CIKM\fin-sent-optimized\data"

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
PRICE_START = "2009-01-01"
PRICE_END   = "2023-12-29"           # FNSPID's effective price end
VAL_YEAR    = 2022
TEST_YEAR   = 2023
TRAIN_END   = "2021-12-31"

# Coverage thresholds
N_NEWS_VAL_MIN  = 30                  # >=30 news articles in VAL year
N_NEWS_TEST_MIN = 30                  # >=30 news articles in TEST year
MIN_DOLLAR_VOL  = 5_000_000           # median daily $ volume in TEST year
MAX_GAP_DAYS    = 7                   # max consecutive missing trading days


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target_n", type=int, default=300,
                   help="Target size of main universe.")
    p.add_argument("--seqvsglob_n", type=int, default=50,
                   help="Size of the sequential-vs-global subset.")
    p.add_argument("--news_file", default=NEWS_FILE)
    p.add_argument("--prices_dir", default=PRICES_DIR)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--news_val_min", type=int, default=N_NEWS_VAL_MIN)
    p.add_argument("--news_test_min", type=int, default=N_NEWS_TEST_MIN)
    p.add_argument("--min_dollar_vol", type=float, default=MIN_DOLLAR_VOL)
    p.add_argument("--max_gap", type=int, default=MAX_GAP_DAYS)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Stage 1: per-stock article counts (one streaming pass through the 23 GB CSV)
# -----------------------------------------------------------------------------
def stream_news_per_stock_per_year(news_file: str) -> pd.DataFrame:
    """Single streaming pass over the news file. Returns DataFrame indexed by
    ticker with columns (n_2022, n_2023, n_total, n_pre_2022).

    We use a defaultdict to accumulate per-(stock, year) counts then pivot.
    """
    print(f"[stage1] streaming {news_file} ...", flush=True)
    counts: dict[tuple[str, int], int] = defaultdict(int)
    chunks_seen = 0
    rows_seen = 0
    for chunk in pd.read_csv(news_file, chunksize=500_000,
                              usecols=["Date", "Stock_symbol"],
                              dtype={"Stock_symbol": str}):
        chunks_seen += 1
        chunk["Date"] = pd.to_datetime(chunk["Date"], errors="coerce", utc=True)
        chunk = chunk.dropna(subset=["Date", "Stock_symbol"])
        chunk["year"] = chunk["Date"].dt.year
        # Some stocks have multiple uppercase/lowercase variants; normalise.
        chunk["Stock_symbol"] = chunk["Stock_symbol"].str.strip().str.upper()
        gb = chunk.groupby(["Stock_symbol", "year"]).size()
        for (sym, yr), n in gb.items():
            counts[(sym, int(yr))] += int(n)
        rows_seen += len(chunk)
        if chunks_seen % 5 == 0:
            print(f"  chunk {chunks_seen}: cum_rows={rows_seen:,}  "
                  f"unique_stocks={len({s for s, _ in counts})}",
                  flush=True)

    # Pivot
    rows = []
    by_stock = defaultdict(dict)
    for (sym, yr), n in counts.items():
        by_stock[sym][yr] = n
    for sym, yrs in by_stock.items():
        rows.append({
            "ticker": sym,
            "n_2022": yrs.get(2022, 0),
            "n_2023": yrs.get(2023, 0),
            "n_pre_2022": sum(n for y, n in yrs.items() if y < 2022),
            "n_total": sum(yrs.values()),
        })
    df = pd.DataFrame(rows).sort_values("n_total", ascending=False)
    print(f"[stage1] done. {len(df)} unique tickers with news.", flush=True)
    return df


# -----------------------------------------------------------------------------
# Stage 2: per-stock price diagnostics
# -----------------------------------------------------------------------------
def price_diag_one(path: str) -> dict:
    """Return diagnostics for one price file, or None if disqualifying."""
    try:
        df = pd.read_csv(path, usecols=["date", "open", "high", "low",
                                         "close", "volume"])
    except Exception as e:
        return {"error": f"read_fail: {e}"}
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if len(df) < 100:
        return {"error": "too_short"}

    px_min, px_max = df["date"].min(), df["date"].max()

    # C1: price coverage span
    if px_min > pd.Timestamp(PRICE_START, tz="UTC"):
        return {"error": "starts_too_late",
                "price_start": str(px_min.date())}
    if px_max < pd.Timestamp(PRICE_END, tz="UTC") - pd.Timedelta(days=14):
        return {"error": "ends_too_early",
                "price_end": str(px_max.date())}

    # Restrict to in-window
    df = df[(df["date"] >= pd.Timestamp(PRICE_START, tz="UTC"))
            & (df["date"] <= pd.Timestamp(PRICE_END, tz="UTC"))]
    if len(df) < 1000:
        return {"error": "in_window_too_short"}

    # C4: gap test
    diff = df["date"].diff().dropna().dt.days
    max_gap = int(diff.max()) if len(diff) else 99
    # Trading-day gap: weekends produce 3-day gaps; we tolerate up to MAX_GAP_DAYS.

    # C3: liquidity in TEST year (median daily $ volume)
    test_df = df[df["date"].dt.year == TEST_YEAR]
    if len(test_df) < 100:
        return {"error": "no_test_data"}
    dollar_vol = test_df["close"] * test_df["volume"]
    median_dvol = float(dollar_vol.median())

    # zero-volume / low-quality test (all closes equal -> bad data)
    if test_df["close"].std() < 1e-3 * test_df["close"].mean():
        return {"error": "constant_test_close"}

    return {
        "price_start": str(px_min.date()),
        "price_end": str(px_max.date()),
        "n_price_rows": int(len(df)),
        "max_gap_days": max_gap,
        "median_dollar_vol_2023": median_dvol,
        "close_test_mean": float(test_df["close"].mean()),
    }


def price_diagnostics_all(prices_dir: str, candidate_tickers: set[str]) -> pd.DataFrame:
    """Compute price diagnostics for every candidate ticker."""
    print(f"[stage2] price diagnostics for {len(candidate_tickers)} candidates ...",
          flush=True)
    rows = []
    files = glob(os.path.join(prices_dir, "*.csv"))
    sym_to_path = {os.path.splitext(os.path.basename(f))[0].upper(): f for f in files}
    found = 0
    for i, sym in enumerate(sorted(candidate_tickers)):
        path = sym_to_path.get(sym)
        if path is None:
            rows.append({"ticker": sym, "error": "no_price_file"})
            continue
        d = price_diag_one(path)
        d["ticker"] = sym
        rows.append(d)
        if d.get("error") is None:
            found += 1
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(candidate_tickers)}  passed_so_far={found}",
                  flush=True)
    df = pd.DataFrame(rows)
    print(f"[stage2] done. {found} tickers passed price filter.", flush=True)
    return df


# -----------------------------------------------------------------------------
# Stage 3: combine + tier + select
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    news_df = stream_news_per_stock_per_year(args.news_file)
    news_df.to_csv(os.path.join(args.out_dir, "_news_per_stock.csv"), index=False)

    # Apply news thresholds
    news_pass = news_df[
        (news_df["n_2022"] >= args.news_val_min)
        & (news_df["n_2023"] >= args.news_test_min)
    ].copy()
    print(f"[stage3] {len(news_pass)} tickers pass news thresholds "
          f"(>= {args.news_val_min} in 2022 AND >= {args.news_test_min} in 2023)",
          flush=True)

    # Stage 2 only on news-passing candidates (saves work)
    candidates = set(news_pass["ticker"])
    price_df = price_diagnostics_all(args.prices_dir, candidates)
    price_df.to_csv(os.path.join(args.out_dir, "_price_diag.csv"), index=False)

    price_pass = price_df[
        (price_df["error"].isna())
        & (price_df["max_gap_days"] <= args.max_gap)
        & (price_df["median_dollar_vol_2023"] >= args.min_dollar_vol)
    ]
    print(f"[stage3] {len(price_pass)} tickers pass price+liquidity filters",
          flush=True)

    # Merge and tier
    merged = news_pass.merge(price_pass, on="ticker", how="inner")
    if len(merged) == 0:
        print("ERROR: zero tickers passed all filters.", file=sys.stderr)
        sys.exit(1)

    # Tier by news_2023 (the test-year coverage that actually matters most)
    q = merged["n_2023"].quantile([0.25, 0.5, 0.75]).to_dict()
    def tier_of(n):
        if n >= q[0.75]:
            return "STRONG"
        if n >= q[0.25]:
            return "MEDIUM"
        return "WEAK"
    merged["tier"] = merged["n_2023"].apply(tier_of)

    # Final selection: rank by composite score (news_2023 + news_2022 + log_dvol)
    # to break ties on equally-covered stocks, prefer ones with higher liquidity.
    merged["composite_score"] = (
        np.log1p(merged["n_2023"])
        + 0.5 * np.log1p(merged["n_2022"])
        + 0.2 * np.log10(merged["median_dollar_vol_2023"].clip(lower=1))
    )
    merged = merged.sort_values("composite_score", ascending=False).reset_index(drop=True)

    # Save full universe
    full_path = os.path.join(args.out_dir, "universe_v1.csv")
    merged.to_csv(full_path, index=False)
    print(f"[stage3] universe_v1.csv: {len(merged)} stocks  -> {full_path}",
          flush=True)

    # Top-N main universe
    main = merged.head(args.target_n).copy()
    main_path = os.path.join(args.out_dir, "universe_main.csv")
    main.to_csv(main_path, index=False)
    print(f"[stage3] universe_main.csv: {len(main)} stocks (top by composite)  "
          f"-> {main_path}", flush=True)

    # Sequential-vs-global subset: 50 from STRONG tier of the main, randomly
    # chosen with seed=2026 for reproducibility.
    strong_pool = main[main["tier"] == "STRONG"]
    rng = np.random.default_rng(2026)
    if len(strong_pool) >= args.seqvsglob_n:
        idx = rng.choice(len(strong_pool), size=args.seqvsglob_n, replace=False)
        seqvsglob = strong_pool.iloc[idx].copy()
    else:
        # If STRONG tier too small, supplement from MEDIUM
        more = main[main["tier"] != "STRONG"].head(
            args.seqvsglob_n - len(strong_pool))
        seqvsglob = pd.concat([strong_pool, more])
    seqvsglob = seqvsglob.sort_values("ticker").reset_index(drop=True)
    seq_path = os.path.join(args.out_dir, "universe_seqvsglob.csv")
    seqvsglob.to_csv(seq_path, index=False)
    print(f"[stage3] universe_seqvsglob.csv: {len(seqvsglob)} stocks  "
          f"-> {seq_path}", flush=True)

    # Summary report
    print()
    print("==================== SUMMARY ====================")
    print(f"Total candidates with news: {len(news_df):,}")
    print(f"Pass news thresholds:       {len(news_pass):,}")
    print(f"Pass price+liquidity:       {len(price_pass):,}")
    print(f"Final universe (intersect): {len(merged):,}")
    print(f"  STRONG tier:              {(merged.tier=='STRONG').sum():,}")
    print(f"  MEDIUM tier:              {(merged.tier=='MEDIUM').sum():,}")
    print(f"  WEAK tier:                {(merged.tier=='WEAK').sum():,}")
    print()
    print(f"Main universe (top {args.target_n}): {len(main)} stocks")
    print(f"  median news_2023:        {int(main['n_2023'].median()):,}")
    print(f"  median news_2022:        {int(main['n_2022'].median()):,}")
    print(f"  median dollar-vol 2023:  ${main['median_dollar_vol_2023'].median()/1e6:.1f}M")
    print()
    print(f"Sequential-vs-Global subset: {len(seqvsglob)} stocks (STRONG tier)")
    print(f"Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
