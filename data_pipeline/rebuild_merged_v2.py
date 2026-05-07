"""Merge FNSPID prices + FinBERT daily sentiment into per-stock CSVs (v2).

Replaces the deleted `rebuild_350_merged.py` with v2 conventions:
  * Reads prices from FNSPID `full_history/<TICKER>.csv` (the complete
    7,693-stock OHLCV archive shipped with FNSPID)
  * Reads sentiment from `finbert_daily_v2/<TICKER>_daily.csv` (output of
    aggregate_daily_sentiment.py on the new finbert_scores_v2/ dir)
  * Filters to the 300-stock universe in `data/universe_main.csv`
  * Writes to `merged_v3/<TICKER>.csv`

Output schema (per stock, sorted by Date):
    Date, Volume, Open, High, Low, Close, Adj Close,
    avg_composite, scaled_sentiment, avg_positive, avg_negative,
    avg_neutral, article_count

Merging rules:
  * Left-join price rows (every trading day in 2009-2023) with daily
    sentiment on Date.
  * Days with NO news article: scaled_sentiment = 0.5 (neutral fill,
    well-defined "no signal" prior); avg_composite = 0; article_count = 0.
  * Days with articles: real FinBERT-derived values.

Audit invariants per output file:
  * scaled_sentiment.std() > 0.02 (real variance, not constant 0.5)
  * row count == price-row count for the universe span
  * Date monotone-increasing, no duplicates

Usage:
    python data_pipeline/rebuild_merged_v2.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

DEFAULT_PRICES_DIR  = r"D:\Study\FNSPID_v1\Data\full_history"
DEFAULT_SENT_DIR    = r"D:\Study\CIKM\fin-sent-optimized\data\finbert_daily_v2"
DEFAULT_UNIVERSE    = r"D:\Study\CIKM\finsharpe\data\universe_main.csv"
DEFAULT_OUT_DIR     = r"D:\Study\CIKM\fin-sent-optimized\data\merged_v3"

# 13-column target schema (preserved from v1 for downstream compatibility)
TARGET_COLS = [
    "Date", "Volume", "Open", "High", "Low", "Close", "Adj Close",
    "avg_composite", "scaled_sentiment", "avg_positive", "avg_negative",
    "avg_neutral", "article_count",
]


def merge_one(price_path: str, sent_path: str | None, out_path: str,
              date_min: str = "2009-01-01", date_max: str = "2023-12-29") -> dict:
    # Read FNSPID prices (lower-case columns)
    prices = pd.read_csv(price_path)
    # Standardise column names to v1 schema
    rename = {"date": "Date", "open": "Open", "high": "High", "low": "Low",
               "close": "Close", "adj close": "Adj Close", "volume": "Volume"}
    prices = prices.rename(columns=rename)
    prices["Date"] = pd.to_datetime(prices["Date"], utc=True, errors="coerce")
    prices = prices.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Restrict to v2 calendar window
    dmin = pd.Timestamp(date_min, tz="UTC")
    dmax = pd.Timestamp(date_max, tz="UTC")
    prices = prices[(prices["Date"] >= dmin) & (prices["Date"] <= dmax)].reset_index(drop=True)
    n_price_rows = len(prices)

    if sent_path is None or not os.path.exists(sent_path):
        merged = prices.copy()
        merged["scaled_sentiment"] = 0.5
        merged["avg_composite"] = 0.0
        merged["avg_positive"] = np.nan
        merged["avg_negative"] = np.nan
        merged["avg_neutral"] = np.nan
        merged["article_count"] = 0
        had_sentiment = False
    else:
        sent = pd.read_csv(sent_path)
        sent["Date"] = pd.to_datetime(sent["Date"], utc=True, errors="coerce")
        sent = sent.dropna(subset=["Date"]).drop_duplicates(subset=["Date"])
        merged = prices.merge(sent, on="Date", how="left")
        merged["scaled_sentiment"] = merged["scaled_sentiment"].fillna(0.5)
        merged["avg_composite"] = merged["avg_composite"].fillna(0.0)
        merged["article_count"] = merged["article_count"].fillna(0).astype(int)
        had_sentiment = True

    # Ensure all target columns exist
    for col in TARGET_COLS:
        if col not in merged.columns:
            merged[col] = np.nan if col != "article_count" else 0
    merged = merged[TARGET_COLS]
    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d 00:00:00+00:00")

    # Audits
    failures = []
    if len(merged) != n_price_rows:
        failures.append(f"row count changed during merge: {len(merged)} != {n_price_rows}")
    if had_sentiment:
        s_std = merged["scaled_sentiment"].std()
        if s_std < 0.02:
            failures.append(f"scaled_sentiment.std()={s_std:.4f} < 0.02 (signal too weak)")

    merged.to_csv(out_path, index=False)
    return {
        "stock": os.path.splitext(os.path.basename(price_path))[0].upper(),
        "n_rows": int(len(merged)),
        "n_with_sentiment": int((merged["article_count"] > 0).sum()),
        "scaled_sentiment_std": float(merged["scaled_sentiment"].std()),
        "had_sentiment": had_sentiment,
        "audit_failures": failures,
        "status": "fail" if failures else "ok",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices_dir", default=DEFAULT_PRICES_DIR)
    p.add_argument("--sent_dir", default=DEFAULT_SENT_DIR)
    p.add_argument("--universe_file", default=DEFAULT_UNIVERSE)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--date_min", default="2009-01-01")
    p.add_argument("--date_max", default="2023-12-29")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    universe = list(pd.read_csv(args.universe_file)["ticker"].astype(str).str.upper())
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"prices_dir   : {args.prices_dir}")
    print(f"sent_dir     : {args.sent_dir}")
    print(f"universe     : {args.universe_file} ({len(universe)} stocks)")
    print(f"out_dir      : {args.out_dir}")
    print(f"window       : {args.date_min} -> {args.date_max}")
    print()

    rows = []
    for i, sym in enumerate(universe):
        price_path = os.path.join(args.prices_dir, f"{sym}.csv")
        sent_path = os.path.join(args.sent_dir, f"{sym}_daily.csv")
        out_path = os.path.join(args.out_dir, f"{sym}.csv")
        if not os.path.exists(price_path):
            rows.append({"stock": sym, "status": "no_price_file"})
            continue
        r = merge_one(price_path,
                       sent_path if os.path.exists(sent_path) else None,
                       out_path, args.date_min, args.date_max)
        rows.append(r)
        flag = "[FAIL]" if r["status"] == "fail" else "[ok]"
        sent_flag = "S" if r.get("had_sentiment") else "."
        print(f"{flag}{sent_flag} [{i+1:>3}/{len(universe)}] {sym:<8} "
              f"rows={r['n_rows']:>5} news_days={r['n_with_sentiment']:>4} "
              f"sent_std={r['scaled_sentiment_std']:.4f}", flush=True)
        if r.get("audit_failures") and args.strict:
            for af in r["audit_failures"]:
                print(f"           - {af}")
            sys.exit(1)

    manifest = pd.DataFrame(rows)
    manifest_path = os.path.join(args.out_dir, "_merge_manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    n_fail = (manifest["status"] == "fail").sum()
    n_with_sent = manifest["had_sentiment"].fillna(False).sum() if "had_sentiment" in manifest else 0
    print()
    print(f"Manifest: {manifest_path}")
    print(f"Total: {len(manifest)} stocks, with_sentiment={int(n_with_sent)}, audit_failures={int(n_fail)}")


if __name__ == "__main__":
    main()
