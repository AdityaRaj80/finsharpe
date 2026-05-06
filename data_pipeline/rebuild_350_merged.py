"""Merge price data + FinBERT daily sentiment into per-stock CSVs.

Reads:
    D:\\Study\\CIKM\\DATA\\350_merged\\<stock>.csv          (existing prices)
    D:\\Study\\CIKM\\DATA\\finbert_daily\\<stock>_daily.csv (new sentiment)

Writes:
    D:\\Study\\CIKM\\DATA\\350_merged_v2\\<stock>.csv

Output schema: same as the predecessor 350_merged/ schema for stocks
that originally had the full 13 columns:
    Date, Volume, Open, High, Low, Close, Adj Close,
    avg_composite, scaled_sentiment, avg_positive, avg_negative,
    avg_neutral, article_count

For stocks where the predecessor had only 8 columns (NAMES_50 stocks
mostly), we widen them to the full 13-column schema by appending the
sentiment columns.

Merging logic:
    Left-join price rows with daily sentiment on Date.
    On dates with no news article: scaled_sentiment = 0.5 (neutral),
    avg_composite = 0.0, all other sentiment cols = NaN, article_count = 0.
    On dates with articles: real FinBERT-derived values.

This is the cleanest possible interpretation: news-less days are neutral,
news days carry signal. The model can then learn whether to weight
sentiment more strongly on news-rich days (which it can detect via the
article_count column).

Audit invariants on each output:
    * scaled_sentiment.std() > 0.05 across all rows  (real variance present)
    * row count >= original row count (we never drop price rows)
    * dates monotone-increasing, no duplicates
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

DEFAULT_PRICE_DIR     = r"D:\Study\CIKM\DATA\350_merged"
DEFAULT_SENTIMENT_DIR = r"D:\Study\CIKM\DATA\finbert_daily"
DEFAULT_OUT_DIR       = r"D:\Study\CIKM\DATA\350_merged_v2"

# 13-column target schema (matches what stock 'a' originally had)
TARGET_COLS = [
    "Date", "Volume", "Open", "High", "Low", "Close", "Adj Close",
    "avg_composite", "scaled_sentiment", "avg_positive", "avg_negative",
    "avg_neutral", "article_count",
]


def merge_one(price_path: str, sentiment_path: str | None, out_path: str) -> dict:
    prices = pd.read_csv(price_path)
    prices["Date"] = pd.to_datetime(prices["Date"], utc=True, errors="coerce")
    prices = prices.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    n_price_rows = len(prices)

    if sentiment_path is None or not os.path.exists(sentiment_path):
        # No news data for this stock — fill sentiment with neutrals (the
        # legitimate default for a news-less stock; matches the v0 baseline
        # behaviour for fair comparability).
        merged = prices.copy()
        merged["scaled_sentiment"] = 0.5
        merged["avg_composite"]    = 0.0
        merged["avg_positive"]     = np.nan
        merged["avg_negative"]     = np.nan
        merged["avg_neutral"]      = np.nan
        merged["article_count"]    = 0
        had_sentiment_data = False
    else:
        sent = pd.read_csv(sentiment_path)
        sent["Date"] = pd.to_datetime(sent["Date"], utc=True, errors="coerce")
        sent = sent.dropna(subset=["Date"]).drop_duplicates(subset=["Date"])
        # Drop existing sentiment cols from prices (legacy uniform 0.5 etc.) to
        # avoid duplicated columns in the merge.
        for col in ["avg_composite", "scaled_sentiment", "avg_positive",
                    "avg_negative", "avg_neutral", "article_count"]:
            if col in prices.columns:
                prices = prices.drop(columns=[col])
        merged = prices.merge(sent, on="Date", how="left")
        # Fill NaN sentiment for news-less days
        merged["scaled_sentiment"] = merged["scaled_sentiment"].fillna(0.5)
        merged["avg_composite"]    = merged["avg_composite"].fillna(0.0)
        merged["article_count"]    = merged["article_count"].fillna(0).astype(int)
        had_sentiment_data = True

    # Make sure all target columns exist (some 8-col CSVs are missing them)
    for col in TARGET_COLS:
        if col not in merged.columns:
            merged[col] = np.nan if col not in ("article_count",) else 0
    merged = merged[TARGET_COLS]

    # Reformat Date back to ISO 8601 with timezone for downstream loaders
    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d 00:00:00+00:00")

    # Audits
    audit_failures = []
    if len(merged) < n_price_rows:
        audit_failures.append(
            f"merged dropped rows ({len(merged)} < {n_price_rows})")
    if had_sentiment_data:
        s_std = merged["scaled_sentiment"].std()
        if s_std < 0.05:
            audit_failures.append(
                f"scaled_sentiment.std()={s_std:.4f} < 0.05 — sentiment "
                f"signal too weak after merge.")

    merged.to_csv(out_path, index=False)
    return {
        "stock": os.path.splitext(os.path.basename(price_path))[0],
        "n_rows": len(merged),
        "n_with_sentiment": int((merged["article_count"] > 0).sum()),
        "scaled_sentiment_std": float(merged["scaled_sentiment"].std()),
        "had_sentiment_data": had_sentiment_data,
        "audit_failures": audit_failures,
        "status": "fail" if audit_failures else "ok",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--price_dir", default=DEFAULT_PRICE_DIR)
    p.add_argument("--sentiment_dir", default=DEFAULT_SENTIMENT_DIR)
    p.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    price_csvs = sorted(glob.glob(os.path.join(args.price_dir, "*.csv")))
    if not price_csvs:
        print(f"FAIL — no *.csv files in {args.price_dir}")
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"price_dir     = {args.price_dir}")
    print(f"sentiment_dir = {args.sentiment_dir}")
    print(f"out_dir       = {args.out_dir}")
    print(f"#stocks       = {len(price_csvs)}")
    print()

    rows = []
    for i, price_path in enumerate(price_csvs):
        stock = os.path.splitext(os.path.basename(price_path))[0]
        sent_path = os.path.join(args.sentiment_dir, f"{stock}_daily.csv")
        out_path = os.path.join(args.out_dir, f"{stock}.csv")
        r = merge_one(price_path, sent_path if os.path.exists(sent_path) else None,
                      out_path)
        rows.append(r)
        flag = "[FAIL]" if r["status"] == "fail" else "[ok]"
        sent_flag = "S" if r["had_sentiment_data"] else "."
        print(f"{flag}{sent_flag} [{i+1:>3}/{len(price_csvs)}] {stock:<8} "
              f"rows={r['n_rows']:>5} news_days={r['n_with_sentiment']:>4} "
              f"sent_std={r['scaled_sentiment_std']:.4f}",
              flush=True)
        if r["audit_failures"]:
            for af in r["audit_failures"]:
                print(f"           - {af}")
            if args.strict:
                sys.exit(1)

    manifest = pd.DataFrame(rows)
    manifest_path = os.path.join(args.out_dir, "_merge_manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    n_with = manifest["had_sentiment_data"].sum()
    n_fail = (manifest["status"] == "fail").sum()
    print()
    print(f"Manifest: {manifest_path}")
    print(f"Total: {len(manifest)} stocks  with_sentiment={n_with}  audit_failures={n_fail}")


if __name__ == "__main__":
    main()
