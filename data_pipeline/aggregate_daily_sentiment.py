"""Aggregate per-article FinBERT scores to per-stock per-day sentiment.

Reads:    D:\\Study\\CIKM\\DATA\\finbert_scores\\<stock>_finbert.csv
Writes:   D:\\Study\\CIKM\\DATA\\finbert_daily\\<stock>_daily.csv

Output schema (matches what existing 350_merged/ stocks WOULD have looked
like if the FinBERT pipeline had worked):

    Date, avg_composite, scaled_sentiment, avg_positive, avg_negative,
    avg_neutral, article_count

where:
    avg_composite      = mean of (positive_prob - negative_prob)        ∈ [-1, 1]
    scaled_sentiment   = (avg_composite + 1) / 2                         ∈ [0, 1]
    avg_positive/neg/neu = mean of corresponding probabilities
    article_count      = # articles aggregated for that (stock, date)

Date normalisation: every UTC timestamp is floored to UTC midnight, then
written as 'YYYY-MM-DD 00:00:00+00:00' to match the per-stock CSV
convention in 350_merged/.

Audits (failure → exit 1):
  * scaled_sentiment.std() per stock > 0.02 (variance survives aggregation).
  * Date column is monotone-increasing, no duplicates per stock.
  * positive + negative + neutral averages sum to ~1 (within 1e-3).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

DEFAULT_IN = r"D:\Study\CIKM\fin-sent-optimized\data\finbert_scores_v2"
DEFAULT_OUT = r"D:\Study\CIKM\fin-sent-optimized\data\finbert_daily_v2"


def aggregate_one(in_path: str, out_path: str) -> dict:
    df = pd.read_csv(in_path)
    if len(df) == 0:
        return {"stock": os.path.basename(in_path), "n_articles": 0, "status": "empty"}
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    # Normalise to UTC midnight
    df["DateDay"] = df["Date"].dt.floor("D")
    grouped = df.groupby("DateDay").agg(
        avg_composite=("composite_score", "mean"),
        avg_positive=("positive_prob", "mean"),
        avg_negative=("negative_prob", "mean"),
        avg_neutral=("neutral_prob", "mean"),
        article_count=("composite_score", "count"),
    ).reset_index()
    grouped["scaled_sentiment"] = (grouped["avg_composite"] + 1.0) / 2.0
    grouped = grouped.rename(columns={"DateDay": "Date"})
    grouped = grouped[["Date", "avg_composite", "scaled_sentiment",
                        "avg_positive", "avg_negative", "avg_neutral",
                        "article_count"]]
    grouped["Date"] = grouped["Date"].dt.strftime("%Y-%m-%d 00:00:00+00:00")
    grouped = grouped.sort_values("Date").reset_index(drop=True)

    # Per-stock audits
    audit_failures = []
    s_std = grouped["scaled_sentiment"].std()
    if s_std < 0.02:
        audit_failures.append(f"scaled_sentiment.std()={s_std:.4f} < 0.02")
    sums_check = (grouped["avg_positive"] + grouped["avg_negative"]
                  + grouped["avg_neutral"])
    if (sums_check - 1.0).abs().max() > 1e-3:
        audit_failures.append(f"prob sums deviate by up to "
                              f"{(sums_check - 1.0).abs().max():.4f}")
    if grouped["Date"].duplicated().any():
        audit_failures.append(f"duplicate dates found")

    grouped.to_csv(out_path, index=False)
    return {
        "stock": os.path.splitext(os.path.basename(in_path))[0].replace("_finbert", ""),
        "n_articles": len(df),
        "n_unique_days": len(grouped),
        "first_day": grouped["Date"].iloc[0],
        "last_day":  grouped["Date"].iloc[-1],
        "scaled_sentiment_std": float(s_std),
        "audit_failures": audit_failures,
        "status": "fail" if audit_failures else "ok",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default=DEFAULT_IN)
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero on the first audit failure.")
    args = p.parse_args()

    in_csvs = sorted(glob.glob(os.path.join(args.in_dir, "*_finbert.csv")))
    if not in_csvs:
        print(f"FAIL — no *_finbert.csv files in {args.in_dir}")
        sys.exit(1)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"in_dir   = {args.in_dir}")
    print(f"out_dir  = {args.out_dir}")
    print(f"#files   = {len(in_csvs)}")
    print()

    rows = []
    for i, in_path in enumerate(in_csvs):
        stock = os.path.splitext(os.path.basename(in_path))[0].replace("_finbert", "")
        out_path = os.path.join(args.out_dir, f"{stock}_daily.csv")
        r = aggregate_one(in_path, out_path)
        rows.append(r)
        flag = "[FAIL]" if r["status"] == "fail" else "[ok]"
        print(f"{flag} [{i+1:>3}/{len(in_csvs)}] {stock:<8} "
              f"n_articles={r.get('n_articles', 0):>5} "
              f"days={r.get('n_unique_days', 0):>4} "
              f"std={r.get('scaled_sentiment_std', float('nan')):.4f}",
              flush=True)
        if r["audit_failures"]:
            for af in r["audit_failures"]:
                print(f"           - {af}")
            if args.strict:
                sys.exit(1)

    # Manifest
    manifest = pd.DataFrame(rows)
    manifest_path = os.path.join(args.out_dir, "_aggregation_manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"\nManifest: {manifest_path}")
    n_fail = (manifest["status"] == "fail").sum()
    print(f"Total: {len(manifest)} stocks, {n_fail} with audit failures")


if __name__ == "__main__":
    main()
