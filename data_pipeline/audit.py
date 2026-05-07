"""Audit invariants for the FinBERT data pipeline.

Single-purpose script: scan a directory tree (per-article scores OR
merged per-stock CSVs) and assert that the sentiment column actually
varies — the check that should have been there from project day-one.

Exit codes:
    0  all audits pass
    1  any audit failed (script prints which file + what value)

Audits:
  A. positive_prob.std() > 1e-3 across each per-article CSV
  B. fraction of (0.333, 0.333, 0.333) uniform-prior rows < 5% per file
  C. scaled_sentiment.std() > 0.05 in each per-stock merged CSV
  D. scaled_sentiment-vs-time correlation < 0.3 per stock (no monotone
     creep, e.g., date leakage from a sequential FinBERT run)
  E. cache (when present) has non-constant sentiment column

Usage:
    python data_pipeline/audit.py per_article  D:\\Study\\CIKM\\DATA\\finbert_scores
    python data_pipeline/audit.py per_stock    D:\\Study\\CIKM\\DATA\\350_merged_v2
    python data_pipeline/audit.py cache        D:\\Study\\CIKM\\finsharpe\\.cache\\global_scaled
    python data_pipeline/audit.py all
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import numpy as np
import pandas as pd

DEFAULT_PER_ARTICLE_DIR = r"D:\Study\CIKM\fin-sent-optimized\data\finbert_scores_v2"
DEFAULT_PER_STOCK_DIR   = r"D:\Study\CIKM\fin-sent-optimized\data\merged_v3"
DEFAULT_CACHE_DIR       = r"D:\Study\CIKM\finsharpe\.cache\global_scaled"


# ------------------------------------------------------------------
# A + B: per-article FinBERT score files
# ------------------------------------------------------------------
def audit_per_article(directory: str, fail_fast: bool = True) -> int:
    failures = 0
    csvs = sorted(glob.glob(os.path.join(directory, "*_finbert.csv")))
    if not csvs:
        print(f"[audit:per_article] FAIL — no *_finbert.csv files in {directory}")
        return 1
    print(f"[audit:per_article] checking {len(csvs)} files in {directory}")
    for f in csvs:
        df = pd.read_csv(f, usecols=["positive_prob", "negative_prob",
                                      "neutral_prob"])
        n = len(df)
        if n == 0:
            print(f"  FAIL  {os.path.basename(f)}: empty")
            failures += 1
            continue
        pos = df["positive_prob"]
        # Audit A: std
        s = pos.std()
        if s < 1e-3:
            print(f"  FAIL  {os.path.basename(f)}: positive_prob std = {s:.6f}")
            failures += 1
            if fail_fast:
                return 1
        # Audit B: uniform-prior fraction
        uniform_mask = (
            (pos.between(0.32, 0.34)) &
            (df["negative_prob"].between(0.32, 0.34)) &
            (df["neutral_prob"].between(0.32, 0.34))
        )
        u = uniform_mask.mean()
        if u > 0.05:
            print(f"  FAIL  {os.path.basename(f)}: {100*u:.1f}% uniform-prior rows")
            failures += 1
            if fail_fast:
                return 1
    if failures == 0:
        print(f"[audit:per_article] OK ({len(csvs)} files)")
    return 1 if failures > 0 else 0


# ------------------------------------------------------------------
# C + D: per-stock merged CSVs
# ------------------------------------------------------------------
def audit_per_stock(directory: str, fail_fast: bool = True) -> int:
    failures = 0
    csvs = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not csvs:
        print(f"[audit:per_stock] FAIL — no *.csv files in {directory}")
        return 1
    print(f"[audit:per_stock] checking {len(csvs)} files in {directory}")
    bad_std, bad_corr = [], []
    for f in csvs:
        try:
            df = pd.read_csv(f, usecols=["Date", "scaled_sentiment"])
        except Exception as e:
            print(f"  WARN  {os.path.basename(f)}: cannot read sentiment columns: {e}")
            continue
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        df = df.dropna(subset=["Date", "scaled_sentiment"]).sort_values("Date")
        if len(df) < 100:
            continue
        s = df["scaled_sentiment"].std()
        # Audit C
        if s < 0.05:
            bad_std.append((os.path.basename(f), s))
        # Audit D: correlation with time (proxy by index after date-sort)
        if s > 1e-6:
            rho = np.corrcoef(np.arange(len(df)), df["scaled_sentiment"].values)[0, 1]
            if abs(rho) > 0.3:
                bad_corr.append((os.path.basename(f), rho))
    if bad_std:
        print(f"  FAIL  scaled_sentiment.std() < 0.05 in {len(bad_std)} files:")
        for name, s in bad_std[:5]:
            print(f"        {name}: std={s:.4f}")
        failures += len(bad_std)
        if fail_fast and len(bad_std) > 0:
            return 1
    if bad_corr:
        print(f"  FAIL  scaled_sentiment-vs-time |rho| > 0.3 in {len(bad_corr)} files:")
        for name, r in bad_corr[:5]:
            print(f"        {name}: rho={r:+.3f}")
        failures += len(bad_corr)
        if fail_fast and len(bad_corr) > 0:
            return 1
    if failures == 0:
        print(f"[audit:per_stock] OK ({len(csvs)} files)")
    return 1 if failures > 0 else 0


# ------------------------------------------------------------------
# E: preprocessed cache
# ------------------------------------------------------------------
def audit_cache(directory: str) -> int:
    manifest_path = os.path.join(directory, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[audit:cache] WARN — no manifest at {manifest_path}; skipping")
        return 0
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    n_features = manifest.get("n_features", 0)
    feature_names = manifest.get("feature_names", [])
    if "scaled_sentiment" not in feature_names:
        print(f"[audit:cache] WARN — cache has no scaled_sentiment column; skipping")
        return 0
    sent_idx = feature_names.index("scaled_sentiment")
    print(f"[audit:cache] checking sentiment column ({sent_idx}) in cache stocks …")
    bad = []
    for entry in manifest["stocks"][:50]:    # spot-check first 50
        arr = np.load(entry["path"], mmap_mode="r")
        col = arr[:, sent_idx]
        s = float(np.std(col))
        if s < 1e-3:
            bad.append((entry["stock"], s))
    if bad:
        print(f"  FAIL  cache sentiment std < 1e-3 in {len(bad)} stocks:")
        for name, s in bad[:5]:
            print(f"        {name}: std={s:.6f}")
        return 1
    print(f"[audit:cache] OK (50 stocks spot-checked)")
    return 0


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["per_article", "per_stock", "cache", "all"])
    p.add_argument("path", nargs="?", default=None,
                   help="Override default directory for the chosen mode.")
    p.add_argument("--lenient", action="store_true",
                   help="Don't bail on first failure; report all.")
    args = p.parse_args()
    fail_fast = not args.lenient
    rc = 0
    if args.mode in ("per_article", "all"):
        d = args.path or DEFAULT_PER_ARTICLE_DIR
        rc |= audit_per_article(d, fail_fast=fail_fast) if os.path.isdir(d) else 0
    if args.mode in ("per_stock", "all"):
        d = args.path or DEFAULT_PER_STOCK_DIR
        rc |= audit_per_stock(d, fail_fast=fail_fast) if os.path.isdir(d) else 0
    if args.mode in ("cache", "all"):
        d = args.path or DEFAULT_CACHE_DIR
        rc |= audit_cache(d) if os.path.isdir(d) else 0
    sys.exit(rc)


if __name__ == "__main__":
    main()
