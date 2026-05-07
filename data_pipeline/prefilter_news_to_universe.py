"""Pre-filter the 23 GB FNSPID news file to only the 300 stocks in our
universe. Reduces transfer + scoring cost dramatically (typical filtered
size: 1-3 GB vs 23 GB).

Single streaming pass over the source CSV, write one merged
filtered_news.csv with all rows whose Stock_symbol is in universe_main.

Output:
    fin-sent-optimized/data/filtered_news_universe_main.csv

Schema preserved from source. Date column stripped of timezone for
smaller storage. Article column kept (truncated to 4096 chars to bound
file size for tokenizer's MAX_LEN=256 truncation later — preserves
plenty of context).

Usage:
    python data_pipeline/prefilter_news_to_universe.py
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd

DEFAULT_NEWS = r"D:\Study\FNSPID_v1\Data\Stock_news\nasdaq_exteral_data.csv"
DEFAULT_UNIVERSE = r"D:\Study\CIKM\finsharpe\data\universe_main.csv"
DEFAULT_OUT = r"D:\Study\CIKM\fin-sent-optimized\data\filtered_news_universe_main.csv"
ARTICLE_TRUNCATE = 4096   # bound article body length (FinBERT only sees first 256 tokens anyway)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--news_file", default=DEFAULT_NEWS)
    p.add_argument("--universe_file", default=DEFAULT_UNIVERSE)
    p.add_argument("--out_file", default=DEFAULT_OUT)
    p.add_argument("--chunk_rows", type=int, default=500_000)
    args = p.parse_args()

    universe = set(pd.read_csv(args.universe_file)["ticker"].astype(str).str.upper())
    print(f"Universe: {len(universe)} tickers")
    print(f"Source:   {args.news_file}")
    print(f"Output:   {args.out_file}")

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    cols_keep = ["Date", "Stock_symbol", "Article_title", "Article", "Url", "Publisher"]
    t0 = time.time()
    written = 0
    chunks = 0
    is_first = True
    for chunk in pd.read_csv(args.news_file, chunksize=args.chunk_rows,
                              usecols=cols_keep,
                              dtype={"Stock_symbol": str}):
        chunks += 1
        chunk["Stock_symbol"] = chunk["Stock_symbol"].astype(str).str.upper()
        chunk = chunk[chunk["Stock_symbol"].isin(universe)]
        if not len(chunk):
            continue

        # Truncate Article column to bound size
        if "Article" in chunk.columns:
            chunk["Article"] = chunk["Article"].astype(str).str.slice(0, ARTICLE_TRUNCATE)

        chunk.to_csv(args.out_file, mode="w" if is_first else "a",
                      index=False, header=is_first, encoding="utf-8")
        is_first = False
        written += len(chunk)
        if chunks % 5 == 0:
            elapsed = time.time() - t0
            print(f"  chunk {chunks}: written={written:,}  elapsed={elapsed:.0f}s",
                  flush=True)

    elapsed = time.time() - t0
    size_mb = os.path.getsize(args.out_file) / 1e6
    print()
    print(f"Done. Wrote {written:,} articles for {len(universe)} tickers")
    print(f"Output size: {size_mb:.1f} MB")
    print(f"Elapsed: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
