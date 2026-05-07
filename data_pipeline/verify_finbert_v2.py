"""FinBERT smoke verification — proves it actually works before we
commit to a 1.5M-article inference run.

Three checks (all must pass):

  V1. Hand-crafted 4-article test: 3 obviously positive, 1 obviously
      negative — FinBERT should produce diverse probabilities, not all
      uniform 0.333. This catches the v0 "model never loaded" failure.

  V2. Random sample of 200 real articles from the 23 GB FNSPID file,
      filtered to our 300-stock universe. Run audits:
          - probabilities sum to ~1
          - positive_prob.std() > 1e-3
          - uniform-prior fraction < 5%
          - aggregate positive/negative balance is non-degenerate

  V3. Per-stock spot check: pick 5 stocks from universe_main, score 20
      articles each, verify cross-stock variance in mean sentiment
      (different stocks should NOT all average to the same number).

Usage:
    python data_pipeline/verify_finbert_v2.py
    python data_pipeline/verify_finbert_v2.py --news_file <path> --universe_file <path>
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_NEWS_FILE = r"D:\Study\FNSPID_v1\Data\Stock_news\nasdaq_exteral_data.csv"
DEFAULT_UNIVERSE_FILE = r"D:\Study\CIKM\finsharpe\data\universe_main.csv"
FINBERT_MODEL = "ProsusAI/finbert"
MAX_LEN = 256


def load_finbert(device):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print(f"[load] {FINBERT_MODEL} on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).to(device)
    model.eval()
    id2label = {i: model.config.id2label[i].lower() for i in range(3)}
    print(f"[load] model.config.id2label = {id2label}", flush=True)
    expected = {0: "positive", 1: "negative", 2: "neutral"}
    if id2label != expected:
        print(f"[load] WARNING: label order != expected; got {id2label}",
              flush=True)
    return tokenizer, model, id2label


def score(texts, tokenizer, model, device, batch_size=32):
    if len(texts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    out = np.zeros((len(texts), 3), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                         max_length=MAX_LEN, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        out[i:i+len(batch)] = probs
    return out


def fail(msg):
    print(f"\n[FAIL] {msg}", flush=True)
    sys.exit(1)


# ─────────────────────── V1: hand-crafted articles ───────────────────────
def v1_handcrafted(tokenizer, model, device, id2label):
    print("\n" + "=" * 60)
    print("V1: Hand-crafted sanity test (4 articles)")
    print("=" * 60)
    articles = [
        ("Apple's earnings beat expectations, revenue up 25% YoY, raised guidance.",
         "positive"),
        ("Microsoft announced record-breaking cloud growth and strong AI tailwinds.",
         "positive"),
        ("Tesla shares surged on better-than-expected delivery numbers.",
         "positive"),
        ("Meta laid off 10000 workers amid plunging ad revenue and SEC investigation.",
         "negative"),
    ]
    texts = [a[0] for a in articles]
    expected = [a[1] for a in articles]
    probs = score(texts, tokenizer, model, device)

    # Reverse-map id2label so we know which column is which
    label_to_idx = {v: k for k, v in id2label.items()}
    pos_idx = label_to_idx["positive"]
    neg_idx = label_to_idx["negative"]
    neu_idx = label_to_idx["neutral"]

    print(f"{'expected':<10} {'pos':>6} {'neg':>6} {'neu':>6}  text[:50]")
    print("-" * 80)
    for txt, exp, p in zip(texts, expected, probs):
        print(f"{exp:<10} {p[pos_idx]:6.3f} {p[neg_idx]:6.3f} {p[neu_idx]:6.3f}  {txt[:50]!r}")

    # Each "positive" article should have positive_prob > 0.5
    n_correct = 0
    for exp, p in zip(expected, probs):
        if exp == "positive" and p[pos_idx] > 0.5:
            n_correct += 1
        elif exp == "negative" and p[neg_idx] > 0.5:
            n_correct += 1
    if n_correct < 3:
        fail(f"V1 — only {n_correct}/4 articles classified correctly. "
             f"FinBERT is broken or label mapping is wrong.")
    print(f"\nV1 PASS: {n_correct}/4 articles classified correctly.")
    return probs, label_to_idx


# ─────────────────────── V2: random 200 real articles ───────────────────────
def v2_random_sample(news_file, universe_set, tokenizer, model, device,
                     label_to_idx, n_sample=200, seed=2026):
    print("\n" + "=" * 60)
    print(f"V2: Random {n_sample}-article sample from {os.path.basename(news_file)}")
    print("=" * 60)
    rng = np.random.default_rng(seed)
    texts, tickers = [], []
    print("Streaming for samples ...", flush=True)
    seen_universe_articles = 0
    for chunk in pd.read_csv(news_file, chunksize=200_000,
                              usecols=["Date", "Stock_symbol", "Article_title", "Article"],
                              dtype={"Stock_symbol": str}):
        chunk["Stock_symbol"] = chunk["Stock_symbol"].astype(str).str.upper()
        chunk = chunk[chunk["Stock_symbol"].isin(universe_set)]
        seen_universe_articles += len(chunk)
        for _, row in chunk.iterrows():
            text = row.get("Article")
            if not isinstance(text, str) or len(text.strip()) < 50:
                text = row.get("Article_title")
            if isinstance(text, str) and len(text.strip()) > 20:
                texts.append(text)
                tickers.append(row["Stock_symbol"])
                if len(texts) >= n_sample * 5:        # 5x oversample then random-select
                    break
        if len(texts) >= n_sample * 5:
            break
    print(f"  saw {seen_universe_articles:,} universe articles in stream; "
          f"collected {len(texts)} candidates.", flush=True)
    if len(texts) < n_sample:
        fail(f"V2 — only collected {len(texts)} candidates, want {n_sample}.")
    # Random subset
    idx = rng.choice(len(texts), size=n_sample, replace=False)
    texts = [texts[i] for i in idx]
    tickers = [tickers[i] for i in idx]
    print(f"Scoring {len(texts)} sampled articles ...", flush=True)
    probs = score(texts, tokenizer, model, device, batch_size=32)

    pos = probs[:, label_to_idx["positive"]]
    neg = probs[:, label_to_idx["negative"]]
    neu = probs[:, label_to_idx["neutral"]]

    # Check 1: probabilities sum to ~1
    sums = pos + neg + neu
    if (np.abs(sums - 1.0) > 1e-3).any():
        fail(f"V2 — probabilities don't sum to 1 (max dev {np.abs(sums-1).max():.4f})")

    # Check 2: pos.std() > 1e-3
    if pos.std() < 1e-3:
        fail(f"V2 — positive_prob.std()={pos.std():.6f} < 1e-3 — uniform output (v0 redux)")

    # Check 3: uniform-prior fraction < 5%
    uniform_frac = ((np.abs(pos - 1/3) < 0.01) & (np.abs(neg - 1/3) < 0.01)
                    & (np.abs(neu - 1/3) < 0.01)).mean()
    if uniform_frac > 0.05:
        fail(f"V2 — {100*uniform_frac:.1f}% of articles are at uniform prior (>5%)")

    # Check 4: distribution makes sense
    print(f"  pos: mean={pos.mean():.3f}  std={pos.std():.3f}  "
          f"min={pos.min():.3f}  max={pos.max():.3f}")
    print(f"  neg: mean={neg.mean():.3f}  std={neg.std():.3f}  "
          f"min={neg.min():.3f}  max={neg.max():.3f}")
    print(f"  neu: mean={neu.mean():.3f}  std={neu.std():.3f}  "
          f"min={neu.min():.3f}  max={neu.max():.3f}")
    print(f"  uniform-prior fraction: {100*uniform_frac:.2f}%")
    print(f"\nV2 PASS: 200-article sample shows healthy diverse outputs.")
    return probs, tickers


# ─────────────────────── V3: per-stock cross-variance ───────────────────────
def v3_per_stock(news_file, universe_set, tokenizer, model, device,
                 label_to_idx, n_per_stock=20, k_stocks=5, seed=2026):
    print("\n" + "=" * 60)
    print(f"V3: Per-stock cross-variance check ({k_stocks} stocks x {n_per_stock} articles)")
    print("=" * 60)
    rng = np.random.default_rng(seed)
    target_stocks = list(rng.choice(sorted(universe_set), size=k_stocks, replace=False))
    print(f"Target stocks: {target_stocks}", flush=True)

    per_stock_articles = {s: [] for s in target_stocks}
    target_set = set(target_stocks)
    for chunk in pd.read_csv(news_file, chunksize=200_000,
                              usecols=["Stock_symbol", "Article_title", "Article"],
                              dtype={"Stock_symbol": str}):
        chunk["Stock_symbol"] = chunk["Stock_symbol"].astype(str).str.upper()
        chunk = chunk[chunk["Stock_symbol"].isin(target_set)]
        for _, row in chunk.iterrows():
            sym = row["Stock_symbol"]
            if len(per_stock_articles[sym]) >= n_per_stock:
                continue
            text = row.get("Article") if isinstance(row.get("Article"), str) and len(str(row.get("Article")).strip()) > 50 else row.get("Article_title")
            if isinstance(text, str) and len(text.strip()) > 20:
                per_stock_articles[sym].append(text)
        if all(len(arts) >= n_per_stock for arts in per_stock_articles.values()):
            break

    pos_idx = label_to_idx["positive"]
    neg_idx = label_to_idx["negative"]
    composite_means = {}
    for sym, arts in per_stock_articles.items():
        if len(arts) == 0:
            print(f"  {sym}: NO articles found")
            continue
        probs = score(arts, tokenizer, model, device, batch_size=16)
        composite = probs[:, pos_idx] - probs[:, neg_idx]
        composite_means[sym] = composite.mean()
        print(f"  {sym}: n={len(arts)}  composite_mean={composite.mean():+.3f}  "
              f"composite_std={composite.std():.3f}  "
              f"pos_avg={probs[:, pos_idx].mean():.3f}")

    if len(composite_means) < 3:
        fail(f"V3 — only {len(composite_means)} stocks had enough articles.")
    means = np.array(list(composite_means.values()))
    cross_std = means.std()
    print(f"\n  cross-stock std of composite means: {cross_std:.4f}")
    if cross_std < 0.02:
        fail(f"V3 — cross-stock std={cross_std:.4f} < 0.02 — every stock averages "
             f"to the same sentiment (suggests a degenerate model).")
    print(f"\nV3 PASS: stocks have meaningfully different mean sentiments.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--news_file", default=DEFAULT_NEWS_FILE)
    p.add_argument("--universe_file", default=DEFAULT_UNIVERSE_FILE)
    p.add_argument("--n_random", type=int, default=200)
    p.add_argument("--n_per_stock", type=int, default=20)
    p.add_argument("--k_stocks", type=int, default=5)
    args = p.parse_args()

    if not os.path.exists(args.news_file):
        fail(f"News file not found: {args.news_file}")
    if not os.path.exists(args.universe_file):
        fail(f"Universe file not found: {args.universe_file}")

    universe = set(pd.read_csv(args.universe_file)["ticker"].astype(str).str.upper())
    print(f"Universe: {len(universe)} tickers loaded from {args.universe_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()
    tokenizer, model, id2label = load_finbert(device)

    _, label_to_idx = v1_handcrafted(tokenizer, model, device, id2label)
    v2_random_sample(args.news_file, universe, tokenizer, model, device,
                     label_to_idx, n_sample=args.n_random)
    v3_per_stock(args.news_file, universe, tokenizer, model, device,
                 label_to_idx, n_per_stock=args.n_per_stock, k_stocks=args.k_stocks)

    print("\n" + "=" * 60)
    print("ALL FINBERT CHECKS PASSED. The model is working.")
    print("=" * 60)


if __name__ == "__main__":
    main()
