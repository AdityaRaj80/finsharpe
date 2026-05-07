"""FinBERT sentiment scoring optimised for HPC H100 GPU.

Reads the pre-filtered news CSV (output of `prefilter_news_to_universe.py`)
and runs ProsusAI/finbert with batch=256 (H100 has 80 GB VRAM, so we go
4x larger than the local RTX 3060 batch of 64).

Output: per-stock CSV in `<out_dir>/<TICKER>_finbert.csv` with columns:
    Date, Stock_symbol, Article_title, Url, Publisher,
    text_used, n_tokens,
    positive_prob, negative_prob, neutral_prob,
    composite_score, scaled_sentiment

Audit invariants on every 4096-article block (FAIL FAST):
    1. positive_prob.std() > 1e-3
    2. probabilities sum to ~1
    3. uniform-prior fraction < 5%

The script is identical to the local finbert_score.py in semantics, but:
- larger batches (256 vs 64)
- writes to a configurable out_dir (so SLURM can put outputs in $TMPDIR)
- reads the pre-filtered CSV instead of streaming the full 23 GB
- prints throughput stats every 10 chunks for HPC log visibility

Expected wall-clock on H100 80GB:
    300-stock filtered subset ≈ 1.5M articles
    Throughput ≈ 1500-2500 articles/sec at batch=256
    Total time ≈ 10-20 minutes
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=FutureWarning)

FINBERT_MODEL = "ProsusAI/finbert"
MAX_LEN = 256


def load_finbert(device, hf_cache_dir=None):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print(f"[load] {FINBERT_MODEL} on {device} ...", flush=True)
    kw = {"cache_dir": hf_cache_dir} if hf_cache_dir else {}
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL, **kw)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL, **kw).to(device)
    model.eval()
    id2label = {i: model.config.id2label[i].lower() for i in range(3)}
    print(f"[load] id2label = {id2label}", flush=True)
    return tokenizer, model, id2label


def select_text(row):
    body = row.get("Article")
    title = row.get("Article_title")
    if isinstance(body, str) and len(body.strip()) > 50:
        return body, "article"
    if isinstance(title, str) and len(title.strip()) > 0:
        return title, "title"
    return "", "empty"


def score_batch(texts, tokenizer, model, device):
    enc = tokenizer(texts, return_tensors="pt", truncation=True,
                     max_length=MAX_LEN, padding=True)
    enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(**enc).logits
    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    n_tokens = enc["input_ids"].shape[1]
    return probs, n_tokens


def audit_block(probs, label_to_idx, block_id):
    pos = probs[:, label_to_idx["positive"]]
    neg = probs[:, label_to_idx["negative"]]
    neu = probs[:, label_to_idx["neutral"]]
    sums = pos + neg + neu
    if (np.abs(sums - 1.0) > 1e-3).any():
        raise RuntimeError(
            f"[audit FAIL block#{block_id}] probs don't sum to 1 "
            f"(max dev {np.abs(sums-1).max():.4f})")
    if pos.std() < 1e-3:
        raise RuntimeError(
            f"[audit FAIL block#{block_id}] positive_prob.std()={pos.std():.6f} < 1e-3 "
            f"(v0 sentinel — uniform-output bug)")
    uniform = ((np.abs(pos - 1/3) < 0.01) & (np.abs(neg - 1/3) < 0.01)
               & (np.abs(neu - 1/3) < 0.01)).mean()
    if uniform > 0.05:
        raise RuntimeError(
            f"[audit FAIL block#{block_id}] {100*uniform:.1f}% uniform-prior rows")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--news_file", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--audit_every", type=int, default=4096)
    p.add_argument("--hf_cache_dir", default=None,
                   help="Pre-downloaded HF cache dir (HPC nodes may lack net access).")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not torch.cuda.is_available():
        sys.exit("ERROR: This script is HPC GPU-only. Run finbert_score.py for local CPU.")
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)

    tokenizer, model, id2label = load_finbert(device, hf_cache_dir=args.hf_cache_dir)
    label_to_idx = {v: k for k, v in id2label.items()}

    print(f"Reading filtered news from {args.news_file} ...", flush=True)
    df = pd.read_csv(args.news_file, dtype={"Stock_symbol": str})
    df["Stock_symbol"] = df["Stock_symbol"].str.upper()
    print(f"  {len(df):,} articles, {df['Stock_symbol'].nunique()} unique tickers",
          flush=True)

    # Build text + tracking
    texts = []
    sources = []
    for _, row in df.iterrows():
        t, src = select_text(row)
        texts.append(t); sources.append(src)
    df["text_used"] = sources

    # Drop empty texts
    keep_mask = np.array([len(t) > 0 for t in texts])
    df = df[keep_mask].reset_index(drop=True)
    texts = [t for t, k in zip(texts, keep_mask) if k]
    print(f"  after empty-text drop: {len(df):,} articles", flush=True)

    # Run inference in batches
    n = len(texts)
    pos = np.zeros(n, dtype=np.float32)
    neg = np.zeros(n, dtype=np.float32)
    neu = np.zeros(n, dtype=np.float32)
    n_tokens_arr = np.zeros(n, dtype=np.int32)

    t0 = time.time()
    audit_buffer = []
    block_id = 0
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        batch_texts = texts[start:end]
        probs, ntok = score_batch(batch_texts, tokenizer, model, device)
        pos[start:end] = probs[:, label_to_idx["positive"]]
        neg[start:end] = probs[:, label_to_idx["negative"]]
        neu[start:end] = probs[:, label_to_idx["neutral"]]
        n_tokens_arr[start:end] = ntok
        audit_buffer.append(probs)

        # Audit every audit_every articles
        if sum(len(b) for b in audit_buffer) >= args.audit_every:
            block = np.concatenate(audit_buffer)
            audit_block(block, label_to_idx, block_id)
            block_id += 1
            audit_buffer = []

        # Throughput logging
        if (start // args.batch_size) % 50 == 0:
            elapsed = time.time() - t0
            tps = (end / max(elapsed, 1e-3))
            eta = (n - end) / max(tps, 1)
            print(f"  {end:,}/{n:,}  tps={tps:.0f}  ETA={eta/60:.1f}min", flush=True)

    # Final audit
    if audit_buffer:
        block = np.concatenate(audit_buffer)
        audit_block(block, label_to_idx, block_id)

    elapsed = time.time() - t0
    print(f"\n[done] {n:,} articles in {elapsed/60:.1f} min "
          f"({n/elapsed:.0f} articles/sec)", flush=True)

    # Build output frame
    df["positive_prob"] = pos
    df["negative_prob"] = neg
    df["neutral_prob"] = neu
    df["composite_score"] = pos - neg                          # in [-1, +1]
    df["scaled_sentiment"] = (df["composite_score"] + 1) / 2   # in [0, 1]
    df["n_tokens"] = n_tokens_arr

    # Write per-stock CSVs
    written_n = 0
    print("Writing per-stock CSVs ...", flush=True)
    for sym, sub in df.groupby("Stock_symbol"):
        sub = sub.sort_values("Date")
        out_path = os.path.join(args.out_dir, f"{sym}_finbert.csv")
        sub.to_csv(out_path, index=False)
        written_n += 1
    print(f"  wrote {written_n} per-stock CSVs to {args.out_dir}", flush=True)

    # Also write a flat manifest
    manifest = df.groupby("Stock_symbol").agg(
        n_articles=("Date", "count"),
        first_date=("Date", "min"),
        last_date=("Date", "max"),
        mean_pos=("positive_prob", "mean"),
        std_pos=("positive_prob", "std"),
        mean_composite=("composite_score", "mean"),
    ).reset_index()
    manifest_path = os.path.join(args.out_dir, "_manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"  manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
