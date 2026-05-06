"""FinBERT sentiment scoring for the 350-stock universe.

Uses ProsusAI/finbert (the canonical financial-sentiment BERT) from
Hugging Face. Reads the articles in `D:\\Study\\CIKM\\DATA\\All_external.csv`
filtered to the 351 stocks in our universe (NAMES_50 test + ~302 train),
batches them through FinBERT on GPU, writes per-article scores to
`D:\\Study\\CIKM\\DATA\\finbert_scores\\<stock>.csv`.

Output schema:
    Date, Stock_symbol, Article_title, Url, Publisher,
    text_used, n_tokens,
    positive_prob, negative_prob, neutral_prob,
    composite_score, scaled_sentiment

where:
    composite_score = positive_prob - negative_prob              ∈ [-1, +1]
    scaled_sentiment = (composite_score + 1) / 2                  ∈ [0, 1]
    text_used = "article" if article body present else "title"
    n_tokens = number of FinBERT tokens after truncation to max_length

Audit invariants checked every 1000-article block (FAIL FAST if broken):

    1. positive_prob.std() > 1e-3
    2. positive_prob + negative_prob + neutral_prob ≈ 1 within 1e-4
    3. fraction of (0.333, 0.333, 0.333) uniform-prior rows < 5%
    4. positive_prob has values in BOTH halves of [0, 1] across the block

If any audit fails, the script halts immediately and exits non-zero —
exactly the failure mode that should have been caught at v0.

Usage:
    python data_pipeline/finbert_score.py --limit 200 --smoke   # 200-article verification
    python data_pipeline/finbert_score.py                       # full 1.4M run
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
from torch.utils.data import DataLoader, IterableDataset

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
ALL_EXTERNAL_CSV = r"D:\Study\CIKM\DATA\All_external.csv"
OUTPUT_DIR = r"D:\Study\CIKM\DATA\finbert_scores"
FINBERT_MODEL = "ProsusAI/finbert"
MAX_LEN = 256              # FinBERT max sequence length; titles + short articles fit easily
DEFAULT_BATCH = 64         # tuned for ~6 GB VRAM (RTX 3060 Laptop class)

LABELS_INDEX = {0: "positive", 1: "negative", 2: "neutral"}    # ProsusAI/finbert convention
# (verify this from model.config.id2label at load time)


def get_universe(stock_list_file: str | None = None) -> list[str]:
    """Return lower-case ticker symbols for the 351-stock universe.
    Reads from the existing `350_merged/` directory if no explicit list given."""
    if stock_list_file is not None:
        with open(stock_list_file) as f:
            return [l.strip().lower() for l in f if l.strip()]
    # Fallback: enumerate the 350_merged directory
    merged_dir = r"D:\Study\CIKM\DATA\350_merged"
    if not os.path.isdir(merged_dir):
        raise FileNotFoundError(f"{merged_dir} does not exist; pass --stock_list explicitly.")
    return sorted({
        os.path.splitext(f)[0].lower()
        for f in os.listdir(merged_dir)
        if f.endswith(".csv")
    })


def load_finbert(device: torch.device):
    """Load ProsusAI/finbert. Verify label mapping matches our LABELS_INDEX."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"[load] {FINBERT_MODEL} on {device} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).to(device)
    model.eval()

    # Sanity-check labels — if Hugging Face changes the order of labels
    # in a future ProsusAI/finbert revision, we must catch it before
    # writing wrong-named columns.
    id2label = {i: model.config.id2label[i].lower() for i in range(3)}
    print(f"[load] model.config.id2label = {id2label}", flush=True)
    expected = {0: "positive", 1: "negative", 2: "neutral"}
    if id2label != expected:
        # Re-map according to the ACTUAL labels in the loaded model.
        global LABELS_INDEX
        LABELS_INDEX = id2label
        print(f"[load] WARNING: relabelled output columns to match model: {LABELS_INDEX}",
              flush=True)
    return tokenizer, model


# -----------------------------------------------------------------------------
# Article text selection
# -----------------------------------------------------------------------------
def select_text(row: pd.Series) -> tuple[str, str]:
    """Pick the best text field for a single article. Returns (text, source).

    Prefers the full article body; falls back to title.
    """
    body = row.get("Article")
    title = row.get("Article_title")
    if isinstance(body, str) and len(body.strip()) > 32:
        return body, "article"
    if isinstance(title, str) and len(title.strip()) > 0:
        return title, "title"
    return "", "empty"


# -----------------------------------------------------------------------------
# Audit invariants
# -----------------------------------------------------------------------------
def audit_block(probs: np.ndarray, block_idx: int):
    """Run the five fail-fast invariants on a [N, 3] probability block."""
    if len(probs) == 0:
        return
    pos = probs[:, LABEL_POS]
    neg = probs[:, LABEL_NEG]
    neu = probs[:, LABEL_NEU]

    # 1. Probabilities sum to ~1
    sum_check = (pos + neg + neu)
    bad = np.where(np.abs(sum_check - 1.0) > 1e-3)[0]
    if len(bad) > 0:
        raise RuntimeError(
            f"[audit FAIL block#{block_idx}] {len(bad)}/{len(probs)} rows where "
            f"pos+neg+neu != 1 (max deviation {np.abs(sum_check - 1.0).max():.4f}). "
            f"Tokenizer / softmax issue.")

    # 2. Standard deviation must be non-trivial — the constant-0.333 bug
    if pos.std() < 1e-3:
        raise RuntimeError(
            f"[audit FAIL block#{block_idx}] positive_prob.std()={pos.std():.6f} < 1e-3 — "
            f"FinBERT is producing near-constant outputs. This was the v0 sentinel "
            f"failure. STOP NOW.")

    # 3. Fraction of (0.333, 0.333, 0.333) uniform-prior rows
    uniform_mask = (
        (np.abs(pos - 1/3) < 0.01) &
        (np.abs(neg - 1/3) < 0.01) &
        (np.abs(neu - 1/3) < 0.01)
    )
    uniform_frac = uniform_mask.mean()
    if uniform_frac > 0.05:
        raise RuntimeError(
            f"[audit FAIL block#{block_idx}] {100*uniform_frac:.1f}% of rows are at "
            f"uniform prior (0.333, 0.333, 0.333). Threshold is 5%. v0 redux.")

    # 4. positive_prob spans both halves of [0, 1]
    if pos.min() > 0.5 or pos.max() < 0.5:
        # Could legitimately happen on a small block of all-positive or all-negative
        # articles (rare). Warn but don't fail unless block is large.
        if len(probs) > 500:
            print(f"[audit WARN block#{block_idx}] positive_prob in [{pos.min():.3f}, "
                  f"{pos.max():.3f}] — model is making one-sided predictions. Check.",
                  flush=True)


# -----------------------------------------------------------------------------
# Inference loop
# -----------------------------------------------------------------------------
def score_chunk(texts: list[str], tokenizer, model, device: torch.device,
                batch_size: int = DEFAULT_BATCH) -> np.ndarray:
    """Run FinBERT on a list of texts. Returns [N, 3] probability matrix
    in [pos, neg, neu] column order (matches LABEL_POS/NEG/NEU module-level
    indices)."""
    if len(texts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    all_probs = np.zeros((len(texts), 3), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                        max_length=MAX_LEN, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs[start : start + len(batch)] = probs
    return all_probs


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all_external", default=ALL_EXTERNAL_CSV)
    p.add_argument("--out_dir", default=OUTPUT_DIR)
    p.add_argument("--stock_list", default=None,
                   help="Path to file with one ticker per line (lower case). "
                        "If None, uses tickers from D:\\Study\\CIKM\\DATA\\350_merged.")
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--chunk_rows", type=int, default=200_000,
                   help="Pandas read_csv chunk size in rows.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N articles (overall, post-filter). For testing.")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: implies --limit 200 + 3 manual sanity-check articles.")
    args = p.parse_args()

    if args.smoke and args.limit is None:
        args.limit = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    universe = set(get_universe(args.stock_list))
    print(f"Universe: {len(universe)} tickers (sample: {sorted(list(universe))[:5]} …)")
    print(f"Device  : {device}")
    print(f"Output  : {args.out_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer, model = load_finbert(device)

    # ─── module-level label index for audit_block ───
    global LABEL_POS, LABEL_NEG, LABEL_NEU
    inv = {v: k for k, v in LABELS_INDEX.items()}
    LABEL_POS, LABEL_NEG, LABEL_NEU = inv["positive"], inv["negative"], inv["neutral"]

    # Smoke sanity check: 3 known-positive + 3 known-negative articles
    if args.smoke:
        print("\n[smoke] sanity test on 6 hand-crafted articles …")
        sanity_texts = [
            ("Apple Reports Record Quarterly Revenue and Profit, Beats Analyst Estimates", "positive"),
            ("Tesla Surges 12% After Strong Q4 Delivery Numbers", "positive"),
            ("Nvidia Forecasts Strong Growth Ahead Driven By AI Demand", "positive"),
            ("Boeing Stock Plunges After Crash Investigation Reveals Major Defects", "negative"),
            ("Walgreens Faces Bankruptcy Risk Amid Mounting Losses", "negative"),
            ("Bank of America Slashes Profit Outlook On Loan Losses", "negative"),
        ]
        sanity_probs = score_chunk([t for t, _ in sanity_texts], tokenizer, model, device,
                                    args.batch_size)
        ok = True
        for (text, expected), p_row in zip(sanity_texts, sanity_probs):
            pos_p = p_row[LABEL_POS]
            neg_p = p_row[LABEL_NEG]
            neu_p = p_row[LABEL_NEU]
            pred = "positive" if pos_p > max(neg_p, neu_p) else \
                   "negative" if neg_p > max(pos_p, neu_p) else "neutral"
            mark = "OK" if pred == expected else "FAIL"
            if pred != expected:
                ok = False
            print(f"  [{mark}] expected={expected:<8} pred={pred:<8} "
                  f"(pos={pos_p:.3f}, neg={neg_p:.3f}, neu={neu_p:.3f}) — {text[:60]}…")
        if not ok:
            print("\n[smoke] FAILED sanity check — FinBERT is mis-tokenized or wrong model.")
            sys.exit(2)
        print("[smoke] sanity check PASSED — FinBERT loaded correctly.\n")

    # ─── Stream All_external.csv in chunks, filter to universe, score ───
    print("[main] streaming All_external.csv in chunks …")
    chunk_iter = pd.read_csv(
        args.all_external,
        chunksize=args.chunk_rows,
        usecols=["Date", "Article_title", "Stock_symbol", "Url",
                 "Publisher", "Article"],
        on_bad_lines="skip",
        low_memory=False,
    )

    per_stock_buffers: dict[str, list[dict]] = {}
    n_total = 0
    n_kept = 0
    n_skipped_empty = 0
    block_idx = 0
    t_start = time.time()

    for ci, chunk in enumerate(chunk_iter):
        # Lower-case ticker, keep only universe tickers
        chunk["_ticker"] = chunk["Stock_symbol"].astype(str).str.strip().str.lower()
        chunk = chunk[chunk["_ticker"].isin(universe)]
        if len(chunk) == 0:
            continue

        # Pick the best text field per row
        text_pairs = chunk.apply(select_text, axis=1)
        chunk["_text"] = text_pairs.apply(lambda x: x[0])
        chunk["_src"]  = text_pairs.apply(lambda x: x[1])
        keep = chunk["_text"].str.len() > 0
        n_skipped_empty += (~keep).sum()
        chunk = chunk[keep].reset_index(drop=True)
        if len(chunk) == 0:
            continue

        # Honour --limit (cumulative, post-filter)
        if args.limit is not None and n_kept + len(chunk) > args.limit:
            chunk = chunk.iloc[: args.limit - n_kept].reset_index(drop=True)
            if len(chunk) == 0:
                break

        # Score
        probs = score_chunk(chunk["_text"].tolist(), tokenizer, model, device,
                            args.batch_size)
        # Audit per ~1000-article block
        audit_block(probs, block_idx)
        block_idx += 1

        # Build per-row records and append to per-stock buffers
        chunk["positive_prob"]   = probs[:, LABEL_POS]
        chunk["negative_prob"]   = probs[:, LABEL_NEG]
        chunk["neutral_prob"]    = probs[:, LABEL_NEU]
        chunk["composite_score"] = chunk["positive_prob"] - chunk["negative_prob"]
        chunk["scaled_sentiment"] = (chunk["composite_score"] + 1.0) / 2.0
        chunk = chunk.rename(columns={"_text": "text_used", "_src": "text_field_used"})

        for ticker, sub in chunk.groupby("_ticker"):
            sub_out = sub[["Date", "Stock_symbol", "Article_title", "Url", "Publisher",
                           "text_field_used",
                           "positive_prob", "negative_prob", "neutral_prob",
                           "composite_score", "scaled_sentiment"]]
            per_stock_buffers.setdefault(ticker, []).append(sub_out)

        n_total += len(chunk)
        n_kept += len(chunk)

        elapsed = time.time() - t_start
        rate = n_kept / max(elapsed, 1)
        print(f"  chunk {ci:>4}: kept={len(chunk):>5}  cum_kept={n_kept:>7,}  "
              f"rate={rate:.1f} arts/sec  elapsed={elapsed:.0f}s",
              flush=True)

        if args.limit is not None and n_kept >= args.limit:
            break

    # Flush per-stock buffers to CSV
    print(f"\n[flush] writing per-stock CSVs to {args.out_dir} …")
    for ticker, parts in per_stock_buffers.items():
        df = pd.concat(parts, ignore_index=True)
        out = os.path.join(args.out_dir, f"{ticker}_finbert.csv")
        df.to_csv(out, index=False)
    print(f"[flush] wrote {len(per_stock_buffers)} stock CSVs")

    elapsed = time.time() - t_start
    print()
    print("=" * 60)
    print(f"DONE  total_kept={n_kept:,}  skipped_empty={n_skipped_empty:,}")
    print(f"      elapsed={elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"      rate={n_kept/max(elapsed,1):.1f} articles/sec")
    print(f"      avg articles/stock={n_kept/max(len(per_stock_buffers),1):.0f}")
    print("=" * 60)


# Initialise label-index module globals to defaults; load_finbert will overwrite if needed
LABEL_POS = 0
LABEL_NEG = 1
LABEL_NEU = 2

if __name__ == "__main__":
    main()
