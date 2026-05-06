# Data Pipeline — Sentiment-Corrected FNSPID for finsharpe

This directory contains the **clean** sentiment pipeline. The predecessor
project's pipeline produced uniform 0.5 sentiment values (root cause:
FinBERT was never actually executed — see
`reports/sentiment_audit.md`). This directory rebuilds it from scratch
with audit invariants at every stage.

---

## Sequence

Run in order. Each stage gates the next via an audit check.

```
1.  finbert_score.py            ── articles → per-article FinBERT scores
2.  audit.py per_article        ── verify std > 1e-3, uniform-prior rate < 5%
3.  aggregate_daily_sentiment.py── per-article scores → per-day per-stock sentiment
4.  audit.py per_stock          ── (run on output dir) verify daily std > 0.02
5.  rebuild_350_merged.py       ── prices + daily sentiment → merged 350_merged_v2/
6.  audit.py per_stock D:\Study\CIKM\DATA\350_merged_v2  ── final merged audit
7.  preprocess_global_cache.py --force  ── rebuild train + valtest mmap caches
8.  audit.py cache              ── confirm cache sentiment column non-constant
```

After step 8 passes, training is safe to launch.

---

## Stage 1: `finbert_score.py`

```bash
# Smoke test (200 articles + 6 hand-crafted sanity articles)
python data_pipeline/finbert_score.py --smoke

# Full run on the 351-stock universe (≈ 1.4M articles, ≈ 4.5h on RTX 3060)
python data_pipeline/finbert_score.py
```

**Inputs:**
* `D:\Study\CIKM\DATA\All_external.csv` — raw FNSPID news (13M articles, 1999-2020)
* Universe: enumerated from `D:\Study\CIKM\DATA\350_merged\*.csv` (351 stocks)

**Outputs:**
* `D:\Study\CIKM\DATA\finbert_scores\<stock>_finbert.csv` — per-article scores

**What it does:**
* Loads `ProsusAI/finbert` from Hugging Face (BERT-base sized, ≈110M params).
* Verifies `model.config.id2label = {0:positive, 1:negative, 2:neutral}` at load time.
* 6-article hand-crafted sanity check (3 positive + 3 negative). Aborts if any is misclassified.
* Streams `All_external.csv` in 200K-row chunks; filters to universe; tokenises (max 256 tokens); softmax over the 3-class output.
* Audits *every block* of articles: std > 1e-3, prob sums ≈ 1, uniform-prior rate < 5%.
* Output schema: `Date, Stock_symbol, Article_title, Url, Publisher, text_field_used, positive_prob, negative_prob, neutral_prob, composite_score, scaled_sentiment`
  where `composite_score = positive_prob − negative_prob ∈ [−1, +1]` and `scaled_sentiment = (composite_score + 1) / 2 ∈ [0, 1]`.

---

## Stage 2: `audit.py per_article`

```bash
python data_pipeline/audit.py per_article
```

Scans every `*_finbert.csv` in the FinBERT output directory. Asserts:
* A. `positive_prob.std() > 1e-3` per file.
* B. `< 5%` of rows are at the uniform `(0.333, 0.333, 0.333)` prior.

Exit non-zero on any failure.

---

## Stage 3: `aggregate_daily_sentiment.py`

```bash
python data_pipeline/aggregate_daily_sentiment.py
```

**Inputs:**
* `D:\Study\CIKM\DATA\finbert_scores\<stock>_finbert.csv` (per-article)

**Outputs:**
* `D:\Study\CIKM\DATA\finbert_daily\<stock>_daily.csv` (per-day)

**Aggregation:** UTC midnight floor, then `groupby(Date)` with `mean` over per-article probabilities. Multiple articles per day → one daily row averaging them. Output schema matches the column names that the predecessor `350_merged/` expected (`avg_composite, scaled_sentiment, avg_positive, avg_negative, avg_neutral, article_count`) so downstream code is unchanged.

**Per-stock audits inside this script:**
* `scaled_sentiment.std() > 0.02`
* `avg_pos + avg_neg + avg_neu ≈ 1` per row (within 1e-3)
* No duplicate dates per stock

---

## Stage 4: `audit.py per_stock`

```bash
python data_pipeline/audit.py per_stock D:\Study\CIKM\DATA\finbert_daily
```

Sweeps the daily files. Asserts:
* `scaled_sentiment.std() > 0.05` per stock.
* `|corr(time_index, scaled_sentiment)| < 0.3` per stock (no monotone creep — date-leakage check).

---

## Stage 5: `rebuild_350_merged.py`

```bash
python data_pipeline/rebuild_350_merged.py
```

Left-joins `350_merged/<stock>.csv` with `finbert_daily/<stock>_daily.csv` on `Date`. Writes the merged result to `350_merged_v2/<stock>.csv` with the full 13-column target schema.

**News-less day handling:** `scaled_sentiment = 0.5` (neutral), `avg_composite = 0.0`, `article_count = 0`. News-rich days carry real FinBERT signal. The model can learn to weight by `article_count`.

---

## Stage 6: `audit.py per_stock` on the merged dir

```bash
python data_pipeline/audit.py per_stock D:\Study\CIKM\DATA\350_merged_v2
```

Final per-stock check on the data the model will actually see.

---

## Stage 7: `preprocess_global_cache.py --force`

```bash
python preprocess_global_cache.py --force
python preprocess_global_cache.py --only-valtest --force
```

Regenerates the mmap cache from `350_merged_v2/`. The `--force` flag is mandatory because the existing cache contains uniform-0.5 sentiment.

`config.py` must be updated to point `DATA_DIR = r"D:\Study\CIKM\DATA\350_merged_v2"` before running this stage. Otherwise the script will silently re-cache the broken `350_merged/`.

---

## Stage 8: `audit.py cache`

```bash
python data_pipeline/audit.py cache
```

Final sanity check: the cache's sentiment column (feature index 5) is not constant. If this passes, training is safe.

---

## Provenance + reproducibility

Every output stage writes a `_*_manifest.csv` summarising what was processed: per-stock article counts, audit-failure flags, sentiment-std diagnostics. Reviewers can spot-check by re-running any stage on a single stock.

The pipeline is intentionally idempotent: re-running each stage with the same inputs produces identical outputs (modulo file-modification timestamps).
