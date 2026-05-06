# finsharpe — Implementation Plan

**Repo created:** 2026-05-06
**Predecessor:** `D:\Study\CIKM\SR_optimization` (kept as-is for archival; do not modify after this date)
**Target venue:** CIKM 2026 Applied Research Track (deadline May 23, 2026 AoE) primary; ICAIF 2026 (Aug 2) fallback.

---

## 0. Why a fresh repo

The predecessor repo trained models on a `scaled_sentiment` column that was uniformly 0.5 across every stock and every date — i.e. a constant feature, not real FinBERT output. Discovered 2026-05-06; documented in `SR_optimization/reports/sentiment_data_integrity_audit.md`.

We're rebuilding the data pipeline and retraining from a clean slate so that every claim in the resulting paper is auditable end-to-end. The architectural and theoretical contributions (`engine/heads.py`, `engine/losses.py`, Theorem A1) are unchanged and are copied over verbatim — they do not depend on sentiment.

---

## 1. Two-arm experimental design

The accidental discovery is also an opportunity. We now have a clean sentiment-vs-no-sentiment ablation built into the experiment design.

| Arm | Sentiment | Status |
|---|---|---|
| **No-sentiment baseline** | 6th feature = 0.5 everywhere (the existing v1 fan-out) | In progress on HPC, ~22h to complete. Results harvested directly into `finsharpe/results/no_sentiment/` |
| **With-sentiment** | 6th feature = real FinBERT-derived daily aggregate | To-be-trained from scratch on FinBERT-corrected `350_merged_v2/` |

For each arm we run the full pipeline:
1. MSE-baseline retrain (35 jobs: 7 backbones × 5 horizons).
2. Track B v1 retrain with `--init_from` on the same-arm MSE baselines (35 jobs).
3. Cross-sectional eval at H ∈ {5, 20, 60} with both `--strategy simple` and `--strategy risk_aware`.
4. Paired Politis-Romano stationary bootstrap on Sharpe differences (5000 reps, block 5).

### Headline tables produced by this design

* **Table 1: Sentiment vs no-sentiment, MSE baseline.** Pure data ablation — does FinBERT-augmented features help when training with MSE? Reported per (model, horizon).
* **Table 2: Track B vs MSE, no-sentiment.** Pure architecture/loss ablation, holding sentiment-status fixed.
* **Table 3: Track B vs MSE, with-sentiment.** Same architecture/loss ablation, holding sentiment-status fixed at "with".
* **Table 4: Track B with-sentiment vs MSE no-sentiment.** Total contribution of the paper (architecture + sentiment combined).

If sentiment helps, Table 1 has positive deltas, Table 4 is the strongest. If sentiment is neutral, Table 1 is roughly zero — that's still an honest result, and Table 2 still gives us the architectural contribution as the headline.

---

## 2. Repo structure

```
finsharpe/
├── IMPLEMENTATION.md              ← this file
├── README.md                      ← (TBD: short public-facing overview)
├── config.py                      ← FEATURES, SEQ_LEN, HORIZONS, CACHE/DATA dirs
├── data_loader.py                 ← UnifiedDataLoader + calendar split + mmap loaders
├── train.py                       ← CLI entry point with --use_risk_head flag etc.
├── preprocess_global_cache.py     ← cache builder; honoured by --force flag
│
├── engine/                        ← Modeling primitives (the contribution)
│   ├── heads.py                   ← RiskAwareHead (verified working)
│   ├── losses.py                  ← CompositeRiskLoss (verified working)
│   ├── trainer.py                 ← Training loop with risk-head dispatch
│   ├── evaluator.py               ← Test-time eval, dict-output handling
│   └── early_stopping.py          ← EarlyStopping + LR schedule
│
├── models/                        ← Backbones (7 architectures)
│   ├── __init__.py                ← model_dict registry
│   ├── dlinear.py
│   ├── itransformer.py
│   ├── gcformer.py
│   ├── patchtst.py
│   ├── adapatch.py
│   ├── tft.py
│   └── vanilla_transformer.py
│
├── data_pipeline/                 ← NEW: clean FinBERT pipeline
│   ├── finbert_score.py           ← Score articles with ProsusAI/finbert
│   ├── aggregate_daily_sentiment.py ← Per-stock per-date aggregation
│   ├── rebuild_350_merged.py      ← Merge prices + sentiment to per-stock CSVs
│   ├── extend_prices_yfinance.py  ← Stage 2 OOT price extension (already verified)
│   └── README.md                  ← Pipeline sequence + audit checks
│
├── scripts/                       ← HPC sbatch templates
│   ├── train_mse_baseline.sbatch  ← Pure MSE retrain (no risk head)
│   ├── train_riskhead.sbatch      ← Track B v1 with --init_from
│   ├── run_finbert_inference.sbatch ← FinBERT pipeline on HPC
│   └── submit_campaign.sh         ← Fan-out helper across models × horizons
│
├── smoke/                         ← Cross-sectional evaluation pipeline
│   ├── cross_sectional_smoke.py   ← --strategy {simple, risk_aware}
│   ├── bootstrap_paired.py        ← Politis-Romano paired Sharpe-diff CI
│   ├── bootstrap_ci.py            ← Marginal Sharpe CI (single arm)
│   └── verify_A1.py               ← A1 empirical verification
│
├── tests/                         ← Unit + integration tests
│   ├── test_track_b.py            ← Architecture + loss core tests
│   ├── test_trainer_risk_head.py  ← Trainer dispatch + autocast tests
│   └── test_finbert_pipeline.py   ← NEW: data-pipeline tests
│
├── reports/                       ← Methodology + findings docs
│   ├── design.md                  ← Architecture spec
│   ├── theorem_A1.md              ← Variance-decomposition theorem
│   ├── findings.md                ← Empirical writeup (built incrementally)
│   └── audits/                    ← Data-integrity audits
│       └── (FinBERT pipeline audit will go here when ready)
│
├── checkpoints/                   ← Trained model artefacts (gitignored)
└── results/                       ← Eval outputs (CSVs + JSONs)
    ├── no_sentiment/              ← from the in-flight v1 fan-out
    └── with_sentiment/            ← from the new sentiment-aware retrain
```

---

## 3. Stage gate sequence

Each stage has explicit pre-conditions and a single-line audit check before we move to the next stage. **No gate is skipped.** This is the lesson from the FinBERT incident.

### Stage 0 — Repo init + code copy + sanity check (TODAY)

1. Initialize `finsharpe/` git repo. ✅
2. Copy verified code from `SR_optimization/`. (this commit)
3. Run unit tests on copied code (no model retraining required for this).
4. **Audit check:** `python tests/test_track_b.py && python tests/test_trainer_risk_head.py` returns exit 0.

### Stage 1 — FinBERT pipeline (TONIGHT-TOMORROW)

1. Write `data_pipeline/finbert_score.py` using `ProsusAI/finbert` from HuggingFace.
2. **Audit check 1:** Score 100 hand-picked articles locally. Verify `positive_prob`/`negative_prob`/`neutral_prob` have **non-zero std** across the sample. If std=0, FinBERT is mis-loaded — fix before scaling.
3. **Audit check 2:** For 3 manually-known-positive articles ("AAPL Hits Record High After Strong Q3"), verify `positive_prob > 0.5`. For 3 manually-known-negative articles ("Boeing Plunges After 737-MAX Crash"), verify `negative_prob > 0.5`. If predictions are wrong, FinBERT is mis-tokenized or wrong model.
4. Submit FinBERT inference job on HPC for our 350-stock universe articles only (~1.4M articles, ~3-5 GPU-hours).

### Stage 2 — Daily-aggregation + cache rebuild

1. `aggregate_daily_sentiment.py` groups per-article scores by (stock, date), computes per-day weighted average.
2. **Audit check:** For 3 stock-date pairs with multiple articles, verify the daily aggregate is between min and max of per-article composites and matches manual spreadsheet calculation.
3. `rebuild_350_merged.py` merges prices (existing) + sentiment (new) into per-stock CSVs in `D:\Study\CIKM\DATA\350_merged_v2\`.
4. **Audit check:** `groupby("Stock").scaled_sentiment.std()` per-stock min should be ≥ 0.05 (real variance present) — the audit that should have been there from day 1.
5. Run `preprocess_global_cache.py --force` to regenerate cache from `350_merged_v2/`.

### Stage 3 — Train MSE baselines from scratch (with-sentiment arm)

1. 35 jobs: 7 backbones × 5 horizons. From-scratch random init. ~3.5 days HPC at 4 jobs in flight.
2. **Audit check (per job):** verify the model converges (val MSE drops monotonically in early epochs) before running full schedule. Catches another stealth bug class.

### Stage 4 — Train Track B v1 (with-sentiment arm)

1. 35 jobs: same matrix, `--init_from <new MSE checkpoint>`. ~2.5 days HPC.
2. **Audit check:** for the H=60 GCFormer instance, verify Phase 3 reaches α=0.7 in the log and final-epoch checkpoint is saved (not best-val).

### Stage 5 — Cross-sectional evaluation

1. xs-eval all checkpoints under both `simple` and `risk_aware` strategies.
2. Bootstrap paired CIs: with-sentiment-RA vs no-sentiment-RA (the headline test) and Track B vs MSE (the architectural test).
3. **Audit check:** Sharpe formula consistency — for any arm, `gross_sharpe` from JSON ≈ recomputed `mean(returns)/std(returns)*sqrt(252/H)` to within 1e-4.

### Stage 6 — Paper writing

CIKM Applied 8-10 page format. New `reports/findings.md` written incrementally as results land, then converted to LaTeX in the final week.

---

## 4. What gets copied verbatim from SR_optimization

These files are battle-tested, verified working, and contain the contribution. They are copied unchanged and re-tested in finsharpe:

| Source | Target | Reason |
|---|---|---|
| `engine/heads.py` | `engine/heads.py` | RiskAwareHead — works |
| `engine/losses.py` | `engine/losses.py` | CompositeRiskLoss — works (B1 path kept for ablation) |
| `engine/trainer.py` | `engine/trainer.py` | Risk-head dispatch — works |
| `engine/evaluator.py` | `engine/evaluator.py` | Dict-output handling — works |
| `engine/early_stopping.py` | `engine/early_stopping.py` | Boilerplate, works |
| `models/__init__.py` + 7 backbones | `models/...` | Architecture defs — work |
| `train.py` | `train.py` | CLI — works |
| `preprocess_global_cache.py` | `preprocess_global_cache.py` | Cache builder — works (DATA_DIR points to new dir) |
| `data_loader.py` | `data_loader.py` | Mmap loaders + calendar split — work |
| `tests/test_track_b.py` | `tests/test_track_b.py` | 5 unit tests pass |
| `tests/test_trainer_risk_head.py` | `tests/test_trainer_risk_head.py` | 6 wiring tests pass |
| `Smoke_test/cross_sectional_smoke.py` | `smoke/cross_sectional_smoke.py` | xs-eval — works (path-self-detecting) |
| `Smoke_test/bootstrap_paired.py` | `smoke/bootstrap_paired.py` | Paired Politis-Romano — works |
| `Smoke_test/bootstrap_ci.py` | `smoke/bootstrap_ci.py` | Single-arm CI — works |
| `Smoke_test/verify_A1.py` | `smoke/verify_A1.py` | A1 empirical check — works |
| `scripts/riskhead_glob.sbatch` | `scripts/train_riskhead.sbatch` | Track B SLURM — works |
| `scripts/submit_riskhead_campaign.sh` | `scripts/submit_campaign.sh` | Fan-out helper — works |
| `reports/track_b_implementation.md` | `reports/design.md` | Implementation report |
| `reports/track_b_theorem_A1.md` | `reports/theorem_A1.md` | Theorem A1 |

What gets DROPPED:
* `models/timesnet.py` — excluded from prior campaign due to FFT/12h-walltime issue.
* The B1 `--use_xs_sharpe` path is *retained* in `engine/losses.py` so the ablation table can include the negative result, but it is not the headline.

What gets MODIFIED:
* `config.py` → DATA_DIR points to `D:\Study\CIKM\DATA\350_merged_v2` (the new sentiment-corrected dir). Original `350_merged/` is read-only after this date.
* `data_pipeline/` is newly created with FinBERT scoring + aggregation + cache rebuild scripts.

---

## 5. Audit invariants — to be run before any retraining

These checks must pass before we commit GPU hours to retraining. They take seconds each.

```python
# Audit 1: scaled_sentiment is non-trivially varying for every stock
for csv in glob("350_merged_v2/*.csv"):
    df = pd.read_csv(csv, usecols=["scaled_sentiment"])
    s = df["scaled_sentiment"].dropna()
    assert s.std() > 0.05, f"{csv}: sentiment std too low ({s.std():.4f})"

# Audit 2: no monotonic feature creep (e.g., date leakage)
# Spot-check: scaled_sentiment shouldn't trend monotonically with date for any stock
for csv in random.sample(glob("350_merged_v2/*.csv"), 5):
    df = pd.read_csv(csv, parse_dates=["Date"]).sort_values("Date")
    s = df["scaled_sentiment"].dropna()
    if len(s) > 100:
        rho = np.corrcoef(np.arange(len(s)), s)[0,1]
        assert abs(rho) < 0.3, f"{csv}: sentiment-vs-time correlation suspiciously high ({rho:.2f})"

# Audit 3: cache fields actually contain values (not all 0.5)
manifest = json.load(open(".cache/global_scaled/manifest.json"))
sample_arr = np.load(manifest["stocks"][0]["path"], mmap_mode="r")
sentiment_col = sample_arr[:, 5]   # 6th feature (CLOSE_IDX=3 + 2)
# Wait: CLOSE_IDX is the index of Close in FEATURES; sentiment is at index 5
# (FEATURES = ["Open","High","Low","Close","Volume","scaled_sentiment"])
assert np.std(sentiment_col) > 0.001, "Cache sentiment column constant — check rebuild"
```

Every retraining run should reference these audit functions in its sbatch preamble:

```bash
python data_pipeline/audit.py || { echo "[FATAL] data audit failed"; exit 1; }
python preprocess_global_cache.py --force-if-stale
```

The fix-the-bug-before-it-happens layer.

---

## 6. Why this is robust against repeat mishaps

1. **Every external pipeline has a one-line audit invariant** (FinBERT std > 0, sentiment-vs-time low correlation, cache column non-constant).
2. **The audit functions live in `data_pipeline/audit.py`** and are runnable standalone — no hidden state.
3. **Stage gates are explicit.** No silent "let me also rebuild the cache" — every cache regeneration is logged + version-stamped.
4. **The two arms (no-sentiment + with-sentiment) cross-check each other.** If with-sentiment results are dramatically different from no-sentiment in unexpected ways, the audit gets re-run.
5. **The fresh repo has no legacy data files.** Every CSV referenced is one we built in this repo, with a known-good provenance.

The first thing that goes into reports/findings.md is a "pipeline audit" section quoting the actual audit assertions and their last successful run timestamps. Reviewers can verify our claims by re-running the audits.
