# Session State — 2026-05-06 (Pre-Compact Snapshot)

**Purpose:** Resumable record of where the project stands so any subsequent session can pick up without re-deriving context.
**Target venue:** CIKM 2026 Applied Research Track. **Deadline: May 23, 2026 AoE** (17 days from project pivot).
**Repo:** https://github.com/AdityaRaj80/finsharpe (clean rebuild) — replaces deprecated `D:\Study\CIKM\SR_optimization`.
**Working dir for sentiment data:** `D:\Study\CIKM\fin-sent-optimized\`.

---

## 1. Why this state exists — the FinBERT mishap

Discovered 2026-05-06: across the predecessor `D:\Study\CIKM\SR_optimization` project's training data (`350_merged/`), the `scaled_sentiment` feature column was **uniformly 0.5 across every stock and every date**. Upstream, the `350_sentiment_results/*_finbert_tone.csv` files showed `(positive_prob=0.333, negative_prob=0.333, neutral_prob=0.333)` for every article — the uniform prior, indicating FinBERT was never actually executed.

Implications:
* Every prior model was trained on **prices only** (the 6th feature was a constant — neural networks extract zero predictive signal from constants).
* The architectural / theoretical contributions (RiskAwareHead + CompositeRiskLoss + Theorem A1) **remain valid** — they don't depend on sentiment as input.
* The "FNSPID + FinBERT-augmented" framing in the paper draft was **factually incorrect** — must be reframed.

Audit document with full evidence: `reports/sentiment_audit.md`.

---

## 2. What's been built in `finsharpe/` (fresh repo)

### 2.1 Code (copied from SR_optimization, all 15 unit tests pass)

```
engine/                    RiskAwareHead, CompositeRiskLoss, Trainer, Evaluator, EarlyStopping
models/                    7 backbones (DLinear, iTransformer, GCFormer, PatchTST, AdaPatch, TFT, VanillaTransformer)
                           (TimesNet excluded — 12h walltime issue)
layers/                    24 transformer building blocks
utils/                     metrics, masking, time features, tools
data_loader.py             UnifiedDataLoader + calendar split + mmap loaders
train.py                   CLI: --use_risk_head, --use_xs_sharpe, --init_from
preprocess_global_cache.py cache builder; --force regenerates
smoke/                     cross_sectional_smoke.py + bootstrap_paired.py + verify_A1.py
tests/                     test_track_b.py, test_trainer_risk_head.py, test_xs_sharpe.py (15/15 pass)
scripts/                   train_mse_baseline.sbatch (NEW), train_riskhead.sbatch, submit_full_campaign.sh
reports/                   design.md, theorem_A1.md, sentiment_audit.md, cikm_positioning.md,
                           findings_v0_no_sentiment.md (predecessor's findings, kept)
```

### 2.2 Data pipeline (`finsharpe/data_pipeline/` + `fin-sent-optimized/scripts/`)

```
finbert_score.py             ProsusAI/finbert inference, 6-article hand-sanity-check, per-block audits
audit.py                     standalone fail-fast invariants (per_article, per_stock, cache modes)
aggregate_daily_sentiment.py per-article scores → per-day per-stock with audits
rebuild_350_merged.py        prices + daily sentiment → 350_merged_v2/ schema
README.md                    8-stage sequential pipeline with audit gates
```

The two-script ownership: `finsharpe/data_pipeline/` is git-tracked source-of-truth; `fin-sent-optimized/scripts/` is the workspace working copy with workspace-local default paths.

### 2.3 Commits pushed to GitHub

| SHA | What |
|---|---|
| `f8041a6` | initial commit: verified code copy from SR_optimization (32 files, 15 tests pass) |
| `111d4dc` | data_pipeline/ — clean FinBERT pipeline with audit invariants |
| `69649f5` | scripts: MSE-baseline-from-scratch sbatch + full-campaign submitter |
| `f60c6a6` | config: switch DATA_DIR to fin-sent-optimized/data/350_merged_v2 |

---

## 3. The sentiment pipeline run (today, 2026-05-06)

### 3.1 Outputs at `D:\Study\CIKM\fin-sent-optimized\data\`

| Stage | Output dir | Count | Audit |
|---|---|---|---|
| Per-article FinBERT scores | `finbert_scores/` | 294 stocks × ~1.8K articles avg = **540,194 articles** | ✅ all pass (std > 1e-3, uniform-prior < 5%) |
| Per-day per-stock aggregation | `finbert_daily/` | 294 stocks | ✅ all pass (std 0.18-0.27 typical) |
| Final merged with prices | `350_merged_v2/` | **351 stocks** (294 with sentiment, 57 with neutral fallback) | 38 stocks below std<0.05 threshold (sparse-news; expected) |

### 3.2 FinBERT inference timing

* Hardware: RTX 3060 Laptop, 6 GB VRAM
* Model: ProsusAI/finbert (BERT-base, ~110M params)
* Settings: max_length=256, batch=64
* Wall clock: **55 min** (much faster than 4-5h projection)
* Sustained rate: ~127-250 articles/sec
* Hand-crafted sanity check: **6/6 articles correctly classified** (3 positive, 3 negative; pos_prob 0.92-0.96, neg_prob 0.94-0.96)

### 3.3 NAMES_50 (test universe) coverage breakdown

| Coverage | n | Stocks | Effect on sentiment-vs-no-sentiment ΔSharpe |
|---|---|---|---|
| **STRONG** (std > 0.05) | **31/49** | aal, abbv, baba, bhp, bidu, biib, c, cat, cmcsa, cmg, cop, crm, dal, ebay, gild, gld, goog, gsk, ko, mrk, mu, nvda, orcl, pep, qcom, qqq, sbux, tm, tsm, wfc, xlf | full effect |
| **WEAK** (0.005-0.05) | 12/49 | aapl, amd, amgn, amzn, dis, intc, t, tgt, tsla, uso, v, wmt | partial effect |
| **NONE** (std < 0.005, 0 articles) | 6/49 | **cost, cvx, ge, msft, nke, pypl** | zero contribution to ΔSharpe by construction |

The 6 NONE stocks have ZERO articles in `All_external.csv` (FNSPID coverage gap before its 2020-06 cutoff). They will produce **identical predictions** in both arms, so they contribute exactly 0 to the headline delta.

### 3.4 Cache state (after `preprocess_global_cache.py --force`)

* Train: 302/303 stocks cached (1 skipped: `_merge_manifest.csv` mistakenly picked up as a stock)
* Val/Test: 48/49 NAMES_50 cached (1 skipped: SBUX has < 745 rows in its history, below `seq_len + max_horizon + 1` threshold)
* Sentiment column variance check: 7 train stocks have std=0 in cache (legitimate zero-news cases — all 0.5 by design)

---

## 4. HPC: v1 fan-out (no-sentiment baseline arm)

* **Jobs 167702-167731**, submitted 2026-05-06 ~03:00 IST.
* Status as of last hourly tick (13:06 IST): **19/30 COMPLETED, 4 RUNNING, 7 PENDING.**
* Models done: DLinear ×5, iTransformer ×5, PatchTST ×4 (H240 still running), AdaPatch ×3 (H120/240 done).
* Models in flight or pending: PatchTST H240 (running), TFT (1 running, others pending), VanillaTransformer ×5 (all pending).
* Estimated remaining: ~2-3h to full completion.
* These 30 checkpoints are the **NO-SENTIMENT** arm of the headline ablation. They were trained on `D:\Study\CIKM\DATA\350_merged\` (uniform-0.5 sentiment), so effectively price-only models. To be harvested as `results/no_sentiment/` once complete.

Monitor: task `bk8wmzzcp` (active, fires every 30 min).

---

## 5. Two-arm experimental design — the headline of the paper

| Arm | Sentiment column | Training data | Status |
|---|---|---|---|
| **No-sentiment** | uniformly 0.5 (predecessor's broken pipeline; effectively price-only) | `350_merged/` | ⏳ HPC fan-out 19/30 done |
| **With-sentiment** | real FinBERT-derived | `fin-sent-optimized/data/350_merged_v2/` | ⏳ TBD on HPC after we sync the new dataset |

For each arm:
1. Train **MSE baselines** from scratch (35 jobs: 7 backbones × 5 horizons).
2. Train **Track B v1** with `--init_from` on same-arm MSE baselines (35 jobs).
3. xs-eval at H ∈ {5, 20, 60} with `--strategy {simple, risk_aware}`.
4. Paired Politis-Romano stationary bootstrap on Sharpe differences.

### Headline tables produced by this design

| # | Comparison | Tells us |
|---|---|---|
| **1** | Sentiment vs no-sentiment, MSE baseline | Pure data ablation — does FinBERT help when training with MSE? |
| **2** | Track B vs MSE, no-sentiment | Pure architecture/loss ablation, no sentiment |
| **3** | Track B vs MSE, with-sentiment | Same architecture/loss ablation, with sentiment |
| **4** | Track B + sentiment vs MSE no-sentiment | Total contribution of the paper |

Each table reported in three slices for stratified rigor:
* All 49 NAMES_50 (pooled, conservative)
* 31 STRONG-coverage stocks (clean signal)
* 43 NAMES_50-with-any-sentiment (excludes 6 NONE stocks)

---

## 6. What needs to happen next

### 6.1 Immediate (within next ~3-6h, mostly compute-blocked)

1. **Wait for v1 fan-out to finish** (~3h). Harvest no-sentiment results into `finsharpe/results/no_sentiment/`.
2. **Push `350_merged_v2/` to HPC.** ~700 MB of CSV. Approx command:
   ```bash
   rsync -avz D:/Study/CIKM/fin-sent-optimized/data/350_merged_v2/ \
              bitshpc:~/data/350_merged_v2/
   ```
3. **HPC sync code:** `cd ~/finsharpe && git pull` (only if HPC clones the repo; otherwise `scp` the relevant files).
4. **HPC rebuild caches:** `python preprocess_global_cache.py --force && python preprocess_global_cache.py --only-valtest --force`.
5. **HPC audit cache:** `python data_pipeline/audit.py cache` (fail fast if sentiment column is constant).

### 6.2 Stage A — From-scratch MSE retrain (with-sentiment arm)

```bash
bash scripts/submit_full_campaign.sh mse ALL
```

* 35 jobs: 7 backbones × 5 horizons.
* From-scratch random init, 60 epochs, lr=1e-4, batch=512.
* Wall clock: ~3-5 days HPC at 4 in flight.
* Output: `<MODEL>_global_H<H>.pth` checkpoints.

### 6.3 Stage B — Track B v1 (with-sentiment arm)

```bash
bash scripts/submit_full_campaign.sh riskhead ALL
```

* 35 jobs: same matrix, `--init_from <Stage-A checkpoint>`.
* 30 epochs, phase-scheduled composite loss.
* Wall clock: ~2.5 days HPC.
* Output: `<MODEL>_global_H<H>_riskhead.pth` checkpoints.

### 6.4 Cross-sectional evaluation

* Run `smoke/cross_sectional_smoke.py` for every (model, horizon, arm, strategy) combination.
* Run `smoke/bootstrap_paired.py` to compute paired Sharpe-difference CIs:
  - Sentiment vs no-sentiment, MSE arm
  - Track B vs MSE, both arms
  - Stratified by coverage tier (STRONG / WEAK / NONE)
* Run `smoke/verify_A1.py` to empirically verify Theorem A1 on the new checkpoints.

### 6.5 Paper writing (CIKM Applied 8-10 page format)

* Section structure per `reports/cikm_positioning.md` §4.
* Lead with architectural primitive + Theorem A1.
* Cite DeepClair (CIKM 2024) as direct precedent.
* Stratified results tables (per §5 above).
* Coverage table in methods section listing every NAMES_50 stock with article count + std.

### 6.6 Final checks before submission

* `python -m tests.test_track_b && python -m tests.test_trainer_risk_head && python -m tests.test_xs_sharpe` all green.
* `python data_pipeline/audit.py all` exits 0.
* All bootstrap CIs computed.
* Two reproducibility manifests committed: `_aggregation_manifest.csv`, `_merge_manifest.csv`.

---

## 7. Decision tree based on Stage 1 outcome

After Stage A (with-sentiment MSE) completes, we'll have a "sentiment vs no-sentiment, MSE" comparison. Three cases:

| Result | Implication | Action |
|---|---|---|
| Sentiment ⇪ MSE Sharpe by ΔS > 0.3 across most architectures | Headline narrative survives: "FinBERT-augmented forecasting + Track B is the contribution" | Proceed to Stage B + paper writing as planned |
| Sentiment helps marginally (ΔS 0.05-0.3) | Honest mixed result | Reframe as "Track B's loss-function design works regardless of sentiment availability — sentiment adds a modest but consistent boost on news-rich stocks" |
| Sentiment doesn't help / hurts | Pure architectural-contribution paper | Frame as "we tested the FinBERT-augmented hypothesis and report it as a null result; the architectural contribution stands independently" |

All three are publishable. The paper should not over-claim.

---

## 8. Open issues + risks

| Issue | Severity | Mitigation |
|---|---|---|
| 6 NAMES_50 stocks (cost, cvx, ge, msft, nke, pypl) have zero FinBERT articles | Medium | Stratified reporting (STRONG / WEAK / NONE tiers); these stocks contribute identically to both arms |
| FNSPID corpus ends 2020-06; no FinBERT-derivable signal for 2020-2024 | Medium | Sentiment is sparse for that window; consistent across both arms; documented as scope statement |
| 12 NAMES_50 stocks have WEAK sentiment (some news but sparse) | Low | Stratified reporting handles this; signal is non-zero for these |
| 7 train stocks have std=0 sentiment in cache | Low | Train signal still strong from the other 295/302 stocks |
| CIKM Applied deadline May 23 (17 days) is TIGHT | High | Compute path is parallelizable; if missed, ICAIF Aug 2 is the fallback. Same paper, minor reframing. |
| `submit_full_campaign.sh` HPC paths assume `~/finsharpe/`; HPC currently has `~/SR_optimization/` | Medium | Need to clone `finsharpe` to HPC + sync `350_merged_v2/` before launching Stage A |
| Stage 3 (walk-forward × bias-free) deferred | Low | Documented in `reports/cikm_positioning.md` §3 — not required for CIKM 2026 submission |

---

## 9. Workspace layout

```
D:\Study\CIKM\
├── SR_optimization\        DEPRECATED (predecessor with broken sentiment); do not modify
├── DATA\                   Raw FNSPID + Yahoo prices
│   ├── All_external.csv          13M raw articles (1999-2020-06)
│   ├── 350_merged\               Original prices + uniform-0.5 sentiment (BROKEN)
│   ├── 350_merged_ext\           Stage 2 extended prices (sentiment=0.5)
│   └── 350_sentiment_results\    Original FinBERT files (also broken; uniform 0.333)
│
├── finsharpe\              CURRENT REPO (the codebase + paper artefacts)
│   ├── engine/, models/, layers/, utils/, smoke/, tests/, scripts/, reports/
│   ├── config.py                  DATA_DIR points to fin-sent-optimized's 350_merged_v2
│   ├── data_pipeline/             scripts (finbert_score, aggregate, rebuild, audit)
│   └── .cache/                    rebuilt with sentiment-corrected data
│
└── fin-sent-optimized\     CURRENT WORKSPACE (sentiment data + working scripts)
    ├── README.md
    ├── scripts/                   workspace copy of pipeline scripts (path-bound)
    ├── data\
    │   ├── finbert_scores\        540,194 per-article scores, 294 stocks
    │   ├── finbert_daily\         294 per-day aggregations
    │   └── 350_merged_v2\         351 merged per-stock CSVs (the model's training data)
    ├── logs\finbert_run.log       55-min run log
    └── manifests\                 (audit manifest CSVs)
```

---

## 10. Resumption procedure for next session

If anyone picks up this work after a chat compact:

1. Read `reports/sentiment_audit.md` and **this file** (`session_state_2026_05_06.md`).
2. Check HPC v1 fan-out status: `ssh bitshpc "sacct -j 167702-167731 -X --format=JobID,JobName,State -n"`.
3. Confirm local data integrity: `cd D:\Study\CIKM\fin-sent-optimized && python scripts/audit.py all`.
4. Push `350_merged_v2/` to HPC if not already done.
5. Resume from §6.

The paper-side TODOs are tracked in `reports/cikm_positioning.md` §4 (CIKM-aligned framing changes).
