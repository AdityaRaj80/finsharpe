# Comprehensive Plan — ICAIF 2026 Submission
**Drafted:** 2026-05-07 04:30 IST
**Synthesis of:** 4 independent jury agents (Hostile Reviewer, Conference Scout, Dataset Auditor, Methodology Auditor v2) + ground-truth verification.
**Goal:** > 50-60% acceptance probability at the chosen target conference.

---

## §0  COURT PROCEEDING — verified facts on the record

Before any plan, the disputed facts. Each verified by direct check, not memory.

**Fact 1: FNSPID news ends 2020-06-11 — no later version exists.**
- Verified: scanned `D:/Study/CIKM/DATA/All_external.csv` (5.7 GB, 13,057,514 rows). Date range: 1914-09-16 to **2020-06-11 13:12:35 UTC**.
- Verified: HuggingFace `Zihan1004/FNSPID` last modified 2024-04-09; only files are `All_external.csv` + `nasdaq_exteral_data.csv` + price archive.
- **Implication:** Jury C's claim that FNSPID covers to 2023 was wrong. There is no "switch files" easy path.

**Fact 2: 350_merged_v2 prices extend through 2024-2025** (via `extend_prices_2024_2025.py` yfinance pull, per fin-sent-optimized README and graphify analysis).
- We have prices for the modern era; we have NO news for 2020-06 onwards in the academic FNSPID dataset.

**Fact 3: Per-year news article counts in FNSPID** (just verified):
| Year | Articles | Year | Articles |
|------|---------:|------|---------:|
| 2007 | 464,439 | 2014 | 1,166,279 |
| 2008 | 1,056,009 | 2015 | 1,364,956 |
| 2009 | 960,491 | 2016 | 686,338 |
| 2010 | 1,200,802 | 2017 | 344,705 |
| 2011 | 1,571,048 | **2018** | **523,476** |
| 2012 | 1,569,853 | **2019** | **578,683** |
| 2013 | 1,192,744 | **2020** (Jan-Jun) | **201,557** |

→ A 2018-2020-06 test window has **1.3M articles** = adequate coverage.

**Fact 4: FNSPID license** is CC BY-NC-4.0 (non-commercial, academic OK, no commercial redistribution).

**Fact 5: Methodology audit — STILL FAILING items** (per Jury D):
- ❌ Predicting MinMax-scaled price, not log-returns (incomparable to PatchTST/iTransformer/MASTER)
- ❌ Train-cache scaler fitted on full history (subtle leakage)
- ❌ n<6 at H≥60 even with all our guards
- ❌ Sequential training mode (catastrophic forgetting)

**Fact 6: Stock split convention** — we use disjoint stocks (300 train / 49 val=test). MASTER, DeepClair, Qlib, FactorVAE all use SAME stocks across all splits with calendar-only split. Our setup is non-standard.

**Fact 7: Universe size** — currently 49 NAMES_50 (43 effective with sentiment). Modern papers use 100-3000. 49 is a known weakness.

**Fact 8: Conference deadlines verified (today is May 7, 2026):**
| Venue | Deadline | Days from today |
|-------|---------:|----------------:|
| CIKM 2026 AR | 2026-05-23 | 16 |
| ICDM 2026 | 2026-06-06 | 30 |
| ICAIF 2026 | 2026-08-02 | **87** |
| KDD 2026 ADS | passed | — |
| NeurIPS 2026 D&B | passed (2026-05-04) | — |

---

## §1  COURT VERDICT — current state estimated probability of acceptance

From Jury A (Hostile Reviewer):

| Venue | P(accept) at current state |
|-------|---------------------------:|
| CIKM AR 2026 | 8% |
| ICAIF 2026 | 18% |
| KDD AR 2026 | 4% |
| ICDM 2026 | 15% |
| AAAI 2026 | 12% |

Jury A's brutal one-liner: *"competent engineering paper with three known components stitched together, evaluated on a universe too small to matter, on a test period where the title's headline feature is literally constant. Fix the data or fix the title."*

From Jury B (Conference Scout) — projected probability AFTER full retrofits:
- **ICAIF 2026: 65%** (top pick, most calendar headroom, lenient on novelty, loves Sharpe + DSR)
- CIKM AR 2026: 55% if retrofits land in 16 days (unrealistic)
- FinNLP @ EMNLP: 60% (workshop-tier; visibility lower)
- ACM TIST: 40% (long review, robust fallback)
- Algorithmic Finance: 50% (lower prestige safety net)

**Consensus: target ICAIF 2026. Drop CIKM AR for this paper — 16 days is insufficient for the retrofits below.**

---

## §2  COMPREHENSIVE PLAN — end-to-end specification

This is the load-bearing section. Every step from dataset to submission, in execution order.

### §2.1  DATASET — final decision

**Primary news source: FNSPID `All_external.csv` (we already have it).** No new vendor.
**Calendar split:**
- Train: 2009-01-01 to 2017-12-31 (≈9 years, all real sentiment)
- Val: 2018-01-01 to 2018-12-31 (≈252 trading days, real sentiment)
- Test: 2019-01-01 to 2020-06-11 (≈365 trading days incl. COVID onset, real sentiment)

Why this works:
- All val and test rows have **real** sentiment, not 0.5 fill — fixes the headline-untestable bug.
- COVID Q1 2020 stress test in the test window is a **strong** defensive narrative.
- 365-day test window = 73 H=5 rebalances, 18 H=20 rebalances — bootstrap valid.
- 1.3M news articles in val+test = adequate FinBERT signal.

**Reject Option B (Alpha Vantage backfill 2020-2024)** — adds $50/mo cost, complicates the methodology ("FinBERT on FNSPID + Alpha-Vantage native sentiment ≠ FinBERT on FNSPID alone"), and 87-day calendar barely accommodates retrofit + extra dataset work. Not worth the complication.

**Optional supplementary table:** keep our existing 2023 v1 fan-out results as a "modern-era price-only baseline" in §6.3 supplementary. No new compute.

### §2.2  STOCK UNIVERSE — final decision

**Switch from disjoint-stock split to same-stocks calendar-only split (matches MASTER / DeepClair / Qlib convention).**

**Universe size:** 200 stocks selected from FNSPID's 350 by:
1. Has prices for entire 2009-2020-06-11 range (no surviorship-relevant gaps)
2. Has > 100 sentiment-bearing days (i.e. days with article_count > 0) in 2018-2019
3. Stratified by sector (finance / tech / energy / consumer / health / industrial)
4. Stratified by market cap tier (large / mid)

**Same 200 stocks** trained, validated, tested. Calendar-only split.

Why 200 not 500: FNSPID's effective coverage cap (294 stocks have *any* sentiment, ~200 have meaningful coverage). We can't credibly claim 500 without bringing in a different price/news universe.

Why same-stocks: matches every modern cross-sectional ranking benchmark. Apples-to-apples vs MASTER. Avoids the OOD-stock-generalization framing which is a *different* paper.

**Survivorship-bias guard:** include delisted stocks if they were in the universe at training start (2009). Use `yfinance + EDGAR delisting dates` to verify.

### §2.3  TARGET REFORMULATION — predict log-returns, not prices

**Current** (FAIL): predict H-step ahead **MinMax-scaled price** ∈ [0,1].
**Replace with:** predict H-step ahead **log-return** = `log(close_{t+H} / close_t)`.

Implementation:
- New `build_sequences_returns(...)` in `data_loader.py`: target is the log-return scalar (not price sequence).
- Targets are stationary → MSE in return-space is **comparable** to MASTER, FactorVAE, DeepClair benchmarks.
- Heteroscedastic σ-head's log_sigma2_H now directly models return variance (no scaled-price floor hack needed).
- L_NLL term becomes the canonical heteroscedastic Gaussian NLL on returns, no derivation gap.

Backward-compat: keep an optional `--target {price, return}` flag for ablation comparisons.

### §2.4  NORMALIZATION — per-stock z-score on training window only

**Current** (FAIL/CONCERN): per-stock MinMax fitted on full history including val/test calendar dates → leakage through learned representation.
**Replace with:**
- Per-stock z-score `(x − μ_train) / σ_train` where μ_train, σ_train are computed from `date < VAL_START_DATE` only.
- Targets (log-returns) are inherently scale-stable; no separate target scaling needed.
- RevIN remains *internal* to PatchTST/GCFormer for runtime regime adaptation. The outer per-stock z-score is the one-time train-fitted normalization.

Drop: outer MinMax. Drop: target scaling. The double-normalization (outer + RevIN) is gone.

### §2.5  HORIZONS — drop H=120, H=240

**Headline horizons: H ∈ {5, 20}.**
**Supplementary: H ∈ {60}** with explicit "n=6 — borderline, do not over-interpret" footnote.
**Drop entirely: H ∈ {120, 240}.**

Rationale: even with bootstrap CIs, n=2 or n=1 is statistically uninterpretable. Including these horizons in the headline invites a desk-rejection.

### §2.6  MODELS — reduce to 3 representative backbones

**Headline: PatchTST + GCFormer + DLinear.**

- PatchTST: strong representative of the 2023 SOTA family (ICLR 2023).
- GCFormer: our best-performing model in v1 fan-out; lets us claim "the proposed method amplifies the best backbone."
- DLinear: deliberate weak-but-strong-baseline (AAAI 2023) — shows the loss helps even on a non-attention model.

Drop from headline: TFT, iTransformer, AdaPatch, VanillaTransformer (move to a single supplementary table).

Why 3 not 7: Jury R3 critique — "7 backbones dilute the message." Cleaner story. Smaller multiple-comparison family for DSR.

### §2.7  COMPETITIVE BASELINES — implement, don't just cite

**Implement on identical data:**
1. **ZZR-2020** (Zhang-Zohren-Roberts): differentiable Sharpe loss with softmax (long-only) weights. Reference implementation: arxiv:2005.13665.
2. **LdP meta-labeling** (Lopez de Prado 2018, Ch. 3): primary forecast (any backbone) → secondary XGBoost classifier on `1{trade profitable}` → meta-labeled position sizing. This is the apples-to-apples "non-end-to-end gate" baseline.
3. **Markowitz mean-variance with shrinkage covariance** (Ledoit-Wolf 2003 shrinkage estimator, sklearn LedoitWolf). The classical optimization baseline.
4. **Naive equal-weight 1/N** (DeMiguel-Garlappi-Uppal 2009 — the canonical hard-to-beat baseline).
5. **Top-N by predicted return** (the "simple" path of cross_sectional_smoke.py) — already implemented.

Six baselines × 3 backbones for headline = comparison table.

### §2.8  ARCHITECTURAL CONTRIBUTION — clarified

**RiskAwareHead** (μ + σ + vol heads) wrapping the backbone.
**CompositeRiskLoss** (5-term: SR_gated + NLL + MSE_R + VOL + GATE_BCE) with phase schedule.
**Sharpe-saturated portfolio layer** at inference: `tanh(α·μ/σ) × gate`, leg-normalized.

Specifically lean into the **profitability-supervised gate** (L_GATE_BCE) as the primary algorithmic novelty — citing Lopez de Prado 2018 meta-labeling explicitly, locating contribution in (i) joint end-to-end vs LdP's sequential, (ii) sigmoid-product over internal uncertainty estimates, (iii) annealed temperature.

Make the **gate ablation** a centerpiece of §5: post-hoc LdP gate vs joint hard gate vs joint annealed gate (ours) vs MASTER market-gate. Show the annealed-joint variant dominates.

### §2.9  THEORETICAL CONTRIBUTION — strengthen Theorem A1 OR demote

**Two paths.** Decide based on whether we can land path (a) in a week.

**Path (a) — strengthen to a finite-sample bound.** State and prove:

> *Theorem A1':* under sub-Gaussian estimation error in σ̂, the realized Sharpe gap of the σ-aware portfolio over equal-weight is lower-bounded by ‖P_{u^⊥} Σ^{−½} μ‖² minus a term linear in σ̂-estimation MSE.

This *binds the architectural choice to estimation quality*. It's a meaningful theorem and exactly answers Jury A's "the identity says nothing about estimation error" critique.

**Path (b) — demote to "geometric remark".** Move A1 to §3.4 as motivation/intuition, not as a Theorem. Stop calling it a "contribution."

**Recommendation: try (a) first**; if proof doesn't land in 5 days, fall back to (b). Either way, the paper does not lead with A1 — it leads with the architecture.

### §2.10  STATISTICAL MACHINERY — final choices

| Test | Choice | Rationale |
|------|--------|-----------|
| Block-length | Politis-White auto (Patton-Politis-White 2009) | Already implemented |
| Bootstrap | Politis-Romano stationary, paired | Already implemented |
| Studentization | Ledoit-Wolf 2008 (HAC-studentized) on headline cell | Already implemented; tighter CI |
| Sharpe annualization | Lo (2002) autocorr-corrected | NEW — replaces naive √(252/H) |
| Cross-sectional IC | Spearman ρ, NW HAC SE at lag=H-1 | Already implemented |
| Multiple-testing | Deflated Sharpe Ratio with FULL trial count | Already implemented; expand N to include model × horizon × top_n × cost × strategy = ~270 trials |
| Pre-registration | Log every config tried in `manifests/dsr_trials.csv` | Transparent N_trials |

### §2.11  COST MODEL — 20bps default headline

**Headline cost: 20 bps round-trip.** Sweep: {0, 1, 5, 10, 20, 50}.

Rationale: Frazzini-Israel-Moskowitz (2018) report 10-20 bps for liquid US equities; with NAMES_50-style ADRs (BABA, BIDU, GSK, BHP, TM) included, 20 is realistic. 0 bps gross is supplementary only.

Add: a one-paragraph §4.2 footnote on market-impact (Almgren et al. 2005 square-root law) explaining we're below the regime where it matters at our notional.

Add: short-borrow fee assumption — ~30 bps/yr for the short leg, applied uniformly.

### §2.12  SENTIMENT — feature ablation, not headline

Title becomes: **"End-to-End Sharpe Optimization with Profitability-Gated Heteroscedastic Heads"** (Jury A's suggestion).

Sentiment becomes a §5.X ablation row:
- v1: 5 features (OHLCV)
- v2: 6 features (+ scaled_sentiment, concat as 6th column)
- v3: separate sentiment branch with cross-attention fusion (MASTER-style)

If sentiment helps, great — it's a feature. If it doesn't, the paper still works because we no longer claim "FinBERT helps" in the headline.

### §2.13  DSR TRIAL COUNT — honest accounting

Trial count for DSR includes every config swept:
- 3 models × 3 horizons (H ∈ {5,20,60}) × 5 top_n × 6 cost levels × 2 strategies = **540 cells**.

This is harsh but honest. Better to under-claim with N=540 than over-claim with N=5. Pre-register via `manifests/dsr_trials.csv` committed to git BEFORE running final eval.

---

## §3  TIMELINE — 87 days to ICAIF Aug 2

| Phase | Days | Tasks |
|-------|-----:|-------|
| **§3.1 Plan freeze + repo refactor** | 1-7 | Land this plan as `COMPREHENSIVE_PLAN.md`. Commit. Update `config.py`: VAL_START=2018-01-01, TEST_START=2019-01-01, TEST_END=2020-06-11. Add `RETURN_TARGET=True` flag and dual code paths. |
| **§3.2 Data layer rewrite** | 8-14 | New `build_sequences_returns(...)`. Per-stock z-score on train-only window. Same-stocks calendar split. Universe selection (200 stocks). Cache rebuild. New `tests/test_data_layer.py`. |
| **§3.3 Loss/head sanity** | 15-17 | Update `engine/heads.py` and `engine/losses.py` to use return targets directly. Drop denom-floor hack. Re-verify all 34+ tests pass. |
| **§3.4 Baseline implementation** | 18-25 | ZZR-2020 (1d), LdP meta-labeling-XGBoost (2d), Markowitz-Ledoit-Wolf shrinkage (1d), naive 1/N (already done). All 4 in `smoke/baselines/`. Tests for each. |
| **§3.5 First training pass** | 26-35 | 3 backbones × 3 horizons × {MSE, Track-B} = 18 jobs on HPC + 4 baselines. ~5 days HPC. |
| **§3.6 Theorem A1 strengthening** | 36-42 | Attempt finite-sample bound. If lands, write proof. If not, demote to remark. |
| **§3.7 Eval pipeline** | 43-52 | DSR with full N=540. Lo (2002) annualization. Cost-sweep. Sentiment ablation. Verify_A1 numerical run. Cross-sectional IC tables. |
| **§3.8 Paper draft v0** | 53-67 | First end-to-end draft. ~15 days for 8 sections at 1-2 days each. |
| **§3.9 Internal review pass** | 68-74 | Self-review against Jury A's checklist. Fix every "concrete attack". Get a colleague review if possible. |
| **§3.10 Polish + supplementary** | 75-84 | Tables, figures, appendix proofs, supplementary methods, code release prep. |
| **§3.11 Submit** | 85-87 | Final formatting checks. Submit to ICAIF Aug 2. |

**Buffer:** ~3 days slack across phases. Plan should hold even with one phase slipping.

---

## §4  RISK REGISTER

| # | Risk | Probability | Mitigation |
|---|------|------------:|------------|
| 1 | Theorem A1 finite-sample bound doesn't close | 50% | Demote to remark — paper still works. |
| 2 | Sentiment ablation null result | 40% | Already accounted for — sentiment is no longer headline. Frame as "FinBERT on FNSPID provides modest signal in some regimes." |
| 3 | Profitability-gate ablation: removing gate < 0.05 Sharpe drop | 30% | Show interaction with σ-head: gate amplifies σ-head benefit. If even that fails, paper degrades to "incremental architecture + benchmark" — still ICAIF-acceptable. |
| 4 | DSR with N=540 fails for headline cell | 25% | Honestly report; tone down "outperforms" → "directionally consistent with theory." Still publishable at ICAIF if DSR ≥ 0.7. |
| 5 | Time slip on retrofits | 30% | 3-day slack baked in. Worst case, drop one supplementary table or one ablation. |
| 6 | Jury A's "DSR p > 0.05" scenario | 20% | Honestly report. Pivot to rank-IC headline (which has more samples). |
| 7 | A reviewer demands H=240 | 5% | We can run it; just refuse to put a CI on it. |
| 8 | Out-of-distribution-stock generalization complaint | 5% | We're now using same-stocks split; not applicable. |

**Highest-impact failure mode:** Theorem A1 doesn't strengthen + sentiment ablation null + DSR borderline. This trifecta drops P(accept) ICAIF from 65% → 35%. Acceptable downside; still better than the current 18%.

---

## §5  CONTRIBUTION CLAIMS — final, defensible wording

**§1.4 Our Contributions** (final):

> 1. **A profitability-gated risk-aware multi-head architecture for cross-sectional return forecasting.** We propose a backbone-agnostic wrapper that augments any standard time-series forecaster with (i) a heteroscedastic σ head (Kendall & Gal 2017), (ii) a forward-vol auxiliary head, and (iii) a continuous confidence gate constructed as a product of sigmoids over (i) and (ii) and supervised end-to-end by binary cross-entropy on realized P&L sign — extending the meta-labeling concept of López de Prado (2018) to a single end-to-end pass with annealed temperature.
>
> 2. **A composite risk-aware loss with curriculum-scheduled phases** combining differentiable Sharpe (Zhang-Zohren-Roberts 2020 generalized to long-short with leg-wise L1 normalisation), heteroscedastic NLL on returns, return-MSE anchor, log-vol-MSE auxiliary, and the L_GATE_BCE term — under a 3-phase warm-up schedule.
>
> 3. **A geometric identity (Theorem 1)** expressing the squared-Sharpe gap between σ-aware and equal-weight cross-sectional portfolios as `‖P_{u^⊥} Σ^{−½} μ‖²` (under diagonal Σ; closed-form Lagrange identity, folkloric mathematics, our novel framing for ML σ-prediction quality), with three corollaries linking the gap to σ-heterogeneity, μ-homogeneity, and the degenerate-equality regime — used as a regime-of-edge predictor that we numerically verify on test.
>
> 4. **A unified, statistically rigorous benchmark** of three time-series forecasters under DSR-deflated Sharpe (Bailey-López de Prado 2014), studentized stationary bootstrap (Ledoit-Wolf 2008 / Politis-Romano 1994), Newey-West HAC rank-IC, Lo (2002) autocorrelation-corrected annualization, and four competitive baselines (ZZR-2020, LdP meta-labeling+XGBoost, Markowitz-Ledoit-Wolf shrinkage, naive 1/N).

The word "FinBERT" / "Sentiment-Augmented" appears in §5.6 ablation table only, not in the title or §1.

---

## §6  EXECUTIVE SUMMARY

**Single most important decision:** target ICAIF 2026 (Aug 2 deadline, 87 days). Skip CIKM AR (May 23, 16 days — infeasible).

**Single most important fix:** switch target to log-returns (not MinMax-scaled prices). This single change reconciles us with PatchTST/iTransformer/MASTER conventions and makes our MSE numbers comparable to recent baselines.

**Single most important narrative shift:** drop "Sentiment-Augmented" from the title. The headline becomes the architecture (RiskAwareHead + CompositeRiskLoss + profitability-gated meta-labeling). Sentiment is a feature ablation row.

**Probability target after full execution:**
- ICAIF 2026: 60-65% (vs current 18%)
- ACM TIST (rolling, journal fallback): 40%
- Algorithmic Finance (rolling fallback): 50%

The plan is realistic. The downside (Theorem A1 doesn't strengthen + sentiment null + DSR borderline) still gives ~35% ICAIF — better than today's 18%. The upside (everything lands) gives 65%+.

**Next action:** approval gate. The user reads this plan. If approved, day 1 begins with config.py changes and stopping the in-flight v1 fan-out (its 27/30 results no longer matter — different calendar window, different target reformulation, different stock split convention).
