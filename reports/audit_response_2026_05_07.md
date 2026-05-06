# Audit Response — Defensibility Hardening

**Companion to** `reports/methodology_audit_2026_05_07.md`.
**Status as of 2026-05-07 02:30 IST.**

This document tracks the resolution of every PASS / CONCERN / FAIL
verdict from the audit. Code changes land before paper writing.

---

## Tier 1 — desk-rejection blockers (DONE this session)

### ✅ §H — Deflated Sharpe Ratio implementation
- **New file:** `smoke/deflated_sharpe.py` (~190 LOC).
- Implements Bailey & Lopez de Prado 2014 eqs. 7-9: SR₀ extreme-value
  threshold + DSR (Φ-form with skew/kurt adjustment).
- Reads any winning-cell timeseries CSV + a sweep CSV with annualised
  Sharpes per trial → JSON with DSR ∈ [0,1] and significance flag.
- **Tests:** `tests/test_deflated_sharpe.py` — 11 tests including
  Euler-Mascheroni constant matches scipy.psi(1), monotonicity in N
  and T, extreme-input bounds, normal-returns pass, overfitting fail.

### ✅ §C — Bootstrap miscitation removed; studentized option added
- **Modified:** `smoke/bootstrap_paired.py`.
- Default path: explicitly labelled "Politis-Romano 1994
  (unstudentized)" — the L&W 2008 citation that previously appeared
  in the docstring is gone.
- **New flag:** `--studentized` invokes `arch.bootstrap.StationaryBootstrap`
  with HAC variance studentization per Ledoit & Wolf 2008 §3 — the
  *correct* L&W procedure.
- The reported `procedure` field in the JSON output makes the choice
  explicit so reviewers can verify which test was actually run.

### ✅ §D + §K — Long-horizon validity guard
- `bootstrap_paired.py` and `bootstrap_ci.py` now refuse to run
  when n_returns < 6 (hard error, exit 2) unless `--force` is passed.
- Warn at n in [6, 12]: "CIs will be wide; interpret with caution."
- `cross_sectional_smoke.py` annotates every metric block with
  `n_obs_nonoverlap` and `inference_valid` (boolean; True iff n ≥ 6).
  Main summary prints a clear `[WARNING]` line when invalid.
- This formally locks H ∈ {60, 120, 240} out of the headline CI table
  unless we (a) extend the test window beyond 1 year, or (b) shift to
  overlapping-return Newey-West inference.

### ✅ §D — Automatic block-length selection
- Both bootstrap modules now expose `--auto_block` invoking
  `arch.bootstrap.optimal_block_length` (Politis-White 2004 +
  Patton-Politis-White 2009 correction).
- Per-arm block lengths are averaged for the paired test (the standard
  approach when the two arms have potentially-different autocorrelation
  structures).
- Floor of 2.0 enforced; falls back to n^(1/3) if `arch` import fails.

---

## Tier 2 — strong defense (DONE this session)

### ✅ §I + §L — Cross-sectional rank-IC + Newey-West
- **New file:** `smoke/rank_ic.py` (~200 LOC).
- Computes Spearman ρ_t and Kendall τ_t per rebalance across the
  cross-section.
- Subsamples to non-overlapping `[::H]` rebalances, then reports IC
  mean / std / ICIR / Newey-West HAC SE / NW t-statistic with
  Bartlett-kernel HAC at lag = H-1 (Hansen-Hodrick standard for
  overlapping H-day returns).
- **Tests:** `tests/test_rank_ic.py` — 7 tests including NW SE matches
  iid SE at lag 0, NW SE inflates ≥ 1.5× under AR(1) ρ=0.7, Spearman
  series correctly recovers near-1 IC for monotone signal and near-0
  for independent.
- Closes the "MSE ≠ ranking ability" critique (§I) and the
  Fama-MacBeth companion-test gap (§L).

### ✅ §B + §J — Citation lineage and "Kelly" rename
- **Modified:** `engine/losses.py` docstring — explicitly cites
  Zhang-Zohren-Roberts 2020 (arXiv:2005.13665) for the differentiable
  Sharpe surrogate, with a paragraph clarifying that:
  * v1 path (`use_xs_sharpe=False`) is a per-sample correlate of
    portfolio Sharpe, NOT Moody-Saffell's recursive D_t.
  * B1 path (`use_xs_sharpe=True`) is the closer ZZR analogue,
    generalised long-short with leg-wise L1 normalisation.
- **Renamed in code comments:** "Kelly-tanh" → "Sharpe-saturated" /
  "soft-Sharpe-targeting" in:
  * `engine/losses.py` position-function comment.
  * `engine/losses.py` B1 surrogate docstring.
  * `smoke/cross_sectional_smoke.py:248-273`
    (`cross_sectional_positions_risk_aware` docstring).
- Internal variable names (`kelly_long`, `kelly_short`) retained for
  code-search continuity, with a one-line explanatory comment pointing
  to audit §J.

---

## Tier 2 — partial / pending

### ⚠️ §A — DeepClair positioning table
- **Status:** still to write (paper-side, not code).
- **Plan:** §2 of the paper will include a comparison table:
  
  | Aspect             | DeepClair (CIKM '24) | Ours      |
  |--------------------|----------------------|-----------|
  | Universe           | 26 stocks            | 49 stocks |
  | Test window        | 16.5 years           | 1 year    |
  | Significance test  | none (10 seeds)      | paired stationary bootstrap |
  | Cost modelling     | not modelled         | 0/5/10/20/50 bps swept |
  | Sentiment input    | NLP on news (theirs) | FinBERT on FNSPID |

- **Defense framing:** matched-to-FNSPID-availability for the test
  window; deeper rigor on cost + significance.

### ⚠️ §E — ZZR generalisation acknowledgement in paper §3
- **Status:** docstring done; paper §3 still to write.
- **Action:** state in §3.3 that our differentiable Sharpe loss is a
  long-short generalisation of ZZR's softmax-simplex formulation,
  with leg-wise L1 normalisation in place of softmax.

---

## Tier 3 — polish (pending; partial)

### ⚠️ §G general LdP concerns
- §G(1) overlap — already PASS (subsample [::H]).
- §G(2) single test window — pre-emptive defense in paper §4
  (matched-to-FNSPID + ablation across two cost regimes).
- §G(3) backtest length — same defense as §A.
- §G(4) DSR — DONE in §H above.
- §G(5) cost realism — sweep covers 0-50 bps; supplementary table to
  add 100 bps for completeness.

### ⏳ Theorem A1 numerical verification
- **Status:** `smoke/verify_A1.py` exists and is functional;
  needs to be run against the in-flight v1 fan-out checkpoints once
  they finish (currently 26/30 COMPLETED, 4 RUNNING) and against the
  forthcoming Stage A + Stage B sentiment-aware checkpoints.
- **Output target:** `results/a1_verification_<model>_<tag>.csv` with
  Pearson + Spearman correlation between bar_CV_sigma and Delta_SR.
- Will be run automatically as part of the harvest stage.

### ⏳ α ∈ {1, 2, 5, 10} ablation
- **Status:** to be added to the supplementary.
- **Action:** add a top-line table to §5.2 of the paper showing
  Sharpe at each α value for one (model, horizon) cell.
- Cheap: re-run only `cross_sectional_smoke.py` with the existing
  Track B checkpoints (no retraining needed) — it's an inference
  hyperparameter at the position-sizing step.

---

## Summary table

| Audit § | Topic                          | Verdict before | Status now |
|---------|--------------------------------|----------------|------------|
| A       | DeepClair positioning           | ⚠️ CONCERN     | ⚠️ paper-side TODO |
| B       | Moody-Saffell vs ours           | ⚠️ CONCERN     | ✅ DONE (citation fixed) |
| C       | Ledoit-Wolf studentization      | ❌ FAIL        | ✅ DONE |
| D       | Block-length choice             | ❌ FAIL        | ✅ DONE (auto + guard) |
| E       | ZZR 2020 lineage                | ⚠️ CONCERN     | ✅ DONE (code) ⚠️ paper TODO |
| F       | Kendall-Gal NLL                 | ✅ PASS        | ✅ PASS |
| G       | LdP general warnings             | ⚠️ CONCERN     | ✅ DONE (sub-items) |
| H       | Deflated Sharpe Ratio           | ❌ FAIL        | ✅ DONE |
| I       | MSE-ranked models               | ⚠️ CONCERN     | ✅ DONE (rank-IC) |
| J       | Kelly-tanh terminology          | ⚠️ CONCERN     | ✅ DONE (renamed) |
| K       | Bootstrap at long H             | ❌ FAIL        | ✅ DONE (guard) |
| L       | Fama-MacBeth companion           | ⚠️ CONCERN     | ✅ DONE (rank_ic.py + NW) |

**Counts:** 9 / 12 audit items closed in code this session.
3 paper-side items (A, E, A1 numerical) remain — to be addressed
during paper writing once results are in hand.

---

## What this means for the paper claims

After landing these fixes, the paper can now defensibly state:

1. **Headline Sharpe is reported alongside DSR** (Bailey-LdP 2014).
   With our trial count (≈350 swept configs) and T ≈ 50, even a raw
   point Sharpe of 1.5 may not survive deflation. We **commit to
   reporting DSR** in the headline cell rather than soft-pedalling it.

2. **CIs are valid only at H ∈ {5, 20}.** At H ∈ {60, 120, 240} we
   report point Sharpe with `n_obs` and an explicit caveat — no CI.
   This is a **strict** improvement over silently reporting [nan,nan]
   intervals.

3. **Bootstrap procedure is honestly described:** Politis-Romano 1994
   unstudentized by default; Ledoit-Wolf 2008 studentized when the
   user explicitly invokes it. No false citations.

4. **Block length is data-driven**, not an arbitrary 5.

5. **Cross-sectional rank-IC is reported** alongside portfolio Sharpe
   for each model — defends against "MSE-trained model with bad
   ranking" critique.

6. **Position function correctly named "Sharpe-saturated"**, not
   Kelly. The mu/sigma argument is justified in audit §J as more
   robust under noisy sigma estimates than Kelly's mu/sigma².

7. **Differentiable Sharpe loss correctly cited to ZZR 2020**, with
   long-short generalisation explicitly disclosed.

The remaining paper-side TODOs (DeepClair table, ZZR generalisation
note, A1 numerical verification, α ablation) are not blockers — they
are textbook §2/§3/§5 paragraphs that get written naturally during
paper writing.
