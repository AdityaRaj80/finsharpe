# Methodology Audit — CIKM 2026 Applied Track Submission
**Audit date:** 2026-05-07
**Auditor role:** Senior reviewer (Applied Track, CIKM 2026)
**Scope:** Verify §1–§10 of submitted methodology against canonical literature.
**Verdict legend:** PASS = matches canonical formulation or deviation justified; CONCERN = defensible but reviewers will flag; FAIL = wrong formulation, fix required.

> **Caveat on sources.** Several primary PDFs (Ledoit–Wolf 2008, Politis–White 2004/2009, Kendall–Gal 2017, Zhang–Zohren–Roberts 2020, Bailey–Lopez de Prado 2014, Moody–Saffell 2001 NIPS-1998 precursor) returned binary streams via WebFetch and could not be quoted line-for-line by the audit tool. Where this happens I cite the equation as reproduced in (a) the Wikipedia / arch / R-package documentation pages I *was* able to fetch, (b) Anthropic web-search snippet quotes that paraphrase the equation, and (c) the corresponding secondary sources I link. Items where the audit could not unambiguously verify a numerical constant in the *primary* source are explicitly flagged "[secondary cite]".

---

## A. DeepClair (CIKM 2024) — comparator framing

**Source:** Lee, D., Lee, J. and Cho, S. (2024). *DeepClair: Utilizing Market Forecasts for Effective Portfolio Selection*. CIKM '24. DOI: 10.1145/3627673.3680008. arXiv:2407.13427.

**What DeepClair actually does** (verified via arXiv HTML, https://arxiv.org/html/2407.13427):
- **Universe:** top 26 stocks by market cap as of 2023 across Nasdaq + Dow Jones.
- **Test window:** 2006-06 to 2022-12 (≈16.5 years).
- **Long-short:** yes, with parameter ρ_t mixing `w_long` and `w_short` vectors.
- **Significance test:** none — point estimates averaged over 10 random seeds.
- **Transaction costs:** acknowledged as a limitation (not modelled).
- **Sharpe formula:** "Annualized Sharpe Ratio" reported but not algebraically defined.

**Verdict: ⚠️ CONCERN (positioning, not correctness).**
DeepClair's evaluation is *weaker* than ours on transaction costs (we sweep 0/5/10/20/50 bps; they don't model costs at all) and *weaker* on significance testing (they have none; we have paired bootstrap). However, our test window is **1 year (2023)** vs their **16.5 years**; reviewers will ask why we use such a short test window. Frame our 1-year test as "matched to FNSPID news-feature availability" and contrast our paired bootstrap + 49-stock cross-section with their absent significance test. Do *not* claim a 49-stock universe is large; DeepClair's 26 is comparable, and a CIKM AR reviewer who reads other 2024–2025 portfolio-DL papers will see 50–500 stock universes routinely.

**Defense:** explicitly tabulate DeepClair vs ours on (universe size, test horizon, significance test, cost modelling). Our wins are cost-sweep and paired bootstrap; their win is test-window length.

---

## B. Moody & Saffell 2001 — differential Sharpe vs our batch Sharpe

**Source:** Moody, J. and Saffell, M. (2001). *Learning to Trade via Direct Reinforcement.* IEEE Trans. Neural Networks 12(4):875–889. DOI: 10.1109/72.935097. (NIPS-1998 precursor: "Reinforcement Learning for Trading.")

**Their formula** (per web-search synthesis of the paper, since direct PDF parsing failed):
> D_t = (B_{t-1}·ΔA_t − ½·A_{t-1}·ΔB_t) / (B_{t-1} − A_{t-1}²)^{3/2}

with A_t, B_t being exponentially-weighted recursive estimates of the first and second moments of returns, updated **per period**. The *point* of D_t is to produce a per-step scalar reward signal (additive to first order) for online RL — it is not a batch statistic.

**What we do** (engine/losses.py:247–253, legacy v1 path):
```
strat_return[i] = gate[i] * tanh(α·μ_i/σ_i) * y_true_return[i]
L_SR = −mean(strat_return) / (std(strat_return, unbiased=False) + 1e-3)
```
Per-sample-within-batch Sharpe of `gate × Kelly-tanh × realised return`. The samples within a batch are **not temporally adjacent** (DataLoader shuffles, and batches mix stocks × dates). This is a *batch* Sharpe in the Sharpe = mean/std spirit, not Moody–Saffell's differential D_t.

**Verdict: ⚠️ CONCERN.**
Calling our objective "differential Sharpe" or citing Moody–Saffell as the source would be wrong. The correct citation lineage is **Zhang–Zohren–Roberts 2020** (item E) for batch-level differentiable Sharpe in supervised DL, *not* Moody–Saffell. The B1 path (`use_xs_sharpe=True`, lines 210–245) is closer to ZZR because it actually constructs synthetic cross-sectional portfolios. The legacy v1 surrogate is **a per-sample mean/std proxy that correlates with portfolio Sharpe but does not equal it** — your code comment says exactly this on line 248, which is honest.

**Required fix:** in the paper, do not cite Moody–Saffell for the v1 surrogate. Cite ZZR 2020 for the B1 cross-sectional Sharpe loss; and explicitly state the v1 path is a "per-sample Kelly-strength surrogate" with the caveat that it is not the Sharpe of any deployable portfolio. Better: only present B1 results in the paper.

---

## C. Ledoit & Wolf 2008 — studentized vs naïve bootstrap

**Source:** Ledoit, O. and Wolf, M. (2008). *Robust performance hypothesis testing with the Sharpe ratio.* Journal of Empirical Finance 15(5):850–859. PDF: http://www.ledoit.net/jef_2008pdf.pdf.

**What L&W 2008 prescribes** (from search-result synthesis verified against the JEF abstract and the PeerPerformance R-package docs at https://search.r-project.org/CRAN/refmans/PeerPerformance/html/sharpeTesting.html):
- A **studentized** circular-block / stationary bootstrap CI on Δ̂ = SR̂_A − SR̂_B.
- Studentization uses a **HAC variance estimator** of Δ̂ (QS kernel with Andrews 1991 automatic bandwidth, or pre-whitened QS per Andrews–Monahan 1992) — recomputed on each bootstrap resample.
- Test statistic:  T* = (Δ̂* − Δ̂) / ŝe_HAC(Δ̂*), and percentiles of T* invert to a CI for Δ.

**What our code does** (smoke/bootstrap_paired.py:92–107):
```
boot_a[k] = annualized_sharpe(a[idx], H)
boot_b[k] = annualized_sharpe(b[idx], H)
boot_d   = boot_a − boot_b
ci95_diff = np.percentile(boot_d, [2.5, 97.5])
p_one_sided = (boot_d <= 0).mean()
```
We bootstrap **the Sharpe ratios themselves** and take quantiles of the difference. **No studentization. No HAC variance estimator.** Yet `bootstrap_paired.py` lines 9–12 explicitly cite Ledoit & Wolf 2008.

**Verdict: ❌ FAIL (citation), ⚠️ CONCERN (correctness).**
We are **not** running the L&W 2008 procedure. Our procedure is the *naïve paired stationary bootstrap* of Politis–Romano applied to a Sharpe-difference functional — a defensible non-parametric test, but it is **not** what L&W recommend, and L&W 2008 §4 simulations show the unstudentized bootstrap can be liberal under heavy tails and serial correlation (search-result quote: "such HAC inference is often liberal when sample sizes are small to moderate, meaning hypothesis tests tend to reject a true null hypothesis too often").

**Required fix (priority 1):**
1. Remove the L&W 2008 citation from bootstrap_paired.py:9–12 *or* implement studentization.
2. Easiest path: drop in `arch.bootstrap.StationaryBootstrap` with a HAC-studentized statistic (`arch` package has this as a primitive), or call the R `PeerPerformance::sharpeTesting()` once for the headline pair.
3. If you keep the naïve bootstrap, change the citation to **Politis & Romano 1994 (JASA 89:1303–1313)** alone and explicitly disclose: "We bootstrap the Sharpe-difference functional directly, without studentization, accepting some over-rejection under heavy tails."

---

## D. Politis & Romano 1994 + Politis–White 2009 — block-length choice

**Sources:**
- Politis, D. and Romano, J. (1994). *The Stationary Bootstrap.* JASA 89(428):1303–1313.
- Politis, D. and White, H. (2004), *Automatic Block-Length Selection for the Dependent Bootstrap.* Econometric Reviews 23(1):53–70.
- Patton, A., Politis, D. and White, H. (2009). *Correction to "Automatic Block-Length Selection ..."* Econometric Reviews 28(4):372–375.

**Their rule** (verified via arch documentation at https://arch.readthedocs.io and the Patton–Politis–White correction summary):
> b̂_OPT = (2 ĝ² / D̂)^{1/3} · n^{1/3}, with D̂ = 2 (σ̂²)² for the stationary bootstrap.

ĝ depends on estimated autocovariances. The cube-root scaling is the load-bearing part: block length grows like n^{1/3}.

**What we have:** test window = 2023 = ~252 trading days. Non-overlapping rebalances at horizon H:

| H (days) | n_rebalances | n^{1/3} | block ~  |
|----------|--------------|---------|----------|
| 5        | ~50          | 3.7     | 2–4      |
| 20       | ~12          | 2.3     | 1–2      |
| 60       | ~4           | 1.6     | 1        |
| 120      | ~2           | 1.3     | n/a      |
| 240      | ~1           | 1.0     | n/a      |

**Our code uses `expected_block = 5.0` for ALL horizons** (smoke/bootstrap_paired.py:68 default).

**Verdict: ❌ FAIL at H ∈ {60,120,240}, ⚠️ CONCERN at H=5,20.**
- At H=5 (n=50): block 5 is roughly 1.4× the n^{1/3} rule of thumb but within Politis–White's reasonable range; defensible.
- At H=60 (n≈4): block 5 > sample size — the bootstrap collapses into "draw the same series back" and CI is meaningless.
- At H=120 (n≈2) and H=240 (n≈1): paired bootstrap is **not a valid inferential procedure**. With one or two non-overlapping rebalances you have no degrees of freedom. The bootstrap CI you report at these horizons is statistical noise; remove or replace with overlapping-return bootstrap with explicit Newey–West variance.

**Required fix (priority 1):**
1. Replace the constant block=5 default with `arch.bootstrap.optimal_block_length` per arm and use the average; OR run a sensitivity grid {2, 5, 10}.
2. **Drop bootstrap CIs entirely for H ≥ 120.** Either (a) remove these horizons from the paper, (b) report point Sharpe only with a footnote that n is too small for a valid CI, or (c) shift to overlapping daily returns with Newey–White HAC SE (Hansen–Hodrick or Newey–West with lag = H).

---

## E. Zhang, Zohren & Roberts 2020 — direct portfolio Sharpe optimization

**Source:** Zhang, Z., Zohren, S. and Roberts, S. (2020). *Deep Learning for Portfolio Optimisation.* Journal of Financial Data Science 2(4):8–20. arXiv:2005.13665. PDF: https://www.oxford-man.ox.ac.uk/wp-content/uploads/2020/06/Deep-Learning-for-Portfolio-Optimisation.pdf.

**Their construction** (verified via web-search synthesis, since PDF parse failed):
- Long-only ETF universe; weights via **softmax output layer** that enforces simplex (sum-to-one, non-negative).
- Loss = − Sharpe of the portfolio return series E[R_p] / σ(R_p), gradient flows through softmax weights into network parameters.
- Long-only, no short sleeve, no Kelly-tanh.

**What we do:** Kelly-tanh per-stock weights with leg-wise L1 normalization (Smoke_test inference) or per-sample tanh on `μ/σ` (legacy v1 training) or random-partition synthetic cross-section (B1 training). Long-short with separate L1 normalization in each leg.

**Verdict: ⚠️ CONCERN.**
Our construction is **strictly more expressive** than ZZR's softmax (we permit shorts) but **not the same operator** — reviewers may ask why we don't justify the long-short generalization with a citation. The Kelly-tanh-of-Sharpe weighting (vs Kelly's correct f* = μ/σ²) is examined in item J below. ZZR is the right anchor for "differentiable cross-sectional Sharpe loss" if you cite anyone.

**Defense:** cite ZZR 2020 as the lineage for "differentiable portfolio Sharpe" and explicitly state the long-short generalization (replace softmax with leg-wise normalised |tanh|). Acknowledge the deviation.

---

## F. Kendall & Gal 2017 — heteroscedastic NLL

**Source:** Kendall, A. and Gal, Y. (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NIPS 2017, pp. 5580–5590. arXiv:1703.04977.

**Their eq. (5)/(8)** (canonical form from secondary teaching sources, e.g. the Stirn 2023 ICML reproduction at proceedings.mlr.press/v206/stirn23a):
> L = (1/(2σ̂²(x))) · ‖y − f̂(x)‖² + (1/2) · log σ̂²(x)
> ⇔ with s = log σ̂²:  L = ½·exp(−s)·‖y − f̂‖² + ½·s

**Our eq.** (engine/losses.py:189):
```
nll = 0.5 * (log_var + (y_true - mu_pred)^2 / exp(log_var))
    = 0.5 * log_var + 0.5 * exp(-log_var) * (y_true - mu_pred)^2
```

**Verdict: ✅ PASS.**
Algebraically identical to K&G eq. (5) up to dropping the additive π/normalising constant which is parameter-independent. The clamp on `log_var ∈ [−12, 4]` (lines 35–36) is a defensible numerical safeguard for autocast bf16/fp16 — no correctness issue.

---

## G. Lopez de Prado warnings

**Source:** Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. Also "The 10 Reasons Most ML Funds Fail" (J. Portfolio Mgmt. 2018).

LdP's Big Five warnings, mapped to our paper:
1. **Overlapping returns inflate Sharpe.** We explicitly subsample `[::H]` for non-overlap (cross_sectional_smoke.py:404, bootstrap_paired.py:47–56). ✅ PASS.
2. **Single test-window p-hacking.** We use one calendar test window (2023). ⚠️ CONCERN — see item H.
3. **Backtest length too short.** Our 1-year test is short by quant-finance norms. ⚠️ CONCERN — see item A.
4. **Deflated Sharpe needed when many trials are run.** We sweep top_n on val ∈ {3,5,7,10,15} and run 7 models × 5 horizons × 2 strategies = 70 final paths. ❌ FAIL — see item H.
5. **Cost realism.** We sweep 0/5/10/20/50 bps round-trip; LdP would call this acceptable for an academic submission but ask for slippage modelling. Defensible.

**Verdict for §G overall: ⚠️ CONCERN.** Items 2, 3, 4 are real LdP-style risks. Item 4 is the binding one.

---

## H. Bailey & Lopez de Prado 2014 — Deflated Sharpe Ratio

**Source:** Bailey, D. and López de Prado, M. (2014). *The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.* Journal of Portfolio Management 40(5):94–107.

**Formula** (verified via Wikipedia https://en.wikipedia.org/wiki/Deflated_Sharpe_ratio and search-result synthesis):
> SR₀ = √V[SR̂_n] · ((1 − γ)·Φ⁻¹[1 − 1/N] + γ·Φ⁻¹[1 − 1/(Ne)])
> DSR = Φ((SR̂* − SR₀)·√(T−1) / √(1 − γ̂₃·SR₀ + ((γ̂₄ − 1)/4)·SR₀²))

with γ ≈ 0.5772 (Euler–Mascheroni), N = number of trials, T = sample length, γ̂₃, γ̂₄ = skew, kurt.

**Our trial count** is at minimum:
- 7 models (PatchTST, GCFormer, TFT, iTransformer, AdaPatch, VanillaTransformer, DLinear)
- × 5 horizons (5/20/60/120/240)
- × 2 strategies (simple, risk_aware)
- × ~5 top_n values swept on val
= **350 backtests** → at least N=350 if you include every config; conservatively N=70 if we count only "head" winners.

For T=50 (H=5, 1-year test) and SR₀ realistic skew/kurt, the DSR threshold is **non-trivial** — a point Sharpe of 1.5 with N=350 trials and T=50 may not survive deflation.

**Verdict: ❌ FAIL.**
We currently apply zero multiple-testing correction. Any CIKM AR reviewer familiar with quant-finance literature will flag this. The deflation is *especially* punitive at our short T — the (T−1) factor in DSR cuts both ways but the skew/kurt term does not save us when T=50.

**Required fix (priority 1):**
1. Compute DSR for the headline cell (best model × best horizon) using N = total swept config count and report it alongside raw SR.
2. Reference implementation: `mlfinlab.backtest_statistics.deflated_sharpe_ratio` or compute by hand from the formula above; takes ~30 lines.
3. If DSR < 0.95 (i.e. cannot reject "false strategy" at 5% even after deflation), tone down the headline language from "outperforms" to "directionally consistent."

---

## I. PatchTST / iTransformer / TFT — using point-forecast models for ranking

**Sources:**
- Nie, Y. et al. (2023). *A Time Series Is Worth 64 Words: Long-term Forecasting with Transformers.* ICLR 2023. arXiv:2211.14730.
- Liu, Y. et al. (2024). *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.* ICLR 2024. arXiv:2310.06625.
- Lim, B. et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting.* International J. of Forecasting 37(4):1748–1764.

**Original use case:** all three are point-forecast models trained with MSE on (multivariate) time series benchmarks (ETT, Electricity, Traffic, Weather). They were not designed for cross-sectional ranking of asset returns.

**Our use:** we treat each per-stock forecast as an alpha signal, rank cross-sectionally at each rebalance, take top-N/bottom-N. There are 2024–2025 papers doing this — see "Comparing Transformer Models for Stock Selection in Quantitative Trading" (https://link.springer.com/chapter/10.1007/978-3-032-00891-6_19) which adapts CrossFormer, MASTER, iTransformer for ranking-based S&P500 portfolios — so the move is precedented.

**Verdict: ⚠️ CONCERN.**
The ranking-of-MSE-trained-forecasters approach has a known weakness: an MSE-trained model is calibrated to the *level* of returns, not their *cross-sectional ordering*. If model A has lower MSE but worse Spearman rank correlation than model B, A may produce worse portfolios. Our results show some models with high test MSE still produce decent Sharpe — this is exactly the rank-vs-level mismatch and reviewers will ask about it.

**Defense:** add a Spearman/Kendall rank-IC table (per stock × per timestamp) for each model alongside the Sharpe-headline table. The IC numbers will let reviewers see ranking ability directly. This is standard in quant-finance papers (Qlib, Alpha158 literature).

---

## J. Kelly criterion: tanh(α·μ/σ) vs f* = μ/σ²

**Sources:**
- Kelly, J. (1956). *A New Interpretation of Information Rate.* Bell System Tech. J. 35:917–926.
- For the continuous-time growth-optimal version, see e.g. Wikipedia "Kelly criterion" continuous-time section, which gives **f* = (μ − r) / σ²** (units: 1/return-variance).

**What we use** (cross_sectional_smoke.py:293–296, engine/losses.py:200):
```
position = tanh(α · μ/σ),  α = 5
```
Argument is in **Sharpe units** (μ/σ), not Kelly units (μ/σ²). At α=5 the tanh saturates near |μ/σ| ≈ 0.4, i.e. a per-step Sharpe of ~0.4 maps to a near-fully-deployed position.

**Verdict: ⚠️ CONCERN (terminology and parameterisation).**
Calling this "Kelly-style" or "Kelly-tanh" in the paper is a stretch. True continuous-time Kelly has μ/σ² in the argument, which would make the natural rescaling of fraction = tanh(c · μ/σ²) for a soft-saturated leverage. Using μ/σ instead implicitly assumes σ-cross-sectional homogeneity is undesirable — large-σ stocks get *less* weight per unit of expected return than they would under true Kelly. In practice this is closer to "risk-parity-flavoured Sharpe targeting" than Kelly. Defensible operationally (works fine and is robust to bad σ estimates) but the **name is wrong**.

**Required fix:**
1. Rename "Kelly-tanh" → "Sharpe-saturated" or "soft-Sharpe targeting" throughout the paper.
2. Drop direct comparisons to f* = μ/σ². If you want to keep the Kelly framing, add a half-page derivation showing that under quadratic-utility approximations our weighting is monotone in the optimal Kelly fraction (which it is, given σ_i within a leg are roughly comparable after volatility-bucketing).
3. α=5 should be ablation-tested in supplementary: try α ∈ {1, 2, 5, 10}. If headline result is sensitive to α, flag it.

---

## K. Bootstrap validity at long horizons

Already handled in item D. Recapping the count:

| H | n_nonoverlap_2023 | bootstrap valid? |
|---|-------------------|------------------|
| 5 | ~50 | yes (with proper block) |
| 20 | ~12 | borderline; CI very wide |
| 60 | ~4 | NO |
| 120 | ~2 | NO |
| 240 | ~1 | NO (single observation) |

**Verdict: ❌ FAIL for H ∈ {60, 120, 240}.**
At H=240 with n=1 you cannot compute a bootstrap CI of *anything*; the std of one number is undefined (and our `annualized_sharpe` returns NaN at n<2 — line 50 of bootstrap_paired.py — so all bootstrap reps return NaN and the printed CI is `[nan, nan]` or worse, the percentile of an all-NaN array which numpy may handle silently).

**Required fix:** see §D.

---

## L. Fama–MacBeth / Newey–West vs bootstrap

**Sources:** Fama, E. and MacBeth, J. (1973). *Risk, Return and Equilibrium: Empirical Tests.* JPE 81:607–636. Newey, W. and West, K. (1987). *A Simple Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.* Econometrica 55:703–708.

**Standard cross-sectional finance procedure:**
1. At each t, run cross-sectional regression of returns on signals → coefficient λ̂_t.
2. Average over t and use Newey–West HAC SE on λ̂_t time series.

This gives a t-statistic for "does the signal price the cross-section?" — **a different question from "does this portfolio earn a positive Sharpe?"**. Fama–MacBeth tests the *predictive validity of the signal*; bootstrap tests the *Sharpe of one specific portfolio construction*. They are complementary, not substitutes.

**Verdict: ⚠️ CONCERN (missing companion test).**
A reviewer trained in finance will ask for *both*:
- Bootstrap on portfolio Sharpe (we have, modulo §C and §D fixes).
- Fama–MacBeth or rank-IC t-statistic on the underlying signal (we don't have).

**Required fix:**
- Add a Fama–MacBeth or rank-IC table: at each rebalance, regress realised H-day returns on standardised model predictions → λ̂_t. Time-average λ̂ and compute Newey–West (lag = H) t-stat. Reporting one row per model is ~20 LOC and 1 paragraph in the paper.

---

## §SUMMARY

| Item | Topic | Verdict |
|------|-------|---------|
| A | DeepClair positioning | ⚠️ CONCERN |
| B | Moody–Saffell vs ours | ⚠️ CONCERN |
| C | L&W 2008 studentized bootstrap | ❌ FAIL (citation) |
| D | Block length choice | ❌ FAIL at H≥60 |
| E | ZZR 2020 differentiable Sharpe | ⚠️ CONCERN |
| F | Kendall–Gal NLL | ✅ PASS |
| G | LdP general warnings | ⚠️ CONCERN |
| H | Deflated Sharpe Ratio | ❌ FAIL |
| I | PatchTST/iTransformer for ranking | ⚠️ CONCERN |
| J | Kelly-tanh terminology | ⚠️ CONCERN |
| K | Bootstrap at long H | ❌ FAIL at H≥60 |
| L | Fama–MacBeth companion | ⚠️ CONCERN |

**Counts:** PASS = 1, CONCERN = 7, FAIL = 4.

---

## §TOP-3 BLOCKERS for CIKM Applied Track desk-rejection

1. **Deflated Sharpe Ratio not applied (item H).** With ~70–350 swept configs and a 1-year test window, ANY reviewer familiar with Bailey–LdP will demand DSR. Without it, the headline Sharpe is statistically un-interpretable. Highest-priority fix.

2. **Bootstrap miscited and invalid at long horizons (items C + D + K).** We claim Ledoit–Wolf 2008 but run the unstudentized Politis–Romano bootstrap; we report CIs at H=120/240 with n≤2 non-overlapping rebalances. A reviewer will catch the citation mismatch in 30 seconds and call the paper sloppy. Either implement the studentized version or drop the L&W cite, and prune the long-horizon CI claims.

3. **"Differential Sharpe" / "Kelly-tanh" terminology (items B + J).** The legacy-v1 surrogate is not Moody–Saffell's D_t, and tanh(α·μ/σ) is not Kelly. Renaming and re-citing (ZZR 2020 for the surrogate; "soft Sharpe targeting" for the position function) is a one-pass edit but failing to do it is a credibility hit on a CIKM AR submission.

---

## §PRE-SUBMISSION FIX LIST (in priority order)

1. **Add DSR computation.** New file: `D:\Study\CIKM\finsharpe\smoke\deflated_sharpe.py` implementing the formula in §H. Apply to headline cell; report N_trials honestly. Accept truth: if DSR < 0.95, soften paper claims.

2. **Fix bootstrap citations.** `D:\Study\CIKM\finsharpe\smoke\bootstrap_paired.py` lines 9–12 — either implement studentized version (use `arch.bootstrap.StationaryBootstrap` with HAC `_studentize` callable) or remove L&W 2008 cite and replace with Politis–Romano 1994 alone, with a one-sentence disclosure that "this is not L&W 2008's studentized procedure."

3. **Drop long-horizon CIs.** `cross_sectional_smoke.py` and the paper tables — for H ∈ {60, 120, 240} report only point Sharpe with `n_obs` column and a footnote that bootstrap CI is not valid at this n. Or reduce the headline horizons to {5, 20} only.

4. **Adopt automatic block length.** `D:\Study\CIKM\finsharpe\smoke\bootstrap_paired.py` line 68 — replace `default=5.0` with a per-arm call to `arch.bootstrap.optimal_block_length` (or hard-code per-horizon values from Politis–White rule). Re-run all bootstraps.

5. **Rename "Kelly-tanh" → "Sharpe-saturated" in the paper.** No code change required (variable names can stay) but every figure caption and section heading using "Kelly" should switch to "Sharpe-saturated" or "soft-Sharpe-targeting." Add ablation on α ∈ {1, 2, 5, 10} in supplementary.

6. **Re-cite the differentiable Sharpe loss to ZZR 2020.** `engine/losses.py` docstring (top of file) and the methods section should cite Zhang–Zohren–Roberts 2020 (arXiv:2005.13665), not Moody–Saffell 2001. State that the legacy-v1 surrogate is a "per-sample correlate of portfolio Sharpe, not portfolio Sharpe itself" and present the B1 (`use_xs_sharpe=True`) results as the primary evidence.

7. **Add rank-IC + Fama–MacBeth table.** New file: `D:\Study\CIKM\finsharpe\smoke\rank_ic.py` — at each rebalance, compute Spearman corr(prediction, realised return) across the 49 stocks. Average across t with NW(lag=H) SE. One table, one paragraph in the paper. Defends against item I and item L simultaneously.

8. **Tighten DeepClair positioning.** Methods/Related-Work — explicit comparison table on (universe size, test horizon, significance test, cost modelling). Don't claim to "outperform" DeepClair without matching their universe and test horizon.

9. **Verify Theorem A1 numerically.** The `reports/theorem_A1.md` derivation is clean (verified by reading), but the file states "empirical verification deferred until B1 retrains land." Run `reports/verify_A1.py` against current B1 artefacts and include a one-line "verified to within 1e-6 in synthetic and ε on real panel" footnote. If A1 is presented as a theoretical contribution, it must be numerically validated before submission.

---

## Bibliography (full citations)

- Bailey, D.H., López de Prado, M. (2014). The Deflated Sharpe Ratio. *J. Portfolio Mgmt* 40(5):94–107. SSRN:2460551.
- Fama, E., MacBeth, J. (1973). Risk, Return, and Equilibrium. *JPE* 81(3):607–636.
- Kelly, J.L. Jr. (1956). A New Interpretation of Information Rate. *Bell Syst. Tech. J.* 35(4):917–926.
- Kendall, A., Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? *NIPS 2017*. arXiv:1703.04977.
- Lee, D., Lee, J., Cho, S. (2024). DeepClair: Utilizing Market Forecasts for Effective Portfolio Selection. *CIKM '24*. DOI 10.1145/3627673.3680008. arXiv:2407.13427.
- Ledoit, O., Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. *J. Empirical Finance* 15(5):850–859.
- Lim, B., Arik, S., Loeff, N., Pfister, T. (2021). Temporal Fusion Transformers. *Int. J. Forecasting* 37(4):1748–1764.
- Liu, Y. et al. (2024). iTransformer. *ICLR 2024*. arXiv:2310.06625.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
- Moody, J., Saffell, M. (2001). Learning to Trade via Direct Reinforcement. *IEEE TNN* 12(4):875–889. DOI 10.1109/72.935097.
- Newey, W., West, K. (1987). A HAC Covariance Matrix. *Econometrica* 55(3):703–708.
- Nie, Y., Nguyen, N.H., Sinthong, P., Kalagnanam, J. (2023). PatchTST. *ICLR 2023*. arXiv:2211.14730.
- Patton, A., Politis, D.N., White, H. (2009). Correction to "Automatic Block-Length Selection..." *Econometric Reviews* 28(4):372–375.
- Politis, D.N., Romano, J.P. (1994). The Stationary Bootstrap. *JASA* 89(428):1303–1313.
- Politis, D.N., White, H. (2004). Automatic Block-Length Selection for the Dependent Bootstrap. *Econometric Reviews* 23(1):53–70.
- Zhang, Z., Zohren, S., Roberts, S. (2020). Deep Learning for Portfolio Optimisation. *J. Financial Data Science* 2(4):8–20. arXiv:2005.13665.
