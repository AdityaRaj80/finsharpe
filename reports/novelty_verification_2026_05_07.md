# Novelty Verification Report
**Date:** 2026-05-07
**Target venue:** CIKM 2026
**Items audited:**
1. Theorem A1 — Variance-decomposition identity for top-K vs EW Sharpe gap
2. L_GATE_BCE — Profitability-supervised confidence gate

**Methodology:** Primary-source verification via WebFetch / WebSearch against the
canonical portfolio-optimisation and financial-ML literature listed in the brief,
plus broader keyword searches on Google Scholar / arXiv. Where a primary PDF could
not be parsed within budget, secondary citations are used and flagged honestly.

---

## ITEM 1 — Theorem A1: Σ-orthogonal-projection identity for the SR² gap

### Verdict: 🟡 NOVEL FRAMING

The closed-form Lagrange/Cauchy–Schwarz form of `SR_K² − SR_EW²` is mathematical
folklore (the authors already concede this), and the geometric content of writing
it as a projection in Σ-Mahalanobis geometry is essentially the
Cauchy–Schwarz / Pythagoras step. **However, no published source in the
financial-ML or portfolio-optimisation literature that I could locate states the
identity explicitly as `‖P_{u^⊥} Σ^{−½} μ‖²` with `u = Σ^{½} 𝟙`, packages it with the
three corollaries A1-1/A1-2/A1-3, or ties it to a learned σ-prediction head.** The
MDPI / arXiv "Tangency-portfolio decomposition" paper (Engineering Proceedings 39,
2023) gives a *related* but distinct decomposition (Mahalanobis-norm × cos-similarity-
with-ones for the *u-coefficient* of the efficient frontier), not the SR-gap
identity.

### Closest prior art

1. **Markowitz (1952), "Portfolio Selection," J. Finance 7(1), 77–91** — gives the
   tangency portfolio `w_K ∝ Σ⁻¹ μ` and `SR_K² = μ^⊤ Σ⁻¹ μ` as the maximum SR². Does
   *not* give the gap against equal-weight; the 1/N portfolio is not a benchmark
   in this paper.
2. **Stevens (1998), "On the Inverse of the Covariance Matrix in Portfolio
   Analysis," J. Finance 53(5), 1821–1827, doi:10.1111/0022-1082.00074** — derives
   `Σ⁻¹` via auxiliary regressions of each asset on the rest; gives a regression
   interpretation of the tangency weights. Does *not* state the SR_K² − SR_EW²
   gap.
3. **Britten-Jones (1999), "The Sampling Error in Estimates of Mean-Variance
   Efficient Portfolio Weights," J. Finance 54(2), 655–671,
   doi:10.1111/0022-1082.00120** — recasts mean-variance estimation as an OLS
   regression; uses squared-Sharpe heavily as a likelihood object. Does *not*
   compare to 1/N or give a gap formula.
4. **DeMiguel, Garlappi, Uppal (2009), "Optimal Versus Naive Diversification: How
   Inefficient is the 1/N Portfolio Strategy?" Review of Financial Studies 22(5),
   1915–1953, doi:10.1093/rfs/hhm075** — the canonical 1/N-vs-MV comparison.
   Reports *empirical* Sharpe gaps and gives an analytical estimation-window
   threshold for when MV beats 1/N out-of-sample, but the in-sample (oracle) gap
   formula `μ^⊤ Σ⁻¹ μ − (𝟙^⊤μ)²/(𝟙^⊤Σ𝟙)` is *not* stated in the paper. Their
   focus is estimation error, not the structural decomposition.
5. **Maillard, Roncalli, Teiletche (2010), J. Portfolio Management 36(4), 60–70**
   — equally-weighted *risk contribution* (ERC) portfolios; expresses portfolio
   variance via marginal risk contributions, not the squared-Sharpe gap. Out of
   scope as direct prior art.
6. **Tu & Zhou (2011), J. Financial Economics 99, 204–215** — "Markowitz Meets
   Talmud" optimally combines 1/N with sophisticated rules using the *expected*
   squared Sharpe of efficient portfolios under estimation error (drawing on Kan
   & Zhou 2007). The expected squared SR appears as a single object in
   propositions, not split as a `SR_K² − SR_EW²` projection identity.
7. **Kan & Zhou (2007), JFQA 42(3), 621–656,
   doi:10.1017/S0022109000004129** — distributional results for the squared
   Sharpe under parameter uncertainty; introduces the three-fund (riskless +
   tangency + GMV) optimal rule. Squared Sharpe of estimated MV portfolios
   is studied, but not as `SR_K² − SR_EW² = ‖P_{u^⊥} Σ^{−½} μ‖²`.
8. **Hansen & Jagannathan (1991)** — bounds the SDF variance by the squared Sharpe
   of any portfolio. Closely related to `SR_K² = μ^⊤Σ⁻¹μ` but in the SDF (not
   1/N) framing; no 1/N-vs-MV gap.
9. **MDPI Eng. Proc. 39(1):34 (2023) "Forecasting Tangency Portfolios and
   Investing in the Minimum Euclidean Distance Portfolio…"** — this is the
   closest geometric framing I found. It writes the *u-coefficient* of the
   efficient frontier as `√(r^⊤V⁻¹r) · g(cos(r, 𝟙))`. That is structurally
   adjacent (it isolates a Mahalanobis-norm factor and a similarity-to-ones
   factor) but it is not the SR_K² − SR_EW² gap and the projection identity is
   not stated in the form (*).
10. **Steven Pav, "Notes on the Sharpe Ratio" (2024), CRAN SharpeR vignette;
    "The Sharpe Ratio: Statistics and Applications" (CRC, 2021)** — connects
    `n μ^⊤Σ⁻¹μ` to Hotelling's T². Squared-Sharpe is treated as Hotelling, but
    the explicit 1/N gap and projection form (*) are not stated.

### Citation gap analysis

* **Defensible §1 (contributions) wording:** "We give a clean projection-identity
  form of the squared-Sharpe gap between the σ-aware (Kelly-under-independence)
  portfolio and equal-weight, expressed in Σ-Mahalanobis geometry, and tie it via
  three closed-form corollaries to (i) cross-sectional CV of σ, (ii) homogeneous-μ
  regimes, and (iii) the degenerate equality case μ ∝ σ². The closed-form Lagrange
  identity itself is folklore (Markowitz 1952, Cauchy-Schwarz); our contribution
  is the projection framing and its empirical use as a regime-of-edge predictor
  for ML-based σ-prediction architectures."

* **Over-claiming wording to avoid:** Anything that calls the *math* itself
  "novel," "the first," or "previously unknown." It is not — Cauchy–Schwarz on
  `Σ^½ 𝟙` and `Σ^{-½}μ` is in any standard MV textbook treatment, and the
  Lagrange-identity form follows in 2 lines. Reviewers from the
  Markowitz/DeMiguel-Garlappi-Uppal / Kan-Zhou tradition will reject "novel
  theorem" claims.

* **Note on "diagonal Σ" assumption:** Theorem A1 as stated requires `Σ` *diagonal*.
  For full Σ (with cross-asset correlation), the same Cauchy-Schwarz gives the
  identical projection form, but the §5.2 Lagrange-identity form gets messier
  (cross-terms in σ_i σ_j cov_ij). The paper should state the diagonal assumption
  prominently (it does, in §1) and remark that the projection form (*) is
  assumption-free even for full Σ.

### Recommended framing for the paper

Replace any "we prove a new theorem" wording with:

> "Theorem A1 (folkloric, our packaging). Under diagonal Σ, the squared-Sharpe
> gap admits the projection-identity form `SR_K² − SR_EW² = ‖P_{u^⊥} Σ^{−½} μ‖²`
> with `u = Σ^{½} 𝟙`. The closed-form follows from a single Cauchy–Schwarz step
> (Markowitz 1952 + Lagrange's identity). To our knowledge, this *projection
> framing* and its three corollaries (A1-1, A1-2, A1-3) — which jointly predict
> when ML-based σ-prediction yields the largest Sharpe edge — have not been used
> previously to *guide architecture choice* for cross-sectional financial
> forecasting models."

That phrasing:
* Honest about folklore status (preempts reviewers who will catch it anyway).
* Locates the actual contribution: ML σ-quality → SR-edge regime predictions.
* Does not invite a Markowitz-1952 citation rebuttal.

---

## ITEM 2 — L_GATE_BCE: Profitability-supervised confidence gate

### Verdict: 🟡 ADJACENT PRIOR ART

**Meta-labeling (López de Prado 2018, *Advances in Financial Machine Learning*,
Wiley) is essentially the same idea executed differently.** It trains a *secondary*
binary classifier whose label is `profitable = 1{position × return > 0}` (the
standard triple-barrier-method label of Lopez de Prado, Ch. 3, p. 50; see also
Joubert & Singh, "Does Meta-Labeling Add to Signal Efficacy?", J. Financial Data
Science 2022). The output of the secondary model is then used to size positions —
i.e., it is functionally a learned gate trained on realised P&L.

**The novelty in the paper is therefore narrower than the brief states**, and
specifically:
1. **Joint end-to-end training** of the primary forecast head (μ, σ) and the
   profitability-supervised gate, with the gate constructed as
   `σ((τ_vol − log_vol_pred)/T) · σ((τ_σ − σ̂)/T)` — a *product of sigmoids on
   internal uncertainty estimates*, not a separate classifier on raw features.
2. **Annealed temperature** `T: 1.0 → 0.13` over training, which converts a soft
   continuous gate into a near-binary switch by the end of optimisation.
3. **Co-supervised by both BCE-on-profitability *and* heteroscedastic NLL on the
   underlying σ̂** — i.e., the gate is forced to be high-confidence iff both σ̂
   *and* the realised position-return signs agree.

(1)+(2)+(3) together do appear to be novel; (1) alone does not — meta-labeling
predates this — and (3) closely resembles a standard auxiliary-loss multi-head
network.

### Closest prior art

1. **López de Prado (2018), *Advances in Financial Machine Learning*, Wiley,
   Ch. 3 §3.6 ("Meta-Labeling," p. 50–51)** — direct prior art. Defines
   meta-labels as binary `1{trade was profitable}` and trains a secondary
   classifier (typically random forest / gradient boosting) on them. Does *not*
   integrate the meta-labeller into a single neural network with a heteroscedastic
   NLL head, does *not* use a sigmoid×sigmoid gate, does *not* train end-to-end.
2. **Joubert & Singh (2022), "Does Meta-Labeling Add to Signal Efficacy? Triple
   Barrier Method," J. Financial Data Science 4(3); Joubert "Meta-Labeling:
   Theory and Framework," SSRN 4032018 (2022)** — formal framework for
   meta-labeling, including ensembles. Reports Sharpe-ratio improvements from
   meta-label filtering. Same architectural decoupling: secondary model
   trained sequentially, not jointly.
3. **MASTER (Li et al., AAAI 2024), "Market-Guided Stock Transformer,"
   arxiv:2312.15235** — has a market-guided *gating* mechanism (sigmoidal
   feature selector) but it is supervised *only* through the downstream MSE
   regression loss; the gate is not trained on profitability outcomes. Verified
   by reading §3.2 of the arXiv HTML version.
4. **DeepClair (Choi et al., CIKM 2024), arxiv:2407.13427** — uses pretrained
   transformer time-series forecaster + LoRA fine-tuning for portfolio RL. Gate
   structure (if any) is not BCE-on-profitability supervised; reward signal is
   the standard RL portfolio return.
5. **Moody & Saffell (2001), "Learning to Trade via Direct Reinforcement,"
   IEEE T. Neural Networks 12(4), 875–889, doi:10.1109/72.935097** — direct
   gradient on differential Sharpe. The output position is learned from the
   Sharpe gradient; there is no separate sigmoid gate trained by BCE on
   profitability — instead the entire policy is trained by the reward.
6. **Zhang, Zohren, Roberts (2020), "Deep Learning for Portfolio Optimisation,"
   J. Financial Data Science 2(4)** — direct Sharpe loss on a softmax of
   position weights; no separate confidence gate.
7. **Theate & Ernst (2021), "An Application of Deep Reinforcement Learning to
   Algorithmic Trading," Expert Systems with Applications 173** — DQN-style RL
   trading. Reward-supervised, no BCE-on-profitability gate.
8. **Deng et al. (2017), IEEE TNNLS 28(3) — "Deep Direct Reinforcement Learning
   for Financial Signal Representation and Trading"** — direct RL training; no
   profitability-BCE gate.
9. **Singh & Joubert PA-BCE / "Profit-Aware BCE" (arxiv 2509.16616, "Learn to
   Rank Risky Investors," 2025)** — name-collision warning: this *is* called
   "Profit-Aware Binary Cross Entropy" but it is a *pairwise ranking loss*
   (LambdaRank-style) over traders, weighted by log-P&L gaps — it is *not* a
   per-trade binary-supervised confidence gate. Different mechanism, same
   buzzwords.
10. **Chalkidis (2021), "Trading via Selective Classification,"
    arxiv:2110.14914** — uses selective classification (abstain mechanism) for
    trade signals. The reject option is calibrated for confidence, but the
    confidence head is supervised on the *direction* label, not on
    `1{position × return > 0}`. Adjacent in spirit (a learned "should I trade?"
    head) but with different supervision.
11. **Kendall & Gal (2017), "What Uncertainties Do We Need in Bayesian Deep
    Learning for Computer Vision?" NeurIPS 2017, arxiv:1703.04977** — the
    heteroscedastic-NLL prior art used by `log_sigma2_H`. Not a gate, not
    profitability.
12. **Lim, Arik, Loeff, Pfister (2021), "Temporal Fusion Transformers,"
    Int. J. Forecasting 37(4), 1748–1764** — quantile-loss multi-horizon
    forecaster. No gate, no profitability supervision.

### Citation gap analysis

* **Defensible §1 wording:** "We jointly train, in a single network, three
  prediction heads (μ, σ, log realised vol) together with a continuous
  uncertainty gate constructed as a product of sigmoids on internal σ̂ and σ̂_vol
  estimates. The gate is supervised by binary cross-entropy against the realised
  per-trade P&L sign, in the spirit of meta-labeling (López de Prado 2018) but
  trained end-to-end through the same backbone. To our knowledge, this *joint*
  formulation — heteroscedastic NLL + profitability-BCE on a sigmoid×sigmoid
  uncertainty gate, with annealed temperature — is new in the financial
  time-series literature."

* **Over-claiming wording to avoid:** "First method to supervise an
  uncertainty / confidence head with realised profitability." This is **wrong**
  — meta-labeling has existed since 2018 with exactly that label. Reviewers
  familiar with López de Prado will reject it instantly.

* **Critical citation to add:** López de Prado (2018) Ch. 3 §3.6 must be cited
  in §3 next to the L_GATE_BCE definition. The paper should explicitly say "this
  is a neural-network-internal, end-to-end-trained variant of meta-labeling."
  Failure to cite this is a near-certain reviewer rejection.

### Recommended framing for the paper

Replace any "novel profitability supervision" claim with:

> "L_GATE_BCE is a per-trade binary-cross-entropy term on the realised P&L sign
> (`1{position × return > 0}`), reminiscent of meta-labeling
> (López de Prado, 2018, *Advances in Financial Machine Learning*, Ch. 3). The
> distinguishing aspects of our formulation are: (i) the gate is *internal to
> the forecasting network* and trained jointly with the heteroscedastic μ-σ-vol
> heads via a single back-propagation pass, rather than being a separately
> trained downstream classifier on engineered features; (ii) the gate is
> structurally a product of sigmoids over the network's *own* uncertainty
> estimates (`log_sigma2_H`, `log_vol_pred`), which makes it interpretable as a
> learned dual-threshold uncertainty filter rather than as a black-box scorer;
> and (iii) the temperature `T` is annealed from 1.0 to 0.13 during training,
> bridging the soft-attention regime (early) and the near-binary kill-switch
> regime (late). To our knowledge, this combination has not been published, but
> the *idea* of supervising a position-sizing head with realised trade
> profitability is a 2018 contribution by López de Prado, which we cite and
> build upon."

That phrasing:
* Cites López de Prado upfront — disarms the strongest reviewer attack.
* Locates novelty in (i)+(ii)+(iii) jointly, which *is* defensible.
* Does not pretend the BCE-on-profitability idea itself is new.

---

## §SUMMARY

* **Theorem A1:** 🟡 NOVEL FRAMING. The Cauchy–Schwarz / Lagrange identity is
  folklore (Markowitz 1952; standard MV textbooks); the projection form
  `‖P_{u^⊥} Σ^{−½} μ‖²` and the three corollaries A1-1/A1-2/A1-3 tied to ML
  σ-prediction quality are not in DeMiguel-Garlappi-Uppal 2009, Tu-Zhou 2011,
  Kan-Zhou 2007, Britten-Jones 1999, Stevens 1998, or Maillard-Roncalli-Teiletche
  2010. The MDPI 2023 tangency-portfolio decomposition paper is the closest
  geometric prior art but addresses a different quantity (the u-coefficient of
  the efficient frontier).
* **L_GATE_BCE:** 🟡 ADJACENT PRIOR ART. **Meta-labeling (López de Prado 2018,
  Wiley, Ch. 3) is the direct precursor** — a secondary binary classifier
  trained on `1{trade profitable}`. Our novelty is the joint end-to-end
  formulation with an internal sigmoid×sigmoid gate over the network's own
  uncertainty estimates plus annealed temperature. **The López de Prado
  citation MUST appear in §3.** Failure to cite is a serious reviewer risk.
* **Recommended paper headline:** Lead with the *combination* — a backbone-
  agnostic risk-aware multi-head architecture (μ, σ, vol, gate) trained with a
  composite loss whose gate term is profitability-supervised à la meta-labeling
  but folded into a single end-to-end pass — *justified* by Theorem A1's
  projection identity, which says architectures that learn σ well realise a
  Sharpe edge that scales with cross-sectional σ-heterogeneity. Neither the
  theorem alone nor the gate alone clears the CIKM novelty bar; the *coupling*
  of "theorem-explains-architecture" is the headline.

### Verification honesty notes

* Several primary PDFs (Pav SharpeR vignette; arxiv 2509.16616 PA-BCE;
  arxiv 2407.13427 DeepClair; Joubert/Singh 2022; arxiv 2604.03948
  tangency-decomposition) returned compressed/encoded content via WebFetch and
  could not be parsed inside the budget; verdicts on those use secondary
  citations (Wikipedia, Hudson & Thames blog, MDPI abstract, search-result
  snippets) and are flagged accordingly. The conclusions on those sources are
  consistent across multiple secondary descriptions and I am confident in them,
  but a final paper-ready citation should re-verify the López de Prado (2018)
  Ch. 3 §3.6 page number against the physical Wiley book and the Joubert-Singh
  (2022) DOI against the *J. Financial Data Science* TOC.
* MASTER (AAAI 2024) was verified directly via the arXiv HTML version — gate
  is end-to-end MSE-supervised, *not* BCE-on-profitability. This finding is
  high-confidence.
* Markowitz 1952, Stevens 1998, Britten-Jones 1999, DeMiguel-Garlappi-Uppal
  2009, Maillard-Roncalli-Teiletche 2010, Tu-Zhou 2011, Kan-Zhou 2007 verdicts
  are based on canonical published abstracts/summaries and well-established
  secondary literature; none of these works is reported in any review or
  textbook citation I located as containing the projection identity (*) in the
  stated form. Confidence: high but not 100% — a reviewer with the Tu-Zhou or
  Kan-Zhou paper open *could* point to a single equation that is algebraically
  equivalent to A1 (the math is just Cauchy-Schwarz). The 🟡 verdict — i.e.,
  *framing*, not *math*, is novel — is robust to this risk.

---

### Sources cited (primary)

* Markowitz, H. (1952). Portfolio Selection. *J. Finance* 7(1), 77–91.
* Stevens, G.V.G. (1998). On the Inverse of the Covariance Matrix in Portfolio
  Analysis. *J. Finance* 53(5), 1821–1827. doi:10.1111/0022-1082.00074
* Britten-Jones, M. (1999). The Sampling Error in Estimates of Mean-Variance
  Efficient Portfolio Weights. *J. Finance* 54(2), 655–671.
  doi:10.1111/0022-1082.00120
* DeMiguel, V., Garlappi, L., Uppal, R. (2009). Optimal Versus Naive
  Diversification. *RFS* 22(5), 1915–1953. doi:10.1093/rfs/hhm075
* Maillard, S., Roncalli, T., Teiletche, J. (2010). The Properties of
  Equally-Weighted Risk Contributions Portfolios. *J. Portfolio Management*
  36(4), 60–70.
* Kan, R., Zhou, G. (2007). Optimal Portfolio Choice with Parameter
  Uncertainty. *JFQA* 42(3), 621–656. doi:10.1017/S0022109000004129
* Tu, J., Zhou, G. (2011). Markowitz Meets Talmud. *JFE* 99(1), 204–215.
* Hansen, L.P., Jagannathan, R. (1991). Implications of Security Market Data
  for Models of Dynamic Economies. *JPE* 99(2), 225–262.
* López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  Ch. 3 §3.6 ("Meta-Labeling").
* Joubert, J. (2022). Meta-Labeling: Theory and Framework. SSRN 4032018.
* Singh, A., Joubert, J. (2022). Does Meta-Labeling Add to Signal Efficacy?
  *J. Financial Data Science.*
* Moody, J., Saffell, M. (2001). Learning to Trade via Direct Reinforcement.
  *IEEE T. Neural Networks* 12(4), 875–889. doi:10.1109/72.935097
* Kendall, A., Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep
  Learning for Computer Vision? *NeurIPS.* arxiv:1703.04977
* Lim, B., Arık, S.Ö., Loeff, N., Pfister, T. (2021). Temporal Fusion
  Transformers for Interpretable Multi-Horizon Time Series Forecasting. *IJF*
  37(4), 1748–1764.
* Li, T. et al. (2024). MASTER: Market-Guided Stock Transformer. *AAAI 2024.*
  arxiv:2312.15235
* Choi, D. et al. (2024). DeepClair. *CIKM 2024.* arxiv:2407.13427
* Chalkidis, N. (2021). Trading via Selective Classification.
  arxiv:2110.14914
* Engineering Proceedings 39(1):34 (2023). Forecasting Tangency Portfolios and
  Investing in the Minimum Euclidean Distance Portfolio.
* Pav, S.E. (2024). Notes on the Sharpe Ratio. CRAN SharpeR vignette.
* "Learn to Rank Risky Investors" (2025). arxiv:2509.16616 (PA-BCE name-collision
  reference).

