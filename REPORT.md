# Responsible AI for Credit Scoring
### A Fairness-Audited Machine Learning System on the German Credit Dataset

*University project — Responsible AI in the financial domain, July 2026*

---

## 1. Executive Summary

This project builds a credit-risk classification system on the German Credit Dataset (1,000 applicants) and subjects it to a full Responsible AI evaluation. Three models spanning the interpretability–performance spectrum were trained with cost-sensitive learning; the best (Random Forest, ROC-AUC 0.770, bad-credit recall 0.75) was then audited for group fairness and explained with SHAP.

**Headline findings:**
1. The model **fails the 80% disparate-impact rule for age** (ratio 0.50): applicants aged ≤25 are wrongly denied credit at 59%, roughly double the rate of 26–60-year-olds (29%).
2. Fairness by sex is borderline (disparate impact 0.80); creditworthy women are wrongly denied at 41.3% vs 31.1% for men.
3. A simple **mitigation — group-specific decision thresholds — repaired the sex and age disparities** (sex 0.80 → 0.96; age 0.50 → 0.82, halving the young-applicant wrongful-denial rate) at little accuracy cost. But the same fix applied to **foreign-worker status backfires** (recall collapses 0.76 → 0.47), because that group is 96% of the data — showing fairness mitigation is not always free and sometimes demands a policy decision, not an automatic rule.
4. Bringing in the **full 20-feature UCI dataset** both raises performance (ROC-AUC 0.770 → 0.783) and exposes a **third bias the 9-feature subset cannot show**: foreign workers are approved at ~54% vs ~76% for non-foreign workers (disparate impact 0.71 — also a failure).
5. SHAP explanations confirm the model reasons primarily from financial signals (checking account, duration, amount) but also uses Sex directly — motivating continuous fairness monitoring regardless of feature choices.

The system is designed as **decision support with a human in the loop**, consistent with the EU AI Act's classification of credit scoring as high-risk and GDPR Art. 22's right to explanation.

---

## 2. Data Analysis (Week 1)

- **Dataset:** 1,000 loan applicants, 9 features + binary target (70% good / 30% bad credit). Kaggle subset of the original 20-feature UCI dataset.
- **Feature types:** numerical (Age, Credit amount, Duration), ordinal (Job, Saving/Checking account tiers), nominal (Sex, Housing, Purpose).
- **Data quality:** no duplicates; missing values only in Saving accounts (~18%) and Checking account (~39%), interpreted as *"no account exists"* and encoded as an explicit `none` category rather than imputed.
- **Key EDA insights:** checking account status is the strongest risk signal; duration and credit amount are strongly correlated (~0.6) and both increase risk; younger applicants show elevated historical default rates; credit amount is right-skewed (log-transformed, not trimmed).
- **Sensitive attributes:** Age and Sex (present); Job as a socioeconomic proxy; foreign-worker status exists in the full UCI file. Personal and financial attributes are entangled, so dropping sensitive columns cannot guarantee fairness ("fairness through unawareness" fails via proxies).
- **Class imbalance:** the 70/30 split plus the dataset's 5:1 misclassification cost matrix make accuracy misleading; **recall on the bad class** is the operationally correct primary metric.

## 3. Methodology (Week 2)

**Pipeline** ([src/preprocessing.py](src/preprocessing.py)): missing-as-category handling → log-transform + scaling of skewed numericals → ordinal encoding of account tiers and job level → one-hot encoding of nominal features → target mapped to *bad = 1* so recall measures risky-applicant detection. A separate audit frame (Age, Sex, Risk) feeds the fairness layer independently of the model's inputs.

**Models** ([src/train_baseline.py](src/train_baseline.py)) — chosen along the interpretability–performance spectrum, all with balanced class weights (cost-sensitive learning), stratified 80/20 hold-out plus 5-fold cross-validation:

| Model | Role |
|---|---|
| Logistic Regression | Fully transparent regulatory baseline |
| Decision Tree (depth 5) | Human-readable decision rules |
| Random Forest (300 trees) | Performance reference |

## 4. Results

**Hold-out test set (n = 200, positive class = bad credit):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.645 | 0.432 | 0.583 | 0.496 | 0.664 |
| Decision Tree | 0.620 | 0.425 | **0.750** | 0.542 | 0.695 |
| Random Forest | **0.690** | **0.489** | 0.750 | **0.592** | **0.770** |

Cross-validated recall (LR 0.65 / DT 0.68 / RF 0.75, ±0.04–0.08) confirms the ranking is stable. Accuracy sits below the naive 70% "approve everyone" baseline *by design*: cost-sensitive weighting trades accuracy for detection of risky applicants, the economically correct trade for a lender. **Random Forest** was selected for the Responsible AI audit (best AUC and F1, near-best recall).

Artifacts: [roc_curves.png](results/roc_curves.png), [confusion_matrices.png](results/confusion_matrices.png), [baseline_metrics.csv](results/baseline_metrics.csv).

### 4.1 Feature-engineering experiment — can more preprocessing raise performance?

Before accepting the baseline, we tested whether engineering new features from the 9 Kaggle columns could improve the model, using 5-fold cross-validation on all 1,000 rows ([experiments/feature_engineering.py](experiments/feature_engineering.py)). Note these are cross-validated estimates, so the baseline reads slightly differently from the single hold-out split in §4.

| Variant | Accuracy | Recall (bad) | ROC-AUC |
|---|---|---|---|
| RF baseline (current pipeline) | 0.688 | **0.753** | 0.759 |
| RF + MonthlyPayment (Credit ÷ Duration) | 0.710 | 0.733 | **0.765** |
| RF + Monthly + CreditPerAge | 0.718 | 0.733 | 0.763 |
| RF + Monthly + DurationBin | 0.708 | 0.700 | 0.761 |
| RF **without** class_weight (accuracy-max) | **0.723** | 0.153 | 0.760 |
| Logistic Regression + engineered features | 0.636 | 0.623 | 0.677 |

Three findings, each of which we treat as a deliberate design decision:

1. **A `MonthlyPayment` feature (installment burden) is the only real gain** — about +1.6 points accuracy and +0.7 points AUC. But the improvement is marginal, within cross-validation noise, and comes with a recall drop at the fixed 0.5 threshold. Stacking further engineered features (`CreditPerAge`, `DurationBin`) adds essentially nothing.
2. **The "accuracy-max" row is the accuracy trap made explicit.** Removing cost-sensitive weighting yields the *highest* accuracy in the table (0.723) while recall on bad credit **collapses to 0.15** — the model would miss 70% of risky applicants. This is direct, quantitative evidence for why we optimise recall, not accuracy.
3. **The genuine performance lever is more features, not transformed features.** Engineering on the 9-column subset cannot beat the information ceiling; the real improvement came from adding the missing 11 features in the full UCI dataset (§5.1, AUC 0.770 → 0.783).

**Decision:** we kept the simpler, more interpretable baseline pipeline. The marginal, recall-costing gain from `MonthlyPayment` did not justify the added complexity, and the full-UCI model already captures the meaningful headroom.

## 5. Fairness Audit

([src/fairness_audit.py](src/fairness_audit.py)) — computed on 5-fold **out-of-fold predictions for all 1,000 applicants** (larger, more stable group slices than the 200-row test set). Convention: predicted bad = credit denied.

**By Sex:**

| Group | n | Approval rate | TPR (bad detected) | FPR (good denied) |
|---|---|---|---|---|
| Male | 690 | 0.572 | 0.733 | 0.311 |
| Female | 310 | 0.455 | 0.789 | **0.413** |

**By Age band:**

| Group | n | Approval rate | TPR (bad detected) | FPR (good denied) |
|---|---|---|---|---|
| 26–60 | 765 | 0.593 | 0.714 | 0.290 |
| ≤25 | 190 | **0.295** | 0.863 | **0.591** |
| >60 | 45 | 0.578 | 0.700 | 0.343 |

**Summary:**

| Attribute | Demographic parity diff | Disparate impact (80% rule) | Equal opportunity diff | Equalized odds diff |
|---|---|---|---|---|
| Sex | 0.117 | 0.795 — borderline | 0.056 | 0.102 |
| Age band | 0.298 | **0.497 — fail** | 0.163 | 0.301 |

**Interpretation.** The age disparity is the dominant fairness failure: a creditworthy applicant under 26 has a **55% chance of being wrongly denied** — the model has learned and amplified the historical penalty against young borrowers identified in the Week 1 EDA. The sex disparity is smaller but real, and concentrated in wrongful denials (FPR gap ~10 pts) rather than risk detection (TPR gap 3 pts) — i.e., the harm falls on creditworthy women.

**Mitigation.** We apply group-specific decision thresholds ([src/fairness_audit.py](src/fairness_audit.py) `equalize_fpr_thresholds`), chosen so each group's wrongful-denial rate (FPR) meets the lowest group's — targeting the *direct harm* to applicants. Applied to **both** Sex and Age band:

| Attribute | Scenario | Disparate impact | Equalized odds diff | Accuracy | Recall (bad) |
|---|---|---|---|---|---|
| Sex | Baseline | 0.795 | 0.102 | 0.688 | 0.753 |
| Sex | **Mitigated** | **0.958** | **0.027** | 0.699 | 0.723 |
| Age band | Baseline | 0.497 | 0.301 | 0.688 | 0.753 |
| Age band | **Mitigated** | **0.821** | 0.414 | 0.697 | 0.667 |

For **Sex**, the disparity is nearly eliminated at zero accuracy cost. For **Age**, the mitigation lifts the disparate-impact ratio **0.50 → 0.82 (now passing the 80% rule)** and roughly **halves the young-applicant wrongful-denial rate (59% → 29%)**, at a ~9-point recall cost. One honest caveat: on this 9-feature model the age equalized-odds *gap widens slightly* (0.30 → 0.41), because the tiny >60 group (n=45) is statistically unstable — an artefact the richer 20-feature model does not share (§5.2). The correction is a documented business/policy decision owned by accountable humans, not an automatic rule.

Artifacts: [fairness_group_metrics.png](results/fairness_group_metrics.png), [fairness_approval_rates.png](results/fairness_approval_rates.png), [mitigation_comparison.csv](results/mitigation_comparison.csv).

### 5.1 Extended model — the full 20-feature UCI dataset

The Kaggle file is a 9-feature subset. To honour the brief's *"if you can find additional information, use it"*, we also decoded the **original UCI file** ([data/german.data](data/german.data)) — 20 features including credit history, employment length, and two sensitive attributes the subset omits: personal status & sex, and **foreign-worker status** ([src/full_uci.py](src/full_uci.py)). The same Random Forest, on the same hold-out split, improves with the richer data:

| Feature set | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Kaggle subset (9 features) | 0.690 | 0.489 | 0.750 | 0.592 | 0.770 |
| **Full UCI (20 features)** | 0.695 | 0.495 | **0.783** | **0.606** | **0.783** |

More consequentially, the full data unlocks a fairness slice **the subset structurally cannot produce**:

| Group | n | Approval rate | TPR (bad detected) | FPR (good denied) |
|---|---|---|---|---|
| Foreign worker = yes | 963 | 0.588 | 0.703 | 0.283 |
| Foreign worker = no | 37 | **0.811** | 0.500 | 0.152 |

Foreign workers are approved at ~54% versus ~76% for non-foreign workers — **disparate impact 0.71, a fresh failure of the 80% rule**. Because ~96% of applicants are foreign workers, the *advantaged* group is a fragile 37-person minority. The lesson reinforces the audit's core thesis: had we simply dropped the sensitive column, this disparity would have been hidden, not removed. Top drivers of the richer model (checking status 0.27, duration 0.13, credit history 0.08, credit amount 0.08, employment length 0.06, age 0.05) confirm the added features carry genuine signal. Artifact: [full_uci_comparison.png](results/full_uci_comparison.png), [full_uci_fairness.csv](results/full_uci_fairness.csv).

### 5.2 Mitigation on the full model — and where it breaks

We applied the same FPR-equalizing threshold mitigation to the 20-feature model, for both Age and Foreign worker ([full_uci_mitigation.csv](results/full_uci_mitigation.csv)):

| Attribute | Scenario | Disparate impact | Equalized odds diff | Accuracy | Recall (bad) |
|---|---|---|---|---|---|
| Age band | Baseline | 0.543 | 0.283 | 0.697 | 0.757 |
| Age band | **Mitigated** | **0.921** | **0.201** | 0.707 | 0.680 |
| Foreign worker | Baseline | 0.707 | 0.247 | 0.697 | 0.757 |
| Foreign worker | **Mitigated** | **0.927** | 0.034 | 0.734 | **0.467** |

**Age is a clean win** on the richer model — disparate impact 0.54 → 0.92 *and* equalized odds *improves* (0.28 → 0.20), at a ~9-point recall cost. The larger feature set stabilises the estimate that the 9-feature model could not.

**Foreign worker is a genuine trade-off, not a free fix.** The threshold adjustment does lift disparate impact (0.71 → 0.93), but because foreign workers are ~96% of all applicants, raising their threshold to cut wrongful denials also makes the model miss far more bad loans — **overall recall collapses from 0.76 to 0.47**. This is the most important nuance in the whole audit: *fairness mitigation is not always free.* When the disadvantaged reference group is a fragile 4% minority, a naive threshold fix trades away the model's core purpose. The responsible response is to treat foreign-worker fairness as a **policy decision** — a partial adjustment, a reweighing / in-processing method, or a governance sign-off — rather than an automatic threshold. We surface the trade-off explicitly rather than reporting a headline "fixed."

## 6. Explainability

([src/explainability.py](src/explainability.py)) — SHAP TreeExplainer on the Random Forest.

**Global drivers (mean |SHAP| toward P(bad)):** Checking account **0.106**, Duration 0.052, Saving accounts 0.023, Credit amount 0.023, Housing=own 0.017, Age 0.015. The model reasons primarily from financial capacity and loan structure — consistent with the EDA — but the non-zero attributions for Age and Sex show sensitive attributes influence individual decisions directly (the beeswarm shows *male* lowers predicted risk), reinforcing the audit's necessity.

**Local explanations:** SHAP waterfall plots for a denied applicant (P(bad)=0.63 — driven by low checking/saving balances and 24-month duration) and an approved one (P(bad)=0.35 — rich savings, moderate checking) provide the per-decision reasoning required for adverse-action notices under GDPR Art. 22 ([shap_local_denied.png](results/shap_local_denied.png), [shap_local_approved.png](results/shap_local_approved.png)).

**Transparent reference model:** Logistic Regression odds ratios ([lr_coefficients.csv](results/lr_coefficients.csv)) — e.g., education-purpose loans nearly double the odds of bad credit (OR 1.98), as does each additional standardized unit of duration (OR 1.95). One nuance: the positive checking-account coefficient reflects that applicants with *no* checking account were historically low-risk in this data, so the tier ordering is not monotone in risk — a limitation of linear models that the Random Forest handles natively and SHAP makes visible.

## 7. Transparency & Accountability

- **Model card** published: [MODEL_CARD.md](MODEL_CARD.md) — intended use, performance, fairness results, and caveats in one auditable document.
- **Human in the loop:** the system emits a *recommendation + explanation + fairness context*; a loan officer holds final authority (EU AI Act high-risk compliance posture).
- **Auditability:** every stage is scripted and reproducible; predictions carry per-decision SHAP explanations suitable for logging and contestation.

## 8. Limitations & Future Work

1. **Foreign-worker mitigation:** threshold equalization improves disparate impact but at an unacceptable recall cost (§5.2), because the disadvantaged group is a 4% minority. This needs a *policy* approach — partial adjustment, reweighing/in-processing, or governance sign-off — which is the top remaining fairness task. (Sex and Age mitigation are implemented, §5–5.2.)
2. **Small subgroups:** >60 (n=45) and non-foreign workers (n=37) estimates are fragile — visible in the age equalized-odds wobble on the 9-feature model; report confidence intervals.
3. **Intersectionality:** audit joint groups (young women, young foreign workers, etc.), where disparities often compound.
4. **Threshold as policy:** formalize the 5:1 cost matrix into an explicit, documented decision-threshold policy.
5. **Productionisation:** implement event logging (AI Act Art. 12) and post-market monitoring (Art. 72) for distribution shift.

## 9. Reproducibility

```
pip install -r requirements.txt   # exact version pins → figures reproduce
python src/train_baseline.py     # models + metrics + ROC/confusion charts
python src/fairness_audit.py     # group fairness + mitigation experiment
python src/explainability.py     # SHAP global/local + LR coefficients
python src/full_uci.py           # 20-feature model + foreign-worker fairness
python -m pytest                 # 43 tests, 98.9% coverage (enforced ≥80%)
```

All outputs land in `results/`; fitted pipelines in `models/`. `requirements.txt` pins exact library versions so a clean install reproduces the exact figures quoted in this report — the whole analysis also runs end-to-end in [Responsible_AI_Credit_Scoring.ipynb](Responsible_AI_Credit_Scoring.ipynb) via *Restart & Run All*.
