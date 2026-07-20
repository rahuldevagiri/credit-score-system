# Responsible AI for Credit Scoring — Final Report

### A Fairness-Audited, Explainable and Adversarially-Tested Machine Learning System on the German Credit Dataset

*Responsible AI & Data Ethics (PEL, SS2026) — Track: The Ethical Credit System*
*Final in-depth documentation — Week 4 deliverable*

---

## Table of Contents

1. Executive Summary
2. Project Objectives & Scope
3. Data Analysis (Week 1)
4. Regulatory Analysis (Week 1)
5. Methodology (Week 2)
6. Results
7. Fairness Audit & Bias Mitigation
8. Explainability (XAI)
9. Adversarial Robustness & Security
10. Risk Assessment
11. Testing & Reproducibility (Week 3)
12. Deployment — Decision-Support Application
13. Transparency & Accountability
14. Limitations & Future Work
15. Conclusion
16. How to Reproduce

---

## 1. Executive Summary

This project builds a credit-risk classification system on the German Credit dataset (1,000 applicants) and subjects it to a complete Responsible-AI evaluation: fairness auditing, bias mitigation, explainability, adversarial testing, and a regulatory analysis mapped to the EU AI Act and GDPR. A Random Forest (ROC-AUC 0.774 on the 9-feature subset, 0.790 on the full 20-feature dataset) is the audited model.

**Headline findings:**

1. **Accuracy is the wrong metric.** With a 70/30 class balance and a 5:1 misclassification-cost asymmetry, the model is deliberately optimised for **recall on bad credit**, not accuracy. Removing cost-sensitive weighting raises accuracy to 0.74 but collapses recall to 0.30 — a quantified proof that the "most accurate" model is the worst lender.
2. **The model fails the 80% fairness rule on two protected attributes:** age (disparate impact 0.56) and — visible only in the full dataset — foreign-worker status (0.73).
3. **Bias mitigation works for sex and age.** Group-specific decision thresholds repair sex (0.83 → 0.98) and age (0.56 → 0.86), roughly halving the wrongful-denial rate for applicants under 26 (54.5% → 24.5%), at a modest recall cost.
4. **Fairness mitigation is not always free.** The same technique on foreign-worker status improves the fairness metric (0.73 → 0.88) but collapses overall recall (0.70 → 0.50), because that group is 96% of applicants. This is documented as a *policy* decision, not an automatic fix — arguably the project's most important insight.
5. **The system is explainable, tested, and threat-modelled:** SHAP global/local explanations, a 38-test suite at 98.9% coverage, and three adversarial studies (poisoning, evasion, membership inference).

The system is designed as **human-in-the-loop decision support**, consistent with the EU AI Act (credit scoring = high-risk) and GDPR Art. 22.

---

## 2. Project Objectives & Scope

**Objective:** demonstrate that a credit-scoring model can be built to be *compliant by construction* — accurate on the metric that matters to a lender, and simultaneously auditable, explainable, fair, robust, and legally defensible.

**Scope:** classical, interpretable machine learning on a small tabular dataset (appropriate given 1,000 rows). The system is a decision-support prototype, not a production deployment; it informs a human loan officer rather than deciding autonomously.

---

## 3. Data Analysis (Week 1)

- **Dataset:** German Credit (UCI / Kaggle). 1,000 loan applicants, binary target (70% good / 30% bad credit). Two versions are used: the simplified **9-feature Kaggle subset** (baseline) and the **original 20-feature UCI file** (extended analysis).
- **Feature types:** numerical (Age, Credit amount, Duration), ordinal (Job, Saving/Checking account tiers), nominal (Sex, Housing, Purpose).
- **Data quality — a key finding.** The *original* UCI dataset has **zero missing values**; "no account" is an explicit category (codes A14 for checking, A65 for savings). The Kaggle version re-encoded those categories as blank cells (183 savings, 394 checking — matching the UCI code counts exactly). We therefore fill these with an explicit `none` category, which *reconstructs the original meaning* rather than imputing.
- **Distribution:** Credit amount is right-skewed (skew 1.95, median 2,320 vs mean 3,271 DM); log-transformed rather than trimmed, since large loans are legitimate. Duration and Credit amount correlate (~0.6); both increase risk.
- **Sensitive attributes:** Age and Sex (present in both versions); foreign-worker status and personal-status/sex (full UCI only); Job as a socioeconomic proxy.
- **Class imbalance:** 70/30 plus a 5:1 cost matrix make accuracy misleading — recall on bad credit is the operationally correct metric.

---

## 4. Regulatory Analysis (Week 1)

Credit scoring is a **named high-risk use case** under the EU AI Act (Annex III(5)(b)). This drives obligations mapped to concrete artifacts: risk management (Art. 9), data governance & bias examination (Art. 10), technical documentation (Art. 11), logging (Art. 12), transparency (Art. 13), human oversight (Art. 14), and robustness/security (Art. 15). GDPR applies in full — most notably **Art. 22** (right not to be subject to a solely automated decision; right to explanation), satisfied by the human-in-the-loop design plus SHAP local explanations. Full mapping in `REGULATORY_ANALYSIS.md`.

---

## 5. Methodology (Week 2)

**Preprocessing pipeline** (`preprocessing.py`): missing-as-category handling → log-transform + standardisation of numerics → ordinal encoding of ordered tiers → one-hot encoding of nominal features → target mapped to *bad = 1* so recall measures risky-applicant detection. A separate audit frame (Age, Sex, Risk) feeds the fairness layer independently of the model's inputs. All preprocessing lives inside the model pipeline, guaranteeing leakage-free cross-validation.

**Models** — three across the interpretability–performance spectrum, all with `class_weight="balanced"` (cost-sensitive learning), evaluated on a stratified 80/20 hold-out plus 5-fold cross-validation:

| Model | Role |
|---|---|
| Logistic Regression | Fully transparent regulatory baseline |
| Decision Tree (depth 5) | Human-readable decision rules |
| Random Forest (300 trees) | Performance reference (selected for audit) |

**Evaluation metrics:** Accuracy, Precision, **Recall (primary)**, F1, ROC-AUC.

---

## 6. Results

**Baseline hold-out (n = 200, positive class = bad credit):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.645 | 0.432 | 0.583 | 0.496 | 0.664 |
| Decision Tree | 0.620 | 0.425 | **0.750** | 0.542 | 0.695 |
| Random Forest | **0.695** | **0.494** | 0.700 | **0.579** | **0.774** |

Cross-validated recall (LR 0.65 / DT 0.68 / RF 0.70, ±0.06–0.08) confirms the ranking is stable. Accuracy sits below the naive 70% "approve everyone" baseline *by design*.

**Extended model — full 20-feature UCI dataset:** the same Random Forest improves to Accuracy 0.705, Recall 0.733, ROC-AUC 0.790 — credit history and employment length carry genuine signal the subset discards.

**Feature-engineering experiment:** engineering new features (e.g. monthly payment burden) yields only marginal gains (+0.7 AUC) within CV noise, at a recall cost. Removing `class_weight` gives the highest accuracy (0.742) but recall collapses to 0.303 — the accuracy trap, quantified. The genuine performance lever is *more features* (full UCI), not transformed features.

---

## 7. Fairness Audit & Bias Mitigation

Computed on 5-fold **out-of-fold predictions across all 1,000 applicants**. Convention: predicted *bad* = credit denied.

**Disparate impact (80% rule) — baseline:**

| Attribute | Disparate impact | Verdict |
|---|---|---|
| Sex | 0.83 | borderline |
| Age band | **0.56** | **fail** |
| Foreign worker (full UCI) | **0.73** | **fail** |

Applicants under 26 face a 54.5% wrongful-denial rate (vs 24.5% for ages 26–60); foreign workers are approved at 58.8% vs 81.1% for non-foreign workers.

**Mitigation** — group-specific FPR-equalizing thresholds:

| Attribute | Model | Disparate impact | Overall recall | Verdict |
|---|---|---|---|---|
| Sex | 9-feature | 0.83 → **0.98** | 0.70 → 0.66 | ✅ clean |
| Age band | 9-feature | 0.56 → **0.86** | 0.70 → 0.62 | ✅ passes; young-denial halved |
| Age band | 20-feature | 0.58 → **0.94** | 0.70 → 0.61 | ✅ clean (eq-odds also improves) |
| Foreign worker | 20-feature | 0.73 → **0.88** | **0.70 → 0.50** | ⚠️ trade-off |

**The critical insight.** For sex and age, mitigation is a clean win. For foreign worker it backfires: because that group is ~96% of applicants, raising their threshold to cut wrongful denials makes the model miss half of *all* bad loans. **Fairness mitigation is not always free.** When the disadvantaged reference group is a fragile minority (n=37), a naive threshold fix trades away the model's core purpose. The responsible response is to treat foreign-worker fairness as a documented **policy decision** (partial adjustment, reweighing/in-processing, or governance sign-off), not an automatic rule.

---

## 8. Explainability (XAI)

SHAP TreeExplainer on the Random Forest.

- **Global drivers (mean |SHAP| toward P(bad)):** Checking account **0.120**, Duration 0.068, Savings 0.035, Credit amount 0.032, Housing=own 0.023, Age 0.021, Sex 0.012. The model reasons primarily from financial capacity — but the non-zero Age/Sex attributions justify the fairness audit.
- **Local explanations:** per-applicant SHAP waterfalls give a contestable reason ("denied because of low account balances and a 24-month term") — the GDPR Art. 22 right-to-explanation made concrete.
- **Transparent reference:** Logistic Regression odds ratios (e.g. education-purpose loans nearly double the odds of bad credit; each standardised unit of duration OR 1.95).

---

## 9. Adversarial Robustness & Security

Three attack studies (`adversarial.py`), because a high-risk system's threat model is part of its risk assessment (AI Act Art. 15):

1. **Evasion / gaming:** ~52% of rejected applicants flip to "approved" by halving the requested loan size/term — a real gaming risk.
2. **Data poisoning:** corrupting training labels degrades ranking quality (ROC-AUC 0.79 → 0.71 as 0–30% of labels are flipped).
3. **Membership inference (privacy):** attack AUC ≈ 0.58 — a mild, quantified leak (confidence gap 0.055), addressing GDPR data-protection-by-design (Art. 25).

---

## 10. Risk Assessment

| Risk | Type | Treatment |
|---|---|---|
| Age discrimination | Fairness / legal | Detected (DI 0.56) → **mitigated** (0.86), young-denial halved |
| Sex discrimination | Fairness / legal | **Mitigated** via group thresholds (0.83 → 0.98) |
| Foreign-worker discrimination | Fairness / legal | Detected (DI 0.73); mitigation is a documented trade-off → policy decision |
| Historical bias in labels | Data | Documented; model audited rather than trusted blindly |
| Score gaming | Security | Quantified (evasion study); flag manipulable features |
| Data poisoning | Security | Quantified; motivates data-pipeline integrity controls |
| Privacy leakage | Security / legal | Membership-inference measured (AUC 0.58) |
| Automation bias | Governance | Human-in-the-loop + per-decision explanation + fairness context |

---

## 11. Testing & Reproducibility (Week 3)

A `pytest` suite of **43 tests** at **98.9% line coverage**, enforced at ≥80% via `pytest.ini`. Tests include hand-computed confusion-matrix checks, fairness-metric ground truth, and **attack-invariant tests** (e.g. shrinking a loan can never lower an applicant's approval chance). `requirements.txt` pins exact library versions so the figures reproduce on a clean install.

---

## 12. Deployment — Decision-Support Application

A local **Streamlit** app (`streamlit_app.py`) turns the architecture into a working artifact: enter an applicant, receive a risk score, a banded recommendation, a **live SHAP explanation**, a **fairness warning** if the applicant falls in a flagged group, and — for declined applicants — a **counterfactual recourse panel** (`recourse.py`). Where SHAP explains *why* an applicant was declined, recourse explains *how* they could be approved: the smallest realistic change to the loan they control (a smaller amount or shorter term) that flips the recommendation — e.g. *"reduce both the amount and term by 55% → approved."* When no loan reduction suffices, it honestly reports that a broader manual review is needed rather than a smaller loan. This implements the *right to recourse* principle in responsible lending: a denial should come with a path, not just a reason. It is explicitly decision *support* — the loan officer holds final authority, and every assessment is logged (inputs, score, explanation) for auditability.

---

## 13. Transparency & Accountability

- **Model card** (`MODEL_CARD.md`): intended use, out-of-scope uses, performance, fairness results, and caveats.
- **Human-in-the-loop:** the system emits a recommendation + explanation + fairness context; a human decides. This keeps it out of GDPR Art. 22's "solely automated" prohibition and satisfies AI Act Art. 14 human oversight.
- **Auditability:** every stage is scripted, tested, and reproducible; predictions carry SHAP explanations suitable for logging and contestation.

---

## 14. Limitations & Future Work

1. **Foreign-worker fairness** needs a policy-level approach (partial adjustment / reweighing / governance sign-off) rather than an automatic threshold — the top remaining fairness task. (Sex and age mitigation are implemented.)
2. **Small subgroups** (>60: n=45; non-foreign workers: n=37) make some estimates fragile — visible in the age equalized-odds wobble on the 9-feature model.
3. **Intersectionality:** audit joint groups (young women, young foreign workers) where disparities compound.
4. **Productionisation:** event logging (AI Act Art. 12) and post-market monitoring (Art. 72) for distribution shift.

---

## 15. Conclusion

The project delivers a credit-scoring system engineered for a high-risk regulatory context: performant on the correct metric, explainable at global and local levels, fair (with two of three disparities mitigated and the third honestly documented as a policy trade-off), threat-modelled, tested to 98.9% coverage, and deployed as a human-in-the-loop decision-support app. Its most valuable contribution is not a single number but a demonstrated *methodology*: measure bias rather than assume it away, mitigate where mitigation is appropriate, and know when a fair-looking fix would do more harm than good.

---

## 16. How to Reproduce

```bash
pip install -r requirements.txt          # exact version pins
python -m pytest                         # 43 tests, 98.9% coverage
# or run stages individually:
python src/train_baseline.py             # models + metrics + charts
python src/fairness_audit.py             # fairness + sex/age mitigation
python src/full_uci.py                   # 20-feature model + FW trade-off
python src/explainability.py             # SHAP + LR coefficients
python src/adversarial.py                # poisoning / evasion / membership
streamlit run app/streamlit_app.py       # decision-support app
```

The complete analysis also runs end-to-end in the presentation notebook (`Kernel → Restart & Run All`). See `00_START_HERE.md` in the deliverables root for the folder map.
