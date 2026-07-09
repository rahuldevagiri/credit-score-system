# Week 2 — Status Update

**Project:** The Ethical Credit System (German Credit dataset) · **Track:** Responsible AI in Credit Scoring
**Week-2 goals (syllabus):** build a baseline model · make a risk analysis · analyze fairness

---

## 1. What we did this week

### a) Baseline model
We trained **three models across the interpretability → performance spectrum**, all with `class_weight="balanced"` (cost-sensitive learning), evaluated on a stratified 80/20 hold-out **plus** 5-fold cross-validation.

| Model | Accuracy | Precision (bad) | **Recall (bad)** | F1 (bad) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.645 | 0.432 | 0.583 | 0.496 | 0.664 |
| Decision Tree (depth 5) | 0.620 | 0.425 | **0.750** | 0.542 | 0.695 |
| **Random Forest (300 trees)** | **0.695** | **0.494** | 0.700 | **0.579** | **0.774** |

Cross-validated recall (LR 0.65 / DT 0.68 / RF 0.70, ±0.06–0.08) confirms the ranking is stable. **We selected Random Forest** (best AUC + F1, near-best recall) as the model to audit.

### b) Risk analysis
The dataset is **70% good / 30% bad**, and the dataset's own cost matrix says approving a bad loan costs a lender **~5× more** than rejecting a good one. Two consequences drive every choice we made:
- **Accuracy is misleading** — a model that approves everyone scores 70%. So **recall on the *bad* class is our primary metric**, and our models' accuracy sits *below* 70% **by design** (we trade accuracy to catch risky applicants — the economically correct trade for a lender).
- We keep an explicit **risk register** (fairness, historical bias, gaming, privacy, automation bias, distribution shift) with a treatment for each.

### c) Fairness analysis
We audited the Random Forest on **5-fold out-of-fold predictions for all 1,000 applicants** (larger, more stable group slices than the 200-row test set). Convention: predicted **bad = credit denied**.

| Attribute | Disparate impact (80% rule) | Equal-opportunity gap (TPR) | Equalized-odds gap |
|---|---|---|---|
| Sex | 0.83 — **borderline** | 0.03 | 0.10 |
| Age band | **0.56 — FAILS** | 0.28 | 0.30 |

- **Age is the dominant failure:** a creditworthy applicant **under 26 is wrongly denied 54.5%** of the time vs 24.5% for ages 26–60 — the model learned and amplified a historical young-borrower penalty.
- **Sex** harm is smaller and falls on **wrongful denials of creditworthy women** (FPR 36.8% vs 26.9% for men), not on risk detection.

**Mitigation demonstrated:** group-specific decision thresholds on Sex (male 0.50, female 0.554) that equalize the wrongful-denial rate raised the disparate-impact ratio **0.83 → 0.98 at no accuracy cost** (0.70 → 0.71) and a small recall cost (0.70 → 0.66). This proves the measured bias is **actionable**, and that setting the threshold is a **documented policy decision owned by an accountable human**.

### d) Extra: full 20-feature UCI dataset ("use additional information if you can find it")
The Kaggle file is a 9-feature subset. We also decoded the **original UCI file** (`data/german.data`, 20 features) and re-ran the same Random Forest. Two payoffs:
- **Better performance:** ROC-AUC **0.774 → 0.790**, recall **0.70 → 0.73** (the extra features — credit history, employment length — carry real signal).
- **A third bias the subset structurally cannot show:** **foreign-worker status** fails the 80% rule (**DI 0.73** — approved 58.8% vs 81.1%). This is the audit's core lesson in one number: dropping a sensitive column would have *hidden* this disparity, not removed it.

---

## 2. Challenges
- **Accuracy below the naive baseline.** Explaining *why* a "worse" accuracy is actually the correct model required framing the whole evaluation around the cost matrix rather than accuracy.
- **~39% missing checking-account values.** We interpreted these as *"no account exists"* and encoded them as an explicit `none` category rather than imputing — a judgement call we documented.
- **Discovering the age bias.** Deciding how to handle a clear 80%-rule failure: we chose to *measure and demonstrate a fix on Sex now* and schedule the same intervention for Age.
- **Small subgroups** (>60: n = 45) make some fairness estimates statistically fragile.

---

## 3. Next steps (Week 3)
- **XAI:** SHAP global + local explanations to understand *why* the model decides as it does (supports the GDPR right-to-explanation).
- **Automated tests** with **≥80% code coverage** to detect regressions and weaknesses.
- **Extend the fairness mitigation to Age bands and foreign-worker status** — the largest measured harms (the Sex technique is already demonstrated; extending it is mechanical).

---

*Artifacts: `results/baseline_metrics.csv`, `roc_curves.png`, `confusion_matrices.png`, `fairness_summary.csv`, `fairness_group_metrics.png`, `mitigation_comparison.csv`. Everything is reproducible from `src/train_baseline.py` and `src/fairness_audit.py`, or the notebook (§3, §4, §7).*
