# Model Card — German Credit Risk Classifier

## Model Details
- **Model:** Random Forest classifier (300 trees, `min_samples_leaf=5`, balanced class weights), selected over Logistic Regression and Decision Tree baselines.
- **Version / date:** v1.0 — July 2026 (university project, Responsible AI in Credit Scoring).
- **Task:** Binary classification of loan applicants into *good* vs *bad* credit risk. Prediction "bad" corresponds to a **recommendation to deny** credit.
- **Developers:** Student project; not a production system.

## Intended Use
- **Primary use:** Decision *support* for a human loan officer — the model outputs a risk score plus a SHAP explanation; it never issues an autonomous final decision.
- **Out-of-scope uses:** Fully automated credit decisions; deployment on populations that differ from 1990s German bank customers; any use without the accompanying fairness monitoring.

## Training Data
- German Credit Dataset (UCI / Kaggle mirror), 1,000 applicants, 9 features, target 70% good / 30% bad.
- **Known data risks:** labels encode historical human lending decisions from 1990s Germany and inherit their biases; missing account values are treated as an explicit "none" category; the Kaggle version omits 11 of the original 20 UCI features (e.g., credit history), capping achievable accuracy.

## Performance (stratified 20% hold-out, positive class = bad)
| Metric | Value |
|---|---|
| Accuracy | 0.695 |
| Precision (bad) | 0.494 |
| **Recall (bad)** — primary metric | **0.700** |
| F1 (bad) | 0.579 |
| ROC-AUC | 0.774 |

Recall is prioritized because approving a bad credit costs the bank roughly 5× more than rejecting a good one (dataset cost matrix). Class imbalance is handled with balanced class weights.

## Fairness Evaluation (5-fold out-of-fold predictions, n = 1,000)
| Attribute | Disparate impact (80% rule) | Equal-opportunity gap (TPR) | Equalized-odds gap |
|---|---|---|---|
| Sex | 0.83 — borderline pass | 0.03 | 0.10 |
| Age band | **0.56 — fail** | 0.28 | 0.30 |
| Foreign worker †| **0.73 — fail** | 0.20 | 0.20 |

† Foreign-worker status exists only in the full 20-feature UCI dataset ([src/full_uci.py](src/full_uci.py)), not the 9-feature Kaggle subset the primary model uses. The 20-feature model also outperforms the subset (ROC-AUC 0.790 vs 0.774, recall 0.73 vs 0.70).

- Female applicants are wrongly denied (FPR) at 36.8% vs 26.9% for males.
- Applicants aged ≤25 face a 54.5% wrongful-denial rate vs 24.5% for ages 26–60 — the dominant fairness problem.
- Foreign workers are approved at 58.8% vs 81.1% for non-foreign workers; since ~96% of applicants are foreign workers, the advantaged group is a fragile 37-person minority.
- **Mitigation demonstrated:** group-specific decision thresholds equalizing FPR across Sex raised the disparate-impact ratio from 0.83 to 0.98 with no accuracy loss (0.70 → 0.71) and a small recall cost (0.70 → 0.66). Equivalent interventions for age bands and foreign-worker status are required before any real-world use.
- The model currently sees Sex and Age as features; SHAP confirms a small direct effect of Sex (mean |SHAP| ≈ 0.012, male lowers predicted risk). Removing them would **not** remove bias (proxies remain) — group auditing stays mandatory either way.

## Explainability
- **Global:** SHAP identifies Checking account status (0.120 mean |SHAP|), loan Duration (0.068), Saving accounts (0.035), and Credit amount (0.032) as dominant drivers.
- **Local:** per-applicant SHAP waterfall explanations support adverse-action notices and the GDPR Art. 22 right to an explanation.
- A fully transparent Logistic Regression (coefficients/odds ratios exported) is maintained as an interpretable reference model.

## Ethical Considerations & Caveats
- Credit scoring is classified **high-risk under the EU AI Act**: human oversight, logging, documentation, and fairness monitoring are legal obligations, not optional features.
- Historical-bias inheritance: the young-applicant penalty reflects and would amplify past lending patterns if deployed unaudited.
- Small subgroups (>60: n=45; non-foreign workers: n=37; female-bad: n=95 across folds) make fairness estimates for those groups statistically fragile.
- Every prediction should be logged with inputs, score, and explanation for auditability and contestability.
