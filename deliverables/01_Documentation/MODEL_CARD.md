# Model Card — German Credit Risk Classifier

## Model Details
- **Model:** Random Forest classifier (300 trees, `min_samples_leaf=20`, balanced class weights), selected over Logistic Regression and Decision Tree baselines.
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
| Accuracy | 0.690 |
| Precision (bad) | 0.489 |
| **Recall (bad)** — primary metric | **0.750** |
| F1 (bad) | 0.592 |
| ROC-AUC | 0.770 |

Recall is prioritized because approving a bad credit costs the bank roughly 5× more than rejecting a good one (dataset cost matrix). Class imbalance is handled with balanced class weights.

## Fairness Evaluation (5-fold out-of-fold predictions, n = 1,000)
| Attribute | Disparate impact (80% rule) | Equal-opportunity gap (TPR) | Equalized-odds gap |
|---|---|---|---|
| Sex | 0.80 — borderline | 0.06 | 0.10 |
| Age band | **0.50 — fail** | 0.16 | 0.30 |
| Foreign worker †| **0.71 — fail** | 0.25 | 0.25 |

† Foreign-worker status exists only in the full 20-feature UCI dataset ([src/full_uci.py](src/full_uci.py)), not the 9-feature Kaggle subset the primary model uses. The 20-feature model also outperforms the subset (ROC-AUC 0.783 vs 0.770, recall 0.78 vs 0.75).

- Female applicants are wrongly denied (FPR) at 41.3% vs 31.1% for males.
- Applicants aged ≤25 face a 59% wrongful-denial rate vs 29% for ages 26–60 — the dominant fairness problem.
- Foreign workers are approved at ~54% vs ~76% for non-foreign workers; since ~96% of applicants are foreign workers, the advantaged group is a fragile 37-person minority.
- **Mitigation implemented for Sex and Age:** group-specific FPR-equalizing thresholds raised disparate impact for Sex (0.80 → 0.96) and Age (0.50 → 0.82, halving the young-applicant wrongful-denial rate) at a small recall cost (~0.75 → 0.67–0.72).
- **Foreign worker — a documented trade-off, not a fix:** the same technique lifts disparate impact (0.71 → 0.93) but collapses overall recall (0.76 → 0.47), because foreign workers are ~96% of applicants. This requires a policy/reweighing approach, not an automatic threshold — the top remaining fairness task.
- The model currently sees Sex and Age as features; SHAP confirms a small direct effect of Sex (mean |SHAP| ≈ 0.012, male lowers predicted risk). Removing them would **not** remove bias (proxies remain) — group auditing stays mandatory either way.

## Explainability
- **Global:** SHAP identifies Checking account status (0.106 mean |SHAP|), loan Duration (0.052), Saving accounts (0.023), and Credit amount (0.023) as dominant drivers.
- **Local:** per-applicant SHAP waterfall explanations support adverse-action notices and the GDPR Art. 22 right to an explanation.
- A fully transparent Logistic Regression (coefficients/odds ratios exported) is maintained as an interpretable reference model.

## Ethical Considerations & Caveats
- Credit scoring is classified **high-risk under the EU AI Act**: human oversight, logging, documentation, and fairness monitoring are legal obligations, not optional features.
- Historical-bias inheritance: the young-applicant penalty reflects and would amplify past lending patterns if deployed unaudited.
- Small subgroups (>60: n=45; non-foreign workers: n=37; female-bad: n=95 across folds) make fairness estimates for those groups statistically fragile.
- Every prediction should be logged with inputs, score, and explanation for auditability and contestability.
