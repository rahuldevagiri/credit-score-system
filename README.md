# Responsible AI for Credit Scoring — German Credit Dataset

University project: a credit-risk classifier with a full Responsible AI layer
(fairness audit, SHAP explainability, model card, human-in-the-loop design).

## Structure

```
data/                          German Credit dataset (Kaggle subset + original UCI file)
src/
  preprocessing.py             loading, cleaning, encoding pipeline + sensitive-attribute audit frame
  train_baseline.py            Logistic Regression / Decision Tree / Random Forest + evaluation
  fairness_audit.py            group fairness metrics (Sex, Age) + threshold mitigation experiment
  explainability.py            SHAP global & local explanations, LR odds ratios
  viz_style.py                 shared chart palette/style
models/                        fitted pipelines (joblib)
results/                       metrics CSVs and presentation-ready charts
REPORT.md                      full project report with results and interpretation
MODEL_CARD.md                  model card (transparency & accountability artifact)
```

## Run

```
pip install pandas scikit-learn matplotlib shap
python src/train_baseline.py
python src/fairness_audit.py
python src/explainability.py
```

## Key results

- Random Forest: ROC-AUC 0.774, bad-credit recall 0.70 (primary metric — cost-sensitive).
- Fairness: age disparate impact 0.56 (fails 80% rule — young applicants wrongly denied at 54.5%);
  sex disparity 0.83, repaired to 0.98 by group-specific thresholds at no accuracy cost.
- SHAP: checking account, duration, and credit amount dominate; sensitive attributes have
  small but non-zero direct influence — see [REPORT.md](REPORT.md) and [MODEL_CARD.md](MODEL_CARD.md).
