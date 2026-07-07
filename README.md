# Responsible AI for Credit Scoring — German Credit Dataset

University project: a credit-risk classifier with a full Responsible AI layer
(fairness audit, SHAP explainability, model card, human-in-the-loop design).

## Structure

```
Responsible_AI_Credit_Scoring.ipynb   documented, runnable presentation notebook (main deliverable)
data/                          German Credit dataset (Kaggle subset + original UCI file)
src/
  preprocessing.py             loading, cleaning, encoding pipeline + sensitive-attribute audit frame
  train_baseline.py            Logistic Regression / Decision Tree / Random Forest + evaluation
  fairness_audit.py            group fairness metrics (Sex, Age) + threshold mitigation experiment
  explainability.py            SHAP global & local explanations, LR odds ratios
  adversarial.py               data-poisoning, evasion/gaming & membership-inference attack studies
  viz_style.py                 shared chart palette/style
tests/                         pytest suite (~99% coverage; enforced >=80% via pytest.ini)
models/                        fitted pipelines (joblib)
results/                       metrics CSVs and presentation-ready charts
REPORT.md                      full project report with results and interpretation
REGULATORY_ANALYSIS.md         EU AI Act (high-risk) + GDPR mapping tied to each artifact
MODEL_CARD.md                  model card (transparency & accountability artifact)
requirements.txt               pinned dependencies
```

## Run

```
pip install -r requirements.txt

# Option A — the presentation notebook (regenerates every result inline):
jupyter notebook Responsible_AI_Credit_Scoring.ipynb   # Kernel > Restart & Run All

# Option B — the scripts directly:
python src/train_baseline.py     # models + metrics + ROC/confusion charts
python src/fairness_audit.py     # group fairness + mitigation experiment
python src/explainability.py     # SHAP global/local + LR coefficients
python src/adversarial.py        # poisoning / evasion / membership-inference

# Tests + coverage (Week-3 requirement):
python -m pytest                 # 30 tests, ~99% coverage
```

## Key results

- Random Forest: ROC-AUC 0.774, bad-credit recall 0.70 (primary metric — cost-sensitive).
- Fairness: age disparate impact 0.56 (fails 80% rule — young applicants wrongly denied at 54.5%);
  sex disparity 0.83, repaired to 0.98 by group-specific thresholds at no accuracy cost.
- SHAP: checking account, duration, and credit amount dominate; sensitive attributes have
  small but non-zero direct influence — see [REPORT.md](REPORT.md) and [MODEL_CARD.md](MODEL_CARD.md).
- Adversarial: ~52% of rejected applicants flip to "approved" by halving the requested loan
  (evasion); poisoning degrades ROC-AUC as labels are corrupted; membership-inference attack
  AUC ≈ 0.58 (mild privacy leak).
