# Deliverables — Responsible AI for Credit Scoring

*Responsible AI & Data Ethics (PEL, SS2026) — Track: The Ethical Credit System*

This folder is the organised, submission-ready package. Start with the numbered
folders below.

## Folder map

| Folder | Contents |
|---|---|
| **01_Documentation/** | The written deliverables. Start with **`FINAL_REPORT.docx`** (the in-depth report; `FINAL_REPORT.md` is the same content in Markdown). Also: `MODEL_CARD.md`, `REGULATORY_ANALYSIS.md` (EU AI Act + GDPR mapping), `PROJECT_README.md`. |
| **02_Code_Notebooks/** | All code as Jupyter notebooks. `Responsible_AI_Credit_Scoring.ipynb` is the main runnable presentation notebook. `modules/` holds one notebook per pipeline stage (preprocessing, training, fairness, explainability, adversarial, full-UCI, styling). |
| **03_Results/** | Every generated figure (`.png`) and metrics table (`.csv`). |
| **04_Application/** | The Streamlit decision-support app + `HOW_TO_RUN.md`. |

## How the code is delivered

The pipeline code is provided as **Jupyter notebooks** in `02_Code_Notebooks/`.
Each module notebook has a self-contained bootstrap cell and can be run with
*Kernel → Restart & Run All*. Two things run as `.py` by necessity, not choice:

- the **test suite** (`pytest` requires `.py`), and
- the **Streamlit app** (Streamlit runs `.py` scripts).

Both live in the parent project repository, which is the reproducible engine
behind these notebooks.

## Reproducing everything

The notebooks import the tested pipeline modules and read the dataset from the
project's `data/` and write to `results/`. To reproduce from a clean environment:

```bash
pip install -r requirements.txt     # exact version pins → figures reproduce
python -m pytest                    # 43 tests, 98.9% coverage
jupyter notebook Responsible_AI_Credit_Scoring.ipynb   # Restart & Run All
```

## Headline results (one glance)

- **Model:** Random Forest — ROC-AUC 0.774 (9-feature) / 0.790 (full 20-feature UCI); bad-credit recall 0.70–0.73 (the primary, cost-sensitive metric).
- **Fairness:** sex (0.83 → 0.98) and age (0.56 → 0.86) disparities **mitigated**; foreign-worker disparity documented as a **policy trade-off** (fix passes fairness but recall 0.70 → 0.50 for a 96% majority).
- **Explainability:** SHAP global + per-applicant local explanations.
- **Robustness:** poisoning / evasion / membership-inference studies.
- **Quality:** 43 tests, 98.9% coverage; live decision-support app.
