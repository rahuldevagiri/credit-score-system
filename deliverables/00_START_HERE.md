# Deliverables — Responsible AI for Credit Scoring

*Responsible AI & Data Ethics (PEL, SS2026) — Track: The Ethical Credit System*

This folder is the organised, submission-ready package. Start with the numbered
folders below.

## Folder map

**This folder is fully self-contained** — the dataset, source modules, tests and
dependencies are all included, so everything runs without any other files.

### Read these (numbered folders)

| Folder | Contents |
|---|---|
| **01_Documentation/** | The written deliverables. Start with **`FINAL_REPORT.docx`** (the in-depth report; `FINAL_REPORT.md` is the same content in Markdown). Also: `MODEL_CARD.md`, `REGULATORY_ANALYSIS.md` (EU AI Act + GDPR mapping), `PROJECT_README.md`. |
| **02_Code_Notebooks/** | All code as Jupyter notebooks. `Responsible_AI_Credit_Scoring.ipynb` is the main runnable presentation notebook. `modules/` holds one notebook per pipeline stage (preprocessing, training, fairness, explainability, adversarial, full-UCI, recourse, styling). |
| **03_Results/** | Every generated figure (`.png`) and metrics table (`.csv`) — the curated snapshot. |
| **04_Application/** | The Streamlit decision-support app + `HOW_TO_RUN.md`. |

### Supporting files (what makes it runnable)

| Item | Purpose |
|---|---|
| `src/` | The tested pipeline modules the notebooks import |
| `data/` | The German Credit dataset (Kaggle subset + original UCI file) |
| `tests/` + `pytest.ini` | The 43-test suite and its ≥80% coverage gate |
| `models/` | Pre-trained pipelines, so notebooks work without retraining first |
| `requirements.txt` | Pinned dependencies |

> Running the notebooks or scripts regenerates outputs into a `results/` folder.
> The curated snapshot for reading is `03_Results/`.

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

- **Model:** Random Forest — ROC-AUC 0.770 (9-feature) / 0.783 (full 20-feature UCI); bad-credit recall 0.75–0.78 (the primary, cost-sensitive metric).
- **Fairness:** sex (0.80 → 0.96) and age (0.50 → 0.82) disparities **mitigated**; foreign-worker disparity documented as a **policy trade-off** (fix passes fairness but recall 0.76 → 0.47 for a 96% majority).
- **Explainability:** SHAP global + per-applicant local explanations.
- **Robustness:** poisoning / evasion / membership-inference studies.
- **Quality:** 43 tests, 98.9% coverage; live decision-support app.
