# Decision-Support App — How to Run

A local Streamlit web app that loads the trained Random Forest and, for any
applicant, shows a risk score, a recommendation, a **live SHAP explanation**,
and a fairness warning for flagged groups. It is decision *support* — the loan
officer holds final authority.

## Run

From the **project root** (so it can find `models/` and `src/`):

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501, enter applicant details in the sidebar, and
click **Assess applicant**.

## Notes

- The app is provided as a `.py` script because Streamlit runs Python scripts,
  not notebooks.
- It loads the pre-trained pipeline from `models/random_forest.joblib`. If you
  retrain (`python src/train_baseline.py`), the app picks up the new model on
  next launch.
- The copy in this folder is a reference snapshot; run the canonical version at
  `app/streamlit_app.py` in the project root.
