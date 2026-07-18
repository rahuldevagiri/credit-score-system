"""
Credit Scoring — Decision Support System (local Streamlit app).

Consistent with the project's Responsible-AI architecture (see MODEL_CARD.md):
the app produces a *recommendation with an explanation and fairness context*
for a human loan officer — it never issues an autonomous decision.

Run:  streamlit run app/streamlit_app.py

Loads the trained Random Forest pipeline from models/random_forest.joblib and,
for each applicant, shows:
  1. a bad-credit risk score + a banded recommendation,
  2. a per-applicant SHAP explanation (the GDPR Art. 22 "why"),
  3. a fairness note when the applicant falls in a group the audit flagged.
"""
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODEL_PATH = ROOT / "models" / "random_forest.joblib"

# Category options must match what the model was trained on. "none" is the
# explicit no-account category the preprocessing creates from missing values.
SAVING_OPTS = ["none", "little", "moderate", "quite rich", "rich"]
CHECKING_OPTS = ["none", "little", "moderate", "rich"]
HOUSING_OPTS = ["own", "rent", "free"]
SEX_OPTS = ["male", "female"]
PURPOSE_OPTS = ["car", "radio/TV", "furniture/equipment", "business", "education",
                "repairs", "domestic appliances", "vacation/others"]
JOB_LABELS = {0: "0 — unskilled / non-resident", 1: "1 — unskilled / resident",
              2: "2 — skilled", 3: "3 — highly skilled / management"}


@st.cache_resource
def load_model():
    pipe = joblib.load(MODEL_PATH)
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    return pipe, prep, model, explainer


def pretty_names(raw):
    """'nom__Purpose_car' -> 'Purpose = car'; 'num__Age' -> 'Age'."""
    out = []
    for n in raw:
        n = n.split("__", 1)[-1]
        for p in ("Sex", "Housing", "Purpose"):
            if n.startswith(p + "_"):
                n = n.replace("_", " = ", 1)
                break
        out.append(n)
    return out


def recommendation(p_bad):
    """Banded, human-facing recommendation around the 0.5 operating threshold."""
    if p_bad < 0.40:
        return "✅ Recommend: APPROVE", "Low estimated risk.", "normal"
    if p_bad < 0.60:
        return "⚠️ BORDERLINE: manual review", "Near the decision threshold — officer judgement required.", "off"
    return "⛔ Recommend: DECLINE / escalate", "High estimated risk of default.", "inverse"


# ---------------------------------------------------------------------------
st.set_page_config(page_title="Credit Risk — Decision Support", page_icon="🏦", layout="wide")
st.title("🏦 Credit Risk — Decision Support System")
st.caption("A **recommendation + explanation** for a human loan officer. Not an autonomous decision. "
           "Model: Random Forest on the German Credit dataset.")

pipe, prep, model, explainer = load_model()

with st.sidebar:
    st.header("Applicant details")
    age = st.slider("Age", 18, 80, 35)
    sex = st.selectbox("Sex", SEX_OPTS)
    job = st.selectbox("Job (skill level)", list(JOB_LABELS), format_func=lambda k: JOB_LABELS[k], index=2)
    housing = st.selectbox("Housing", HOUSING_OPTS)
    saving = st.selectbox("Saving account", SAVING_OPTS, index=1)
    checking = st.selectbox("Checking account", CHECKING_OPTS, index=1)
    credit_amount = st.number_input("Credit amount (DM)", 250, 20000, 3000, step=250)
    duration = st.slider("Duration (months)", 4, 72, 24)
    purpose = st.selectbox("Purpose", PURPOSE_OPTS)
    go = st.button("Assess applicant", type="primary", use_container_width=True)

row = pd.DataFrame([{
    "Age": age, "Sex": sex, "Job": job, "Housing": housing,
    "Saving accounts": saving, "Checking account": checking,
    "Credit amount": credit_amount, "Duration": duration, "Purpose": purpose,
}])

if not go:
    st.info("Enter applicant details in the sidebar and click **Assess applicant**.")
    st.stop()

# --- Prediction -------------------------------------------------------------
p_bad = float(pipe.predict_proba(row)[0, 1])
label, note, style = recommendation(p_bad)

c1, c2 = st.columns([1, 1.4])
with c1:
    st.subheader("Risk assessment")
    st.metric("Estimated probability of BAD credit", f"{p_bad:.0%}",
              delta=note, delta_color=style)
    st.progress(min(p_bad, 1.0))
    st.markdown(f"### {label}")
    if age <= 25:
        st.warning("**Fairness note:** the fairness audit found this model wrongly "
                   "denies creditworthy applicants **under 26 at ~55%** (vs ~25% for "
                   "ages 26–60). Weight this recommendation with extra caution.")

# --- SHAP explanation -------------------------------------------------------
Xt = prep.transform(row)
if hasattr(Xt, "toarray"):
    Xt = Xt.toarray()
names = pretty_names(prep.get_feature_names_out())
sv = explainer(Xt)[:, :, 1]
sv.feature_names = names

with c2:
    st.subheader("Why — per-applicant explanation (SHAP)")
    st.caption("Red pushes risk **up**, blue pushes it **down**, starting from the average applicant.")
    fig = plt.figure()
    shap.plots.waterfall(sv[0], max_display=9, show=False)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# --- Plain-language top factors --------------------------------------------
contrib = pd.Series(sv.values[0], index=names)
top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(5)
st.subheader("Top factors in this decision")
for feat, val in top.items():
    direction = "increased" if val > 0 else "decreased"
    st.markdown(f"- **{feat}** {direction} the risk estimate "
                f"({'+' if val > 0 else ''}{val:.3f})")

st.divider()
st.caption("⚖️ Decision support only. The final credit decision rests with the human loan "
           "officer, who must weigh this recommendation, its explanation, and the fairness "
           "context. Every assessment should be logged (inputs, score, explanation) for "
           "auditability — EU AI Act high-risk / GDPR Art. 22.")
