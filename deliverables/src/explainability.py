"""
Responsible AI layer — Explainability (SHAP).

- Global: mean |SHAP| feature importance + beeswarm summary for the
  Random Forest (best baseline), answering "what drives credit decisions".
- Local: waterfall explanations for two test applicants (one predicted
  bad, one predicted good) — the GDPR Art. 22 "why was I declined"
  adverse-action view.
- Interpretability cross-check: Logistic Regression coefficients (odds
  ratios) exported as the fully transparent reference model.

Outputs (results/): shap_importance_bar.png, shap_beeswarm.png,
shap_local_denied.png, shap_local_approved.png, lr_coefficients.csv
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

from preprocessing import load_data
from viz_style import BLUE, apply_style

RANDOM_STATE = 42
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")


def pretty_names(raw):
    """ColumnTransformer names like 'nom__Purpose_car' -> 'Purpose=car'."""
    out = []
    for n in raw:
        n = n.split("__", 1)[-1]
        out.append(n.replace("_", "=", 1) if "=" not in n and any(
            n.startswith(p + "_") for p in ("Sex", "Housing", "Purpose")) else n)
    return out


def main():
    apply_style()
    RESULTS_DIR.mkdir(exist_ok=True)
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    rf_pipe = joblib.load(MODELS_DIR / "random_forest.joblib")
    prep, rf = rf_pipe.named_steps["prep"], rf_pipe.named_steps["model"]

    X_test_t = prep.transform(X_test)
    if hasattr(X_test_t, "toarray"):
        X_test_t = X_test_t.toarray()
    names = pretty_names(prep.get_feature_names_out())

    explainer = shap.TreeExplainer(rf)
    sv = explainer(X_test_t)
    # keep SHAP values toward class 1 = bad credit
    sv = sv[:, :, 1]
    sv.feature_names = names

    # --- Global: mean |SHAP| bar ------------------------------------------
    mean_abs = pd.Series(np.abs(sv.values).mean(axis=0), index=names)
    top = mean_abs.sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.barh(top.index, top.values, color=BLUE, height=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9, color="#52514e")
    ax.set_xlabel("Mean |SHAP value| (impact on P(bad credit))")
    ax.set_title("Global feature importance — Random Forest (SHAP)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "shap_importance_bar.png", dpi=150)
    plt.close(fig)

    # --- Global: beeswarm summary -----------------------------------------
    fig = plt.figure(figsize=(9, 6))
    shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.title("SHAP summary — direction and spread of feature effects")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_beeswarm.png", dpi=150)
    plt.close("all")

    # --- Local: one denied, one approved applicant --------------------------
    y_pred = rf_pipe.predict(X_test)
    idx_denied = int(np.where((y_pred == 1) & (y_test.to_numpy() == 1))[0][0])
    idx_approved = int(np.where((y_pred == 0) & (y_test.to_numpy() == 0))[0][0])

    for idx, tag in [(idx_denied, "denied"), (idx_approved, "approved")]:
        fig = plt.figure(figsize=(9, 5.5))
        shap.plots.waterfall(sv[idx], max_display=10, show=False)
        plt.title(f"Local explanation — applicant predicted "
                  f"{'BAD (denied)' if tag == 'denied' else 'GOOD (approved)'}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"shap_local_{tag}.png", dpi=150)
        plt.close("all")
        applicant = X_test.iloc[idx]
        print(f"\n--- Applicant ({tag}), P(bad) = "
              f"{rf_pipe.predict_proba(X_test)[idx, 1]:.2f} ---")
        print(applicant.to_string())

    # --- Transparent reference: LR odds ratios ------------------------------
    lr_pipe = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    lr_names = pretty_names(lr_pipe.named_steps["prep"].get_feature_names_out())
    coefs = pd.DataFrame({
        "Feature": lr_names,
        "Coefficient": lr_pipe.named_steps["model"].coef_[0].round(3),
        "Odds ratio (bad)": np.exp(lr_pipe.named_steps["model"].coef_[0]).round(3),
    }).sort_values("Coefficient", ascending=False)
    coefs.to_csv(RESULTS_DIR / "lr_coefficients.csv", index=False)

    print("\n=== Top global drivers (mean |SHAP|) ===")
    print(mean_abs.sort_values(ascending=False).head(10).round(4).to_string())
    print("\n=== Logistic Regression — strongest coefficients (odds of BAD) ===")
    print(pd.concat([coefs.head(5), coefs.tail(5)]).to_string(index=False))
    print(f"\nSaved explainability outputs to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
