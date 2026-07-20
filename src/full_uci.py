"""
Full UCI German Credit dataset (20 features) — the "additional information".

The Kaggle file used for the baseline is a 9-feature subset. The original
UCI file (data/german.data) adds 11 attributes — most importantly Credit
history, Employment length, and two regulatory-critical sensitive ones:
Personal status & sex, and Foreign worker. This module:

1. loads the raw symbolic file (A11...A204 codes) into readable categories,
2. trains the same Random-Forest configuration on all 20 features,
3. compares hold-out performance against the 9-feature baseline, and
4. extends the fairness audit to Foreign worker — the protected group that
   simply does not exist in the Kaggle subset.

Target encoding follows the project convention: bad = 1 (positive class).

Outputs (results/): full_uci_metrics.csv, full_uci_fairness.csv,
full_uci_importances.csv, full_uci_comparison.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from fairness_audit import group_metrics, summarize
from viz_style import AQUA, BLUE, apply_style

RANDOM_STATE = 42
RESULTS_DIR = Path("results")
DATA_PATH = "data/german.data"

COLUMNS = [
    "Checking status", "Duration", "Credit history", "Purpose", "Credit amount",
    "Savings", "Employment since", "Installment rate", "Personal status & sex",
    "Other debtors", "Residence since", "Property", "Age", "Other installment plans",
    "Housing", "Existing credits", "Job", "People liable", "Telephone",
    "Foreign worker", "Target",
]

# Official UCI code book, abbreviated to readable labels.
CODE_MAP = {
    "Checking status": {"A11": "<0 DM", "A12": "0-200 DM", "A13": ">=200 DM", "A14": "no account"},
    "Credit history": {"A30": "no credits/all paid", "A31": "all paid (this bank)",
                       "A32": "existing paid duly", "A33": "past delays",
                       "A34": "critical/other credits"},
    "Purpose": {"A40": "new car", "A41": "used car", "A42": "furniture", "A43": "radio/TV",
                "A44": "appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
                "A48": "retraining", "A49": "business", "A410": "others"},
    "Savings": {"A61": "<100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
                "A64": ">=1000 DM", "A65": "none/unknown"},
    "Employment since": {"A71": "unemployed", "A72": "<1 yr", "A73": "1-4 yrs",
                         "A74": "4-7 yrs", "A75": ">=7 yrs"},
    "Personal status & sex": {"A91": "male div/sep", "A92": "female div/sep/married",
                              "A93": "male single", "A94": "male married/wid",
                              "A95": "female single"},
    "Other debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "Property": {"A121": "real estate", "A122": "savings contract/insurance",
                 "A123": "car or other", "A124": "none/unknown"},
    "Other installment plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "Housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "Job": {"A171": "unemployed non-res", "A172": "unskilled", "A173": "skilled",
            "A174": "management/self-emp"},
    "Telephone": {"A191": "none", "A192": "yes"},
    "Foreign worker": {"A201": "yes", "A202": "no"},
}

NUMERIC = ["Duration", "Credit amount", "Installment rate", "Residence since",
           "Age", "Existing credits", "People liable"]
# tiers with a natural order -> ordinal; everything else categorical -> one-hot
ORDINAL = {
    "Checking status": ["no account", "<0 DM", "0-200 DM", ">=200 DM"],
    "Savings": ["none/unknown", "<100 DM", "100-500 DM", "500-1000 DM", ">=1000 DM"],
    "Employment since": ["unemployed", "<1 yr", "1-4 yrs", "4-7 yrs", ">=7 yrs"],
}
NOMINAL = ["Credit history", "Purpose", "Personal status & sex", "Other debtors",
           "Property", "Other installment plans", "Housing", "Job", "Telephone",
           "Foreign worker"]

SEX_FROM_STATUS = {
    "male div/sep": "male", "male single": "male", "male married/wid": "male",
    "female div/sep/married": "female", "female single": "female",
}

AGE_BINS = [0, 25, 60, 120]
AGE_LABELS = ["<=25", "26-60", ">60"]


def load_full_uci(path: str = DATA_PATH):
    """Load the 20-feature UCI file with readable categories.

    Returns X (20 features), y (bad=1), and an audit frame with the three
    protected attributes: Sex (derived from Personal status), Age, and
    Foreign worker.
    """
    df = pd.read_csv(path, sep=" ", header=None, names=COLUMNS)
    for col, mapping in CODE_MAP.items():
        df[col] = df[col].map(mapping)
    y = (df["Target"] == 2).astype(int)  # UCI: 1 = good, 2 = bad
    X = df.drop(columns=["Target"])
    audit = pd.DataFrame({
        "Sex": X["Personal status & sex"].map(SEX_FROM_STATUS),
        "Age": X["Age"],
        "Foreign worker": X["Foreign worker"],
    })
    return X, y, audit


def build_full_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    ordinal = OrdinalEncoder(categories=[ORDINAL[c] for c in ORDINAL],
                             handle_unknown="use_encoded_value", unknown_value=-1)
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("ord", ordinal, list(ORDINAL)),
        ("nom", OneHotEncoder(handle_unknown="ignore"), NOMINAL),
    ])


def make_full_rf() -> Pipeline:
    """Same RF configuration as the audited 9-feature baseline."""
    return Pipeline([
        ("prep", build_full_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20,
            class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def holdout_metrics(pipe, X_test, y_test, label) -> dict:
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    return {
        "Feature set": label,
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision (bad)": round(precision_score(y_test, y_pred), 3),
        "Recall (bad)": round(recall_score(y_test, y_pred), 3),
        "F1 (bad)": round(f1_score(y_test, y_pred), 3),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 3),
    }


def main():
    apply_style()
    RESULTS_DIR.mkdir(exist_ok=True)
    X, y, audit = load_full_uci()

    # identical split indices to the baseline scripts (same y, same seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    full_pipe = make_full_rf().fit(X_train, y_train)
    rows = [holdout_metrics(full_pipe, X_test, y_test, "Full UCI (20 features)")]

    # 9-feature baseline on the same rows, for an apples-to-apples comparison
    from preprocessing import load_data
    from train_baseline import MODELS
    from sklearn.base import clone
    from preprocessing import build_preprocessor
    Xk, yk, _ = load_data()
    Xk_train, Xk_test, yk_train, yk_test = train_test_split(
        Xk, yk, test_size=0.2, stratify=yk, random_state=RANDOM_STATE)
    base_pipe = Pipeline([("prep", build_preprocessor()),
                          ("model", clone(MODELS["Random Forest"]))]).fit(Xk_train, yk_train)
    rows.append(holdout_metrics(base_pipe, Xk_test, yk_test, "Kaggle subset (9 features)"))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS_DIR / "full_uci_metrics.csv", index=False)
    print("=== Random Forest: full UCI vs Kaggle subset (same split, same config) ===")
    print(metrics.to_string(index=False))

    # --- Fairness on the full data: now including Foreign worker ----------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_prob = cross_val_predict(make_full_rf(), X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    age_band = pd.cut(audit["Age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    tables, summaries = {}, []
    for attr, groups in [("Sex", audit["Sex"]), ("Age band", age_band),
                         ("Foreign worker", audit["Foreign worker"])]:
        t = group_metrics(y, y_pred, groups)
        tables[attr] = t
        summaries.append(summarize(t, attr))
        print(f"\n=== Fairness by {attr} (full UCI model, out-of-fold) ===")
        print(t.to_string(index=False))
    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS_DIR / "full_uci_fairness.csv", index=False)
    print("\n=== Fairness summary (full UCI model) ===")
    print(summary.to_string(index=False))

    # --- Mitigation: Age (clean) vs Foreign worker (trade-off) -------------
    # Age: FPR-equalizing thresholds fix the disparate impact cleanly.
    # Foreign worker: because ~96% of applicants ARE foreign workers, raising
    # their threshold to cut wrongful denials also collapses overall recall —
    # a documented trade-off requiring a policy decision, not an auto-fix.
    from fairness_audit import mitigation_rows
    mit = pd.DataFrame(
        mitigation_rows(y, y_prob, tables["Age band"], age_band, "Age band", threshold=0.5)[0]
        + mitigation_rows(y, y_prob, tables["Foreign worker"],
                          audit["Foreign worker"], "Foreign worker", threshold=0.5)[0])
    mit.to_csv(RESULTS_DIR / "full_uci_mitigation.csv", index=False)
    print("\n=== Mitigation on the full model (Age = clean fix, Foreign worker = trade-off) ===")
    print(mit.to_string(index=False))

    # --- Feature importances of the richer model ---------------------------
    names = [n.split("__", 1)[-1] for n in
             full_pipe.named_steps["prep"].get_feature_names_out()]
    imp = pd.Series(full_pipe.named_steps["model"].feature_importances_,
                    index=names).sort_values(ascending=False)
    imp.head(15).round(4).to_csv(RESULTS_DIR / "full_uci_importances.csv",
                                 header=["Importance"])
    print("\n=== Top 10 feature importances (full UCI model) ===")
    print(imp.head(10).round(4).to_string())

    # --- Chart: performance comparison + importances -----------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    metric_cols = ["Accuracy", "Recall (bad)", "F1 (bad)", "ROC-AUC"]
    xpos = np.arange(len(metric_cols))
    w = 0.38
    for off, color, (_, row) in zip((-w / 2, w / 2), (BLUE, AQUA), metrics.iterrows()):
        bars = ax1.bar(xpos + off, [row[c] for c in metric_cols], w,
                       label=row["Feature set"], color=color)
        ax1.bar_label(bars, fmt="%.2f", padding=2, fontsize=9, color="#52514e")
    ax1.set_xticks(xpos, metric_cols)
    ax1.set_ylim(0, 1)
    ax1.set_title("Random Forest: 20-feature UCI vs 9-feature subset")
    ax1.legend(loc="lower right")

    top10 = imp.head(10).iloc[::-1]
    bars = ax2.barh(top10.index, top10.values, color=BLUE, height=0.6)
    ax2.bar_label(bars, fmt="%.3f", padding=3, fontsize=9, color="#52514e")
    ax2.set_xlabel("Random-Forest importance")
    ax2.set_title("Top drivers with the full feature set")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "full_uci_comparison.png", dpi=150)

    print(f"\nSaved full-UCI artifacts to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
