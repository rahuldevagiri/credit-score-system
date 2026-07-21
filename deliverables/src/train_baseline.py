"""
Week 2 — Baseline model training and evaluation.

Models: Logistic Regression, Decision Tree, Random Forest
(interpretability -> performance spectrum, per the Step-2 project plan).

Evaluation: stratified 80/20 hold-out + 5-fold cross-validation.
Primary metric: recall on the 'bad' class (bad=1), reflecting the
asymmetric cost of approving a risky applicant. class_weight='balanced'
handles the 70/30 imbalance via cost-sensitive learning.

Outputs (results/):
- baseline_metrics.csv       hold-out metrics per model
- cv_recall_bad.csv          5-fold CV recall (mean +/- std)
- roc_curves.png             ROC comparison
- confusion_matrices.png     confusion matrix per model
"""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocessing import build_preprocessor, load_data
from viz_style import SERIES, apply_style

RANDOM_STATE = 42
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE
    ),
}


def main():
    apply_style()
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    X, y, _ = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | bad-rate: {y.mean():.0%}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    metrics_rows, cv_rows, fitted = [], [], {}

    for name, model in MODELS.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])

        cv_recall = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="recall")
        cv_rows.append({"Model": name,
                        "CV Recall (bad) mean": round(cv_recall.mean(), 3),
                        "CV Recall (bad) std": round(cv_recall.std(), 3)})

        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        joblib.dump(pipe, MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        metrics_rows.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "Precision (bad)": round(precision_score(y_test, y_pred), 3),
            "Recall (bad)": round(recall_score(y_test, y_pred), 3),
            "F1 (bad)": round(f1_score(y_test, y_pred), 3),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 3),
        })

    metrics = pd.DataFrame(metrics_rows)
    cv_table = pd.DataFrame(cv_rows)
    metrics.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)
    cv_table.to_csv(RESULTS_DIR / "cv_recall_bad.csv", index=False)

    print("\n=== Hold-out test metrics (positive class = bad credit) ===")
    print(metrics.to_string(index=False))
    print("\n=== 5-fold CV recall on training data ===")
    print(cv_table.to_string(index=False))

    # ROC curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for color, (name, pipe) in zip(SERIES, fitted.items()):
        RocCurveDisplay.from_estimator(
            pipe, X_test, y_test, name=name, ax=ax, curve_kwargs={"color": color}
        )
    ax.plot([0, 1], [0, 1], color="#c3c2b7", ls="--", lw=1)
    ax.set_title("ROC Curves — Baseline Models (bad credit = positive)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)

    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, pipe) in zip(axes, fitted.items()):
        cm = confusion_matrix(y_test, pipe.predict(X_test))
        ConfusionMatrixDisplay(cm, display_labels=["good", "bad"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(name)
        ax.grid(False)
    fig.suptitle("Confusion Matrices — Hold-out Test Set")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)

    print(f"\nSaved metrics and plots to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
