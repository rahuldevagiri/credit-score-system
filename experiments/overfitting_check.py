"""
Overfitting diagnostic — is the model memorising or generalising?

Two complementary checks:
1. Train vs 5-fold cross-validation AUC gap for each model. A large gap
   (train >> validation) is the signature of overfitting.
2. A learning curve for the Random Forest: if the validation score rises and
   plateaus while the train/validation gap narrows with more data, the model
   is generalising, not memorising.

Run:  python experiments/overfitting_check.py   (writes results/overfitting_check.png)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import build_preprocessor, load_data
from viz_style import AQUA, BLUE, apply_style

apply_style()
RESULTS = ROOT / "results"
RS = 42
X, y, _ = load_data()
cv = StratifiedKFold(5, shuffle=True, random_state=RS)

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RS),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=20, class_weight="balanced", random_state=RS),
}

# --- 1) train vs CV AUC gap -------------------------------------------------
rows = []
for name, model in MODELS.items():
    pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
    r = cross_validate(pipe, X, y, cv=cv, scoring="roc_auc", return_train_score=True)
    tr, va = r["train_score"].mean(), r["test_score"].mean()
    rows.append({"Model": name, "Train AUC": round(tr, 3),
                 "CV AUC": round(va, 3), "CV std": round(r["test_score"].std(), 3),
                 "Gap": round(tr - va, 3)})
tbl = pd.DataFrame(rows)
print("=== Train vs 5-fold CV AUC (gap = overfitting signal) ===")
print(tbl.to_string(index=False))
tbl.to_csv(RESULTS / "overfitting_gap.csv", index=False)

# --- 2) learning curve for the Random Forest --------------------------------
rf = Pipeline([("prep", build_preprocessor()),
               ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                 class_weight="balanced", random_state=RS))])
sizes, train_sc, val_sc = learning_curve(
    rf, X, y, cv=cv, scoring="roc_auc",
    train_sizes=np.linspace(0.15, 1.0, 7), shuffle=True, random_state=RS)
tr_m, tr_s = train_sc.mean(1), train_sc.std(1)
va_m, va_s = val_sc.mean(1), val_sc.std(1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sizes, tr_m, "o-", color=BLUE, label="Training AUC")
ax.fill_between(sizes, tr_m - tr_s, tr_m + tr_s, color=BLUE, alpha=0.12)
ax.plot(sizes, va_m, "s-", color=AQUA, label="Cross-validation AUC")
ax.fill_between(sizes, va_m - va_s, va_m + va_s, color=AQUA, alpha=0.12)
ax.set_xlabel("Training set size")
ax.set_ylabel("ROC-AUC")
ax.set_ylim(0.5, 1.02)
ax.set_title("Learning curve — Random Forest (gap narrows → generalising)")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(RESULTS / "overfitting_check.png", dpi=150)
print(f"\nLearning curve: train AUC {tr_m[-1]:.3f} vs CV AUC {va_m[-1]:.3f} at full size")
print(f"Saved to {RESULTS / 'overfitting_check.png'}")
