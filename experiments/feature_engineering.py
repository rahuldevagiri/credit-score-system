"""
Feature-engineering experiment — does extra preprocessing improve the model?

Motivation: a natural question is whether we can raise performance by
engineering new features from the 9 Kaggle columns, rather than by adding the
full UCI feature set. This script tests that with 5-fold stratified CV on all
1,000 rows and reports accuracy, recall (bad) and ROC-AUC side by side.

It is exploratory and deliberately kept OUT of src/ — it does not change the
committed pipeline. Findings are written up in REPORT.md section 4.1.

Run:  python experiments/feature_engineering.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, KBinsDiscretizer,
                                   OneHotEncoder, OrdinalEncoder, StandardScaler)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import ACCOUNT_ORDER, load_data  # noqa: E402

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SCORING = {"accuracy": "accuracy", "recall_bad": "recall", "roc_auc": "roc_auc"}
NUM = ["Age", "Credit amount", "Duration"]
ORD = ["Saving accounts", "Checking account", "Job"]
NOM = ["Sex", "Housing", "Purpose"]


def build_prep(numeric, ordinal, nominal):
    num = Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                    ("scale", StandardScaler())])
    ord_cats = [ACCOUNT_ORDER, ACCOUNT_ORDER, [0, 1, 2, 3]][:len(ordinal)]
    return ColumnTransformer([
        ("num", num, numeric),
        ("ord", OrdinalEncoder(categories=ord_cats, handle_unknown="use_encoded_value",
                               unknown_value=-1), ordinal),
        ("nom", OneHotEncoder(handle_unknown="ignore"), nominal),
    ])


def engineer(X, monthly=False, credit_age=False, dur_bins=False):
    X = X.copy()
    extra = []
    if monthly:
        X["MonthlyPayment"] = X["Credit amount"] / X["Duration"]
        extra.append("MonthlyPayment")
    if credit_age:
        X["CreditPerAge"] = X["Credit amount"] / X["Age"]
        extra.append("CreditPerAge")
    if dur_bins:
        kb = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
        X["DurationBin"] = kb.fit_transform(X[["Duration"]]).ravel()
        extra.append("DurationBin")
    return X, extra


def run(name, X, y, prep, model):
    r = cross_validate(Pipeline([("prep", prep), ("model", model)]), X, y,
                       cv=CV, scoring=SCORING)
    return {"Variant": name,
            "Accuracy": round(r["test_accuracy"].mean(), 3),
            "Recall(bad)": round(r["test_recall_bad"].mean(), 3),
            "ROC-AUC": round(r["test_roc_auc"].mean(), 3)}


def rf(balanced=True):
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                  class_weight="balanced" if balanced else None,
                                  random_state=42)


def main():
    X0, y, _ = load_data()
    rows = [run("RF baseline (current pipeline)", X0, y, build_prep(NUM, ORD, NOM), rf())]

    Xm, ex = engineer(X0, monthly=True)
    rows.append(run("RF + MonthlyPayment", Xm, y, build_prep(NUM + ex, ORD, NOM), rf()))

    Xc, ex = engineer(X0, monthly=True, credit_age=True)
    rows.append(run("RF + Monthly + CreditPerAge", Xc, y, build_prep(NUM + ex, ORD, NOM), rf()))

    Xd, ex = engineer(X0, monthly=True, dur_bins=True)
    rows.append(run("RF + Monthly + DurationBin", Xd, y, build_prep(NUM + ex, ORD, NOM), rf()))

    rows.append(run("RF NO class_weight (accuracy-max)", X0, y,
                    build_prep(NUM, ORD, NOM), rf(balanced=False)))

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    rows.append(run("LR baseline", X0, y, build_prep(NUM, ORD, NOM), lr))

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
