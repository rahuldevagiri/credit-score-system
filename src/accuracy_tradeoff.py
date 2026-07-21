"""
The core decision, quantified — why we do NOT optimise accuracy.

Backs the presentation's central claim: the *most accurate* credit model is
the *worst lender*. We compare, on the same stratified hold-out split:

  A. Accuracy-optimised model  — a plain Random Forest with NO cost weighting
                                 and the default 0.5 threshold.
  B. Our cost-sensitive model  — the audited Random Forest with balanced class
                                 weights (recall-optimised).

For each we report overall accuracy and, crucially, recall on the *bad* class
(the share of risky applicants actually caught). The accuracy-optimised model
posts the higher accuracy but catches only a small fraction of bad loans —
exactly the trade-off slide 4 makes.

Output (results/): accuracy_tradeoff.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor, load_data

RANDOM_STATE = 42
RESULTS_DIR = Path("results")


def _rf(class_weight):
    return Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20,
            class_weight=class_weight, random_state=RANDOM_STATE)),
    ])


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    rows = []
    for label, cw in [("Accuracy-optimised (no cost weighting)", None),
                      ("Cost-sensitive (balanced) — our model", "balanced")]:
        pipe = _rf(cw).fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        rows.append({
            "Model": label,
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "Recall (bad) — bad loans caught": round(recall_score(y_test, y_pred), 3),
        })

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "accuracy_tradeoff.csv", index=False)

    print("=== The accuracy trap: most accurate = worst lender ===")
    print(table.to_string(index=False))
    acc, cost = rows[0], rows[1]
    print(f"\nThe accuracy-optimised model is {acc['Accuracy']:.0%} accurate but catches "
          f"only {acc['Recall (bad) — bad loans caught']:.0%} of bad loans.")
    print(f"Our cost-sensitive model trades accuracy ({cost['Accuracy']:.0%}) to catch "
          f"{cost['Recall (bad) — bad loans caught']:.0%} of bad loans — the economically correct choice.")
    print(f"\nSaved {RESULTS_DIR / 'accuracy_tradeoff.csv'}")


if __name__ == "__main__":
    main()
