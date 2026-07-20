"""
Responsible AI layer — Adversarial robustness & security analysis.

Credit scoring is an EU AI Act *high-risk* system, so its threat model is
part of the risk assessment, not an afterthought. This module implements
three concrete, quantitative attack studies against the Random Forest
credit model and writes reproducible artifacts to ``results/``:

1. Data-poisoning (training-time integrity attack)
   Flip a fraction of the *training* labels and re-fit. Measures how much
   an insider or a compromised data pipeline can silently degrade the
   model's ability to detect bad-credit applicants (recall / ROC-AUC).

2. Evasion / score-gaming (inference-time attack)
   A motivated applicant who is predicted "bad" can only realistically
   change a few features (request a shorter Duration and a smaller Credit
   amount). We perturb exactly those features within a bounded budget and
   measure how many rejected applicants can flip themselves to "approved".

3. Membership inference (privacy attack)
   The gap between the model's confidence on rows it was trained on versus
   unseen rows tells an attacker whether a specific person was in the
   training data — a GDPR-relevant privacy leak. We quantify it as the AUC
   of a confidence-threshold membership classifier.

Outputs (results/):
- adversarial_poisoning.csv        recall/AUC vs. poisoning rate
- adversarial_evasion.csv          flip rate vs. manipulation budget
- adversarial_membership.csv       membership-inference summary
- adversarial_summary.png          combined poisoning + evasion chart
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor, load_data
from viz_style import BLUE, RED, YELLOW, apply_style

RANDOM_STATE = 42
RESULTS_DIR = Path("results")

# Features a real applicant can plausibly manipulate to game the score.
GAMEABLE_FEATURES = ["Duration", "Credit amount"]


def make_rf_pipeline() -> Pipeline:
    """Fresh Random-Forest pipeline matching the audited baseline model."""
    return Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20,
            class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def label_flip_poisoning(X_train, y_train, X_test, y_test, fractions) -> pd.DataFrame:
    """Flip ``fraction`` of training labels, re-fit, report test degradation.

    Returns one row per poisoning fraction with recall (bad) and ROC-AUC,
    so the defender can see how quickly detection collapses under a
    training-data integrity attack.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    y_train = np.asarray(y_train)
    rows = []
    for frac in fractions:
        y_poisoned = y_train.copy()
        n_flip = int(round(frac * len(y_poisoned)))
        if n_flip:
            idx = rng.choice(len(y_poisoned), size=n_flip, replace=False)
            y_poisoned[idx] = 1 - y_poisoned[idx]  # flip 0<->1
        pipe = make_rf_pipeline().fit(X_train, y_poisoned)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "Poisoning rate": round(frac, 3),
            "Labels flipped": n_flip,
            "Recall (bad)": round(recall_score(y_test, y_pred), 3),
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 3),
        })
    return pd.DataFrame(rows)


def evasion_robustness(pipe, X, y, budgets, threshold=0.5) -> pd.DataFrame:
    """Bounded score-gaming attack on applicants predicted 'bad'.

    For each budget b, shrink the gameable features to ``(1 - b)`` of their
    original value (a shorter/smaller loan request) and measure the share of
    originally-rejected applicants whose prediction flips to 'good'.
    """
    prob = pipe.predict_proba(X)[:, 1]
    rejected = prob >= threshold
    X_rej = X[rejected].copy()
    n_rej = len(X_rej)
    rows = []
    for b in budgets:
        X_adv = X_rej.copy()
        for col in GAMEABLE_FEATURES:
            X_adv[col] = X_adv[col] * (1.0 - b)
        flipped = (pipe.predict_proba(X_adv)[:, 1] < threshold).mean() if n_rej else 0.0
        rows.append({
            "Manipulation budget": round(b, 3),
            "Rejected applicants": n_rej,
            "Flipped to approved": round(float(flipped), 3),
        })
    return pd.DataFrame(rows)


def membership_inference(pipe, X_train, y_train, X_test, y_test) -> dict:
    """Confidence-gap membership inference attack.

    An attacker labels high-confidence rows as 'members'. The attack AUC
    (using the model's probability of the *true* class as the score) measures
    how distinguishable training rows are from unseen rows. ~0.5 = no leak.
    """
    def true_class_conf(X, y):
        p = pipe.predict_proba(X)
        y = np.asarray(y)
        return p[np.arange(len(y)), y]

    conf_train = true_class_conf(X_train, y_train)
    conf_test = true_class_conf(X_test, y_test)
    scores = np.concatenate([conf_train, conf_test])
    is_member = np.concatenate([np.ones_like(conf_train), np.zeros_like(conf_test)])
    return {
        "Mean confidence (train/members)": round(float(conf_train.mean()), 3),
        "Mean confidence (test/non-members)": round(float(conf_test.mean()), 3),
        "Confidence gap": round(float(conf_train.mean() - conf_test.mean()), 3),
        "Membership-inference attack AUC": round(float(roc_auc_score(is_member, scores)), 3),
    }


def main():
    apply_style()
    RESULTS_DIR.mkdir(exist_ok=True)
    X, y, _ = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    clean_pipe = make_rf_pipeline().fit(X_train, y_train)

    # 1. Poisoning ---------------------------------------------------------
    poison = label_flip_poisoning(
        X_train, y_train, X_test, y_test,
        fractions=[0.0, 0.05, 0.10, 0.20, 0.30])
    poison.to_csv(RESULTS_DIR / "adversarial_poisoning.csv", index=False)

    # 2. Evasion -----------------------------------------------------------
    evasion = evasion_robustness(
        clean_pipe, X_test, y_test, budgets=[0.0, 0.1, 0.2, 0.3, 0.5])
    evasion.to_csv(RESULTS_DIR / "adversarial_evasion.csv", index=False)

    # 3. Membership inference ---------------------------------------------
    membership = membership_inference(clean_pipe, X_train, y_train, X_test, y_test)
    pd.DataFrame([membership]).to_csv(
        RESULTS_DIR / "adversarial_membership.csv", index=False)

    print("=== Data-poisoning attack (training-label flips) ===")
    print(poison.to_string(index=False))
    print("\n=== Evasion / score-gaming attack (shrink Duration & Credit amount) ===")
    print(evasion.to_string(index=False))
    print("\n=== Membership-inference (privacy) attack ===")
    for k, v in membership.items():
        print(f"{k}: {v}")

    # --- Chart ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    ax1.plot(poison["Poisoning rate"], poison["Recall (bad)"],
             marker="o", color=RED, label="Recall (bad)")
    ax1.plot(poison["Poisoning rate"], poison["ROC-AUC"],
             marker="s", color=BLUE, label="ROC-AUC")
    ax1.set_xlabel("Fraction of training labels flipped")
    ax1.set_ylabel("Test-set metric")
    ax1.set_ylim(0, 1)
    ax1.set_title("Data-poisoning: detection degrades with corrupted labels")
    ax1.legend(loc="lower left")

    bars = ax2.bar(evasion["Manipulation budget"].astype(str),
                   evasion["Flipped to approved"], color=YELLOW, width=0.6)
    ax2.bar_label(bars, fmt="%.2f", padding=2, fontsize=9, color="#52514e")
    ax2.set_xlabel("Manipulation budget (relative reduction of loan size/term)")
    ax2.set_ylabel("Share of rejected applicants flipped to approved")
    ax2.set_ylim(0, 1)
    ax2.set_title("Evasion: how easily a rejection can be gamed")
    fig.suptitle("Adversarial robustness of the credit model")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "adversarial_summary.png", dpi=150)

    print(f"\nSaved adversarial artifacts to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
