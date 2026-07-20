"""
Responsible AI layer — Fairness audit of the credit scoring model.

Approach:
- Best baseline model (Random Forest pipeline) evaluated with 5-fold
  out-of-fold predictions over all 1,000 applicants, so every applicant
  receives an unbiased prediction and group slices stay large enough
  for stable fairness estimates.
- Protected groups: Sex (male/female) and Age band (<=25 / 26-60 / >60).
- Convention: prediction 'bad' (1) = credit DENIED; 'good' (0) = APPROVED.

Metrics per group:
- Approval rate           -> demographic parity / disparate impact (80% rule)
- TPR (recall of bad)     -> equal opportunity (equal detection of true risk)
- FPR (good predicted bad)-> wrongful-denial rate; with TPR -> equalized odds

Mitigation experiment:
- Group-specific decision thresholds on Sex chosen to equalize FPR
  (the wrongful-denial rate — the direct harm to applicants), then
  fairness and performance re-measured.

Outputs (results/): fairness_by_sex.csv, fairness_by_age.csv,
fairness_summary.csv, mitigation_comparison.csv,
fairness_group_metrics.png, fairness_approval_rates.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor, load_data
from viz_style import AQUA, BLUE, RED, YELLOW, apply_style

RANDOM_STATE = 42
RESULTS_DIR = Path("results")
THRESHOLD = 0.5

AGE_BINS = [0, 25, 60, 120]
AGE_LABELS = ["<=25", "26-60", ">60"]


def group_metrics(y_true, y_pred, groups) -> pd.DataFrame:
    """Per-group confusion-based fairness metrics. Denied = predicted bad."""
    rows = []
    for g in groups.unique():
        m = groups == g
        yt, yp = y_true[m], y_pred[m]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        rows.append({
            "Group": g,
            "n": int(m.sum()),
            "Approval rate": round((yp == 0).mean(), 3),
            "TPR (bad detected)": round(tp / (tp + fn), 3) if tp + fn else np.nan,
            "FPR (good denied)": round(fp / (fp + tn), 3) if fp + tn else np.nan,
            "Accuracy": round((yt == yp).mean(), 3),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def summarize(df: pd.DataFrame, attribute: str) -> dict:
    """Gap and ratio summary across the groups of one protected attribute."""
    ar = df["Approval rate"]
    return {
        "Attribute": attribute,
        "Demographic parity diff (approval)": round(ar.max() - ar.min(), 3),
        "Disparate impact ratio (80% rule)": round(ar.min() / ar.max(), 3),
        "Equal opportunity diff (TPR gap)": round(
            df["TPR (bad detected)"].max() - df["TPR (bad detected)"].min(), 3),
        "Equalized odds diff (max TPR/FPR gap)": round(max(
            df["TPR (bad detected)"].max() - df["TPR (bad detected)"].min(),
            df["FPR (good denied)"].max() - df["FPR (good denied)"].min()), 3),
    }


def equalize_fpr_thresholds(y_true, y_prob, groups, target_fpr) -> pd.Series:
    """Per-group thresholds so each group's FPR (wrongful denials) ~= target."""
    thresholds = {}
    for g in groups.unique():
        m = (groups == g) & (y_true == 0)
        probs_good = np.sort(y_prob[m])[::-1]
        k = int(round(target_fpr * len(probs_good)))
        thresholds[g] = probs_good[k - 1] if k > 0 else 1.01
    return pd.Series(thresholds)


def mitigation_rows(y, y_prob, base_table, groups, label, threshold=THRESHOLD):
    """Baseline vs FPR-equalized-threshold mitigation for one protected attribute.

    Returns (list of two comparison-row dicts, per-group threshold Series).
    The mitigation raises each group's threshold so its wrongful-denial rate
    (FPR) meets the lowest group's — the direct harm-equalizing target.
    """
    y_base = (y_prob >= threshold).astype(int)
    thr = equalize_fpr_thresholds(y, y_prob, groups, base_table["FPR (good denied)"].min())
    y_mit = (y_prob >= groups.map(thr).to_numpy()).astype(int)
    mit_table = group_metrics(y, y_mit, groups)

    def _row(scenario, table, yhat):
        row = {"Attribute": label, "Scenario": scenario}
        row.update({k: v for k, v in summarize(table, label).items() if k != "Attribute"})
        row["Overall accuracy"] = round(accuracy_score(y, yhat), 3)
        row["Overall recall (bad)"] = round(recall_score(y, yhat), 3)
        return row

    return [_row("Baseline (0.5 threshold)", base_table, y_base),
            _row("Mitigated (group thresholds)", mit_table, y_mit)], thr


def main():
    apply_style()
    RESULTS_DIR.mkdir(exist_ok=True)
    X, y, audit = load_data()

    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20,
            class_weight="balanced", random_state=RANDOM_STATE)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    sex = audit["Sex"]
    age_band = pd.cut(audit["Age"], bins=AGE_BINS, labels=AGE_LABELS).astype(str)

    by_sex = group_metrics(y, y_pred, sex)
    by_age = group_metrics(y, y_pred, age_band)
    summary = pd.DataFrame([summarize(by_sex, "Sex"), summarize(by_age, "Age band")])

    by_sex.to_csv(RESULTS_DIR / "fairness_by_sex.csv", index=False)
    by_age.to_csv(RESULTS_DIR / "fairness_by_age.csv", index=False)
    summary.to_csv(RESULTS_DIR / "fairness_summary.csv", index=False)

    print("=== Fairness by Sex (out-of-fold predictions, denied = predicted bad) ===")
    print(by_sex.to_string(index=False))
    print("\n=== Fairness by Age band ===")
    print(by_age.to_string(index=False))
    print("\n=== Fairness summary ===")
    print(summary.to_string(index=False))

    # --- Mitigation: FPR-equalizing group thresholds for Sex AND Age -----
    rows_sex, thr_sex = mitigation_rows(y, y_prob, by_sex, sex, "Sex")
    rows_age, thr_age = mitigation_rows(y, y_prob, by_age, age_band, "Age band")
    comparison = pd.DataFrame(rows_sex + rows_age)
    comparison.to_csv(RESULTS_DIR / "mitigation_comparison.csv", index=False)
    print("\n=== Mitigation experiment (Sex & Age band) ===")
    print(f"Sex thresholds: { {k: round(v, 3) for k, v in thr_sex.items()} }")
    print(f"Age thresholds: { {k: round(v, 3) for k, v in thr_age.items()} }")
    print(comparison.to_string(index=False))

    # --- Charts ----------------------------------------------------------
    groups_all = pd.concat([by_sex.assign(Attr="Sex"), by_age.assign(Attr="Age band")])
    labels = groups_all["Group"].tolist()
    xpos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8, 4.8))
    w = 0.38
    b1 = ax.bar(xpos - w / 2, groups_all["TPR (bad detected)"], w,
                label="TPR — true bad detected", color=BLUE)
    b2 = ax.bar(xpos + w / 2, groups_all["FPR (good denied)"], w,
                label="FPR — good applicants denied", color=YELLOW)
    for bars in (b1, b2):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9, color="#52514e")
    ax.set_xticks(xpos, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Error-rate fairness by group — Random Forest (out-of-fold)")
    ax.legend(loc="upper right")
    ax.axvline(1.5, color="#e1e0d9", lw=1)
    ax.text(0.5, 0.97, "Sex", ha="center", color="#898781", transform=ax.get_xaxis_transform())
    ax.text(3.0, 0.97, "Age band", ha="center", color="#898781",
            transform=ax.get_xaxis_transform())
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fairness_group_metrics.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(xpos, groups_all["Approval rate"], 0.55, color=BLUE)
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9, color="#52514e")
    ax.set_xticks(xpos, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Approval rate (predicted good)")
    ax.set_title("Approval rates by group — demographic parity view")
    ax.axvline(1.5, color="#e1e0d9", lw=1)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fairness_approval_rates.png", dpi=150)

    print(f"\nSaved fairness tables and charts to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
