"""Tests for the fairness-audit metrics (src/fairness_audit.py)."""

import numpy as np
import pandas as pd

from fairness_audit import equalize_fpr_thresholds, group_metrics, summarize


def test_group_metrics_known_confusion():
    # single group, hand-computable confusion matrix
    y_true = pd.Series([1, 1, 0, 0])
    y_pred = pd.Series([1, 0, 1, 0])  # tp=1 fn=1 fp=1 tn=1
    groups = pd.Series(["A", "A", "A", "A"])
    out = group_metrics(y_true, y_pred, groups)
    row = out.iloc[0]
    assert row["n"] == 4
    assert row["TPR (bad detected)"] == 0.5
    assert row["FPR (good denied)"] == 0.5
    assert row["Approval rate"] == 0.5     # two predicted good
    assert row["Accuracy"] == 0.5


def test_group_metrics_sorted_by_size_and_all_groups_present():
    y_true = pd.Series([1, 0, 1, 0, 0])
    y_pred = pd.Series([1, 0, 0, 0, 1])
    groups = pd.Series(["big", "big", "big", "small", "small"])
    out = group_metrics(y_true, y_pred, groups)
    assert list(out["Group"]) == ["big", "small"]  # n desc
    assert set(out["Group"]) == {"big", "small"}


def test_group_metrics_handles_group_without_positives():
    y_true = pd.Series([0, 0])
    y_pred = pd.Series([0, 1])
    groups = pd.Series(["G", "G"])
    out = group_metrics(y_true, y_pred, groups)
    assert np.isnan(out.iloc[0]["TPR (bad detected)"])  # no true positives possible


def test_summarize_ratios_and_gaps():
    df = pd.DataFrame({
        "Group": ["A", "B"],
        "Approval rate": [0.8, 0.4],
        "TPR (bad detected)": [0.7, 0.6],
        "FPR (good denied)": [0.2, 0.5],
    })
    s = summarize(df, "Sex")
    assert s["Attribute"] == "Sex"
    assert s["Demographic parity diff (approval)"] == 0.4      # 0.8 - 0.4
    assert s["Disparate impact ratio (80% rule)"] == 0.5       # 0.4 / 0.8
    assert s["Equal opportunity diff (TPR gap)"] == 0.1        # 0.7 - 0.6
    # equalized odds = max(TPR gap 0.1, FPR gap 0.3)
    assert s["Equalized odds diff (max TPR/FPR gap)"] == 0.3


def test_equalize_fpr_thresholds_meets_target():
    rng = np.random.default_rng(0)
    y_true = pd.Series(np.r_[np.zeros(100), np.ones(50)].astype(int))
    y_prob = pd.Series(rng.random(150))
    groups = pd.Series(["M"] * 75 + ["F"] * 75)
    target = 0.2
    thr = equalize_fpr_thresholds(y_true, y_prob, groups, target)
    y_adj = (y_prob.to_numpy() >= groups.map(thr).to_numpy()).astype(int)
    for g in ["M", "F"]:
        m = (groups == g) & (y_true == 0)
        fpr = (y_adj[m.to_numpy()] == 1).mean()
        assert fpr <= target + 0.05  # within one applicant of the target


def test_equalize_fpr_zero_target_blocks_all_denials():
    y_true = pd.Series([0, 0, 0])
    y_prob = pd.Series([0.1, 0.5, 0.9])
    groups = pd.Series(["A", "A", "A"])
    thr = equalize_fpr_thresholds(y_true, y_prob, groups, target_fpr=0.0)
    assert thr["A"] > 1.0  # threshold above any probability -> no wrongful denials
