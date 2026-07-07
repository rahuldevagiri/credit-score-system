"""Tests for the adversarial robustness analyses (src/adversarial.py)."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from adversarial import (
    GAMEABLE_FEATURES,
    evasion_robustness,
    label_flip_poisoning,
    make_rf_pipeline,
    membership_inference,
)


def test_make_rf_pipeline_structure():
    pipe = make_rf_pipeline()
    assert list(pipe.named_steps) == ["prep", "model"]
    assert pipe.named_steps["model"].n_estimators == 300


def test_label_flip_poisoning_counts_and_bounds(sample):
    X, y = sample
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0)
    out = label_flip_poisoning(Xtr, ytr, Xte, yte, fractions=[0.0, 0.2])
    assert out.iloc[0]["Labels flipped"] == 0
    assert out.iloc[1]["Labels flipped"] == round(0.2 * len(ytr))
    assert out["Recall (bad)"].between(0, 1).all()
    assert out["ROC-AUC"].between(0, 1).all()


def test_evasion_zero_budget_flips_nothing(small_rf, sample):
    X, y = sample
    out = evasion_robustness(small_rf, X, y, budgets=[0.0])
    # with no manipulation, no rejected applicant should flip to approved
    assert out.iloc[0]["Flipped to approved"] == 0.0


def test_evasion_larger_budget_flips_some(small_rf, sample):
    X, y = sample
    out = evasion_robustness(small_rf, X, y, budgets=[0.0, 0.5])
    assert out["Flipped to approved"].between(0, 1).all()
    # shrinking loan size/term can only help an applicant, never hurt
    assert out.iloc[1]["Flipped to approved"] >= out.iloc[0]["Flipped to approved"]


def test_gameable_features_are_manipulable_numerics():
    assert GAMEABLE_FEATURES == ["Duration", "Credit amount"]


def test_membership_inference_keys_and_ranges(small_rf, sample):
    X, y = sample
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0)
    res = membership_inference(small_rf, Xtr, ytr, Xte, yte)
    assert 0.0 <= res["Membership-inference attack AUC"] <= 1.0
    gap = (res["Mean confidence (train/members)"]
           - res["Mean confidence (test/non-members)"])
    assert abs(gap - res["Confidence gap"]) < 1e-6
