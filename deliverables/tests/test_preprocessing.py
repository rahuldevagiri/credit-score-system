"""Tests for the preprocessing pipeline (src/preprocessing.py)."""

import numpy as np
import pandas as pd
import pytest

from preprocessing import (
    ACCOUNT_ORDER,
    NUMERIC_FEATURES,
    SENSITIVE_ATTRIBUTES,
    build_preprocessor,
    load_data,
)


def test_load_data_shapes_and_target(dataset):
    X, y, audit = dataset
    assert X.shape == (1000, 9)
    assert set(y.unique()) == {0, 1}
    # target must map bad -> 1 (positive class) at the known 30% base rate
    assert round(y.mean(), 2) == 0.30


def test_missing_accounts_become_none_category(dataset):
    X, _, _ = dataset
    for col in ["Saving accounts", "Checking account"]:
        assert X[col].isna().sum() == 0
        assert "none" in set(X[col].unique())


def test_audit_frame_isolated_from_features(dataset):
    X, _, audit = dataset
    assert list(audit.columns) == SENSITIVE_ATTRIBUTES + ["Risk"]
    # audit keeps the raw label; features never expose the target
    assert "Risk" not in X.columns


def test_load_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("data/does_not_exist.csv")


def test_preprocessor_output_shape_and_finite(dataset):
    X, _, _ = dataset
    Xt = build_preprocessor().fit_transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    assert Xt.shape[0] == 1000
    assert Xt.shape[1] == 19
    assert np.isfinite(Xt).all()


def test_numeric_features_are_scaled(dataset):
    X, _, _ = dataset
    pre = build_preprocessor().fit(X)
    Xt = pre.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    # first three transformed columns are the standardized numerics
    num_block = Xt[:, : len(NUMERIC_FEATURES)]
    assert np.allclose(num_block.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(num_block.std(axis=0), 1, atol=1e-2)


def test_ordinal_unknown_category_maps_to_minus_one(dataset):
    X, _, _ = dataset
    pre = build_preprocessor().fit(X)
    row = X.iloc[[0]].copy()
    row["Saving accounts"] = "platinum"  # unseen tier
    Xt = pre.transform(row)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    ordinal_start = len(NUMERIC_FEATURES)
    assert (Xt[0, ordinal_start:ordinal_start + 3] == -1).any()


def test_account_order_is_monotone_reference():
    # 'none' must sort below real balances so the ordinal encoding is meaningful
    assert ACCOUNT_ORDER[0] == "none"
    assert ACCOUNT_ORDER == sorted(
        ACCOUNT_ORDER, key=["none", "little", "moderate", "quite rich", "rich"].index)
