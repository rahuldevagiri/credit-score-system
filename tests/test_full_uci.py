"""Tests for the full 20-feature UCI loader and pipeline (src/full_uci.py)."""

import numpy as np
import pytest

from full_uci import (
    NOMINAL,
    ORDINAL,
    SEX_FROM_STATUS,
    build_full_preprocessor,
    load_full_uci,
    make_full_rf,
)


@pytest.fixture(scope="module")
def full_data():
    return load_full_uci()


def test_load_shapes_and_target(full_data):
    X, y, audit = full_data
    assert X.shape == (1000, 20)          # 20 features (target dropped)
    assert set(y.unique()) == {0, 1}
    # UCI base rate is the well-known 30% bad
    assert round(y.mean(), 2) == 0.30


def test_symbolic_codes_decoded_to_readable(full_data):
    X, _, _ = full_data
    # raw A-codes must be gone, replaced by the code-book labels
    assert set(X["Foreign worker"].unique()) <= {"yes", "no"}
    assert "no account" in set(X["Checking status"].unique())
    assert not X["Purpose"].astype(str).str.match(r"A\d+").any()


def test_audit_frame_has_three_protected_attributes(full_data):
    _, _, audit = full_data
    assert list(audit.columns) == ["Sex", "Age", "Foreign worker"]
    # Sex derived from Personal status & sex resolves to exactly male/female
    assert set(audit["Sex"].unique()) == {"male", "female"}


def test_sex_mapping_covers_all_status_codes():
    # every personal-status label must map to a sex (no silent NaNs)
    assert set(SEX_FROM_STATUS.values()) == {"male", "female"}
    assert all(v in ("male", "female") for v in SEX_FROM_STATUS.values())


def test_preprocessor_is_finite_and_wider_than_baseline(full_data):
    X, _, _ = full_data
    Xt = build_full_preprocessor().fit_transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    assert Xt.shape[0] == 1000
    # richer feature set must yield more columns than the 19-col baseline
    assert Xt.shape[1] > 19
    assert np.isfinite(Xt).all()


def test_ordinal_tiers_are_ordered_reference():
    # 'no account' / 'none' style floor must sort first in each ordinal ramp
    assert ORDINAL["Checking status"][0] == "no account"
    assert ORDINAL["Savings"][0] == "none/unknown"
    assert "Foreign worker" in NOMINAL


def test_full_rf_fits_and_predicts(full_data):
    X, y, _ = full_data
    pipe = make_full_rf().fit(X.iloc[:200], y.iloc[:200])
    proba = pipe.predict_proba(X.iloc[:10])
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
