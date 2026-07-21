"""Tests for algorithmic recourse (src/recourse.py)."""

import pandas as pd

from recourse import ACTIONABLE, FLOOR, find_recourse


def _risky_applicant():
    """A deliberately high-risk profile: young, large long loan, thin accounts."""
    return pd.DataFrame([{
        "Age": 22, "Sex": "female", "Job": 2, "Housing": "rent",
        "Saving accounts": "little", "Checking account": "little",
        "Credit amount": 9000, "Duration": 48, "Purpose": "business",
    }])


def test_actionable_are_loan_structure_features():
    assert ACTIONABLE == ["Credit amount", "Duration"]
    assert set(FLOOR) == set(ACTIONABLE)


def test_already_approved_returns_no_options(small_rf):
    # a very permissive threshold makes any applicant "approved"
    r = find_recourse(small_rf, _risky_applicant(), threshold=0.999)
    assert r["status"] == "approved"
    assert r["options"] == []


def test_infeasible_when_threshold_impossibly_strict(small_rf):
    # no loan reduction can push risk below 0.1% -> infeasible
    r = find_recourse(small_rf, _risky_applicant(), threshold=0.001)
    assert r["status"] == "infeasible"
    assert r["options"] == []


def test_recourse_options_actually_flip_and_are_bounded(small_rf):
    row = _risky_applicant()
    # threshold at the applicant's own risk => any risk-reducing change flips it
    p0 = small_rf.predict_proba(row)[0, 1]
    r = find_recourse(small_rf, row, threshold=float(p0))
    assert r["status"] in {"recourse", "infeasible"}
    for o in r["options"]:
        assert o["p_bad"] < p0                 # the option really is an approval
        assert 0 < o["reduction_pct"] <= 60    # within the search budget
        if o["to"] is not None:                # single-feature option
            assert o["to"] < o["from"]         # it is a reduction
            assert o["to"] >= FLOOR[o["feature"]]


def test_never_suggests_below_floor(small_rf):
    row = _risky_applicant()
    row.iloc[0, row.columns.get_loc("Credit amount")] = 300  # already near the floor
    r = find_recourse(small_rf, row, threshold=0.001, max_reduction=0.6)
    # can't reduce below 250 -> no valid single-feature amount option
    for o in r["options"]:
        if o["feature"] == "Credit amount":
            assert o["to"] >= FLOOR["Credit amount"]
