"""Tests for the explainability helpers (src/explainability.py)."""

from explainability import pretty_names


def test_pretty_names_strips_transformer_prefix():
    assert pretty_names(["num__Age", "ord__Job"]) == ["Age", "Job"]


def test_pretty_names_formats_onehot_as_equals():
    out = pretty_names(["nom__Sex_male", "nom__Purpose_car", "nom__Housing_own"])
    assert out == ["Sex=male", "Purpose=car", "Housing=own"]


def test_pretty_names_leaves_numeric_names_untouched():
    # a numeric/ordinal feature name with no nominal prefix keeps its underscores
    assert pretty_names(["num__Credit amount"]) == ["Credit amount"]
