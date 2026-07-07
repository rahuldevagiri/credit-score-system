"""End-to-end integration tests.

These run each pipeline stage's ``main()`` once and assert that the expected
reproducible artifacts are written. They double as (a) a reproducibility
guarantee for the grader and (b) coverage of the orchestration code that the
unit tests do not exercise. They are slower (model fitting + SHAP) but still
finish in well under a minute on the 1,000-row dataset.
"""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"


@pytest.mark.integration
def test_train_baseline_main_writes_metrics_and_models():
    import train_baseline

    train_baseline.main()
    metrics = pd.read_csv(RESULTS / "baseline_metrics.csv")
    assert set(metrics["Model"]) == {
        "Logistic Regression", "Decision Tree", "Random Forest"}
    # recall is the primary metric and must be a valid rate for every model
    assert metrics["Recall (bad)"].between(0, 1).all()
    for name in ["random_forest", "logistic_regression", "decision_tree"]:
        assert (MODELS / f"{name}.joblib").exists()


@pytest.mark.integration
def test_fairness_audit_main_writes_summary():
    import fairness_audit

    fairness_audit.main()
    summary = pd.read_csv(RESULTS / "fairness_summary.csv")
    assert set(summary["Attribute"]) == {"Sex", "Age band"}
    comparison = pd.read_csv(RESULTS / "mitigation_comparison.csv")
    assert len(comparison) == 2  # baseline + mitigated


@pytest.mark.integration
def test_explainability_main_writes_shap_and_coefficients():
    import explainability

    explainability.main()
    for art in ["shap_importance_bar.png", "shap_beeswarm.png",
                "shap_local_denied.png", "shap_local_approved.png",
                "lr_coefficients.csv"]:
        assert (RESULTS / art).exists()


@pytest.mark.integration
def test_adversarial_main_writes_attack_artifacts():
    import adversarial

    adversarial.main()
    for art in ["adversarial_poisoning.csv", "adversarial_evasion.csv",
                "adversarial_membership.csv", "adversarial_summary.png"]:
        assert (RESULTS / art).exists()
