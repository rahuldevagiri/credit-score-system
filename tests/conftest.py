"""Shared test configuration and fixtures.

The ``src/`` modules import each other by bare name (``from preprocessing
import ...``) and read data via paths relative to the project root, so we
put ``src/`` on ``sys.path`` and pin the working directory to the repo root
for the whole test session.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
os.chdir(ROOT)

from preprocessing import build_preprocessor, load_data  # noqa: E402


@pytest.fixture(scope="session")
def dataset():
    """(X, y, audit) loaded once for the whole session."""
    return load_data()


@pytest.fixture(scope="session")
def sample(dataset):
    """A small, class-stratified subset for fast model-based tests."""
    X, y, _ = dataset
    good = y[y == 0].index[:120]
    bad = y[y == 1].index[:60]
    idx = list(good) + list(bad)
    return X.loc[idx], y.loc[idx]


@pytest.fixture(scope="session")
def small_rf(sample):
    """A cheap fitted Random-Forest pipeline for evasion/membership tests."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    X, y = sample
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=40, min_samples_leaf=5,
            class_weight="balanced", random_state=0)),
    ])
    return pipe.fit(X, y)
