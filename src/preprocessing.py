"""
Preprocessing pipeline for the German Credit dataset (Kaggle version + Risk target).

Design decisions (documented for the Responsible AI report):
- Missing 'Saving accounts' / 'Checking account' values are treated as an
  informative category "none" (no account exists), not imputed or dropped.
- 'Credit amount' is log-transformed (right-skewed distribution).
- Account tiers are ordinally encoded (they have a natural order);
  nominal features are one-hot encoded.
- 'Risk' target is mapped to bad=1 (positive class), good=0, so that
  recall directly measures detection of risky applicants.
- Sensitive attributes (Age, Sex) are kept in the feature set for the
  baseline; a separate audit copy is always returned so fairness can be
  evaluated regardless of what the model sees.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

DATA_PATH = "data/german_credit_data_with_target.csv"

ACCOUNT_ORDER = ["none", "little", "moderate", "quite rich", "rich"]

NUMERIC_FEATURES = ["Age", "Credit amount", "Duration"]
ORDINAL_FEATURES = ["Saving accounts", "Checking account", "Job"]
NOMINAL_FEATURES = ["Sex", "Housing", "Purpose"]

SENSITIVE_ATTRIBUTES = ["Age", "Sex"]


def load_data(path: str = DATA_PATH):
    """Load dataset; return features X, binary target y (bad=1), and an
    untouched audit frame with sensitive attributes for fairness analysis."""
    df = pd.read_csv(path, index_col=0)

    # Missing account values mean "no account" -> explicit category
    for col in ["Saving accounts", "Checking account"]:
        df[col] = df[col].fillna("none")

    y = (df["Risk"] == "bad").astype(int)
    X = df.drop(columns=["Risk"])
    audit = df[SENSITIVE_ATTRIBUTES + ["Risk"]].copy()
    return X, y, audit


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer implementing the Step-2 preprocessing design."""
    numeric = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    ordinal = OrdinalEncoder(
        categories=[ACCOUNT_ORDER, ACCOUNT_ORDER, [0, 1, 2, 3]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    nominal = OneHotEncoder(handle_unknown="ignore")

    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("ord", ordinal, ORDINAL_FEATURES),
        ("nom", nominal, NOMINAL_FEATURES),
    ])
