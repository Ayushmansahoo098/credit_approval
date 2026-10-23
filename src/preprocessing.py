"""
Full sklearn Pipeline for credit approval data.

Pipeline steps:
  1. Missing value indicators
  2. Imputation (mode for cat, median for num)
  3. Feature engineering (log1p, interaction, ratio)
  4. Encoding (label-encode binaries, OHE multi-cats)
  5. Scaling (StandardScaler on continuous)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.utils import (BINARY_CATS, CONTINUOUS_COLS, MULTI_CATS, SKEWED_COLS,
                       COLUMN_NAMES, TARGET_COL)

# Columns with missingness to flag
MISSING_INDICATOR_COLS = ["A1", "A2", "A4", "A5", "A6", "A7", "A14"]


class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    """Adds binary indicator columns for each column with missingness."""

    def __init__(self, cols=None):
        self.cols = cols or MISSING_INDICATOR_COLS

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.cols_ = [c for c in self.cols if c in X.columns]
        else:
            self.cols_ = self.cols
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        for col in self.cols_:
            if col in df.columns:
                df[f"{col}_missing"] = df[col].isna().astype(int)
        return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates derived features:
      - log1p of skewed continuous columns
      - Age × Income proxy interaction (A2 × A3)
      - Savings-to-payment ratio (A15 / (A14 + 1))
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        for col in SKEWED_COLS:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(pd.to_numeric(df[col], errors="coerce").fillna(0))
        if "A2" in df.columns and "A3" in df.columns:
            df["A2_x_A3"] = (pd.to_numeric(df["A2"], errors="coerce").fillna(0) *
                             pd.to_numeric(df["A3"], errors="coerce").fillna(0))
        if "A15" in df.columns and "A14" in df.columns:
            a15 = pd.to_numeric(df["A15"], errors="coerce").fillna(0)
            a14 = pd.to_numeric(df["A14"], errors="coerce").fillna(0)
            df["savings_to_payment"] = a15 / (a14 + 1)
        return df


class BinaryLabelEncoder(BaseEstimator, TransformerMixin):
    """Label-encodes a list of binary categorical columns in-place."""

    def __init__(self, cols=None):
        self.cols = cols or BINARY_CATS

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.encoders_ = {}
        for col in self.cols:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].dropna().astype(str))
                self.encoders_[col] = le
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        for col, le in self.encoders_.items():
            if col in df.columns:
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(
                    lambda v: le.transform([v])[0] if v in known else 0
                )
        return df


def build_preprocessing_pipeline() -> Pipeline:
    """
    Returns a full preprocessing pipeline.
    Input: raw DataFrame (after load_raw_data, without target column).
    Output: numpy array ready for modelling.
    """
    # After MissingIndicatorAdder + FeatureEngineer the column set is dynamic.
    # We use a custom all-in-one transformer that feeds into ColumnTransformer.
    return Pipeline([
        ("missing_indicators", MissingIndicatorAdder()),
        ("feature_engineer", FeatureEngineer()),
        ("encode_binaries", BinaryLabelEncoder()),
        ("col_transform", _build_column_transformer()),
    ])


def _build_column_transformer() -> ColumnTransformer:
    """
    ColumnTransformer that handles:
      - Categorical imputation + OHE for multi-value cats
      - Numeric imputation + scaling for continuous features
      - Pass-through for already-encoded binary cats + indicator cols
    """
    # Multi-value categoricals → impute mode → OHE
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Continuous → impute median → scale
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    # Engineered continuous features (already numeric; median impute + scale)
    eng_cols = ["A15_log", "A11_log", "A2_x_A3", "savings_to_payment"]
    eng_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    # Binary cats + missing indicators → pass through (already 0/1 int)
    passthrough_cols = BINARY_CATS + [f"{c}_missing" for c in MISSING_INDICATOR_COLS]

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, MULTI_CATS),
            ("num", num_pipe, CONTINUOUS_COLS),
            ("eng", eng_pipe, eng_cols),
            ("pass", "passthrough", passthrough_cols),
        ],
        remainder="drop",
    )


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Extract feature names after transformation (best-effort)."""
    ct: ColumnTransformer = pipeline.named_steps["col_transform"]
    names = []
    for name, trans, cols in ct.transformers_:
        if name == "cat":
            ohe = trans.named_steps["ohe"]
            names.extend(ohe.get_feature_names_out(cols).tolist())
        elif name in ("num", "eng"):
            names.extend(cols if isinstance(cols, list) else list(cols))
        elif name == "pass":
            names.extend(cols if isinstance(cols, list) else list(cols))
    return names
