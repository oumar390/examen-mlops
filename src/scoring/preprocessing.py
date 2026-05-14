"""Preprocessing pipeline shared by training and inference.

We build a single ColumnTransformer that:
- StandardScaler-s the numeric variables (essential for LogReg, harmless for RF/XGBoost)
- One-hot encodes the small set of nominal categoricals (SEX, EDUCATION, MARRIAGE)

PAY_0..PAY_6 are treated as numeric (they are ordinal: from -2 to 8).
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Columns that must be one-hot encoded (nominal categoricals).
NOMINAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]


def get_feature_columns(df: pd.DataFrame, target: str = "default") -> tuple[list[str], list[str]]:
    """Split the columns of `df` into (numeric_cols, nominal_cols).

    The function is data-driven: it returns the actual columns present in
    `df`, so it stays valid as new engineered features are added.
    """
    nominal = [c for c in NOMINAL_COLS if c in df.columns]
    numeric = [c for c in df.columns if c != target and c not in nominal]
    return numeric, nominal


def build_preprocessor(
    numeric_cols: Iterable[str],
    nominal_cols: Iterable[str],
) -> ColumnTransformer:
    """Return a ColumnTransformer ready to be used inside an imblearn Pipeline."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(numeric_cols)),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                list(nominal_cols),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_output_feature_names(
    preprocessor: ColumnTransformer,
    numeric_cols: Iterable[str],
    nominal_cols: Iterable[str],
) -> list[str]:
    """Names of the columns produced by `preprocessor` once fit.

    Helper used by SHAP to display readable feature names instead of
    "x0", "x1", ...
    """
    cat_step = preprocessor.named_transformers_.get("cat")
    if cat_step is None:
        cat_names: list[str] = []
    else:
        cat_names = list(cat_step.get_feature_names_out(list(nominal_cols)))
    return list(numeric_cols) + cat_names
