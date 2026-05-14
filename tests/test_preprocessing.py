"""Tests for the preprocessing module."""
import numpy as np
import pandas as pd
import pytest

from scoring.preprocessing import (
    NOMINAL_COLS,
    build_preprocessor,
    get_feature_columns,
    get_output_feature_names,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "LIMIT_BAL": [10000.0, 50000.0, 30000.0],
            "AGE": [25, 40, 35],
            "SEX": [1, 2, 1],
            "EDUCATION": [1, 2, 3],
            "MARRIAGE": [1, 2, 3],
            "default": [0, 1, 0],
        }
    )


class TestGetFeatureColumns:
    def test_separates_numeric_from_nominal(self, sample_df):
        numeric, nominal = get_feature_columns(sample_df)
        assert set(nominal) == set(NOMINAL_COLS)
        assert "LIMIT_BAL" in numeric and "AGE" in numeric
        assert "default" not in numeric and "default" not in nominal

    def test_skips_missing_nominal_cols(self):
        df = pd.DataFrame({"LIMIT_BAL": [1.0], "default": [0]})
        numeric, nominal = get_feature_columns(df)
        assert nominal == []
        assert numeric == ["LIMIT_BAL"]


class TestBuildPreprocessor:
    def test_fit_transform_produces_dense_array(self, sample_df):
        numeric, nominal = get_feature_columns(sample_df)
        prep = build_preprocessor(numeric, nominal)
        X = sample_df.drop(columns=["default"])
        out = prep.fit_transform(X)
        assert isinstance(out, np.ndarray)
        assert out.shape[0] == 3
        # Should have more columns after OHE expansion
        assert out.shape[1] >= len(numeric) + len(nominal)

    def test_handles_unknown_categories(self, sample_df):
        numeric, nominal = get_feature_columns(sample_df)
        prep = build_preprocessor(numeric, nominal)
        X = sample_df.drop(columns=["default"])
        prep.fit(X)
        # Unknown EDUCATION code should not break
        new_row = X.iloc[[0]].copy()
        new_row["EDUCATION"] = 99
        out = prep.transform(new_row)
        assert out.shape[0] == 1


class TestGetOutputFeatureNames:
    def test_names_match_array_columns(self, sample_df):
        numeric, nominal = get_feature_columns(sample_df)
        prep = build_preprocessor(numeric, nominal)
        X = sample_df.drop(columns=["default"])
        out = prep.fit_transform(X)
        names = get_output_feature_names(prep, numeric, nominal)
        assert len(names) == out.shape[1]
