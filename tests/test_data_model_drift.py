import pandas as pd
import pytest

from scoring_app.data import example_payload, load_scoring_dataset
from scoring_app.drift import analyze_drift, population_stability_index
from scoring_app.model_io import (
    load_model,
    load_reference_profile,
    save_model,
    save_reference_profile,
)


def test_load_scoring_dataset_has_binary_target_and_stable_columns():
    dataset = load_scoring_dataset()

    assert dataset.x_train.shape[1] == len(dataset.feature_names)
    assert set(dataset.y_train.unique()) == {0, 1}
    assert "mean_radius" in dataset.feature_names


def test_example_payload_contains_float_features():
    payload = example_payload()

    assert "mean_radius" in payload
    assert all(isinstance(value, float) for value in payload.values())


def test_model_bundle_roundtrip(tmp_path):
    path = tmp_path / "model.joblib"
    bundle = {"feature_names": ["a"], "metrics": {"business_score": 0.9}}

    save_model(bundle, path)

    assert load_model(path) == bundle


def test_load_model_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")


def test_reference_profile_roundtrip(tmp_path):
    path = tmp_path / "reference.json"
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    save_reference_profile(frame, path)
    profile = load_reference_profile(path)

    assert profile["columns"] == ["a"]
    assert profile["summary"]["a"]["mean"] == 2.0


def test_population_stability_index_detects_shift():
    reference = pd.Series(range(100))
    current = pd.Series(range(50, 150))

    assert population_stability_index(reference, current) > 0


def test_analyze_drift_marks_missing_columns(tmp_path):
    reference_path = tmp_path / "reference.json"
    save_reference_profile(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}), reference_path)

    report = analyze_drift(pd.DataFrame({"a": [1.0, 1.5, 2.0]}), reference_path)

    missing_row = report.loc[report["feature"] == "b"].iloc[0]
    assert missing_row["status"] == "missing"
