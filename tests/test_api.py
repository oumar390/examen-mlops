"""Tests for the FastAPI credit scoring service."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"

requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="best_model.joblib not present — run `python -m scoring.train` first",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Boot the app within a TestClient context so the lifespan loader runs."""
    from api.main import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Sample applications
# ---------------------------------------------------------------------------
LOW_RISK = {
    "LIMIT_BAL": 500000.0,
    "SEX": 2,
    "EDUCATION": 1,
    "MARRIAGE": 1,
    "AGE": 40,
    "PAY_0": -1, "PAY_2": -1, "PAY_3": -1,
    "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
    "BILL_AMT1": 5000.0, "BILL_AMT2": 4000.0, "BILL_AMT3": 3000.0,
    "BILL_AMT4": 2000.0, "BILL_AMT5": 1000.0, "BILL_AMT6": 0.0,
    "PAY_AMT1": 5000.0, "PAY_AMT2": 4000.0, "PAY_AMT3": 3000.0,
    "PAY_AMT4": 2000.0, "PAY_AMT5": 1000.0, "PAY_AMT6": 0.0,
}

HIGH_RISK = {
    "LIMIT_BAL": 20000.0,
    "SEX": 1,
    "EDUCATION": 3,
    "MARRIAGE": 2,
    "AGE": 22,
    "PAY_0": 4, "PAY_2": 3, "PAY_3": 3,
    "PAY_4": 3, "PAY_5": 2, "PAY_6": 2,
    "BILL_AMT1": 19000.0, "BILL_AMT2": 18500.0, "BILL_AMT3": 18000.0,
    "BILL_AMT4": 17000.0, "BILL_AMT5": 16000.0, "BILL_AMT6": 15000.0,
    "PAY_AMT1": 0.0, "PAY_AMT2": 0.0, "PAY_AMT3": 0.0,
    "PAY_AMT4": 100.0, "PAY_AMT5": 100.0, "PAY_AMT6": 100.0,
}


# ---------------------------------------------------------------------------
# Meta endpoints
# ---------------------------------------------------------------------------
@requires_model
class TestMetaEndpoints:
    def test_root_returns_service_info(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert r.json()["service"] == "credit-scoring-api"

    def test_health_reports_model_loaded(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_model_info_returns_metadata(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200
        info = r.json()
        assert info["model_name"]
        assert 0.0 < info["threshold"] < 1.0
        assert info["fn_cost"] == 5.0
        assert info["fp_cost"] == 1.0
        assert len(info["feature_columns"]) > 0


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------
@requires_model
class TestPredictEndpoint:
    def test_low_risk_application_yields_low_probability(self, client):
        r = client.post("/predict", json=LOW_RISK)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["probability_default"] <= 1.0
        assert body["risk_level"] in ("low", "medium")

    def test_high_risk_application_yields_high_probability(self, client):
        r = client.post("/predict", json=HIGH_RISK)
        assert r.status_code == 200
        body = r.json()
        assert body["probability_default"] > 0.5
        assert body["prediction"] == 1
        assert body["label"] == "reject"

    def test_high_risk_proba_above_low_risk_proba(self, client):
        low = client.post("/predict", json=LOW_RISK).json()
        high = client.post("/predict", json=HIGH_RISK).json()
        assert high["probability_default"] > low["probability_default"]

    def test_invalid_sex_rejected(self, client):
        bad = {**LOW_RISK, "SEX": 3}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_negative_age_rejected(self, client):
        bad = {**LOW_RISK, "AGE": -5}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_missing_field_rejected(self, client):
        bad = {k: v for k, v in LOW_RISK.items() if k != "AGE"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422


@requires_model
class TestBatchEndpoint:
    def test_batch_returns_predictions_for_each_input(self, client):
        r = client.post("/predict/batch", json={"applications": [LOW_RISK, HIGH_RISK]})
        assert r.status_code == 200
        body = r.json()
        assert body["count"] == 2
        assert len(body["predictions"]) == 2

    def test_batch_size_limit_enforced(self, client):
        # max_length is 1000
        r = client.post("/predict/batch", json={"applications": [LOW_RISK] * 1001})
        assert r.status_code == 422
