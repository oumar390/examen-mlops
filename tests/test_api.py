from fastapi.testclient import TestClient

from scoring_app import api


class DummyModel:
    def predict_proba(self, frame):
        return [[0.2, 0.8 if frame.iloc[0]["feature_a"] > 0 else 0.1]]


def test_predict_returns_probability(monkeypatch):
    monkeypatch.setattr(
        api,
        "get_model_bundle",
        lambda: {"model": DummyModel(), "feature_names": ["feature_a"]},
    )
    client = TestClient(api.app)

    response = client.post("/predict", json={"features": {"feature_a": 1.0}})

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert body["probability_high_risk"] == 0.8


def test_features_endpoint_returns_model_columns(monkeypatch):
    monkeypatch.setattr(
        api,
        "get_model_bundle",
        lambda: {"model": DummyModel(), "feature_names": ["feature_a"]},
    )
    client = TestClient(api.app)

    response = client.get("/features")

    assert response.status_code == 200
    assert response.json() == {"features": ["feature_a"]}


def test_predict_rejects_missing_features(monkeypatch):
    monkeypatch.setattr(
        api,
        "get_model_bundle",
        lambda: {"model": DummyModel(), "feature_names": ["feature_a", "feature_b"]},
    )
    client = TestClient(api.app)

    response = client.post("/predict", json={"features": {"feature_a": 1.0}})

    assert response.status_code == 422
    assert response.json()["detail"]["missing_features"] == ["feature_b"]
