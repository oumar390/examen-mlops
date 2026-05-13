from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from scoring_app.config import MODEL_PATH
from scoring_app.data import example_payload
from scoring_app.model_io import load_model


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Feature values keyed by training column name",
    )


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability_high_risk: float
    threshold: float


app = FastAPI(title="MLOps Scoring API", version="0.1.0")


@lru_cache(maxsize=1)
def get_model_bundle() -> dict[str, Any]:
    return load_model(MODEL_PATH)


def make_frame(features: dict[str, float], feature_names: list[str]) -> pd.DataFrame:
    missing = sorted(set(feature_names) - set(features))
    extra = sorted(set(features) - set(feature_names))
    if missing or extra:
        detail = {"missing_features": missing, "extra_features": extra}
        raise HTTPException(status_code=422, detail=detail)
    return pd.DataFrame([{name: float(features[name]) for name in feature_names}])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/features")
def features() -> dict[str, list[str]]:
    bundle = get_model_bundle()
    return {"features": bundle["feature_names"]}


@app.get("/example")
def example() -> dict[str, dict[str, float]]:
    return {"features": example_payload()}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    bundle = get_model_bundle()
    frame = make_frame(request.features, bundle["feature_names"])
    probability = float(bundle["model"].predict_proba(frame)[0][1])
    threshold = 0.5
    prediction = int(probability >= threshold)
    return PredictionResponse(
        prediction=prediction,
        label="high_risk_malignant" if prediction else "low_risk_benign",
        probability_high_risk=probability,
        threshold=threshold,
    )
