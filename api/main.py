"""FastAPI application — credit default scoring service.

Endpoints
---------
- GET  /            : service info
- GET  /health      : health probe (used by Docker / Render)
- GET  /model/info  : model metadata (name, threshold, metrics)
- POST /predict     : single prediction
- POST /predict/batch : batch prediction (up to 1000 records)

The model is loaded once at startup and shared across requests.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Make src/ importable so we can reuse data.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scoring.data import clean, engineer_features  # noqa: E402

from .schemas import (  # noqa: E402
    BatchRequest,
    BatchResponse,
    CreditApplication,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ROOT / "models" / "best_model.joblib")))
METADATA_PATH = Path(
    os.getenv("METADATA_PATH", str(ROOT / "models" / "metadata.json"))
)


# ---------------------------------------------------------------------------
# Model loading (lifespan)
# ---------------------------------------------------------------------------
class ModelState:
    """Shared model state — populated at startup."""

    model: Any = None
    metadata: dict | None = None


state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the app boots."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Run `python -m scoring.train` to train and persist a model."
        )
    state.model = joblib.load(MODEL_PATH)
    state.metadata = json.loads(METADATA_PATH.read_text())
    print(f"✓ Model loaded: {state.metadata.get('model_name')} from {MODEL_PATH}")
    print(f"✓ Decision threshold: {state.metadata.get('threshold')}")
    yield
    # No special teardown
    state.model = None
    state.metadata = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Credit Scoring API",
    description=(
        "Predict probability of credit default for a client. "
        "Trained on UCI Credit Card Default (Taiwan, 2005)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Permissive CORS — useful for Streamlit and local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_loaded() -> None:
    if state.model is None or state.metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )


def _risk_level(proba: float) -> str:
    if proba < 0.20:
        return "low"
    if proba < 0.40:
        return "medium"
    if proba < 0.65:
        return "high"
    return "very_high"


def _predict_one(application: CreditApplication) -> PredictionResponse:
    """Run the full prep pipeline + model on a single application."""
    _ensure_loaded()
    raw = pd.DataFrame([application.model_dump()])
    # Re-use the exact same cleaning + FE rules as training (no ID or target
    # columns here — clean() handles their absence gracefully).
    cleaned = clean(raw)
    enriched = engineer_features(cleaned)

    proba = float(state.model.predict_proba(enriched)[:, 1][0])
    threshold = float(state.metadata["threshold"])
    prediction = int(proba >= threshold)
    label = "reject" if prediction == 1 else "approve"

    return PredictionResponse(
        probability_default=proba,
        threshold=threshold,
        prediction=prediction,
        label=label,
        risk_level=_risk_level(proba),
        model_name=state.metadata["model_name"],
        model_trained_at=state.metadata.get("trained_at"),
        timestamp=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "service": "credit-scoring-api",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["meta"], response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy" if state.model is not None else "unhealthy",
        model_loaded=state.model is not None,
        timestamp=datetime.now(timezone.utc),
    )


@app.get("/model/info", tags=["meta"], response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    _ensure_loaded()
    meta = state.metadata
    return ModelInfoResponse(
        model_name=meta["model_name"],
        threshold=meta["threshold"],
        best_params=meta.get("best_params", {}),
        test_metrics=meta.get("test_metrics", {}),
        trained_at=meta.get("trained_at", ""),
        fn_cost=meta.get("fn_cost", 5.0),
        fp_cost=meta.get("fp_cost", 1.0),
        feature_columns=meta.get("feature_columns", []),
    )


@app.post("/predict", tags=["prediction"], response_model=PredictionResponse)
def predict(application: CreditApplication) -> PredictionResponse:
    """Predict default probability for a single client."""
    return _predict_one(application)


@app.post("/predict/batch", tags=["prediction"], response_model=BatchResponse)
def predict_batch(request: BatchRequest) -> BatchResponse:
    """Predict default probability for up to 1000 clients in one call."""
    preds = [_predict_one(app) for app in request.applications]
    return BatchResponse(predictions=preds, count=len(preds))
