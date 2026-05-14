"""Pydantic schemas — strongly-typed request/response payloads.

These are the only data shapes accepted at the API boundary.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Silence pydantic warnings about fields starting with "model_" (protected namespace).
_NO_PROTECTED = ConfigDict(protected_namespaces=())


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
class CreditApplication(BaseModel):
    """One client's data — exactly the 23 raw input features.

    Engineered features (PAY_DELAY_COUNT, MEAN_UTIL, …) are computed
    server-side from these raw values, so clients never need to know about them.
    """

    LIMIT_BAL: float = Field(..., ge=0, description="Credit limit (NT$)")
    SEX: Literal[1, 2] = Field(..., description="1 = male, 2 = female")
    EDUCATION: int = Field(..., ge=0, le=6, description="1-4 standard, 0/5/6 → others")
    MARRIAGE: int = Field(..., ge=0, le=3, description="1=married, 2=single, 3=others, 0→others")
    AGE: int = Field(..., ge=18, le=120)

    # PAY status (most recent → oldest)
    PAY_0: int = Field(..., ge=-2, le=8, description="September status")
    PAY_2: int = Field(..., ge=-2, le=8, description="August status")
    PAY_3: int = Field(..., ge=-2, le=8, description="July status")
    PAY_4: int = Field(..., ge=-2, le=8, description="June status")
    PAY_5: int = Field(..., ge=-2, le=8, description="May status")
    PAY_6: int = Field(..., ge=-2, le=8, description="April status")

    # Bill amounts
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float

    # Payment amounts
    PAY_AMT1: float = Field(..., ge=0)
    PAY_AMT2: float = Field(..., ge=0)
    PAY_AMT3: float = Field(..., ge=0)
    PAY_AMT4: float = Field(..., ge=0)
    PAY_AMT5: float = Field(..., ge=0)
    PAY_AMT6: float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "LIMIT_BAL": 200000.0,
                "SEX": 2,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 35,
                "PAY_0": 0, "PAY_2": 0, "PAY_3": 0,
                "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 50000.0, "BILL_AMT2": 48000.0, "BILL_AMT3": 45000.0,
                "BILL_AMT4": 42000.0, "BILL_AMT5": 40000.0, "BILL_AMT6": 38000.0,
                "PAY_AMT1": 5000.0, "PAY_AMT2": 5000.0, "PAY_AMT3": 5000.0,
                "PAY_AMT4": 5000.0, "PAY_AMT5": 5000.0, "PAY_AMT6": 5000.0,
            }
        }
    }


class BatchRequest(BaseModel):
    """Batch prediction — list of applications."""

    applications: list[CreditApplication] = Field(..., min_length=1, max_length=1000)


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    """Prediction result for one client."""

    model_config = _NO_PROTECTED

    probability_default: float = Field(..., ge=0, le=1, description="P(default = 1)")
    threshold: float = Field(..., ge=0, le=1, description="Decision threshold applied")
    prediction: int = Field(..., description="0 = approve, 1 = reject")
    label: Literal["approve", "reject"]
    risk_level: Literal["low", "medium", "high", "very_high"]
    model_name: str
    model_trained_at: str | None = None
    timestamp: datetime


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


# ---------------------------------------------------------------------------
# Health / Info
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    model_config = _NO_PROTECTED

    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    timestamp: datetime


class ModelInfoResponse(BaseModel):
    model_config = _NO_PROTECTED

    model_name: str
    threshold: float
    best_params: dict
    test_metrics: dict
    trained_at: str
    fn_cost: float
    fp_cost: float
    feature_columns: list[str]
