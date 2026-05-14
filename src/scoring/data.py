"""Data loading, cleaning and feature engineering for UCI Credit Card Default.

Used by both the EDA notebook and the training pipeline so cleaning rules
stay consistent between exploration and production.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "credit_clean.parquet"

# ---------------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------------
TARGET = "default"  # renamed from "default.payment.next.month"

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

DEMOGRAPHIC_COLS = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]

# ---------------------------------------------------------------------------
# Categorical decoding maps (for human-readable EDA)
# ---------------------------------------------------------------------------
SEX_LABELS = {1: "male", 2: "female"}

EDUCATION_LABELS = {
    1: "graduate_school",
    2: "university",
    3: "high_school",
    4: "others",
}

MARRIAGE_LABELS = {1: "married", 2: "single", 3: "others"}

PAY_STATUS_LABELS = {
    -2: "no_use",
    -1: "paid_duly",
    0: "revolving",
    1: "delay_1m",
    2: "delay_2m",
    3: "delay_3m",
    4: "delay_4m",
    5: "delay_5m",
    6: "delay_6m",
    7: "delay_7m",
    8: "delay_8m_plus",
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw CSV exactly as published by UCI."""
    path = Path(path) if path else RAW_PATH
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataset.

    Operations
    ----------
    - Drop the `ID` column (no predictive value).
    - Rename the target from `default.payment.next.month` to `default`.
    - Map undocumented EDUCATION codes (0, 5, 6) to 4 = "others".
    - Map undocumented MARRIAGE code (0) to 3 = "others".

    The function returns a new DataFrame; the input is left untouched.
    """
    df = df.copy()

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    df = df.rename(columns={"default.payment.next.month": TARGET})

    df.loc[~df["EDUCATION"].isin([1, 2, 3, 4]), "EDUCATION"] = 4
    df.loc[~df["MARRIAGE"].isin([1, 2, 3]), "MARRIAGE"] = 3

    return df


def decode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Replace numeric codes with human-readable labels (EDA only)."""
    df = df.copy()
    df["SEX"] = df["SEX"].map(SEX_LABELS)
    df["EDUCATION"] = df["EDUCATION"].map(EDUCATION_LABELS)
    df["MARRIAGE"] = df["MARRIAGE"].map(MARRIAGE_LABELS)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features rooted in credit-scoring domain knowledge.

    Adds the following columns:
    - PAY_DELAY_COUNT  : number of months with PAY status >= 1
    - MAX_DELAY        : worst delay observed across 6 months
    - MEAN_PAY_STATUS  : average PAY status (signals chronic behaviour)
    - HAS_EVER_DELAYED : binary flag, any delay in last 6 months
    - UTIL_RATIO_1     : BILL_AMT1 / LIMIT_BAL (latest utilisation)
    - MEAN_UTIL        : mean utilisation over 6 months
    - MAX_UTIL         : worst-month utilisation
    - TOTAL_PAID       : sum of PAY_AMT over 6 months
    - TOTAL_BILLED     : sum of BILL_AMT over 6 months
    - PAY_TO_BILL_RATIO: TOTAL_PAID / TOTAL_BILLED (capacity to repay)
    - BILL_TREND       : (BILL_AMT1 - BILL_AMT6) / 6 (debt slope)
    - PAY_TREND        : (PAY_AMT1 - PAY_AMT6) / 6 (payment slope)
    """
    df = df.copy()

    # --- Payment behaviour ---
    pay_block = df[PAY_COLS]
    df["PAY_DELAY_COUNT"] = (pay_block >= 1).sum(axis=1).astype(int)
    df["MAX_DELAY"] = pay_block.max(axis=1)
    df["MEAN_PAY_STATUS"] = pay_block.mean(axis=1)
    df["HAS_EVER_DELAYED"] = (df["PAY_DELAY_COUNT"] > 0).astype(int)

    # --- Credit utilisation ---
    limit = df["LIMIT_BAL"].replace(0, np.nan)
    df["UTIL_RATIO_1"] = df["BILL_AMT1"] / limit
    util_block = df[BILL_COLS].div(limit, axis=0)
    df["MEAN_UTIL"] = util_block.mean(axis=1)
    df["MAX_UTIL"] = util_block.max(axis=1)

    # --- Repayment capacity (totals over 6 months) ---
    df["TOTAL_PAID"] = df[PAY_AMT_COLS].sum(axis=1)
    df["TOTAL_BILLED"] = df[BILL_COLS].sum(axis=1)
    safe_billed = df["TOTAL_BILLED"].where(df["TOTAL_BILLED"] > 0, np.nan)
    df["PAY_TO_BILL_RATIO"] = df["TOTAL_PAID"] / safe_billed

    # --- Trends across the 6-month window ---
    df["BILL_TREND"] = (df["BILL_AMT1"] - df["BILL_AMT6"]) / 6
    df["PAY_TREND"] = (df["PAY_AMT1"] - df["PAY_AMT6"]) / 6

    # Replace NaN (from divisions) with 0 — those rows had no credit usage.
    new_cols = [
        "UTIL_RATIO_1",
        "MEAN_UTIL",
        "MAX_UTIL",
        "PAY_TO_BILL_RATIO",
    ]
    df[new_cols] = df[new_cols].fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# Convenience pipeline
# ---------------------------------------------------------------------------
def prepare(path: Path | str | None = None) -> pd.DataFrame:
    """Full pipeline: load raw → clean → engineer features."""
    return engineer_features(clean(load_raw(path)))


def split_x_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) from target (y)."""
    return df.drop(columns=[TARGET]), df[TARGET]
