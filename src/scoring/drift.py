"""Data drift monitoring using Evidently AI.

Compares a *reference* dataset (the training data) against *current* data
(new production samples) and generates a Data Drift Report:
- Per-feature statistical test (Wasserstein for numeric, chi² for categorical)
- Aggregated drift score
- HTML report saved to docs/drift_report.html

Run as a script:

    .venv/bin/python -m scoring.drift

The script simulates "current" data by injecting noise + a distribution
shift into the test split, then runs the drift detection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scoring.data import PROCESSED_PATH, TARGET  # noqa: E402

RANDOM_STATE = 42
REPORT_PATH = ROOT / "docs" / "drift_report.html"
REPORT_JSON_PATH = ROOT / "docs" / "drift_report.json"


def simulate_drifted_data(df_test: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Inject a realistic distribution shift into the test data.

    Three changes that we'd plausibly see in production:
    - Customer base ages a bit (mean +3 years).
    - Credit limits trend upward (mean +20%).
    - Payment status worsens (more delays on PAY_0).
    """
    rng = np.random.default_rng(seed)
    df = df_test.copy()

    # Aging
    df["AGE"] = (df["AGE"] + 3 + rng.normal(0, 1, len(df))).round().astype(int).clip(18, 90)

    # Higher credit limits
    multiplier = np.clip(1.20 + rng.normal(0, 0.05, len(df)), 0.5, None)
    df["LIMIT_BAL"] = df["LIMIT_BAL"] * multiplier

    # More delays
    df["PAY_0"] = (df["PAY_0"] + rng.choice([0, 0, 1, 1, 2], len(df))).clip(-2, 8)

    return df


def build_reference_and_current() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the processed dataset into a reference and a (synthetically drifted) current.

    Reference = the train half of an 80/20 stratified split.
    Current = drifted version of the test half.
    """
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_PATH} not found. Run the EDA notebook or `python -m scoring.train` first."
        )
    df = pd.read_parquet(PROCESSED_PATH)
    df_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    current = simulate_drifted_data(df_test)
    return df_train, current


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Generate Evidently DataDrift + TargetDrift report and save it as HTML."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    report.save_html(str(REPORT_PATH))
    print(f"✓ HTML report saved to {REPORT_PATH.relative_to(ROOT)}")

    # Also save a JSON dump of the metrics for programmatic consumption
    payload = report.as_dict()
    REPORT_JSON_PATH.write_text(json.dumps(payload, default=str, indent=2))
    print(f"✓ JSON metrics saved to {REPORT_JSON_PATH.relative_to(ROOT)}")

    return payload


def summarise(payload: dict) -> None:
    """Print a short summary on stdout."""
    for metric in payload.get("metrics", []):
        name = metric.get("metric")
        result = metric.get("result", {})
        if name == "DatasetDriftMetric":
            print()
            print("=== Dataset Drift ===")
            print(f"  Dataset drift detected : {result.get('dataset_drift')}")
            print(f"  Drift share            : {result.get('drift_share', 0):.2%}")
            print(f"  Drifted columns count  : {result.get('number_of_drifted_columns')}")
        if name == "ColumnDriftMetric":
            col = result.get("column_name")
            drift = result.get("drift_detected")
            score = result.get("drift_score")
            if drift:
                print(f"  → DRIFT on {col}  score={score:.3f}")


def main() -> dict:
    reference, current = build_reference_and_current()
    print(f"Reference shape : {reference.shape}")
    print(f"Current shape   : {current.shape}")
    payload = run_drift_report(reference, current)
    summarise(payload)
    return payload


if __name__ == "__main__":
    main()
