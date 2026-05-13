from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def save_model(bundle: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run: python -m scoring_app.train")
    return joblib.load(path)


def save_reference_profile(frame: pd.DataFrame, path: Path) -> None:
    profile = {
        "columns": list(frame.columns),
        "summary": {
            column: {
                "mean": float(frame[column].mean()),
                "std": float(frame[column].std()),
                "min": float(frame[column].min()),
                "max": float(frame[column].max()),
            }
            for column in frame.columns
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")


def load_reference_profile(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
