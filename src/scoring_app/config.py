from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "models" / "best_model.joblib"))
REFERENCE_PATH = Path(os.getenv("REFERENCE_PATH", ROOT_DIR / "models" / "reference_profile.json"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
ENABLE_MLFLOW = os.getenv("ENABLE_MLFLOW", "false").lower() == "true"
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-scoring-exam")
