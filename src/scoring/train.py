"""End-to-end training pipeline for the credit scoring model.

Pipeline executed when this module is run as a script:
1. Load processed data (or run prepare() if missing).
2. Stratified 80/20 train/test split.
3. Build an imblearn Pipeline: ColumnTransformer + SMOTE + classifier.
4. GridSearchCV over a small hyperparameter grid for each of the three
   models (LogReg, RandomForest, XGBoost) optimising the business gain.
5. Optimise the decision threshold on the train predictions of each
   model.
6. Evaluate every model on the held-out test set; log everything to
   MLflow.
7. Persist the winning model + threshold + metadata to `models/`.

Run from the project root:

    .venv/bin/python -m scoring.train

The MLflow tracking URI is read from the `MLFLOW_TRACKING_URI` env var
or falls back to http://localhost:5050.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Make `src` importable when run as a script
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from scoring.business_score import (  # noqa: E402
    DEFAULT_FN_COST,
    DEFAULT_FP_COST,
    business_cost,
    business_gain,
    find_optimal_threshold,
    make_business_scorer,
)
from scoring.data import PROCESSED_PATH, TARGET, prepare, split_x_y  # noqa: E402
from scoring.preprocessing import (  # noqa: E402
    build_preprocessor,
    get_feature_columns,
    get_output_feature_names,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
EXPERIMENT_NAME = "credit-scoring"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "docs" / "figures"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

FN_COST = float(os.getenv("BUSINESS_FN_COST", DEFAULT_FN_COST))
FP_COST = float(os.getenv("BUSINESS_FP_COST", DEFAULT_FP_COST))


# ---------------------------------------------------------------------------
# Model definitions — kept compact for a 1-day project
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str
    estimator: Any
    param_grid: dict[str, list]
    notes: str = ""


def get_model_specs() -> list[ModelSpec]:
    """Return the three models to compare."""
    return [
        ModelSpec(
            name="logreg",
            estimator=LogisticRegression(
                solver="liblinear",
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            param_grid={
                "clf__C": [0.1, 1.0, 10.0],
                "clf__penalty": ["l2"],
            },
            notes="Baseline interpretable",
        ),
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            param_grid={
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [8, 16],
                "clf__min_samples_leaf": [5],
            },
            notes="Robust non-linear baseline",
        ),
        ModelSpec(
            name="xgboost",
            estimator=XGBClassifier(
                eval_metric="logloss",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                tree_method="hist",
            ),
            param_grid={
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [3, 6],
                "clf__learning_rate": [0.1],
            },
            notes="State-of-the-art tabular",
        ),
    ]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load the processed dataset (regenerate from raw if missing)."""
    if PROCESSED_PATH.exists():
        print(f"Loading processed dataset from {PROCESSED_PATH.relative_to(ROOT)}")
        return pd.read_parquet(PROCESSED_PATH)
    print("Processed dataset missing — running prepare() from raw")
    df = prepare()
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split."""
    X, y = split_x_y(df)
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# Pipeline assembly
# ---------------------------------------------------------------------------
def build_pipeline(estimator: Any, numeric_cols: list[str], nominal_cols: list[str]) -> ImbPipeline:
    """Assemble the full imblearn Pipeline: preprocessor -> SMOTE -> clf.

    Using `imblearn.pipeline.Pipeline` is critical: it ensures SMOTE
    is applied only during fit (per fold during CV), never at predict time.
    """
    preprocessor = build_preprocessor(numeric_cols, nominal_cols)
    return ImbPipeline(
        steps=[
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("clf", estimator),
        ]
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    name: str
    best_params: dict
    cv_business_gain: float
    cv_business_gain_std: float
    threshold: float
    test_metrics: dict
    fitted_pipeline: ImbPipeline | None = None
    cv_results: pd.DataFrame | None = None
    artifact_files: list[Path] = field(default_factory=list)


def compute_test_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    return {
        "test_business_gain": business_gain(y_true, y_pred, FN_COST, FP_COST),
        "test_business_cost": business_cost(y_true, y_pred, FN_COST, FP_COST),
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision": precision_score(y_true, y_pred, zero_division=0),
        "test_recall": recall_score(y_true, y_pred, zero_division=0),
        "test_f1": f1_score(y_true, y_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_true, y_proba),
    }


def save_confusion_matrix(
    name: str, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["non-défaut", "défaut"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Matrice de confusion — {name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(name: str, y_true: np.ndarray, y_proba: np.ndarray, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — {name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numeric_cols: list[str],
    nominal_cols: list[str],
    scorer,
    cv,
) -> ModelResult:
    """Run a full GridSearchCV + test evaluation for one model spec."""
    print(f"\n── Training {spec.name} ─────────────────────────")
    pipe = build_pipeline(spec.estimator, numeric_cols, nominal_cols)

    grid = GridSearchCV(
        pipe,
        spec.param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
        return_train_score=False,
    )

    grid.fit(X_train, y_train)

    best_idx = grid.best_index_
    cv_results = pd.DataFrame(grid.cv_results_)
    cv_std = float(cv_results["std_test_score"].iloc[best_idx])

    # Find optimal threshold on train predictions
    y_train_proba = grid.predict_proba(X_train)[:, 1]
    threshold_res = find_optimal_threshold(
        y_train.to_numpy(),
        y_train_proba,
        fn_cost=FN_COST,
        fp_cost=FP_COST,
    )

    # Evaluate on test
    y_test_proba = grid.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold_res.best_threshold).astype(int)
    test_metrics = compute_test_metrics(y_test.to_numpy(), y_test_pred, y_test_proba)

    # Persist artifacts
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = FIG_DIR / f"20_cm_{spec.name}.png"
    roc_path = FIG_DIR / f"20_roc_{spec.name}.png"
    save_confusion_matrix(spec.name, y_test.to_numpy(), y_test_pred, cm_path)
    save_roc_curve(spec.name, y_test.to_numpy(), y_test_proba, roc_path)

    print(
        f"  cv_gain={grid.best_score_:.3f} ± {cv_std:.3f}  "
        f"thr={threshold_res.best_threshold:.2f}  "
        f"test_gain={test_metrics['test_business_gain']:.3f}  "
        f"AUC={test_metrics['test_roc_auc']:.3f}"
    )

    return ModelResult(
        name=spec.name,
        best_params=grid.best_params_,
        cv_business_gain=float(grid.best_score_),
        cv_business_gain_std=cv_std,
        threshold=threshold_res.best_threshold,
        test_metrics=test_metrics,
        fitted_pipeline=grid.best_estimator_,
        cv_results=cv_results,
        artifact_files=[cm_path, roc_path],
    )


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------
def log_to_mlflow(result: ModelResult, spec: ModelSpec) -> str:
    """Log a model result to MLflow and return the run_id."""
    with mlflow.start_run(run_name=result.name) as run:
        mlflow.log_params({**result.best_params, "model_family": result.name})
        mlflow.log_params({
            "fn_cost": FN_COST,
            "fp_cost": FP_COST,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
        })
        mlflow.log_metric("cv_business_gain", result.cv_business_gain)
        mlflow.log_metric("cv_business_gain_std", result.cv_business_gain_std)
        mlflow.log_metric("optimal_threshold", result.threshold)
        for k, v in result.test_metrics.items():
            mlflow.log_metric(k, float(v))
        for path in result.artifact_files:
            mlflow.log_artifact(str(path), artifact_path="plots")
        if result.fitted_pipeline is not None:
            mlflow.sklearn.log_model(result.fitted_pipeline, "model")
        return run.info.run_id


# ---------------------------------------------------------------------------
# Best model persistence
# ---------------------------------------------------------------------------
def save_best_model(
    result: ModelResult,
    feature_columns: list[str],
    numeric_cols: list[str],
    nominal_cols: list[str],
) -> None:
    """Persist the winning model + metadata for the API."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "best_model.joblib"
    joblib.dump(result.fitted_pipeline, model_path)

    metadata = {
        "model_name": result.name,
        "best_params": result.best_params,
        "threshold": result.threshold,
        "cv_business_gain": result.cv_business_gain,
        "test_metrics": result.test_metrics,
        "fn_cost": FN_COST,
        "fp_cost": FP_COST,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_columns": feature_columns,
        "numeric_columns": numeric_cols,
        "nominal_columns": nominal_cols,
    }
    (MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, default=float))
    print(f"\n✓ Best model saved to {model_path.relative_to(ROOT)}")
    print(f"✓ Metadata saved to {(MODELS_DIR / 'metadata.json').relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> dict:
    """End-to-end training. Returns a summary dict."""
    print(f"MLflow tracking URI : {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data()
    print(f"Dataset : {df.shape}")

    numeric_cols, nominal_cols = get_feature_columns(df, target=TARGET)
    feature_columns = numeric_cols + nominal_cols
    print(f"Numeric  : {len(numeric_cols)} cols")
    print(f"Nominal  : {len(nominal_cols)} cols ({nominal_cols})")

    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train    : {X_train.shape}  (default rate {y_train.mean():.3f})")
    print(f"Test     : {X_test.shape}   (default rate {y_test.mean():.3f})")

    scorer = make_business_scorer(fn_cost=FN_COST, fp_cost=FP_COST)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results: list[ModelResult] = []
    run_ids: dict[str, str] = {}
    for spec in get_model_specs():
        result = train_one_model(
            spec=spec,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            numeric_cols=numeric_cols,
            nominal_cols=nominal_cols,
            scorer=scorer,
            cv=cv,
        )
        run_ids[result.name] = log_to_mlflow(result, spec)
        results.append(result)

    # Select winner on the business gain on the TEST set
    # (CV is the deciding factor for hyperparam tuning, but at the end we
    # compare families on what matters: test set business gain).
    winner = max(results, key=lambda r: r.test_metrics["test_business_gain"])
    print(f"\n🏆 Winner : {winner.name} "
          f"(test_business_gain={winner.test_metrics['test_business_gain']:.3f})")

    save_best_model(winner, feature_columns, numeric_cols, nominal_cols)

    return {
        "winner": winner.name,
        "winner_metrics": winner.test_metrics,
        "all_results": {r.name: r.test_metrics for r in results},
        "mlflow_runs": run_ids,
    }


if __name__ == "__main__":
    summary = main()
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, default=float))
