from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scoring_app.config import (
    ENABLE_MLFLOW,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    REFERENCE_PATH,
)
from scoring_app.data import Dataset, load_scoring_dataset
from scoring_app.model_io import save_model, save_reference_profile
from scoring_app.scoring import business_score, business_scorer


@dataclass(frozen=True)
class TrainingResult:
    model_name: str
    metrics: dict[str, float]
    feature_importance: pd.DataFrame


def candidate_models() -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    return {
        "logistic_regression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42),
                    ),
                ]
            ),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            Pipeline(
                [
                    (
                        "model",
                        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1),
                    )
                ]
            ),
            {
                "model__n_estimators": [100, 250],
                "model__max_depth": [None, 6, 10],
                "model__min_samples_leaf": [1, 3],
            },
        ),
    }


def evaluate_model(model: Pipeline, dataset: Dataset) -> dict[str, float]:
    predictions = model.predict(dataset.x_test)
    probabilities = model.predict_proba(dataset.x_test)[:, 1]
    return {
        "accuracy": accuracy_score(dataset.y_test, predictions),
        "recall_high_risk": recall_score(dataset.y_test, predictions),
        "f1": f1_score(dataset.y_test, predictions),
        "roc_auc": roc_auc_score(dataset.y_test, probabilities),
        "business_score": business_score(dataset.y_test.to_numpy(), predictions),
    }


def compute_feature_importance(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        values = np.zeros(len(feature_names))

    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def log_with_mlflow(result: TrainingResult, params: dict[str, object], model: Pipeline) -> None:
    if not ENABLE_MLFLOW:
        return

    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        return

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=result.model_name):
            mlflow.log_params(params)
            mlflow.log_metrics(result.metrics)
            importance_path = "reports/feature_importance.csv"
            result.feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
            mlflow.sklearn.log_model(model, artifact_path="model")
    except Exception as exc:
        print(f"MLflow logging skipped: {exc}")


def train_best_model() -> TrainingResult:
    dataset = load_scoring_dataset()
    best_search: GridSearchCV | None = None
    best_name = ""
    best_metrics: dict[str, float] = {}

    for name, (pipeline, grid) in candidate_models().items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring={
                "business_score": business_scorer,
                "roc_auc": "roc_auc",
                "recall": "recall",
            },
            refit="business_score",
            cv=5,
            n_jobs=1,
        )
        search.fit(dataset.x_train, dataset.y_train)
        metrics = evaluate_model(search.best_estimator_, dataset)

        feature_importance = compute_feature_importance(
            search.best_estimator_,
            dataset.feature_names,
        )
        result = TrainingResult(name, metrics, feature_importance)
        log_with_mlflow(result, search.best_params_, search.best_estimator_)

        if best_search is None or metrics["business_score"] > best_metrics["business_score"]:
            best_search = search
            best_name = name
            best_metrics = metrics

    if best_search is None:
        raise RuntimeError("No model was trained.")

    best_model = best_search.best_estimator_
    feature_importance = compute_feature_importance(best_model, dataset.feature_names)
    bundle = {
        "model": best_model,
        "feature_names": dataset.feature_names,
        "target_name": "is_malignant",
        "positive_label": "high_risk_malignant",
        "metrics": best_metrics,
        "business_context": {
            "false_positive_cost": 1.0,
            "false_negative_cost": 5.0,
            "reason": "A missed malignant case is much more costly than a false alarm.",
        },
    }
    save_model(bundle, MODEL_PATH)
    save_reference_profile(dataset.x_train, REFERENCE_PATH)
    feature_importance.to_csv("reports/feature_importance.csv", index=False)
    print(json.dumps({"best_model": best_name, "metrics": best_metrics}, indent=2))
    return TrainingResult(best_name, best_metrics, feature_importance)


if __name__ == "__main__":
    train_best_model()
