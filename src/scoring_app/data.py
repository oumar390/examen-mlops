from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Dataset:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]


def load_scoring_dataset(test_size: float = 0.2, random_state: int = 42) -> Dataset:
    """Load a reproducible binary scoring dataset.

    The sklearn breast cancer dataset is used as an offline substitute for a Kaggle
    binary-classification kernel. The target is inverted so that 1 means a high-risk
    malignant case, which makes false negatives the most expensive business error.
    """

    raw = load_breast_cancer(as_frame=True)
    features = raw.data.rename(columns=lambda name: name.replace(" ", "_"))
    target = (raw.target == 0).astype(int)
    target.name = "is_malignant"

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    return Dataset(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(features.columns),
    )


def example_payload() -> dict[str, float]:
    dataset = load_scoring_dataset()
    return dataset.x_test.iloc[0].astype(float).to_dict()
