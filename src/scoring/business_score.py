"""Asymmetric business score for credit default prediction.

In credit scoring, a False Negative (predicting "no default" for a client
who will actually default) is far more costly than a False Positive
(refusing credit to a creditworthy client). Standard banking practice
(Basel II/III) estimates this ratio at roughly 5:1.

This module provides:
- `business_cost`      : total weighted misclassification cost (lower is better)
- `business_gain`      : normalised score in [0, 1]   (higher is better)
- `make_business_scorer`: sklearn-compatible scorer for GridSearchCV
- `find_optimal_threshold`: probability threshold that minimises cost

Default costs (FN=5, FP=1) are overridable so the same code works for any
business context with different misclassification economics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer

# ---------------------------------------------------------------------------
# Default cost ratio — credit scoring (Basel II/III convention)
# ---------------------------------------------------------------------------
DEFAULT_FN_COST: float = 5.0
DEFAULT_FP_COST: float = 1.0

ArrayLike = Sequence[int] | np.ndarray


def business_cost(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
) -> float:
    """Total weighted misclassification cost (lower is better).

    Parameters
    ----------
    y_true : True binary labels (0 = non-default, 1 = default).
    y_pred : Predicted binary labels.
    fn_cost : Cost of a false negative (default 5.0).
    fp_cost : Cost of a false positive (default 1.0).

    Returns
    -------
    Total cost = FP * fp_cost + FN * fn_cost.
    A perfect classifier returns 0.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp * fp_cost + fn * fn_cost)


def business_gain(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
) -> float:
    """Normalised business score in [0, 1] where 1 is perfect.

    The denominator is the worst plausible cost: predicting every sample
    with the most damaging single class. For asymmetric costs that means
    either "always predict 0" (all positives become FN) or "always
    predict 1" (all negatives become FP), whichever is more expensive.
    """
    y_true_arr = np.asarray(y_true)
    actual_cost = business_cost(y_true_arr, y_pred, fn_cost, fp_cost)

    n_positives = int((y_true_arr == 1).sum())
    n_negatives = int((y_true_arr == 0).sum())

    worst_always_neg = n_positives * fn_cost
    worst_always_pos = n_negatives * fp_cost
    worst_cost = max(worst_always_neg, worst_always_pos)

    if worst_cost == 0:
        return 1.0
    return 1.0 - actual_cost / worst_cost


def make_business_scorer(
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
):
    """Return an sklearn-compatible scorer (higher is better).

    Usage:
        scorer = make_business_scorer(fn_cost=5, fp_cost=1)
        grid = GridSearchCV(model, param_grid, scoring=scorer)
    """
    return make_scorer(
        business_gain,
        greater_is_better=True,
        fn_cost=fn_cost,
        fp_cost=fp_cost,
    )


@dataclass
class ThresholdResult:
    """Result of a threshold optimisation."""

    best_threshold: float
    best_cost: float
    best_gain: float
    all_results: list[dict]


def find_optimal_threshold(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
    thresholds: np.ndarray | None = None,
) -> ThresholdResult:
    """Find the probability threshold that minimises business cost.

    The default classifier threshold (0.5) optimises accuracy. For
    asymmetric costs we usually want a *lower* threshold so the model
    flags more positives — even at the price of more FP — because each
    FN avoided is worth 5 FP avoided.

    Parameters
    ----------
    y_true     : True labels.
    y_proba    : Predicted probability of class 1.
    fn_cost, fp_cost : Cost weights.
    thresholds : Grid of thresholds to test (default: 0.05 → 0.95 by 0.01).

    Returns
    -------
    ThresholdResult dataclass with best_threshold, best_cost, best_gain
    and the full grid of (threshold, cost, gain).
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    y_proba_arr = np.asarray(y_proba)
    results: list[dict] = []
    for t in thresholds:
        y_pred = (y_proba_arr >= t).astype(int)
        results.append(
            {
                "threshold": float(t),
                "cost": business_cost(y_true, y_pred, fn_cost, fp_cost),
                "gain": business_gain(y_true, y_pred, fn_cost, fp_cost),
            }
        )

    best = min(results, key=lambda r: r["cost"])
    return ThresholdResult(
        best_threshold=best["threshold"],
        best_cost=best["cost"],
        best_gain=best["gain"],
        all_results=results,
    )
