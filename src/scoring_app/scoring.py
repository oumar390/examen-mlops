from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer

FALSE_POSITIVE_COST = 1.0
FALSE_NEGATIVE_COST = 5.0


def business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    false_positive_cost: float = FALSE_POSITIVE_COST,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
) -> float:
    """Return the weighted error cost for a high-risk binary classifier."""

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp * false_positive_cost + fn * false_negative_cost)


def business_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    false_positive_cost: float = FALSE_POSITIVE_COST,
    false_negative_cost: float = FALSE_NEGATIVE_COST,
) -> float:
    """Return a normalized score where 1 is perfect and 0 is as bad as always wrong."""

    max_cost = len(y_true) * max(false_positive_cost, false_negative_cost)
    if max_cost == 0:
        return 1.0
    cost = business_cost(y_true, y_pred, false_positive_cost, false_negative_cost)
    return 1.0 - (cost / max_cost)


business_scorer = make_scorer(business_score, greater_is_better=True)
