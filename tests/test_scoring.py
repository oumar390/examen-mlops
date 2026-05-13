import numpy as np

from scoring_app.scoring import business_cost, business_score


def test_business_cost_penalizes_false_negatives_more_than_false_positives():
    y_true = np.array([1, 1, 0, 0])
    false_negative_case = np.array([0, 1, 0, 0])
    false_positive_case = np.array([1, 1, 1, 0])

    assert business_cost(y_true, false_negative_case) == 5.0
    assert business_cost(y_true, false_positive_case) == 1.0


def test_business_score_is_normalized():
    y_true = np.array([1, 0, 1, 0])

    assert business_score(y_true, y_true) == 1.0
    assert 0.0 <= business_score(y_true, 1 - y_true) <= 1.0
