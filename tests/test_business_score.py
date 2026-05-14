"""Tests for the business score module."""
import numpy as np
import pytest
from sklearn.metrics import make_scorer

from scoring.business_score import (
    DEFAULT_FN_COST,
    DEFAULT_FP_COST,
    business_cost,
    business_gain,
    find_optimal_threshold,
    make_business_scorer,
)


# ---------------------------------------------------------------------------
# business_cost
# ---------------------------------------------------------------------------
class TestBusinessCost:
    def test_perfect_predictions_zero_cost(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert business_cost(y_true, y_pred) == 0.0

    def test_only_fp_uses_fp_cost(self):
        # 1 FP, 0 FN -> cost = 1 * fp_cost
        y_true = np.array([0, 1])
        y_pred = np.array([1, 1])
        assert business_cost(y_true, y_pred, fn_cost=5, fp_cost=1) == 1.0

    def test_only_fn_uses_fn_cost(self):
        # 0 FP, 1 FN -> cost = 1 * fn_cost
        y_true = np.array([0, 1])
        y_pred = np.array([0, 0])
        assert business_cost(y_true, y_pred, fn_cost=5, fp_cost=1) == 5.0

    def test_default_ratio_is_five_to_one(self):
        assert DEFAULT_FN_COST / DEFAULT_FP_COST == 5.0

    def test_custom_ratio_propagates(self):
        # 2 FN, 0 FP with 10x ratio -> cost = 20
        y_true = np.array([0, 1, 1])
        y_pred = np.array([0, 0, 0])
        assert business_cost(y_true, y_pred, fn_cost=10, fp_cost=1) == 20.0

    def test_works_with_python_lists(self):
        # Should accept plain lists, not only numpy arrays
        assert business_cost([0, 1, 1, 0], [0, 0, 1, 1]) == 5.0 + 1.0


# ---------------------------------------------------------------------------
# business_gain
# ---------------------------------------------------------------------------
class TestBusinessGain:
    def test_perfect_predictions_gain_one(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert business_gain(y_true, y_pred) == 1.0

    def test_always_predict_negative_gain_zero(self):
        # When n_positives * fn_cost > n_negatives * fp_cost (true at 5:1),
        # predicting all 0 yields the worst cost -> gain = 0.
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        assert business_gain(y_true, y_pred) == 0.0

    def test_gain_is_bounded(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        y_pred = rng.integers(0, 2, size=200)
        g = business_gain(y_true, y_pred)
        assert 0.0 <= g <= 1.0

    def test_better_predictions_higher_gain(self):
        # Same y_true, different predictions: closer-to-truth -> higher gain.
        y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1])

        good = np.array([0, 0, 1, 1, 1, 0, 1, 1])  # 1 FP
        bad = np.array([0, 0, 0, 0, 0, 0, 0, 0])  # 4 FN

        assert business_gain(y_true, good) > business_gain(y_true, bad)


# ---------------------------------------------------------------------------
# make_business_scorer
# ---------------------------------------------------------------------------
class TestMakeBusinessScorer:
    def test_returns_sklearn_scorer(self):
        scorer = make_business_scorer()
        # sklearn scorers are callables with a specific interface
        assert callable(scorer)

    def test_scorer_uses_custom_costs(self):
        # Build a tiny "model" that returns fixed predictions
        class StubModel:
            def predict(self, X):
                return np.array([0, 0])

        scorer = make_business_scorer(fn_cost=10, fp_cost=1)
        # y_true = [1, 1] -> 2 FN -> cost 20, worst = 2 * 10 = 20, gain = 0
        score = scorer(StubModel(), np.array([[0], [0]]), np.array([1, 1]))
        assert score == 0.0


# ---------------------------------------------------------------------------
# find_optimal_threshold
# ---------------------------------------------------------------------------
class TestFindOptimalThreshold:
    def test_perfect_model_finds_clean_split(self):
        # If proba perfectly ranks classes, any threshold in (max_neg, min_pos]
        # gives cost = 0.
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        result = find_optimal_threshold(y_true, y_proba)
        assert result.best_cost == 0.0
        assert result.best_gain == 1.0

    def test_returns_grid_of_results(self):
        y_true = np.array([0, 1, 1, 0])
        y_proba = np.array([0.3, 0.6, 0.7, 0.4])
        result = find_optimal_threshold(y_true, y_proba)
        assert len(result.all_results) > 10
        # All results should have the expected keys
        for r in result.all_results:
            assert {"threshold", "cost", "gain"} <= set(r.keys())

    def test_asymmetric_cost_lowers_threshold(self):
        # With FN >> FP, the optimal threshold should be LOWER than 0.5
        # (we accept more FP to catch more FN).
        rng = np.random.default_rng(0)
        n = 1000
        y_true = rng.binomial(1, 0.25, size=n)
        # Noisy but informative probability
        y_proba = np.clip(y_true * 0.4 + rng.normal(0.3, 0.2, size=n), 0, 1)

        # Symmetric cost — optimal threshold near 0.5
        sym = find_optimal_threshold(y_true, y_proba, fn_cost=1.0, fp_cost=1.0)
        # Asymmetric cost — optimal threshold below 0.5
        asym = find_optimal_threshold(y_true, y_proba, fn_cost=5.0, fp_cost=1.0)

        assert asym.best_threshold < sym.best_threshold
