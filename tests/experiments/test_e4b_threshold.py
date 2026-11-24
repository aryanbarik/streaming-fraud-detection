import numpy as np
import pytest

from fraudstream.config import CostConfig
from fraudstream.experiments.e4b_threshold import choose_threshold, scaled_costs


def test_scaled_costs_leave_fraud_costs_unchanged():
    base = CostConfig(review_usd=5, fp_friction_usd=3, fraud_fixed_fee_usd=1)
    scaled = scaled_costs(base, 2.0)
    assert (scaled.review_usd, scaled.fp_friction_usd) == (10, 6)
    assert scaled.fraud_fixed_fee_usd == 1
    assert (base.review_usd, base.fp_friction_usd) == (5, 3)


def test_min_loss_threshold_rises_with_alert_cost():
    # Hand-checked. Savings at the cut that alerts the top k rows:
    #   0.9 (k=1): 100 - 40 missed - 5 = 95 at ratio 1; 80 at ratio 4 (review 20, no friction)
    #   0.7 (k=3): 140 - 0 missed - 3*5 review - 1*3 friction = 122 at ratio 1 (best);
    #              at ratio 4 it is 68, so 0.9 wins.
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.7, 0.1])
    amount = np.array([100.0, 20.0, 40.0, 10.0])
    base = CostConfig()
    assert choose_threshold("min_loss", y, score, amount,
                            scaled_costs(base, 1.0)) == pytest.approx(0.7)
    assert choose_threshold("min_loss", y, score, amount,
                            scaled_costs(base, 4.0)) == pytest.approx(0.9)
