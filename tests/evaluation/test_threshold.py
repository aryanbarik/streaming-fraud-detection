import numpy as np

from fraudstream.config import CostConfig
from fraudstream.evaluation.cost import net_savings
from fraudstream.evaluation.metrics import pick_threshold


def test_minimum_loss_selects_best_complete_score_group():
    y = [1, 0, 1, 0]
    scores = [.9, .8, .7, .1]
    amounts = [100, 2, 40, 2]
    costs = CostConfig()
    threshold = pick_threshold("min_loss", y, scores, amounts, costs)
    assert threshold == .7
    choices = [np.inf, *scores]
    assert net_savings(y, scores, amounts, threshold, costs) == max(
        net_savings(y, scores, amounts, choice, costs) for choice in choices
    )


def test_capacity_does_not_split_ties():
    costs = CostConfig()
    scores = [.9, .9, .8, .1]
    assert np.isinf(pick_threshold("capacity", [1, 0, 1, 0], scores, [1] * 4, costs, capacity=.25))
    assert pick_threshold("capacity", [1, 0, 1, 0], scores, [1] * 4, costs, capacity=.5) == .9


def test_f1_threshold_uses_validation_labels():
    assert pick_threshold("f1", [1, 0, 1], [.9, .8, .7], [1, 1, 1], CostConfig()) == .7
