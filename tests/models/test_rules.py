import numpy as np

from fraudstream.models.rules import RulesScorer


def test_rules_use_named_features_and_count_triggers():
    scorer = RulesScorer(["amount_zscore", "card_count_1h", "terminal_report_rate_7d"])
    scores = scorer.score([[2, 3, .05], [3, 4, .1], [4, 1, 0]])
    np.testing.assert_array_equal(scores, [0, 3, 1])
