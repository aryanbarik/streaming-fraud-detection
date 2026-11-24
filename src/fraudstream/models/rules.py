"""Interpretable hand-written fraud risk baseline."""

import numpy as np


class RulesScorer:
    """Count distinct rules triggered; scores are ranks, not probabilities."""

    def __init__(self, feature_list: list[str]):
        self.columns = {name: feature_list.index(name) for name in (
            "card_count_1h", "amount_zscore", "terminal_report_rate_7d"
        )}

    def score(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return (
            (X[:, self.columns["card_count_1h"]] >= 4).astype(float)
            + (X[:, self.columns["amount_zscore"]] >= 3)
            + (X[:, self.columns["terminal_report_rate_7d"]] >= 0.1)
        )
