import itertools
from datetime import datetime

import numpy as np
import pytest

from fraudstream.features.spec import FEATURES
from fraudstream.models.registry import load_model_config
from fraudstream.models.train import train
from fraudstream.serving.scorer import LGBMScorer, RulesScorer, score_row
from fraudstream.stream.card_stage import FeatureRow

NAMES = [spec.name for spec in FEATURES]


def trained_bundle():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, len(NAMES)))
    y = (X[:, 0] > 0).astype(int)
    return train(load_model_config("lgbm"), X, y, NAMES), X


def test_lgbm_scorer_matches_bundle_row_by_row():
    bundle, X = trained_bundle()
    scorer = LGBMScorer(bundle, NAMES)
    for row in X[:5]:
        assert scorer.score(row) == pytest.approx(bundle.score(row.reshape(1, -1))[0], rel=1e-12)


def test_lgbm_scorer_rejects_feature_order_mismatch():
    bundle, _ = trained_bundle()
    with pytest.raises(ValueError, match="feature_list"):
        LGBMScorer(bundle, NAMES[::-1])


def test_rules_scorer_counts_fired_rules():
    x = np.zeros(len(NAMES))
    x[NAMES.index("card_count_1h")] = 5
    x[NAMES.index("terminal_report_rate_7d")] = 0.2
    x[NAMES.index("amount_zscore")] = 1
    assert RulesScorer(NAMES).score(x) == 2.0


class FixedScorer:
    def __init__(self, value: float) -> None:
        self.value = value

    def score(self, x: np.ndarray) -> float:
        return self.value


def test_score_row_alerts_at_threshold_and_keeps_timings():
    row = FeatureRow(
        tx_id=1, event_ts=datetime(2025, 1, 1), card_id=2, terminal_id=3, is_fraud=0,
        t_produce_ns=10, features=np.zeros(len(NAMES)), timings={"t_stageA_in": 1},
    )
    scored = score_row(FixedScorer(0.5), row, threshold=0.5, clock=itertools.count(100).__next__)
    assert scored.alert is True
    assert scored.timings == {"t_stageA_in": 1, "t_score": 100, "t_emit": 101}
