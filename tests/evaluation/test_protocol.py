from datetime import datetime, timedelta

import numpy as np
import pyarrow as pa
import pytest

from fraudstream.config import CostConfig
from fraudstream.evaluation.cost import net_savings
from fraudstream.evaluation.labelview import LabelView
from fraudstream.evaluation.splits import temporal_split


def test_label_view_hides_future_and_unreported():
    cutoff = datetime(2025, 2, 1)
    table = pa.table({
        "event_ts": [cutoff - timedelta(days=20), cutoff - timedelta(days=2), cutoff],
        "available_ts": [cutoff, cutoff + timedelta(seconds=1), None],
        "is_fraud": [1, 1, 1],
    })
    view = LabelView(table, cutoff)
    assert view.labels().tolist() == [1, 0, 0]
    assert view.mature().tolist() == [True, False, False]


def test_temporal_windows_exclude_gap_and_do_not_overlap():
    days = np.arange(130)
    split = temporal_split(days, T=90, G=14, V=14)
    assert (split.train[0], split.train[-1]) == (0, 61)
    assert (split.validation[0], split.validation[-1]) == (62, 75)
    assert (split.test[0], split.test[-1]) == (90, 119)
    assert not set(split.train) & set(split.validation)
    assert not set(split.validation) & set(split.test)


def test_net_savings_accounts_for_missed_fraud_and_reviews():
    y = [1, 0, 1, 0]
    amount = [100, 20, 40, 10]
    costs = CostConfig(review_usd=5, fp_friction_usd=3)
    assert net_savings(y, [1, 0, 1, 0], amount, .5, costs) == 130
    assert net_savings(y, [1, 1, 0, 0], amount, .5, costs) == 87
    with pytest.raises(ValueError):
        net_savings(y, [1], amount, .5, costs)


def test_perfect_oracle_savings_equal_baseline_loss_minus_review_cost():
    y = [1, 0, 1, 0]
    amount = [100, 20, 40, 10]
    costs = CostConfig(review_usd=5, fp_friction_usd=3, fraud_fixed_fee_usd=2)
    baseline_loss = (100 + 40) + 2 * 2
    assert net_savings(y, y, amount, .5, costs) == baseline_loss - 2 * 5
