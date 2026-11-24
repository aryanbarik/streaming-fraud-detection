import pytest

from fraudstream.evaluation.quality import (
    calibration_metrics,
    daily_top_k_precision,
    recall_at_fpr,
)


def test_recall_at_fpr_rejects_false_positive_before_second_fraud():
    assert recall_at_fpr([1, 0, 1, 0], [.9, .8, .7, .1]) == .5


def test_calibration_with_hand_computed_bins():
    result = calibration_metrics([0, 1], [.2, .8], bins=2)
    assert result["brier"] == pytest.approx(.04)
    assert result["ece"] == pytest.approx(.2)


def test_daily_precision_averages_days_not_transactions():
    assert daily_top_k_precision([1, 0, 0, 0], [.9, .2, .9, .1], [1, 1, 2, 2], .5) == .5
