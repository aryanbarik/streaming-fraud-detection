from datetime import datetime, timedelta

import pytest

from fraudstream.features.derive import derive
from fraudstream.features.spec import FEATURES
from fraudstream.features.state import CardState, TerminalState, Tx


def test_derived_features_use_preupdate_history():
    t = datetime(2025, 1, 1, 8)
    card, terminal = CardState(), TerminalState()
    for minutes, amount in [(0, 10), (30, 30)]:
        tx = Tx(t + timedelta(minutes=minutes), 1, 2, amount, 0, 0)
        card.update(tx)
        terminal.update_tx(tx)
    current = Tx(t + timedelta(hours=1, minutes=30), 1, 2, 40, 3, 4)
    features = derive(current, card.read(current.event_ts), terminal.read(current.event_ts))
    assert set(features) == {spec.name for spec in FEATURES}
    assert features["amount_mean_ratio"] == pytest.approx(2.0)
    assert features["amount_zscore"] == pytest.approx(2.0)
    assert features["travel_units_per_hour"] == pytest.approx(5.0)
    assert features["hour"] == 9
    assert features["weekday"] == 2
