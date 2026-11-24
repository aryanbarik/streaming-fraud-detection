from datetime import datetime, timedelta

import pytest

from fraudstream.features.state import CardState, TerminalState, Tx


def test_card_preupdate_and_window_boundaries():
    t = datetime(2025, 1, 1)
    card = CardState()
    tx = Tx(t, 1, 2, 10.0, 0.0, 1.0)
    assert card.read(t)["card_count_1h"] == 0
    card.update(tx)
    assert card.read(t + timedelta(minutes=59))["card_count_1h"] == 1
    assert card.read(t + timedelta(hours=1))["card_count_1h"] == 0
    assert card.read(t + timedelta(days=7))["card_count_7d"] == 0


def test_terminal_labels_are_visible_only_after_availability():
    t = datetime(2025, 1, 1)
    terminal = TerminalState()
    terminal.update_tx(Tx(t, 1, 2, 10.0, 0.0, 1.0))
    assert terminal.read(t + timedelta(hours=1))["terminal_fraud_reports_7d"] == 0
    terminal.apply_label(t + timedelta(hours=2))
    assert terminal.read(t + timedelta(hours=2))["terminal_fraud_reports_7d"] == 1
    with pytest.raises(ValueError):
        terminal.read(t + timedelta(hours=1))
