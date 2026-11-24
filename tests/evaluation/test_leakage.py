from datetime import datetime, timedelta

import pandas as pd
import pyarrow as pa

from fraudstream.evaluation.leakage import future_terminal_features, immediate_fraud_labels


def test_immediate_labels_expose_even_unreported_fraud():
    t = datetime(2025, 1, 1)
    tx = pa.table({"tx_id": [1, 2], "terminal_id": [10, 10],
                   "event_ts": [t, t + timedelta(hours=1)], "is_fraud": [1, 0]})
    assert immediate_fraud_labels(tx).to_pylist() == [
        {"tx_id": 1, "terminal_id": 10, "available_ts": t, "kind": "fraud"}
    ]


def test_non_pit_variant_uses_each_terminals_future_state_only():
    t = datetime(2025, 1, 1)
    frame = pd.DataFrame({"tx_id": [1, 2, 3], "terminal_id": [5, 6, 5],
                          "event_ts": [t, t, t + timedelta(days=1)],
                          "terminal_fraud_reports_7d": [0, 4, 3],
                          "card_count_7d": [1, 2, 3]})
    leaked = future_terminal_features(frame, ["terminal_fraud_reports_7d"])
    assert leaked["terminal_fraud_reports_7d"].tolist() == [3, 4, 3]
    assert leaked["card_count_7d"].tolist() == [1, 2, 3]
    assert frame["terminal_fraud_reports_7d"].tolist() == [0, 4, 3]
