from datetime import datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fraudstream.features.backfill import backfill
from fraudstream.features.spec import FEATURES


def test_replay_preupdate_and_delayed_label_visibility(tmp_path):
    t = datetime(2025, 1, 1)
    transactions = pa.Table.from_pylist([
        {"tx_id": i, "event_ts": t + timedelta(hours=i), "card_id": 1,
         "terminal_id": 2, "amount": 10.0 + i, "x": 0.1, "y": 0.2}
        for i in range(3)
    ])
    labels = pa.Table.from_pylist([
        {"tx_id": 0, "terminal_id": 2, "available_ts": t + timedelta(hours=2), "kind": "fraud"}
    ])
    path = tmp_path / "features.parquet"
    backfill(transactions, labels, path)
    rows = pq.read_table(path).to_pylist()
    assert [row["card_count_7d"] for row in rows] == [0, 1, 2]
    assert [row["terminal_fraud_reports_7d"] for row in rows] == [0, 0, 1]
    assert set(rows[0]) == {"tx_id", "event_ts", "card_id", "terminal_id"} | {s.name for s in FEATURES}
    prefix = tmp_path / "prefix.parquet"
    backfill(transactions.slice(0, 2), labels, prefix)
    assert pq.read_table(prefix).to_pylist() == rows[:2]


def test_unsorted_transactions_rejected(tmp_path):
    t = datetime(2025, 1, 1)
    rows = [{"tx_id": i, "event_ts": t + timedelta(hours=i), "card_id": 1,
             "terminal_id": 2, "amount": 1., "x": 0., "y": 0.} for i in (1, 0)]
    labels = pa.table({"available_ts": pa.array([], type=pa.timestamp("us"))})
    with pytest.raises(ValueError, match="must be sorted"):
        backfill(pa.Table.from_pylist(rows), labels, tmp_path / "out.parquet")
