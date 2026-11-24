from datetime import datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq

from fraudstream.features.backfill import backfill
from fraudstream.features.derive import derive
from fraudstream.features.state import CardState, TerminalState, Tx


def test_backfill_matches_incremental_state_with_interleaved_labels(tmp_path):
    start = datetime(2025, 1, 1)
    transactions = pa.Table.from_pylist([
        {"tx_id": 1, "event_ts": start, "card_id": 1, "terminal_id": 2,
         "amount": 10., "x": 0., "y": 0.},
        {"tx_id": 2, "event_ts": start + timedelta(hours=2), "card_id": 1,
         "terminal_id": 2, "amount": 15., "x": 1., "y": 0.},
        {"tx_id": 3, "event_ts": start + timedelta(hours=3), "card_id": 1,
         "terminal_id": 2, "amount": 20., "x": 0., "y": 1.},
    ])
    label_time = start + timedelta(hours=1)
    labels = pa.Table.from_pylist([
        {"tx_id": 1, "terminal_id": 2, "available_ts": label_time, "kind": "fraud"}
    ])
    out = tmp_path / "features.parquet"
    backfill(transactions, labels, out)
    recorded = pq.read_table(out).to_pylist()
    card, terminal = CardState(), TerminalState()
    for i, row in enumerate(transactions.to_pylist()):
        if i == 1:
            terminal.apply_label(label_time)
        tx = Tx(row["event_ts"], row["card_id"], row["terminal_id"],
                row["amount"], row["x"], row["y"])
        actual = derive(tx, card.read(tx.event_ts), terminal.read(tx.event_ts))
        assert actual == {key: recorded[i][key] for key in actual}
        card.update(tx)
        terminal.update_tx(tx)
