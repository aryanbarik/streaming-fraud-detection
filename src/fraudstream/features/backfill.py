"""Replay transaction and label streams in availability order."""

import heapq
from collections import defaultdict
from itertools import chain
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fraudstream.features.derive import derive
from fraudstream.features.spec import FEATURES
from fraudstream.features.state import CardState, TerminalState, Tx


def _events(table: pa.Table, time_field: str, priority: int):
    previous = None
    for batch in table.to_batches(max_chunksize=4096):
        for row in batch.to_pylist():
            timestamp = row[time_field]
            if previous is not None and timestamp < previous:
                raise ValueError(f"{time_field} input must be sorted")
            previous = timestamp
            yield timestamp, priority, row


def backfill(transactions: pa.Table, labels: pa.Table, out: Path) -> None:
    """Write one feature row per transaction using only available information.

    Labels at the same timestamp as a transaction are visible to that transaction.
    Both input tables must be sorted by their respective time columns.
    """
    fields = [
        pa.field("tx_id", pa.int64()),
        pa.field("event_ts", pa.timestamp("us")),
        pa.field("card_id", pa.int32()),
        pa.field("terminal_id", pa.int32()),
    ]
    fields.extend(pa.field(spec.name, pa.type_for_alias(spec.dtype)) for spec in FEATURES)
    schema = pa.schema(fields)
    cards: defaultdict[int, CardState] = defaultdict(CardState)
    terminals: defaultdict[int, TerminalState] = defaultdict(TerminalState)
    rows = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(out, schema) as writer:
        events = heapq.merge(_events(labels, "available_ts", 0),
                             _events(transactions, "event_ts", 1), key=lambda e: (e[0], e[1]))
        for timestamp, priority, row in events:
            if priority == 0:
                if row["kind"] != "fraud":
                    raise ValueError(f"unsupported label kind: {row['kind']}")
                terminals[row["terminal_id"]].apply_label(timestamp)
                continue
            tx = Tx(timestamp, row["card_id"], row["terminal_id"], row["amount"], row["x"], row["y"])
            card = cards[tx.card_id]
            terminal = terminals[tx.terminal_id]
            values = derive(tx, card.read(timestamp), terminal.read(timestamp))
            rows.append(dict(chain((("tx_id", row["tx_id"]), ("event_ts", timestamp),
                                    ("card_id", tx.card_id), ("terminal_id", tx.terminal_id)),
                                   values.items())))
            card.update(tx)
            terminal.update_tx(tx)
            if len(rows) == 4096:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                rows.clear()
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
