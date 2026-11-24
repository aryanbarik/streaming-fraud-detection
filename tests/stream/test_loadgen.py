from datetime import datetime, timedelta

import pyarrow as pa
import pytest

from fraudstream.stream.loadgen import parse_ramp, rate_at, send_paced, timeline_messages
from fraudstream.stream.schemas import LabelEvent, TxEvent, Watermark, decode_raw
from fraudstream.stream.topics import RAW_PARTITIONS, source_of

T0 = datetime(2025, 1, 1)


def tx_table(txs):
    """txs: (tx_id, event_ts, card_id, terminal_id)."""
    return pa.table({
        "tx_id": pa.array([t[0] for t in txs], pa.int64()),
        "event_ts": pa.array([t[1] for t in txs], pa.timestamp("us")),
        "card_id": pa.array([t[2] for t in txs], pa.int32()),
        "terminal_id": pa.array([t[3] for t in txs], pa.int32()),
        "amount": pa.array([1.0] * len(txs), pa.float64()),
        "x": pa.array([0.0] * len(txs), pa.float64()),
        "y": pa.array([0.0] * len(txs), pa.float64()),
        "is_fraud": pa.array([0] * len(txs), pa.int8()),
    })


def label_table(labels, kind="fraud"):
    """labels: (tx_id, terminal_id, available_ts)."""
    return pa.table({
        "tx_id": pa.array([x[0] for x in labels], pa.int64()),
        "terminal_id": pa.array([x[1] for x in labels], pa.int32()),
        "available_ts": pa.array([x[2] for x in labels], pa.timestamp("us")),
        "kind": pa.array([kind] * len(labels), pa.string()),
    })


class FakeTime:
    """Monotonic clock that advances only when the sender sleeps."""

    def __init__(self):
        self.now = 0

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.now += round(seconds * 1e9)


def tx_events(n):
    return iter(
        TxEvent(source=0, tx_id=i, event_ts=T0, card_id=0, terminal_id=0, amount=1.0,
                x=0.0, y=0.0, is_fraud=0)
        for i in range(n)
    )


def test_label_precedes_same_timestamp_transaction_and_routes_by_terminal():
    txs = tx_table([(0, T0, 5, 7), (1, T0 + timedelta(seconds=1), 6, 3)])
    events = list(timeline_messages(txs, label_table([(0, 7, T0)])))
    assert [(type(e), e.source) for e in events] == [
        (LabelEvent, source_of(7)),
        (TxEvent, source_of(7)),
        (TxEvent, source_of(3)),
    ]


def test_unknown_label_kind_is_rejected():
    txs = tx_table([(0, T0, 5, 7)])
    with pytest.raises(ValueError, match="unsupported label kind"):
        list(timeline_messages(txs, label_table([(0, 7, T0)], kind="chargeback")))


def test_send_paced_follows_the_ramp_and_ends_every_partition():
    fake = FakeTime()
    produced = []

    def produce(partition, key, value):
        produced.append((fake.now, partition, decode_raw(value)))

    sent = send_paced(
        tx_events(100), tps=10.0, duration_s=2.0, ramp=((1.0, 20.0),), produce=produce,
        clock=fake.clock, sleep=fake.sleep, wall_ns=fake.clock,
    )
    assert sent == 30
    data = produced[:30]
    expected_times = [k * 10**8 for k in range(10)] + [10**9 + k * 5 * 10**7 for k in range(20)]
    assert [t for t, _, _ in data] == expected_times
    assert all(message.t_produce_ns == t for t, _, message in data)
    ends = produced[30:]
    assert [partition for _, partition, _ in ends] == list(range(RAW_PARTITIONS))
    assert all(isinstance(m, Watermark) and m.ts is None for _, _, m in ends)


def test_rate_schedule_and_ramp_parsing():
    ramp = parse_ramp("30:1000,45.5:2000")
    assert ramp == ((30.0, 1000.0), (45.5, 2000.0))
    assert parse_ramp("") == ()
    assert rate_at(29.9, 500.0, ramp) == 500.0
    assert rate_at(30.0, 500.0, ramp) == 1000.0
    assert rate_at(50.0, 500.0, ramp) == 2000.0


def test_unsorted_ramp_is_rejected():
    with pytest.raises(ValueError, match="ramp"):
        send_paced(
            iter([]), tps=10.0, duration_s=1.0, ramp=((2.0, 5.0), (1.0, 5.0)),
            produce=lambda *args: None,
        )
