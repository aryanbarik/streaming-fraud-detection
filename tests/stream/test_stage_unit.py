import random
from collections import deque
from datetime import datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fraudstream.features.backfill import backfill
from fraudstream.features.state import TerminalState
from fraudstream.stream.card_stage import FEATURE_NAMES, CardStage, FeatureRow, ReorderBuffer
from fraudstream.stream.loadgen import timeline_messages
from fraudstream.stream.schemas import DeadLetter, TermTx, TxEvent, Watermark
from fraudstream.stream.terminal_stage import TerminalStage
from fraudstream.stream.topics import RAW_PARTITIONS

T0 = datetime(2025, 1, 1)


def zero() -> int:
    return 0


def tx(tx_id, ts, source=0, card=0, terminal=0):
    return TxEvent(source=source, tx_id=tx_id, event_ts=ts, card_id=card, terminal_id=terminal,
                   amount=10.0, x=0.0, y=0.0, is_fraud=0)


def term_tx(tx_id, ts, source=0, card=0):
    return TermTx(source=source, tx_id=tx_id, event_ts=ts, card_id=card, terminal_id=0,
                  amount=10.0, x=0.0, y=0.0, is_fraud=0,
                  term_feats=TerminalState().read(ts), t_produce_ns=0)


def test_terminal_stage_heartbeat_promises_last_processed_time_then_stops():
    stage = TerminalStage(0, clock=zero)
    assert stage.heartbeat() == []
    stage.process(tx(0, T0))
    assert stage.heartbeat() == [Watermark(source=0, ts=T0)]
    assert stage.process(Watermark(source=0)) == [Watermark(source=0)]
    assert stage.heartbeat() == []
    (rejected,) = stage.process(tx(1, T0 + timedelta(seconds=1)))
    assert isinstance(rejected, DeadLetter)


def test_terminal_stage_dead_letters_source_regression():
    stage = TerminalStage(0, clock=zero)
    (accepted,) = stage.process(tx(0, T0 + timedelta(minutes=2)))
    assert isinstance(accepted, TermTx)
    (rejected,) = stage.process(tx(1, T0))
    assert isinstance(rejected, DeadLetter) and rejected.stage == "terminal"
    assert "regressed" in rejected.reason


def test_terminal_stage_dead_letters_event_for_another_source():
    (rejected,) = TerminalStage(1, clock=zero).process(tx(0, T0, source=0))
    assert isinstance(rejected, DeadLetter)


def test_reorder_buffer_releases_only_when_every_source_has_passed_the_time():
    t1, t2 = T0 + timedelta(seconds=1), T0 + timedelta(seconds=2)
    buffer = ReorderBuffer[str](sources=2)
    assert buffer.add(0, t1, 1, "a") == []  # source 1 has not spoken yet
    assert buffer.watermark(1, t1) == []  # source 1 may still send at t1
    assert buffer.watermark(1, t2) == []  # source 0 may still send at t1
    assert buffer.watermark(0, t2) == ["a"]  # both sources have passed t1


def test_reorder_buffer_orders_ties_by_tx_id():
    t1, t2 = T0 + timedelta(seconds=1), T0 + timedelta(seconds=2)
    buffer = ReorderBuffer[str](sources=2)
    buffer.add(0, t1, 5, "five")
    buffer.add(1, t1, 3, "three")
    assert buffer.watermark(0, t2) == []
    assert buffer.watermark(1, t2) == ["three", "five"]


def test_card_stage_dead_letters_source_regression():
    stage = CardStage(sources=2, clock=zero)
    assert stage.process(term_tx(0, T0 + timedelta(seconds=2))) == []
    (rejected,) = stage.process(term_tx(1, T0 + timedelta(seconds=1)))
    assert isinstance(rejected, DeadLetter) and rejected.stage == "card"


def synthetic_tables(seed, n_tx=4000, n_cards=10, n_terminals=12):
    """Whole-second timestamps over two days, so txs tie with each other and labels with txs."""
    rng = random.Random(seed)
    seconds = sorted(rng.randint(0, 2 * 86400) for _ in range(n_tx))
    txs = [
        (i, T0 + timedelta(seconds=s), rng.randrange(n_cards), rng.randrange(n_terminals))
        for i, s in enumerate(seconds)
    ]
    transactions = pa.table({
        "tx_id": pa.array([t[0] for t in txs], pa.int64()),
        "event_ts": pa.array([t[1] for t in txs], pa.timestamp("us")),
        "card_id": pa.array([t[2] for t in txs], pa.int32()),
        "terminal_id": pa.array([t[3] for t in txs], pa.int32()),
        "amount": pa.array([round(rng.uniform(1, 500), 2) for _ in txs], pa.float64()),
        "x": pa.array([rng.uniform(0, 100) for _ in txs], pa.float64()),
        "y": pa.array([rng.uniform(0, 100) for _ in txs], pa.float64()),
        "is_fraud": pa.array([int(rng.random() < 0.1) for _ in txs], pa.int8()),
    })
    labelled = []
    for i in rng.sample(range(n_tx), n_tx // 4):
        delay = rng.choice([0, 0, rng.randint(1, 3 * 86400)])
        labelled.append((txs[i][1] + timedelta(seconds=delay), i))
    labelled.sort()
    labels = pa.table({
        "tx_id": pa.array([i for _, i in labelled], pa.int64()),
        "terminal_id": pa.array([txs[i][3] for _, i in labelled], pa.int32()),
        "available_ts": pa.array([ts for ts, _ in labelled], pa.timestamp("us")),
        "kind": pa.array(["fraud"] * len(labelled), pa.string()),
    })
    return transactions, labels


def source_queues(transactions, labels):
    queues = {s: deque() for s in range(RAW_PARTITIONS)}
    for event in timeline_messages(transactions, labels):
        queues[event.source].append(event)
    for source, queue in queues.items():
        queue.append(Watermark(source=source))
    return queues


def replay(queues, seed):
    """Deliver each source in order, with sources and heartbeats interleaved at random."""
    rng = random.Random(seed)
    terminals = {s: TerminalStage(s, clock=zero) for s in queues}
    card = CardStage(len(queues), clock=zero)
    rows: list[FeatureRow] = []

    def deliver(messages):
        for message in messages:
            assert not isinstance(message, DeadLetter), message
            for out in card.process(message):
                assert not isinstance(out, DeadLetter), out
                rows.append(out)

    while ready := [s for s in queues if queues[s]]:
        s = rng.choice(ready)
        deliver(terminals[s].process(queues[s].popleft()))
        if rng.random() < 0.3:
            deliver(terminals[rng.choice(list(terminals))].heartbeat())
    return rows


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_card_features_match_backfill_under_random_interleaving(seed, tmp_path):
    transactions, labels = synthetic_tables(seed)
    out = tmp_path / "features.parquet"
    backfill(transactions, labels, out)
    expected = pq.read_table(out).to_pylist()
    rows = replay(source_queues(transactions, labels), seed)
    assert len(rows) == len(expected) == transactions.num_rows
    assert [row.tx_id for row in rows] == [exp["tx_id"] for exp in expected]
    for row, exp in zip(rows, expected, strict=True):
        assert list(row.features) == [exp[name] for name in FEATURE_NAMES]
