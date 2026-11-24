from datetime import datetime

from fraudstream.stream.schemas import DeadLetter, ScoredTx, TermTx, Watermark, decode_term
from fraudstream.stream.sinks import to_records
from fraudstream.stream.topics import DLQ_TOPIC, SCORED_TOPIC, TERM_PARTITIONS, TERM_TOPIC

T0 = datetime(2026, 1, 1, 12, 0, 0)


def _term_tx(*, tx_id: int = 1, card_id: int = 1) -> TermTx:
    return TermTx(
        source=0,
        tx_id=tx_id,
        event_ts=T0,
        card_id=card_id,
        terminal_id=3,
        amount=12.5,
        x=0.1,
        y=0.2,
        is_fraud=0,
        term_feats={"fraud_rate_1h": 0.0},
        t_produce_ns=1,
    )


def test_watermark_reaches_each_term_partition_once():
    mark = Watermark(source=2, ts=T0)

    records = to_records([mark])

    assert [r.partition for r in records] == list(range(TERM_PARTITIONS))
    assert {r.topic for r in records} == {TERM_TOPIC}
    assert all(r.key is None for r in records)
    assert all(decode_term(r.value) == mark for r in records)


def test_transactions_are_keyed_by_card():
    tx = _term_tx(card_id=42)

    [record] = to_records([tx])

    assert (record.topic, record.key, record.partition) == (TERM_TOPIC, b"42", None)
    assert decode_term(record.value) == tx


def test_watermarks_keep_their_place_behind_the_transactions_they_cover():
    first = _term_tx(tx_id=1, card_id=1)
    second = _term_tx(tx_id=2, card_id=2)
    mark = Watermark(source=0, ts=T0)

    decoded = [decode_term(r.value) for r in to_records([first, mark, second])]

    assert decoded == [first, *([mark] * TERM_PARTITIONS), second]


def test_scored_transactions_and_dead_letters_route_to_their_topics():
    scored = ScoredTx(
        tx_id=7,
        event_ts=T0,
        card_id=1,
        terminal_id=3,
        score=0.9,
        alert=True,
        is_fraud=1,
        t_produce_ns=5,
        timings={"t_score": 3},
    )
    dead = DeadLetter(stage="card", reason="regression", payload="{}")

    scored_record, dlq_record = to_records([scored, dead])

    assert (scored_record.topic, scored_record.key) == (SCORED_TOPIC, b"7")
    assert dlq_record.topic == DLQ_TOPIC
    assert DeadLetter.model_validate_json(dlq_record.value) == dead
