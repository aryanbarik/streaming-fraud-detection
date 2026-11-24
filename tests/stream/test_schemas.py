from datetime import datetime

import pytest
from pydantic import ValidationError

from fraudstream.stream.schemas import (
    LabelEvent,
    TermTx,
    TxEvent,
    Watermark,
    decode_raw,
    decode_term,
    encode,
)

TS = datetime(2025, 1, 2, 3, 4, 5, 678901)


def test_tx_event_roundtrips_exactly():
    event = TxEvent(
        source=2, tx_id=7, event_ts=TS, card_id=11, terminal_id=8, amount=12.34,
        x=1.5, y=-2.25, is_fraud=1, t_produce_ns=123,
    )
    assert decode_raw(encode(event)) == event


def test_label_and_end_of_stream_roundtrip():
    label = LabelEvent(source=0, tx_id=1, terminal_id=0, available_ts=TS)
    end = Watermark(source=3)
    assert decode_raw(encode(label)) == label
    assert decode_raw(encode(end)) == end
    assert end.ts is None


def test_heartbeat_and_term_tx_roundtrip():
    heartbeat = Watermark(source=1, ts=TS)
    message = TermTx(
        source=1, tx_id=2, event_ts=TS, card_id=3, terminal_id=4, amount=5.0, x=0.0, y=0.0,
        is_fraud=0, term_feats={"terminal_count_1h": 2.0}, t_produce_ns=9,
        timings={"t_stageA_in": 1},
    )
    assert decode_term(encode(heartbeat)) == heartbeat
    assert decode_term(encode(message)) == message


@pytest.mark.parametrize(
    "payload",
    [
        b'{"type": "unknown", "source": 0}',
        b'{"type": "watermark", "source": -1}',
        b'{"type": "tx", "source": 0}',
    ],
)
def test_invalid_messages_are_rejected(payload):
    with pytest.raises(ValidationError):
        decode_raw(payload)
