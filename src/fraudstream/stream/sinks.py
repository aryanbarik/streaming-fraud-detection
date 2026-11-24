"""Map stage outputs to producer records.

One producer sends a stage's records in emission order. A watermark promises that every
transaction it covers was emitted before it, so watermarks keep that place in the order and go to
every tx.term partition (ADR 0006).
"""

from collections.abc import Iterable
from dataclasses import dataclass

from fraudstream.stream.schemas import DeadLetter, ScoredTx, TermTx, Watermark, encode
from fraudstream.stream.topics import DLQ_TOPIC, SCORED_TOPIC, TERM_PARTITIONS, TERM_TOPIC


@dataclass(frozen=True)
class Record:
    topic: str
    key: bytes | None
    value: bytes
    partition: int | None = None


def to_records(outputs: Iterable[TermTx | ScoredTx | Watermark | DeadLetter]) -> list[Record]:
    """Records for `outputs`, in the same order."""
    records: list[Record] = []
    for item in outputs:
        if isinstance(item, TermTx):
            records.append(Record(TERM_TOPIC, str(item.card_id).encode(), encode(item)))
        elif isinstance(item, ScoredTx):
            records.append(Record(SCORED_TOPIC, str(item.tx_id).encode(), encode(item)))
        elif isinstance(item, Watermark):
            value = encode(item)
            records.extend(Record(TERM_TOPIC, None, value, p) for p in range(TERM_PARTITIONS))
        elif isinstance(item, DeadLetter):
            records.append(Record(DLQ_TOPIC, None, encode(item)))
        else:
            raise TypeError(f"no topic for {type(item).__name__}")
    return records
