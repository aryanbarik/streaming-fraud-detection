"""Open-loop load generator: replays transactions and labels onto events.raw.

Messages follow `features.backfill.availability_timeline`, so each events.raw partition receives
its events in event-time order. A message goes to the partition of its terminal's source
(`source_of`). After the last message, every partition gets an end-of-stream watermark.

The sender follows a rate schedule and does not slow down when the broker is slow, so downstream
latency includes queueing.
Usage: python -m fraudstream.stream.loadgen --parquet PATH --tps N --duration-s S [--ramp T:N,...]
labels.parquet must sit beside transactions.parquet. The broker is $BROKER, else StreamConfig.
"""

import argparse
import itertools
import os
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from fraudstream.config import StreamConfig
from fraudstream.features.backfill import availability_timeline
from fraudstream.stream.schemas import LabelEvent, TxEvent, Watermark, encode
from fraudstream.stream.topics import RAW_PARTITIONS, RAW_TOPIC, source_of

Produce = Callable[[int, bytes, bytes], None]
Ramp = Sequence[tuple[float, float]]


def timeline_messages(transactions: pa.Table, labels: pa.Table) -> Iterator[LabelEvent | TxEvent]:
    """Labels and transactions in availability order, each tagged with its source."""
    for timestamp, priority, row in availability_timeline(transactions, labels):
        source = source_of(row["terminal_id"])
        if priority == 0:
            if row["kind"] != "fraud":
                raise ValueError(f"unsupported label kind: {row['kind']}")
            yield LabelEvent(
                source=source,
                tx_id=row["tx_id"],
                terminal_id=row["terminal_id"],
                available_ts=timestamp,
            )
        else:
            yield TxEvent(
                source=source,
                tx_id=row["tx_id"],
                event_ts=timestamp,
                card_id=row["card_id"],
                terminal_id=row["terminal_id"],
                amount=row["amount"],
                x=row["x"],
                y=row["y"],
                is_fraud=row["is_fraud"],
            )


def rate_at(elapsed_s: float, tps: float, ramp: Ramp) -> float:
    """Target rate at `elapsed_s`: the last ramp step that has started, else `tps`."""
    rate = tps
    for at_s, step_tps in ramp:
        if elapsed_s < at_s:
            break
        rate = step_tps
    return rate


def parse_ramp(text: str) -> Ramp:
    """Parse "at_s:tps,at_s:tps" into ramp steps."""
    if not text:
        return ()
    steps = []
    for item in text.split(","):
        at_s, _, step_tps = item.partition(":")
        steps.append((float(at_s), float(step_tps)))
    return tuple(steps)


def send_paced(
    messages: Iterable[LabelEvent | TxEvent],
    *,
    tps: float,
    duration_s: float,
    ramp: Ramp,
    produce: Produce,
    clock: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
    wall_ns: Callable[[], int] = time.time_ns,
) -> int:
    """Send messages on schedule until the schedule passes `duration_s`, then end every partition.

    Returns the number of messages sent; end-of-stream watermarks are not counted.
    """
    if tps <= 0 or duration_s <= 0:
        raise ValueError("tps and duration_s must be positive")
    if any(rate <= 0 for _, rate in ramp) or any(b[0] <= a[0] for a, b in itertools.pairwise(ramp)):
        raise ValueError("ramp must be sorted by time with positive rates")
    start = clock()
    next_offset_ns = 0
    sent = 0
    for message in messages:
        scheduled_s = next_offset_ns / 1e9
        if scheduled_s >= duration_s:
            break
        delay_ns = start + next_offset_ns - clock()
        if delay_ns > 0:
            sleep(delay_ns / 1e9)
        if isinstance(message, TxEvent):
            message = message.model_copy(update={"t_produce_ns": wall_ns()})
        produce(message.source, str(message.terminal_id).encode(), encode(message))
        sent += 1
        next_offset_ns += round(1e9 / rate_at(scheduled_s, tps, ramp))
    for source in range(RAW_PARTITIONS):
        produce(source, str(source).encode(), encode(Watermark(source=source)))
    return sent


def run(parquet: Path, tps: float, duration_s: float, ramp: Ramp = (), *, broker: str) -> int:
    """Replay `parquet` (transactions) and the labels.parquet beside it onto events.raw."""
    # Imported here so the modules above load without the `stream` dependency group.
    from quixstreams.kafka import Producer

    transactions = pq.read_table(parquet)
    labels = pq.read_table(parquet.with_name("labels.parquet"))
    producer = Producer(broker, extra_config={"enable.idempotence": True})

    def produce(partition: int, key: bytes, value: bytes) -> None:
        producer.produce(topic=RAW_TOPIC, key=key, value=value, partition=partition)
        producer.poll(0)

    sent = send_paced(
        timeline_messages(transactions, labels),
        tps=tps,
        duration_s=duration_s,
        ramp=ramp,
        produce=produce,
    )
    producer.flush()
    return sent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--tps", type=float, required=True,
                        help="messages per second (transactions and labels)")
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--ramp", type=parse_ramp, default=(), help="steps as at_s:tps")
    args = parser.parse_args()
    broker = os.environ.get("BROKER") or StreamConfig().broker
    sent = run(args.parquet, args.tps, args.duration_s, args.ramp, broker=broker)
    print(f"sent {sent} transactions and labels to {RAW_TOPIC}")


if __name__ == "__main__":
    main()
