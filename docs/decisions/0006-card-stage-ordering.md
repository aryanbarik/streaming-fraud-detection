# 0006: Card-stage ordering across sources

Status: accepted (P4, non-Docker parts). Unit tests cover the ordering logic. The Kafka path is
verified only in the Docker phase.

## Context

- ADR 0002 keeps labels and transactions on one terminal-keyed topic. The plan keys `events.raw`
  by terminal and `tx.term` by card.
- A card's transactions come from several terminals, so one card reaches the card stage from
  several `events.raw` partitions. Arrival order across partitions is not event order.
- `CardState` and `TerminalState` raise on any time regression (`_check_time`). The backfill
  (`features/backfill.py`) defines the reference history: availability order, with labels before
  transactions at equal timestamps.

## Decision

1. The load generator replays `availability_timeline` order and sends each message to partition
   `source_of(terminal_id) = terminal_id % 6` of `events.raw`, keyed by terminal id. Each
   partition is therefore in event-time order.
2. Stage A (`TerminalStage`) handles one source. It rejects an out-of-order event to `tx.dlq`
   before touching state. It reads terminal features, updates the terminal, and forwards a
   `TermTx` to `tx.term` keyed by card id.
3. Stage B (`CardStage`) runs one `tx.term` partition. `ReorderBuffer` holds each transaction until
   every source has promised to send nothing earlier, then releases in `(event_ts, tx_id)` order.
   This is the backfill's order.
4. Promises come from data (a source's last event time), from heartbeats (Stage A reports the last
   event time it has processed; sent periodically), and from end-of-stream watermarks (`ts = None`).
   The load generator sends an end-of-stream watermark to every `events.raw` partition. Stage A
   forwards it to every `tx.term` partition with explicit partition numbers.
5. Producers are idempotent (`enable.idempotence=True`), so retries cannot reorder a partition.
6. The rules scorer returns a count of fired rules. Its alert threshold is 1.0 (any rule fires).
   This is a serving convention, not a tuned value.

## Consequences

- When one source is idle, release waits for its heartbeat. Card-stage latency is therefore bounded
  below by the heartbeat interval. The interval is a config value to be measured in the Docker
  phase; it is not wired yet.
- Heartbeat traffic is sources × partitions per interval (6 × 6).
- Rejected alternatives:
  - Dead-letter on out-of-order arrival: arrival order is not event order by design, so this would
    reject most cross-terminal traffic and fail the 10k-scored smoke test.
  - Single-partition `events.raw` and `tx.term`: deviates from the six-partition topology in the
    plan and would need its own ADR.
  - Time-window sorting: adds latency and still needs a completeness signal, which the watermark
    already provides.
- Verification: `tests/stream/test_stage_unit.py` replays random interleavings across six sources
  and checks that feature rows, and their order, equal `backfill` output for three seeds. Not yet
  verified over Kafka.
- Duplicates are not removed by this design. Exactly-once processing (`processing_guarantee`) is the
  planned mitigation and is measured in the Docker phase.
