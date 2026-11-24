"""Topic names and partition layout shared by the load generator and the stages."""

RAW_TOPIC = "events.raw"

# One source per events.raw partition. A source owns the terminals with terminal_id % 6 == source.
RAW_PARTITIONS = 6


def source_of(terminal_id: int) -> int:
    """Source id, which is also the events.raw partition, for a terminal."""
    return terminal_id % RAW_PARTITIONS


TERM_TOPIC = "tx.term"
# Keyed by card_id. Watermarks fan out to all partitions (ADR 0006).
TERM_PARTITIONS = 6
SCORED_TOPIC = "tx.scored"
DLQ_TOPIC = "tx.dlq"
