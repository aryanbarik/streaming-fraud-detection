"""Stage B: card state and feature rows, released in event-time order across sources.

A tx.term partition receives transactions from every events.raw source, interleaved in arrival
order, because tx.term is keyed by card and events.raw by terminal. `ReorderBuffer` holds a
transaction until every source has promised to send nothing earlier, then releases in
(event_ts, tx_id) order. That is the order of transactions in
`features.backfill.availability_timeline`, so card state sees the same history as the backfill.
See docs/decisions/0006-card-stage-ordering.md.
"""

import heapq
import itertools
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from fraudstream.features.derive import derive
from fraudstream.features.spec import FEATURES
from fraudstream.features.state import CardState, Tx
from fraudstream.stream.schemas import DeadLetter, TermTx, Watermark

FEATURE_NAMES = [spec.name for spec in FEATURES]
Clock = Callable[[], int]
_END = datetime.max


class SourceOrderError(ValueError):
    """A source sent an event earlier than one it had already sent."""


class ReorderBuffer[T]:
    """Release items in event-time order once every source has passed their time."""

    def __init__(self, sources: int) -> None:
        self._last: list[datetime | None] = [None] * sources
        self._heap: list[tuple[datetime, int, int, T]] = []
        self._seq = itertools.count()

    def add(self, source: int, ts: datetime, tx_id: int, item: T) -> list[T]:
        """Hold `item` and return the items that are now safe to release."""
        self._advance(source, ts)
        heapq.heappush(self._heap, (ts, tx_id, next(self._seq), item))
        return self._release()

    def watermark(self, source: int, ts: datetime | None) -> list[T]:
        """Record that `source` sends nothing before `ts`. `None` ends the source."""
        self._advance(source, _END if ts is None else ts)
        return self._release()

    def _advance(self, source: int, ts: datetime) -> None:
        last = self._last[source]
        if last is not None and ts < last:
            raise SourceOrderError(f"source {source} regressed from {last} to {ts}")
        self._last[source] = ts

    def _release(self) -> list[T]:
        if any(last is None for last in self._last):
            return []
        bound = min(self._last)
        released: list[T] = []
        while self._heap and self._heap[0][0] < bound:
            released.append(heapq.heappop(self._heap)[3])
        return released


@dataclass(frozen=True)
class FeatureRow:
    """A transaction's feature vector in `FEATURES` order, ready to score."""

    tx_id: int
    event_ts: datetime
    card_id: int
    terminal_id: int
    is_fraud: int
    t_produce_ns: int
    features: np.ndarray
    timings: dict[str, int]


class CardStage:
    """Card state for one tx.term partition. Feed it every message that partition receives."""

    def __init__(
        self,
        sources: int,
        *,
        clock: Clock = time.time_ns,
        state_factory: Callable[[], CardState] = CardState,
    ) -> None:
        self._buffer: ReorderBuffer[tuple[TermTx, int]] = ReorderBuffer(sources)
        self._clock = clock
        self._cards: defaultdict[int, CardState] = defaultdict(state_factory)

    def process(self, message: TermTx | Watermark) -> list[FeatureRow | DeadLetter]:
        t_in = self._clock()
        try:
            if isinstance(message, Watermark):
                released = self._buffer.watermark(message.source, message.ts)
            else:
                released = self._buffer.add(
                    message.source, message.event_ts, message.tx_id, (message, t_in)
                )
        except SourceOrderError as exc:
            return [DeadLetter(stage="card", reason=str(exc), payload=message.model_dump_json())]
        return [self._features(tx, t_arrive) for tx, t_arrive in released]

    def _features(self, message: TermTx, t_arrive: int) -> FeatureRow:
        tx = Tx(
            message.event_ts,
            message.card_id,
            message.terminal_id,
            message.amount,
            message.x,
            message.y,
        )
        card = self._cards[message.card_id]
        values = derive(tx, card.read(message.event_ts), message.term_feats)
        card.update(tx)
        return FeatureRow(
            tx_id=message.tx_id,
            event_ts=message.event_ts,
            card_id=message.card_id,
            terminal_id=message.terminal_id,
            is_fraud=message.is_fraud,
            t_produce_ns=message.t_produce_ns,
            features=np.array([values[name] for name in FEATURE_NAMES], dtype=np.float64),
            timings={**message.timings, "t_stageB_in": t_arrive, "t_feat": self._clock()},
        )
