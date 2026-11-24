"""Stage A: terminal state for one events.raw partition, which carries one source.

Within a source, events arrive in event-time order. A label is applied before a transaction at
the same timestamp, as in `features.backfill.availability_timeline`. Terminal features are read
before the transaction updates the terminal, as in the backfill.
"""

import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime

from fraudstream.features.state import TerminalState, Tx
from fraudstream.stream.schemas import DeadLetter, LabelEvent, TermTx, TxEvent, Watermark

Clock = Callable[[], int]
RawMessage = TxEvent | LabelEvent | Watermark


class TerminalStage:
    def __init__(
        self,
        source: int,
        *,
        clock: Clock = time.time_ns,
        state_factory: Callable[[], TerminalState] = TerminalState,
    ) -> None:
        self.source = source
        self._clock = clock
        self._terminals: defaultdict[int, TerminalState] = defaultdict(state_factory)
        self._last_ts: datetime | None = None
        self._ended = False

    def process(self, event: RawMessage) -> list[TermTx | Watermark | DeadLetter]:
        """Apply one events.raw message and return the messages to forward."""
        t_in = self._clock()
        if event.source != self.source:
            return [self._reject(event, f"event for source {event.source} on {self.source}")]
        if isinstance(event, Watermark):
            if event.ts is not None:
                return [self._reject(event, "watermark with a timestamp on events.raw")]
            self._ended = True
            return [Watermark(source=self.source)]
        if self._ended:
            return [self._reject(event, "event after end of stream")]
        ts = event.event_ts if isinstance(event, TxEvent) else event.available_ts
        if self._last_ts is not None and ts < self._last_ts:
            return [self._reject(event, f"source regressed from {self._last_ts} to {ts}")]
        self._last_ts = ts
        terminal = self._terminals[event.terminal_id]
        if isinstance(event, LabelEvent):
            terminal.apply_label(ts)
            return []
        tx = Tx(ts, event.card_id, event.terminal_id, event.amount, event.x, event.y)
        feats = terminal.read(ts)
        terminal.update_tx(tx)
        return [
            TermTx(
                source=self.source,
                tx_id=event.tx_id,
                event_ts=ts,
                card_id=event.card_id,
                terminal_id=event.terminal_id,
                amount=event.amount,
                x=event.x,
                y=event.y,
                is_fraud=event.is_fraud,
                term_feats=feats,
                t_produce_ns=event.t_produce_ns,
                timings={"t_stageA_in": t_in, "t_stageA_out": self._clock()},
            )
        ]

    def heartbeat(self) -> list[Watermark]:
        """Promise that later events from this source are at or after the last one processed.

        Must not run concurrently with `process`; the caller serializes the two.
        """
        if self._last_ts is None or self._ended:
            return []
        return [Watermark(source=self.source, ts=self._last_ts)]

    def _reject(self, event: RawMessage, reason: str) -> DeadLetter:
        return DeadLetter(stage="terminal", reason=reason, payload=event.model_dump_json())
