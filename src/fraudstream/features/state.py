"""Pre-update event-time state shared by feature consumers."""

import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class Tx:
    event_ts: datetime
    card_id: int
    terminal_id: int
    amount: float
    x: float
    y: float


def _check_time(previous: datetime | None, current: datetime) -> None:
    if previous is not None and current < previous:
        raise ValueError("events must be processed in nondecreasing availability order")


class CardState:
    """Sliding card activity; reads never include the transaction being scored."""

    def __init__(self) -> None:
        self.events: deque[Tx] = deque()
        self.last_ts: datetime | None = None
        self.last_xy: tuple[float, float] | None = None
        self.clock: datetime | None = None

    def read(self, as_of: datetime) -> dict:
        _check_time(self.clock, as_of)
        self.clock = as_of
        while self.events and self.events[0].event_ts <= as_of - timedelta(days=7):
            self.events.popleft()
        recent_1h = [tx for tx in self.events if tx.event_ts > as_of - timedelta(hours=1)]
        recent_24h = [tx for tx in self.events if tx.event_ts > as_of - timedelta(days=1)]
        amounts = [tx.amount for tx in self.events]
        mean = sum(amounts) / len(amounts) if amounts else 0.0
        variance = sum((amount - mean) ** 2 for amount in amounts) / len(amounts) if amounts else 0.0
        return {
            "card_count_1h": len(recent_1h),
            "card_count_24h": len(recent_24h),
            "card_count_7d": len(self.events),
            "card_amount_mean_7d": mean,
            "card_amount_std_7d": math.sqrt(variance),
            "card_distinct_terminals_7d": len({tx.terminal_id for tx in self.events}),
            "card_last_ts": self.last_ts,
            "card_last_xy": self.last_xy,
        }

    def update(self, tx: Tx) -> None:
        _check_time(self.clock, tx.event_ts)
        self.clock = tx.event_ts
        self.events.append(tx)
        self.last_ts = tx.event_ts
        self.last_xy = (tx.x, tx.y)


class TerminalState:
    """Terminal activity and fraud reports visible as of the read time."""

    def __init__(self) -> None:
        self.transactions: deque[datetime] = deque()
        self.labels: deque[datetime] = deque()
        self.clock: datetime | None = None

    def read(self, as_of: datetime) -> dict:
        _check_time(self.clock, as_of)
        self.clock = as_of
        while self.transactions and self.transactions[0] <= as_of - timedelta(days=7):
            self.transactions.popleft()
        while self.labels and self.labels[0] <= as_of - timedelta(days=7):
            self.labels.popleft()
        total = len(self.transactions)
        return {
            "terminal_count_1h": sum(ts > as_of - timedelta(hours=1) for ts in self.transactions),
            "terminal_count_24h": sum(ts > as_of - timedelta(days=1) for ts in self.transactions),
            "terminal_count_7d": total,
            "terminal_fraud_reports_7d": len(self.labels),
            "terminal_report_rate_7d": len(self.labels) / total if total else 0.0,
        }

    def update_tx(self, tx: Tx) -> None:
        _check_time(self.clock, tx.event_ts)
        self.clock = tx.event_ts
        self.transactions.append(tx.event_ts)

    def apply_label(self, available_ts: datetime) -> None:
        _check_time(self.clock, available_ts)
        self.clock = available_ts
        self.labels.append(available_ts)
