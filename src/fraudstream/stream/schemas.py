"""Wire formats for every topic. Each message is JSON with a `type` discriminator."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class TxEvent(BaseModel):
    """A transaction on events.raw. The load generator sets `t_produce_ns` at send time."""

    type: Literal["tx"] = "tx"
    source: int = Field(ge=0)
    tx_id: int
    event_ts: datetime
    card_id: int
    terminal_id: int
    amount: float
    x: float
    y: float
    is_fraud: int
    t_produce_ns: int = 0


class LabelEvent(BaseModel):
    """A fraud label on events.raw, visible to transactions from `available_ts` onward."""

    type: Literal["label"] = "label"
    source: int = Field(ge=0)
    tx_id: int
    terminal_id: int
    available_ts: datetime


class Watermark(BaseModel):
    """No later message from `source` has event time before `ts`. `ts=None` ends the source."""

    type: Literal["watermark"] = "watermark"
    source: int = Field(ge=0)
    ts: datetime | None = None


RawEvent = Annotated[TxEvent | LabelEvent | Watermark, Field(discriminator="type")]


class TermTx(BaseModel):
    """A transaction on tx.term, carrying the terminal features read at its source."""

    type: Literal["tx"] = "tx"
    source: int = Field(ge=0)
    tx_id: int
    event_ts: datetime
    card_id: int
    terminal_id: int
    amount: float
    x: float
    y: float
    is_fraud: int
    term_feats: dict[str, float]
    t_produce_ns: int
    timings: dict[str, int] = Field(default_factory=dict)


TermMessage = Annotated[TermTx | Watermark, Field(discriminator="type")]


class ScoredTx(BaseModel):
    """A scored transaction on tx.scored."""

    tx_id: int
    event_ts: datetime
    card_id: int
    terminal_id: int
    score: float
    alert: bool
    is_fraud: int
    t_produce_ns: int
    timings: dict[str, int]


class DeadLetter(BaseModel):
    """A message a stage rejected, and why."""

    stage: str
    reason: str
    payload: str


_RAW = TypeAdapter(RawEvent)
_TERM = TypeAdapter(TermMessage)


def encode(message: BaseModel) -> bytes:
    return message.model_dump_json().encode()


def decode_raw(data: bytes) -> TxEvent | LabelEvent | Watermark:
    return _RAW.validate_json(data)


def decode_term(data: bytes) -> TermTx | Watermark:
    return _TERM.validate_json(data)
