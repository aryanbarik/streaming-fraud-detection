"""Static temporal protocol shared by the fraud experiments (E1, E4)."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fraudstream.config import CostConfig
from fraudstream.evaluation.labelview import LabelView
from fraudstream.evaluation.splits import SplitIdx, temporal_split
from fraudstream.features.spec import FEATURES
from fraudstream.timeline import ORIGIN


@dataclass(frozen=True)
class Protocol:
    feature_names: list[str]
    X: np.ndarray
    amount: np.ndarray
    truth: np.ndarray
    days: np.ndarray
    visible: np.ndarray
    split: SplitIdx
    as_of: datetime
    costs: CostConfig


def load_protocol(data: Path, *, test_start: int = 20, gap: int = 2, validation: int = 4) -> Protocol:
    """Split simulator rows by day; training and validation labels are visible as of test_start."""
    transactions = pq.read_table(data / "transactions.parquet").to_pandas()
    feature_names = [feature.name for feature in FEATURES]
    features = pq.read_table(data / "features.parquet", columns=["tx_id", *feature_names]).to_pandas()
    labels = pq.read_table(data / "labels.parquet", columns=["tx_id", "available_ts"]).to_pandas()
    frame = transactions.merge(features, on="tx_id", validate="one_to_one")
    frame = frame.merge(labels, on="tx_id", how="left", validate="one_to_one")
    days = (frame["event_ts"] - ORIGIN).dt.days.to_numpy()
    split = temporal_split(days, test_start, gap, validation, test_days=30)
    if not all(len(indices) for indices in (split.train, split.validation, split.test)):
        raise ValueError("train, validation and test splits must all be nonempty")
    as_of = (ORIGIN + timedelta(days=test_start)).to_pydatetime()
    visible = LabelView(pa.Table.from_pandas(frame[["event_ts", "available_ts"]]), as_of).labels()
    return Protocol(
        feature_names=feature_names,
        X=frame[feature_names].to_numpy(dtype=float),
        amount=frame["amount"].to_numpy(),
        truth=frame["is_fraud"].to_numpy(dtype=int),
        days=days,
        visible=visible,
        split=split,
        as_of=as_of,
        costs=CostConfig(),
    )
