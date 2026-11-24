"""Generate customer-report labels with explicit availability times."""

import numpy as np
import pyarrow as pa

from fraudstream.config import LabelConfig


def make_label_events(tx: pa.Table, cfg: LabelConfig, seed: int) -> pa.Table:
    rng = np.random.default_rng(seed)
    fraud = tx["is_fraud"].to_numpy() == 1
    report = fraud & (rng.random(tx.num_rows) < cfg.p_report)
    indices = np.flatnonzero(report)
    delay_days = rng.lognormal(np.log(cfg.median_delay_days), cfg.sigma, len(indices))
    event_us = tx["event_ts"].to_numpy()[indices].astype("datetime64[us]")
    delay_us = np.maximum(1, np.rint(delay_days * 86_400_000_000)).astype("timedelta64[us]")
    available = event_us + delay_us
    order = np.argsort(available, kind="stable")
    return pa.table({
        "tx_id": pa.array(tx["tx_id"].to_numpy()[indices][order], type=pa.int64()),
        "terminal_id": pa.array(tx["terminal_id"].to_numpy()[indices][order], type=pa.int32()),
        "available_ts": pa.array(available[order], type=pa.timestamp("us")),
        "kind": pa.array(["fraud"] * len(order), type=pa.string()),
    })
