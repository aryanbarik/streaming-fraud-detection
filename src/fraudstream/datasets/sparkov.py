"""Convert the Kaggle Sparkov credit-card fraud CSVs into simulator-format tables.

Sparkov reports no label delays, so labels come from the simulator's delay model with
LabelConfig defaults (seed + 1, as in the simulator CLI). Event times are shifted so the
first transaction sits at ORIGIN. x and y are merchant longitude and latitude min-max
scaled over the whole file; cardholder coordinates are fixed per card, so they cannot
carry travel features. Columns the feature backfill never reads (mcc, channel, scenario)
are omitted rather than filled with constants. See docs/decisions/0005-sparkov-adapter.md.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fraudstream.config import LabelConfig
from fraudstream.sim.labels import make_label_events
from fraudstream.timeline import ORIGIN

RAW_FILES = ("fraudTrain.csv", "fraudTest.csv")
REQUIRED_COLUMNS = ("unix_time", "cc_num", "merchant", "amt", "merch_lat", "merch_long", "is_fraud")


def load_raw(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for name in RAW_FILES:
        path = raw_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}; download the Sparkov dataset first")
        frame = pd.read_csv(path, usecols=lambda column: column in REQUIRED_COLUMNS)
        missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} lacks required columns: {missing}")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise ValueError("Sparkov CSVs contain no transactions")
    return combined


def _unit_interval(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    low, high = array.min(), array.max()
    if high == low:
        return np.zeros_like(array)
    return (array - low) / (high - low)


def to_transactions(frame: pd.DataFrame) -> pa.Table:
    frame = frame.sort_values("unix_time", kind="stable", ignore_index=True)
    seconds = frame["unix_time"].to_numpy(dtype=np.int64)
    event_ts = ORIGIN + pd.to_timedelta(seconds - seconds[0], unit="s")
    cards, _ = pd.factorize(frame["cc_num"])
    terminals, _ = pd.factorize(frame["merchant"])
    return pa.table({
        "tx_id": pa.array(np.arange(len(frame), dtype=np.int64)),
        "event_ts": pa.array(event_ts.to_numpy().astype("datetime64[us]"),
                             type=pa.timestamp("us")),
        "card_id": pa.array(cards.astype(np.int32)),
        "terminal_id": pa.array(terminals.astype(np.int32)),
        "amount": pa.array(frame["amt"].to_numpy(dtype=np.float32)),
        "x": pa.array(_unit_interval(frame["merch_long"]).astype(np.float32)),
        "y": pa.array(_unit_interval(frame["merch_lat"]).astype(np.float32)),
        "is_fraud": pa.array(frame["is_fraud"].to_numpy(dtype=np.int8)),
    })


def convert(raw_dir: Path, out: Path, *, seed: int, labels: LabelConfig) -> None:
    transactions = to_transactions(load_raw(raw_dir))
    label_events = make_label_events(transactions, labels, seed + 1)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(transactions, out / "transactions.parquet")
    pq.write_table(label_events, out / "labels.parquet")
    print(f"wrote {transactions.num_rows} transactions and {label_events.num_rows} labels "
          f"to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True,
                        help="directory holding fraudTrain.csv and fraudTest.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    convert(args.raw, args.out, seed=args.seed, labels=LabelConfig())


if __name__ == "__main__":
    main()
