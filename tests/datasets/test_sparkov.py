import pandas as pd
import pyarrow.parquet as pq
import pytest

from fraudstream.config import LabelConfig
from fraudstream.datasets.sparkov import convert
from fraudstream.timeline import ORIGIN

ROWS = pd.DataFrame({
    "cc_num": [111, 222, 111, 111],
    "merchant": ["m1", "m2", "m2", "m1"],
    "amt": [10.0, 20.0, 30.0, 40.0],
    "unix_time": [1_000_100, 1_000_000, 1_000_300, 1_000_200],
    "merch_lat": [40.0, 41.0, 42.0, 40.5],
    "merch_long": [-70.0, -71.0, -72.0, -70.5],
    "is_fraud": [1, 0, 1, 0],
})


def _write_raw(raw, rows):
    raw.mkdir(parents=True, exist_ok=True)
    rows.iloc[:2].to_csv(raw / "fraudTrain.csv", index=False)
    rows.iloc[2:].to_csv(raw / "fraudTest.csv", index=False)
    return raw


def test_transactions_are_sorted_across_files_and_start_at_origin(tmp_path):
    out = tmp_path / "out"
    convert(_write_raw(tmp_path / "raw", ROWS), out, seed=0, labels=LabelConfig(p_report=1.0))
    tx = pq.read_table(out / "transactions.parquet").to_pandas()
    assert tx["event_ts"].iloc[0] == ORIGIN
    assert list(tx["tx_id"]) == [0, 1, 2, 3]
    assert list((tx["event_ts"] - ORIGIN).dt.total_seconds()) == [0, 100, 200, 300]
    assert list(tx["amount"]) == [20.0, 10.0, 40.0, 30.0]
    assert tx["card_id"].iloc[0] != tx["card_id"].iloc[1]
    assert tx["card_id"].iloc[1] == tx["card_id"].iloc[2] == tx["card_id"].iloc[3]
    assert tx[["x", "y"]].min().min() >= 0 and tx[["x", "y"]].max().max() <= 1


def test_labels_cover_frauds_and_arrive_after_the_event(tmp_path):
    out = tmp_path / "out"
    convert(_write_raw(tmp_path / "raw", ROWS), out, seed=0, labels=LabelConfig(p_report=1.0))
    tx = pq.read_table(out / "transactions.parquet").to_pandas()
    labels = pq.read_table(out / "labels.parquet").to_pandas()
    assert set(labels["tx_id"]) == set(tx.loc[tx["is_fraud"] == 1, "tx_id"])
    merged = labels.merge(tx[["tx_id", "event_ts"]], on="tx_id")
    assert (merged["available_ts"] > merged["event_ts"]).all()


def test_missing_required_column_is_named(tmp_path):
    raw = _write_raw(tmp_path / "raw", ROWS.drop(columns=["merch_lat"]))
    with pytest.raises(ValueError, match="merch_lat"):
        convert(raw, tmp_path / "out", seed=0, labels=LabelConfig())
