import pyarrow.parquet as pq

from fraudstream.config import SimConfig, load_config
from fraudstream.sim.generator import generate


def test_tiny_parquet_is_reproducible_and_fraud_rate_bounded(tmp_path):
    from pathlib import Path

    cfg = load_config(Path("configs/sim_tiny.yaml"), SimConfig)
    first = generate(cfg, 0)
    second = generate(cfg, 0)
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    pq.write_table(first, a)
    pq.write_table(second, b)
    assert a.read_bytes() == b.read_bytes()
    rate = first["is_fraud"].to_numpy().mean()
    assert 0.002 <= rate <= 0.02
