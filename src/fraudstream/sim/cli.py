"""Generate transaction and delayed-label Parquet files."""

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from fraudstream.config import SimConfig, load_config
from fraudstream.sim.generator import generate
from fraudstream.sim.labels import make_label_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, SimConfig)
    tx = generate(cfg, args.seed)
    labels = make_label_events(tx, cfg.labels, args.seed + 1)
    args.out.mkdir(parents=True, exist_ok=True)
    pq.write_table(tx, args.out / "transactions.parquet")
    pq.write_table(labels, args.out / "labels.parquet")
    print(f"wrote {tx.num_rows} transactions and {labels.num_rows} labels to {args.out}")


if __name__ == "__main__":
    main()
