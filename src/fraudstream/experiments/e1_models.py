"""Train and evaluate initial model baselines on a chronological simulator split."""

import argparse
import json
import subprocess
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sklearn.metrics import average_precision_score

from fraudstream.config import CostConfig
from fraudstream.evaluation.cost import net_savings
from fraudstream.evaluation.labelview import LabelView
from fraudstream.evaluation.metrics import pick_threshold
from fraudstream.evaluation.splits import temporal_split
from fraudstream.features.spec import FEATURES
from fraudstream.models.train import train


def run(data: Path, out: Path, *, test_start: int = 20, gap: int = 2, validation: int = 4) -> None:
    transactions = pq.read_table(data / "transactions.parquet").to_pandas()
    feature_names = [feature.name for feature in FEATURES]
    features = pq.read_table(data / "features.parquet", columns=["tx_id", *feature_names]).to_pandas()
    labels = pq.read_table(data / "labels.parquet", columns=["tx_id", "available_ts"]).to_pandas()
    frame = transactions.merge(features, on="tx_id", validate="one_to_one")
    frame = frame.merge(labels, on="tx_id", how="left", validate="one_to_one")
    origin = pd.Timestamp("2025-01-01")
    days = (frame["event_ts"] - origin).dt.days.to_numpy()
    split = temporal_split(days, test_start, gap, validation, test_days=30)
    if not all(len(indices) for indices in (split.train, split.validation, split.test)):
        raise ValueError("train, validation and test splits must all be nonempty")
    as_of = (origin + timedelta(days=test_start)).to_pydatetime()
    visible = LabelView(pa.Table.from_pandas(frame[["event_ts", "available_ts"]]), as_of).labels()
    X = frame[feature_names].to_numpy(dtype=float)
    amount = frame["amount"].to_numpy()
    truth = frame["is_fraud"].to_numpy(dtype=int)
    costs = CostConfig()
    metrics = {}
    for kind in ("lr", "lgbm", "iforest"):
        bundle = train(kind, X[split.train], visible[split.train], feature_names)
        val_score = bundle.score(X[split.validation])
        threshold = pick_threshold("min_loss", visible[split.validation], val_score,
                                   amount[split.validation], costs)
        test_score = bundle.score(X[split.test])
        metrics[kind] = {
            "average_precision": float(average_precision_score(truth[split.test], test_score)),
            "net_savings_usd": net_savings(truth[split.test], test_score,
                                           amount[split.test], threshold, costs),
            "threshold": float(threshold) if np.isfinite(threshold) else None,
        }
    config = {"test_start_day": test_start, "gap_days": gap, "validation_days": validation,
              "seed": 0, "train_rows": len(split.train), "validation_rows": len(split.validation),
              "test_rows": len(split.test), "cost": costs.model_dump(),
              "training_labels_as_of": as_of.isoformat()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (out / "git_sha.txt").write_text(sha + "\n")
    print(f"wrote {len(metrics)} model evaluations to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out)


if __name__ == "__main__":
    main()
