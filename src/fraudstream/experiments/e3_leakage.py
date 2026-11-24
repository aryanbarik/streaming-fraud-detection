"""Compare temporal evaluation against intentionally leaky alternatives."""

import argparse
import json
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from fraudstream.config import CostConfig
from fraudstream.evaluation.cost import net_savings
from fraudstream.evaluation.labelview import LabelView
from fraudstream.evaluation.leakage import future_terminal_features, immediate_fraud_labels
from fraudstream.evaluation.metrics import pick_threshold
from fraudstream.evaluation.splits import temporal_split
from fraudstream.features.backfill import backfill
from fraudstream.features.spec import FEATURES
from fraudstream.models.registry import load_model_config
from fraudstream.models.train import train
from fraudstream.timeline import ORIGIN


def run(data: Path, out: Path, *, test_start: int = 20, gap: int = 2, validation: int = 4) -> None:
    tx_table = pq.read_table(data / "transactions.parquet")
    tx = tx_table.to_pandas()
    features = pq.read_table(data / "features.parquet").to_pandas()
    reports = pq.read_table(data / "labels.parquet", columns=["tx_id", "available_ts"]).to_pandas()
    names = [spec.name for spec in FEATURES]
    terminal_names = [spec.name for spec in FEATURES if spec.entity == "terminal"]
    frame = tx.merge(features[["tx_id", *names]], on="tx_id", validate="one_to_one")
    frame = frame.merge(reports, on="tx_id", how="left", validate="one_to_one")
    days = (frame["event_ts"] - ORIGIN).dt.days.to_numpy()
    split = temporal_split(days, test_start, gap, validation)
    if not all(len(part) for part in (split.train, split.validation, split.test)):
        raise ValueError("train, validation and test splits must all be nonempty")
    cutoff = (ORIGIN + timedelta(days=test_start)).to_pydatetime()
    visible = LabelView(pa.Table.from_pandas(frame[["event_ts", "available_ts"]]), cutoff).labels()
    truth = frame["is_fraud"].to_numpy(dtype=int)
    amounts = frame["amount"].to_numpy()
    costs = CostConfig()
    with tempfile.TemporaryDirectory(prefix="fraudstream-leakage-") as temp:
        path = Path(temp) / "immediate.parquet"
        backfill(tx_table, immediate_fraud_labels(tx_table), path)
        immediate = pq.read_table(path, columns=["tx_id", *names]).to_pandas()
    variants = {
        "correct": frame[names].to_numpy(dtype=float),
        "immediate_label_features": frame[["tx_id"]].merge(
            immediate, on="tx_id", validate="one_to_one"
        )[names].to_numpy(dtype=float),
        "non_pit_terminal_features": future_terminal_features(frame, terminal_names)[names].to_numpy(dtype=float),
    }
    results = {}
    for name, X in variants.items():
        bundle = train(load_model_config("lgbm"), X[split.train], visible[split.train], names)
        val_score = bundle.score(X[split.validation])
        threshold = pick_threshold("min_loss", visible[split.validation], val_score,
                                   amounts[split.validation], costs)
        score = bundle.score(X[split.test])
        results[name] = {
            "average_precision": float(average_precision_score(truth[split.test], score)),
            "net_savings_per_1000_usd": net_savings(truth[split.test], score,
                                                     amounts[split.test], threshold, costs)
                                         * 1000 / len(split.test),
        }
    # This deliberately leaky control trains on future rows and fully observed labels.
    X = variants["correct"]
    oof_score = np.empty(len(frame))
    alerted = np.zeros(len(frame), dtype=np.int8)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for outer_train, outer_test in cv.split(X, truth):
        training, holdout = train_test_split(outer_train, test_size=.2, stratify=truth[outer_train],
                                            random_state=0)
        bundle = train(load_model_config("lgbm"), X[training], truth[training], names)
        threshold = pick_threshold("min_loss", truth[holdout], bundle.score(X[holdout]),
                                   amounts[holdout], costs)
        score = bundle.score(X[outer_test])
        oof_score[outer_test] = score
        alerted[outer_test] = score >= threshold
    results["random_stratified_kfold"] = {
        "average_precision": float(average_precision_score(truth, oof_score)),
        "net_savings_per_1000_usd": net_savings(truth, alerted, amounts, .5, costs) * 1000 / len(truth),
    }
    baseline = results["correct"]
    for value in results.values():
        value["average_precision_delta_vs_correct"] = value["average_precision"] - baseline["average_precision"]
        value["savings_per_1000_delta_vs_correct_usd"] = (
            value["net_savings_per_1000_usd"] - baseline["net_savings_per_1000_usd"]
        )
    out.mkdir(parents=True, exist_ok=True)
    config = {"test_start_day": test_start, "gap_days": gap, "validation_days": validation,
              "seed": 0, "random_cv_folds": 5, "random_cv_labels": "fully observed ground truth",
              "note": "random CV covers all days; temporal controls use the held-out test window"}
    (out / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (out / "git_sha.txt").write_text(sha + "\n")
    print(f"wrote {len(results)} leakage protocols to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out)


if __name__ == "__main__":
    main()
