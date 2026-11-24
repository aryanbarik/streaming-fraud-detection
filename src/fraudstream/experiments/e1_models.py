"""Train and evaluate initial model baselines on a chronological simulator split."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from fraudstream.evaluation.cost import net_savings
from fraudstream.evaluation.metrics import pick_threshold
from fraudstream.evaluation.quality import (
    calibration_metrics,
    daily_top_k_precision,
    ranking_metrics,
)
from fraudstream.experiments.protocol import load_protocol
from fraudstream.models.calibrate import PlattCalibrator
from fraudstream.models.rules import RulesScorer
from fraudstream.models.train import fit_lgbm_variant, train


def run(data: Path, out: Path, *, imbalance: str, test_start: int = 20, gap: int = 2,
        validation: int = 4) -> None:
    p = load_protocol(data, test_start=test_start, gap=gap, validation=validation)
    split = p.split
    X_train, y_train = p.X[split.train], p.visible[split.train]
    scorers = {
        "rules": RulesScorer(p.feature_names).score,
        "lr": train("lr", X_train, y_train, p.feature_names).score,
        "lgbm": fit_lgbm_variant(imbalance, X_train, y_train),
        "iforest": train("iforest", X_train, y_train, p.feature_names).score,
    }
    y_val = p.visible[split.validation]
    y_test = p.truth[split.test]
    metrics = {}
    for kind, score in scorers.items():
        val_score = score(p.X[split.validation])
        test_score = score(p.X[split.test])
        if kind in {"lr", "lgbm"}:
            calibrator = PlattCalibrator().fit(val_score, y_val)
            val_score = calibrator.predict(val_score)
            test_score = calibrator.predict(test_score)
        threshold = pick_threshold("min_loss", y_val, val_score, p.amount[split.validation], p.costs)
        metrics[kind] = {
            "calibration": "platt" if kind in {"lr", "lgbm"} else "none",
            **ranking_metrics(y_test, test_score),
            "daily_top_k_precision": daily_top_k_precision(y_test, test_score, p.days[split.test]),
            "net_savings_usd": net_savings(y_test, test_score, p.amount[split.test], threshold, p.costs),
            "threshold": float(threshold) if np.isfinite(threshold) else None,
        }
        if kind in {"lr", "lgbm"}:
            metrics[kind].update(calibration_metrics(y_test, test_score))
    config = {"test_start_day": test_start, "gap_days": gap, "validation_days": validation,
              "seed": 0, "lgbm_imbalance": imbalance, "train_rows": len(split.train),
              "validation_rows": len(split.validation), "test_rows": len(split.test),
              "cost": p.costs.model_dump(), "training_labels_as_of": p.as_of.isoformat()}
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
    parser.add_argument("--imbalance-from", type=Path, required=True,
                        help="E4 metrics.json; its selected_variant fixes the LightGBM arm")
    args = parser.parse_args()
    imbalance = json.loads(args.imbalance_from.read_text())["selected_variant"]
    run(args.data, args.out, imbalance=imbalance)


if __name__ == "__main__":
    main()
