"""Compare LightGBM imbalance strategies on the static temporal protocol (E4)."""

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
from fraudstream.models.registry import load_model_config
from fraudstream.models.train import IMBALANCE_VARIANTS, fit_lgbm_variant

SELECTION_RULE = "max validation average precision"


def run(data: Path, out: Path, *, test_start: int = 20, gap: int = 2, validation: int = 4) -> None:
    p = load_protocol(data, test_start=test_start, gap=gap, validation=validation)
    split = p.split
    y_val = p.visible[split.validation]
    y_test = p.truth[split.test]
    variants = {}
    lgbm_config = load_model_config("lgbm")
    for variant in IMBALANCE_VARIANTS:
        scorer = fit_lgbm_variant(variant, p.X[split.train], p.visible[split.train], lgbm_config)
        val_prob = scorer(p.X[split.validation])
        test_prob = scorer(p.X[split.test])
        calibrator = PlattCalibrator().fit(val_prob, y_val)
        val_cal = calibrator.predict(val_prob)
        test_cal = calibrator.predict(test_prob)
        threshold = pick_threshold("min_loss", y_val, val_cal, p.amount[split.validation], p.costs)
        calibrated = calibration_metrics(y_test, test_cal)
        uncalibrated = calibration_metrics(y_test, test_prob)
        variants[variant] = {
            "validation_average_precision": ranking_metrics(y_val, val_cal)["average_precision"],
            **ranking_metrics(y_test, test_cal),
            "brier": calibrated["brier"],
            "ece": calibrated["ece"],
            "uncalibrated_brier": uncalibrated["brier"],
            "uncalibrated_ece": uncalibrated["ece"],
            "daily_top_k_precision": daily_top_k_precision(y_test, test_cal, p.days[split.test]),
            "net_savings_usd": net_savings(y_test, test_cal, p.amount[split.test], threshold, p.costs),
            "threshold": float(threshold) if np.isfinite(threshold) else None,
        }
    selected = max(IMBALANCE_VARIANTS, key=lambda name: variants[name]["validation_average_precision"])
    metrics = {"selected_variant": selected, "selection_rule": SELECTION_RULE, "variants": variants}
    config = {"test_start_day": test_start, "gap_days": gap, "validation_days": validation,
              "seed": 0, "undersample_negative_per_positive": 10,
              "train_rows": len(split.train), "validation_rows": len(split.validation),
              "test_rows": len(split.test), "cost": p.costs.model_dump(),
              "training_labels_as_of": p.as_of.isoformat()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (out / "git_sha.txt").write_text(sha + "\n")
    print(f"selected {selected}; wrote {len(variants)} variants to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.out)


if __name__ == "__main__":
    main()
