"""Alert-threshold rules and review-cost sweep on the E4-selected LightGBM (E4b).

Each rule picks a threshold on validation data under costs scaled by a ratio, then reports
test savings at that threshold. Scaling multiplies only the per-alert costs (review and
false-positive friction); fraud amounts and the fraud fixed fee are unchanged.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import yaml

from fraudstream.config import CostConfig
from fraudstream.evaluation.cost import net_savings
from fraudstream.evaluation.metrics import pick_threshold
from fraudstream.experiments.protocol import load_protocol
from fraudstream.models.calibrate import PlattCalibrator
from fraudstream.models.registry import load_model_config
from fraudstream.models.train import fit_lgbm_variant

RULES = ("half", "f1", "min_loss", "capacity")
COST_RATIOS = (0.25, 0.5, 1.0, 2.0, 4.0)
CAPACITY = 0.001  # alert budget as a fraction of validation rows (capacity rule)


def scaled_costs(costs: CostConfig, ratio: float) -> CostConfig:
    """Scale per-alert costs by ratio; fraud amounts and fraud_fixed_fee_usd are unchanged."""
    return costs.model_copy(update={"review_usd": costs.review_usd * ratio,
                                    "fp_friction_usd": costs.fp_friction_usd * ratio})


def choose_threshold(rule: str, y_val, val_score, amount_val, costs: CostConfig) -> float:
    """Threshold for one rule: 'half' is fixed at 0.5; the others are fit on validation."""
    if rule == "half":
        return 0.5
    return pick_threshold(rule, y_val, val_score, amount_val, costs, capacity=CAPACITY)


def run(data: Path, out: Path, *, imbalance: str, test_start: int = 20, gap: int = 2,
        validation: int = 4) -> None:
    p = load_protocol(data, test_start=test_start, gap=gap, validation=validation)
    split = p.split
    scorer = fit_lgbm_variant(imbalance, p.X[split.train], p.visible[split.train],
                              load_model_config("lgbm"))
    y_val = p.visible[split.validation]
    y_test = p.truth[split.test]
    val_raw = scorer(p.X[split.validation])
    test_raw = scorer(p.X[split.test])
    calibrator = PlattCalibrator().fit(val_raw, y_val)
    val_score = calibrator.predict(val_raw)
    test_score = calibrator.predict(test_raw)
    results = []
    for rule in RULES:
        for ratio in COST_RATIOS:
            costs = scaled_costs(p.costs, ratio)
            threshold = choose_threshold(rule, y_val, val_score,
                                         p.amount[split.validation], costs)
            results.append({
                "rule": rule,
                "cost_ratio": ratio,
                "threshold": float(threshold) if np.isfinite(threshold) else None,
                "alerts": int((test_score >= threshold).sum()),
                "net_savings_usd": net_savings(y_test, test_score, p.amount[split.test],
                                               threshold, costs),
            })
    metrics = {"lgbm_imbalance": imbalance, "calibration": "platt", "results": results}
    config = {"test_start_day": test_start, "gap_days": gap, "validation_days": validation,
              "seed": 0, "cost_ratios": list(COST_RATIOS),
              "scaled_cost_fields": ["review_usd", "fp_friction_usd"], "capacity": CAPACITY,
              "base_cost": p.costs.model_dump(), "train_rows": len(split.train),
              "validation_rows": len(split.validation), "test_rows": len(split.test),
              "training_labels_as_of": p.as_of.isoformat()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (out / "git_sha.txt").write_text(sha + "\n")
    print(f"wrote {len(results)} rule x cost-ratio results to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imbalance-from", type=Path, required=True,
                        help="E4 metrics.json; its selected_variant fixes the LightGBM model")
    args = parser.parse_args()
    imbalance = json.loads(args.imbalance_from.read_text())["selected_variant"]
    run(args.data, args.out, imbalance=imbalance)


if __name__ == "__main__":
    main()
