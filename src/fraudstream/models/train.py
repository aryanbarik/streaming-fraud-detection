"""Train CPU baselines on explicitly supplied, leakage-safe feature rows."""

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraudstream.config import ModelConfig
from fraudstream.models.calibrate import PlattCalibrator
from fraudstream.models.focal import focal_loss_grad_hess
from fraudstream.models.imbalance import correct_prior, undersample_indices

IMBALANCE_VARIANTS = ("none", "class_weight", "scale_pos_weight", "undersample_prior", "focal")


def _sigmoid(raw) -> np.ndarray:
    # tanh form stays finite for large |raw|; exp(-raw) would overflow.
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(raw, dtype=float)))


@dataclass
class ModelBundle:
    """A trained baseline plus the metadata needed to reproduce its scores.

    raw_score is the uncalibrated model output: a probability for lr and lgbm, and an
    anomaly score (higher means more anomalous) for iforest. score applies the calibration.
    """

    model: object  # lgb.Booster for lgbm; a fitted sklearn estimator otherwise
    kind: str
    feature_list: list[str]
    threshold: float = 0.5
    train_window: str = "unspecified"
    calibration: dict = field(default_factory=lambda: {"method": "none"})

    def raw_score(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.kind == "iforest":
            return -self.model.decision_function(X)
        if self.kind == "lgbm":
            return self.model.predict(X)
        return self.model.predict_proba(X)[:, 1]

    def score(self, X) -> np.ndarray:
        raw = self.raw_score(X)
        method = self.calibration["method"]
        if method == "none":
            return raw
        if method == "platt":
            return _sigmoid(self.calibration["coef"] * raw + self.calibration["intercept"])
        raise ValueError(f"unknown calibration method: {method}")

    def calibrate(self, X_val, y_val) -> "ModelBundle":
        """Fit Platt scaling on held-out validation rows; never pass training rows here."""
        calibrator = PlattCalibrator().fit(self.raw_score(X_val), y_val)
        self.calibration = {"method": "platt",
                            "coef": float(calibrator.model.coef_[0, 0]),
                            "intercept": float(calibrator.model.intercept_[0])}
        return self

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self.kind == "lgbm":
            self.model.save_model(str(directory / "model.txt"))
        else:
            joblib.dump(self.model, directory / "model.joblib")
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        manifest = {"kind": self.kind, "git_sha": sha, "feature_list": self.feature_list,
                    "threshold": self.threshold, "train_window": self.train_window,
                    "calibration": self.calibration}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path) -> "ModelBundle":
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["kind"] == "lgbm":
            model = lgb.Booster(model_file=str(directory / "model.txt"))
        else:
            model = joblib.load(directory / "model.joblib")
        return cls(model, manifest["kind"], manifest["feature_list"], manifest["threshold"],
                   manifest["train_window"], manifest["calibration"])


def _lgbm(config: ModelConfig, seed: int, **overrides) -> lgb.LGBMClassifier:
    params = {"n_estimators": config.n_estimators, "learning_rate": config.learning_rate,
              "num_leaves": config.num_leaves, "random_state": seed, "verbosity": -1, "n_jobs": 1}
    return lgb.LGBMClassifier(**params, **overrides)


def train(config: ModelConfig, X, y, feature_list: list[str], *, seed: int = 0) -> ModelBundle:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or y.shape != (len(X),) or X.shape[1] != len(feature_list):
        raise ValueError("training arrays and feature names must align")
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("features must be finite and labels must be binary")
    y = y.astype(np.int8)
    if config.kind != "iforest" and len(np.unique(y)) != 2:
        raise ValueError("supervised training requires both label classes")
    if config.kind == "lr":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
        model.fit(X, y)
    elif config.kind == "lgbm":
        model = _lgbm(config, seed).fit(X, y).booster_
    else:
        model = IsolationForest(n_estimators=config.n_estimators, random_state=seed,
                                n_jobs=1).fit(X, y)
    return ModelBundle(model, config.kind, list(feature_list))


def _positive_probability(model, X) -> np.ndarray:
    return model.predict_proba(np.asarray(X, dtype=float))[:, 1]


def _focal_objective(y_true, y_pred):
    # LightGBM passes (labels, preds, weight, group) for >2 parameters; keep two.
    return focal_loss_grad_hess(y_true, y_pred)


def fit_lgbm_variant(variant: str, X, y, config: ModelConfig, *,
                     seed: int = 0) -> Callable[[object], np.ndarray]:
    """Fit LightGBM under one imbalance strategy and return a probability-scale scorer.

    Undersampled models are prior-corrected to the original class balance; focal
    scores are sigmoid(raw margin). Callers calibrate on held-out validation data.
    """
    if config.kind != "lgbm":
        raise ValueError("imbalance variants are defined for lgbm configs only")
    if variant not in IMBALANCE_VARIANTS:
        raise ValueError(f"unknown imbalance variant: {variant}")
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or y.shape != (len(X),):
        raise ValueError("training arrays must be aligned")
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("training features must be finite and labels binary")
    y = y.astype(np.int8)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("supervised training requires both label classes")
    if variant == "undersample_prior":
        chosen, retention = undersample_indices(y, negative_per_positive=10, seed=seed)
        model = _lgbm(config, seed).fit(X[chosen], y[chosen])
        return lambda Z: correct_prior(_positive_probability(model, Z), retention)
    if variant == "focal":
        model = _lgbm(config, seed, objective=_focal_objective).fit(X, y)
        booster = model.booster_
        return lambda Z: _sigmoid(booster.predict(np.asarray(Z, dtype=float), raw_score=True))
    overrides = {
        "none": {},
        "class_weight": {"class_weight": "balanced"},
        "scale_pos_weight": {"scale_pos_weight": negatives / positives},
    }[variant]
    model = _lgbm(config, seed, **overrides).fit(X, y)
    return lambda Z: _positive_probability(model, Z)
