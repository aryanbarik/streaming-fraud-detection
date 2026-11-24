"""Train CPU baselines on explicitly supplied, leakage-safe feature rows."""

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelBundle:
    model: object
    kind: str
    feature_list: list[str]
    threshold: float = 0.5
    train_window: str = "unspecified"

    def score(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.kind == "iforest":
            return -self.model.decision_function(X)
        return self.model.predict_proba(X)[:, 1]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory / "model.joblib")
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        manifest = {"kind": self.kind, "git_sha": sha, "feature_list": self.feature_list,
                    "threshold": self.threshold, "train_window": self.train_window,
                    "calibration": "none"}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path) -> "ModelBundle":
        manifest = json.loads((directory / "manifest.json").read_text())
        return cls(joblib.load(directory / "model.joblib"), manifest["kind"],
                   manifest["feature_list"], manifest["threshold"], manifest["train_window"])


def train(model: str, X, y, feature_list: list[str], *, seed: int = 0) -> ModelBundle:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=np.int8)
    if X.ndim != 2 or y.shape != (len(X),) or X.shape[1] != len(feature_list):
        raise ValueError("training arrays and feature names must align")
    if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("features must be finite and labels must be binary")
    if model != "iforest" and len(np.unique(y)) != 2:
        raise ValueError("supervised training requires both label classes")
    if model == "lr":
        estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    elif model == "lgbm":
        estimator = lgb.LGBMClassifier(n_estimators=100, random_state=seed, verbosity=-1, n_jobs=1)
    elif model == "iforest":
        estimator = IsolationForest(n_estimators=100, random_state=seed, n_jobs=1)
    else:
        raise ValueError(f"unknown model: {model}")
    estimator.fit(X, y)
    return ModelBundle(estimator, model, list(feature_list))
