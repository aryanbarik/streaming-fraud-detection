"""Platt calibration fitted only on a held-out validation partition."""

import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(random_state=0)

    def fit(self, scores, y) -> "PlattCalibrator":
        values = np.asarray(scores, dtype=float)
        labels = np.asarray(y)
        if values.ndim != 1 or labels.shape != values.shape or not np.isfinite(values).all():
            raise ValueError("scores and labels must be aligned finite vectors")
        if not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
            raise ValueError("calibration requires both binary classes")
        self.model.fit(values.reshape(-1, 1), labels)
        return self

    def predict(self, scores) -> np.ndarray:
        values = np.asarray(scores, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("scores must be a finite vector")
        return self.model.predict_proba(values.reshape(-1, 1))[:, 1]
