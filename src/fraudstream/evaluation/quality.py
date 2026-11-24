"""Offline ranking, operating-point and calibration metrics."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_curve


def recall_at_fpr(y, score, max_fpr: float = 0.01) -> float:
    y = np.asarray(y)
    score = np.asarray(score)
    if len(np.unique(y)) != 2:
        raise ValueError("recall at FPR requires both classes")
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be between zero and one")
    fpr, tpr, _ = roc_curve(y, score, drop_intermediate=False)
    return float(tpr[fpr <= max_fpr].max())


def calibration_metrics(y, probability, bins: int = 10) -> dict[str, float]:
    y = np.asarray(y)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("labels and probabilities must be aligned nonempty vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("labels must be binary and probabilities must lie in [0, 1]")
    if bins <= 0:
        raise ValueError("bins must be positive")
    indices = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for index in range(bins):
        selected = indices == index
        if selected.any():
            ece += selected.mean() * abs(p[selected].mean() - y[selected].mean())
    return {"brier": float(np.mean((p - y) ** 2)), "ece": float(ece)}


def daily_top_k_precision(y, score, day, fraction: float = 0.001) -> float:
    y = np.asarray(y)
    score = np.asarray(score)
    day = np.asarray(day)
    if y.ndim != 1 or y.shape != score.shape or y.shape != day.shape or not len(y):
        raise ValueError("labels, scores and days must be aligned nonempty vectors")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    precision = []
    for value in np.unique(day):
        selected = np.flatnonzero(day == value)
        k = max(1, int(np.ceil(len(selected) * fraction)))
        order = np.argsort(-score[selected], kind="stable")[:k]
        precision.append(y[selected[order]].mean())
    return float(np.mean(precision))


def ranking_metrics(y, score) -> dict[str, float]:
    return {"average_precision": float(average_precision_score(y, score)),
            "recall_at_fpr_1pct": recall_at_fpr(y, score)}
