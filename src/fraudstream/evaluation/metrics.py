"""Operating-point selection on a validation set."""

from typing import Literal

import numpy as np

from fraudstream.config import CostConfig


def pick_threshold(
    rule: Literal["min_loss", "f1", "capacity"],
    y,
    score,
    amount,
    cost: CostConfig,
    *,
    capacity: float = 0.001,
) -> float:
    """Select an inclusive score cutoff without splitting equal scores.

    Capacity is the maximum fraction of validation rows reviewed. Infinite
    threshold represents reviewing nobody when no score group fits.
    """
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    amount = np.asarray(amount, dtype=float)
    if y.ndim != 1 or y.shape != score.shape or y.shape != amount.shape:
        raise ValueError("labels, scores and amounts must be one-dimensional and aligned")
    if not np.isfinite(score).all() or not np.isfinite(amount).all():
        raise ValueError("scores and amounts must be finite")
    if not 0 <= capacity <= 1:
        raise ValueError("capacity must be in [0, 1]")
    if rule not in {"min_loss", "f1", "capacity"}:
        raise ValueError(f"unknown threshold rule: {rule}")
    if not len(y):
        raise ValueError("validation set must not be empty")
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    fraud = y[order]
    values = amount[order]
    ends = np.r_[np.flatnonzero(sorted_score[:-1] != sorted_score[1:]), len(y) - 1]
    count = ends + 1
    if rule == "capacity":
        fits = count <= int(np.floor(capacity * len(y)))
        return float(sorted_score[ends[fits][-1]]) if fits.any() else float("inf")
    if rule == "min_loss":
        savings = np.cumsum(
            fraud * (values + cost.fraud_fixed_fee_usd)
            - cost.review_usd - (~fraud) * cost.fp_friction_usd
        )[ends]
        best = int(np.argmax(savings))
        return float(sorted_score[ends[best]]) if savings[best] > 0 else float("inf")
    true_positives = np.cumsum(fraud)[ends]
    denom = count + fraud.sum()
    f1 = 2 * true_positives / denom
    best = int(np.argmax(f1))
    return float(sorted_score[ends[best]]) if f1[best] > 0 else float("inf")
