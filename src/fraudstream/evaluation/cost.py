"""Explicit review, friction and missed-fraud costs."""

import numpy as np

from fraudstream.config import CostConfig


def net_savings(y, score, amount, thr: float, cost: CostConfig) -> float:
    """Return savings against approving every transaction without review."""
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score)
    amount = np.asarray(amount)
    if y.shape != score.shape or y.shape != amount.shape:
        raise ValueError("labels, scores and amounts must have identical shapes")
    alert = score >= thr
    baseline = amount[y].sum() + y.sum() * cost.fraud_fixed_fee_usd
    missed = y & ~alert
    loss = amount[missed].sum() + missed.sum() * cost.fraud_fixed_fee_usd
    loss += alert.sum() * cost.review_usd
    loss += (alert & ~y).sum() * cost.fp_friction_usd
    return float(baseline - loss)
