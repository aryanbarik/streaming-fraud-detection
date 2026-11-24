"""Binary focal-loss derivatives for LightGBM raw logits."""

import numpy as np
from scipy.special import expit


def focal_loss_grad_hess(y, raw_score, *, alpha: float = .25, gamma: float = 2.0):
    """Return first and second derivatives with respect to raw logits.

    LightGBM needs positive curvature, so the Hessian is floored at 1e-6
    where focal loss is locally non-convex.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(expit(np.asarray(raw_score, dtype=float)), 1e-12, 1 - 1e-12)
    q = 1 - p
    pos_grad = alpha * (gamma * p * q ** gamma * np.log(p) - q ** (gamma + 1))
    neg_grad = (1 - alpha) * (p ** (gamma + 1) - gamma * p ** gamma * q * np.log(q))
    pos_hess = alpha * (
        gamma * p * q ** gamma * ((1 - (gamma + 1) * p) * np.log(p) + q)
        + (gamma + 1) * p * q ** (gamma + 1)
    )
    neg_hess = (1 - alpha) * (
        (gamma + 1) * p ** (gamma + 1) * q
        - gamma * p ** gamma * q * ((gamma - (gamma + 1) * p) * np.log(q) - p)
    )
    return np.where(y == 1, pos_grad, neg_grad), np.maximum(
        np.where(y == 1, pos_hess, neg_hess), 1e-6
    )
