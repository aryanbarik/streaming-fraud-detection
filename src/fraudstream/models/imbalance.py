"""Training-only class undersampling and posterior prior correction.

The correction assumes selection is independent of features conditional on
class and that every positive is retained. See Dal Pozzolo et al., 2015,
https://dalpozz.github.io/static/pdf/SSCI_calib_final_noCC.pdf, section III.
"""

import numpy as np


def undersample_indices(y, *, negative_per_positive: int = 10, seed: int = 0):
    """Return selected row indices and the retained negative fraction."""
    y = np.asarray(y)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be a one-dimensional binary array")
    if negative_per_positive < 1:
        raise ValueError("negative_per_positive must be positive")
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        raise ValueError("both classes are required")
    count = min(len(negative), negative_per_positive * len(positive))
    chosen = np.random.default_rng(seed).choice(negative, size=count, replace=False)
    return np.sort(np.concatenate([positive, chosen])), count / len(negative)


def correct_prior(sampled_probability, negative_retention: float):
    """Undo known random negative sampling: beta*p_s/(1-p_s+beta*p_s)."""
    p = np.asarray(sampled_probability, dtype=float)
    if not 0 < negative_retention <= 1:
        raise ValueError("negative_retention must be in (0, 1]")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return negative_retention * p / (1 - p + negative_retention * p)
