"""Chronological static evaluation split."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitIdx:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def temporal_split(days, T: int, G: int, V: int, test_days: int = 30) -> SplitIdx:
    """Return row indices for train, validation and test; gap rows are excluded."""
    if G < 0 or V <= 0 or test_days <= 0:
        raise ValueError("gap must be nonnegative and windows must be positive")
    days = np.asarray(days)
    return SplitIdx(
        train=np.flatnonzero(days < T - G - V),
        validation=np.flatnonzero((days >= T - G - V) & (days < T - G)),
        test=np.flatnonzero((days >= T) & (days < T + test_days)),
    )
