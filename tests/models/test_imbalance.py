import numpy as np
import pytest

from fraudstream.models.imbalance import correct_prior, undersample_indices


def test_undersampling_keeps_positives_and_corrects_base_rate():
    y = np.array([1] * 10 + [0] * 990)
    rows, beta = undersample_indices(y, negative_per_positive=10, seed=0)
    assert len(rows) == 110
    assert y[rows].sum() == 10
    sampled_prior = y[rows].mean()
    assert correct_prior(sampled_prior, beta) == pytest.approx(y.mean())
    assert np.array_equal(rows, undersample_indices(y, negative_per_positive=10, seed=0)[0])


def test_prior_correction_endpoints_and_bad_inputs():
    np.testing.assert_allclose(correct_prior([0, .5, 1], .1), [0, 1 / 11, 1])
    with pytest.raises(ValueError):
        correct_prior([.5], 0)
    with pytest.raises(ValueError):
        correct_prior([1.1], .1)
