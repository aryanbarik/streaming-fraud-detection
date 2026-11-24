import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from fraudstream.models.train import IMBALANCE_VARIANTS, fit_lgbm_variant


@pytest.mark.parametrize("variant", IMBALANCE_VARIANTS)
def test_variant_gives_probabilities_that_rank_a_real_signal(variant):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 4))
    y = (X[:, 0] + 0.3 * rng.normal(size=4000) > 1.8).astype(int)
    scorer = fit_lgbm_variant(variant, X[:3000], y[:3000])
    probability = scorer(X[3000:])
    assert probability.shape == (1000,)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    assert roc_auc_score(y[3000:], probability) > 0.9
