import numpy as np
import pytest

from fraudstream.models.train import ModelBundle, train


@pytest.mark.parametrize("kind", ["lr", "lgbm", "iforest"])
def test_model_scores_survive_roundtrip(kind, tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(int)
    bundle = train(kind, X, y, ["a", "b"])
    before = bundle.score(X[:5])
    bundle.save(tmp_path)
    loaded = ModelBundle.load(tmp_path)
    np.testing.assert_allclose(loaded.score(X[:5]), before)
    assert loaded.feature_list == ["a", "b"]


def test_supervised_models_reject_single_class():
    with pytest.raises(ValueError, match="both label classes"):
        train("lr", [[0], [1]], [0, 0], ["feature"])
