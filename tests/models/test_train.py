import numpy as np
import pytest

from fraudstream.models.calibrate import PlattCalibrator
from fraudstream.models.registry import load_model_config
from fraudstream.models.train import ModelBundle, train


@pytest.mark.parametrize("kind", ["lr", "lgbm", "iforest"])
def test_model_scores_survive_roundtrip(kind, tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(int)
    bundle = train(load_model_config(kind), X, y, ["a", "b"])
    before = bundle.score(X[:5])
    bundle.save(tmp_path)
    loaded = ModelBundle.load(tmp_path)
    np.testing.assert_allclose(loaded.score(X[:5]), before)
    assert loaded.feature_list == ["a", "b"]


def test_platt_calibration_matches_calibrator_and_survives_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 2))
    y = (X[:, 0] + 0.5 * rng.normal(size=400) > 0).astype(int)
    bundle = train(load_model_config("lgbm"), X[:200], y[:200], ["a", "b"])
    expected = PlattCalibrator().fit(bundle.raw_score(X[200:]), y[200:]).predict(
        bundle.raw_score(X[:20]))
    bundle.calibrate(X[200:], y[200:])
    np.testing.assert_allclose(bundle.score(X[:20]), expected, rtol=1e-9)
    bundle.save(tmp_path)
    loaded = ModelBundle.load(tmp_path)
    np.testing.assert_allclose(loaded.score(X[:20]), expected, rtol=1e-9)
    assert loaded.calibration["method"] == "platt"


def test_supervised_models_reject_single_class():
    with pytest.raises(ValueError, match="both label classes"):
        train(load_model_config("lr"), [[0], [1]], [0, 0], ["feature"])
