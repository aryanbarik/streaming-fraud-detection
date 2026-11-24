import numpy as np
import pytest

from fraudstream.models.calibrate import PlattCalibrator


def test_calibration_returns_ordered_probability_estimates():
    scores = np.linspace(-3, 3, 101)
    labels = (scores > 0).astype(int)
    calibrator = PlattCalibrator().fit(scores, labels)
    probabilities = calibrator.predict([-2, 0, 2])
    assert 0 < probabilities[0] < probabilities[1] < probabilities[2] < 1
    assert probabilities[0] < .5 < probabilities[2]


def test_calibration_rejects_single_class():
    with pytest.raises(ValueError, match="both binary classes"):
        PlattCalibrator().fit([0, 1], [0, 0])
