import numpy as np
from scipy.special import expit

from fraudstream.models.focal import focal_loss_grad_hess


def test_focal_gradients_match_finite_differences():
    logits = np.array([-.5, .5, -.5, .5])
    labels = np.array([0, 0, 1, 1])
    def loss(x):
        p = expit(x)
        return np.where(labels == 1, -.25 * (1 - p) ** 2 * np.log(p),
                        -.75 * p ** 2 * np.log(1 - p))
    epsilon = 1e-4
    gradient, hessian = focal_loss_grad_hess(labels, logits)
    numeric_gradient = (loss(logits + epsilon) - loss(logits - epsilon)) / (2 * epsilon)
    numeric_hessian = (loss(logits + epsilon) - 2 * loss(logits) + loss(logits - epsilon)) / epsilon ** 2
    np.testing.assert_allclose(gradient, numeric_gradient, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(hessian, numeric_hessian, rtol=1e-5, atol=1e-7)
