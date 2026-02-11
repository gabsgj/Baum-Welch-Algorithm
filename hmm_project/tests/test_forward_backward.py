"""Tests for the forward-backward algorithm and responsibility computations."""

import numpy as np
import pytest

from hmm_core.model.hmm import HiddenMarkovModel
from hmm_core.model.parameters import HMMParameters
from hmm_core.inference.forward_backward import run_forward_backward
from hmm_core.inference.responsibilities import compute_responsibilities


def _make_weather_model() -> HiddenMarkovModel:
    """2-state (Rainy/Sunny), 3-observation (Walk/Shop/Clean) model."""
    A = np.array([
        [0.7, 0.3],
        [0.4, 0.6],
    ])
    B = np.array([
        [0.1, 0.4, 0.5],  # Rainy: Walk=0.1, Shop=0.4, Clean=0.5
        [0.6, 0.3, 0.1],  # Sunny: Walk=0.6, Shop=0.3, Clean=0.1
    ])
    pi = np.array([0.6, 0.4])
    return HiddenMarkovModel(HMMParameters(A=A, B=B, pi=pi))


class TestForwardBackward:
    """Tests for the scaled forward-backward pass."""

    def test_alpha_shape(self):
        model = _make_weather_model()
        obs = np.array([0, 1, 2, 0], dtype=np.intp)
        fb = run_forward_backward(model, obs)
        assert fb.alpha.shape == (4, 2)

    def test_beta_shape(self):
        model = _make_weather_model()
        obs = np.array([0, 1, 2, 0], dtype=np.intp)
        fb = run_forward_backward(model, obs)
        assert fb.beta.shape == (4, 2)

    def test_scaling_factors_shape(self):
        model = _make_weather_model()
        obs = np.array([0, 1, 2, 0], dtype=np.intp)
        fb = run_forward_backward(model, obs)
        assert fb.scaling_factors.shape == (4,)

    def test_log_likelihood_is_negative(self):
        model = _make_weather_model()
        obs = np.array([0, 1, 2, 0], dtype=np.intp)
        fb = run_forward_backward(model, obs)
        assert fb.log_likelihood < 0.0, "Log-likelihood must be negative."

    def test_alpha_rows_sum_to_one(self):
        """After scaling, each α row should sum ≈ 1."""
        model = _make_weather_model()
        obs = np.array([0, 1, 2, 0, 1], dtype=np.intp)
        fb = run_forward_backward(model, obs)
        row_sums = fb.alpha.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


class TestResponsibilities:
    """Tests for gamma and xi computation."""

    def test_gamma_shape_and_sum(self):
        model = _make_weather_model()
        obs = np.array([0, 1, 2], dtype=np.intp)
        fb = run_forward_backward(model, obs)
        gamma, xi = compute_responsibilities(
            fb.alpha, fb.beta, model.A, model.B, obs,
        )
        assert gamma.shape == (3, 2)
        # Each gamma row should sum to 1.
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-10)

    def test_xi_shape(self):
        model = _make_weather_model()
        obs = np.array([0, 1, 2, 0], dtype=np.intp)
        fb = run_forward_backward(model, obs)
        gamma, xi = compute_responsibilities(
            fb.alpha, fb.beta, model.A, model.B, obs,
        )
        assert xi.shape == (3, 2, 2)  # T-1 = 3

    def test_xi_slices_sum_to_one(self):
        """Each ξ_t slice should sum to ≈ 1."""
        model = _make_weather_model()
        obs = np.array([0, 1, 2, 0], dtype=np.intp)
        fb = run_forward_backward(model, obs)
        _, xi = compute_responsibilities(
            fb.alpha, fb.beta, model.A, model.B, obs,
        )
        for t in range(xi.shape[0]):
            np.testing.assert_allclose(xi[t].sum(), 1.0, atol=1e-10)
