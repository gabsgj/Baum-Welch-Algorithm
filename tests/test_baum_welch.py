"""Tests for a single Baum-Welch EM step."""

import numpy as np
import pytest

from hmm_core.model.hmm import HiddenMarkovModel
from hmm_core.model.parameters import HMMParameters
from hmm_core.inference.forward_backward import run_forward_backward
from hmm_core.inference.responsibilities import compute_responsibilities
from hmm_core.optimization.baum_welch_step import baum_welch_update


def _make_model_and_obs():
    """Small 2-state, 3-obs model with a fixed observation sequence."""
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
    pi = np.array([0.6, 0.4])
    model = HiddenMarkovModel(HMMParameters(A=A, B=B, pi=pi))
    obs = np.array([0, 1, 2, 0, 1, 2, 0], dtype=np.intp)
    return model, obs


class TestBaumWelchStep:
    """Tests for a single M-step update."""

    def test_updated_params_are_stochastic(self):
        model, obs = _make_model_and_obs()
        fb = run_forward_backward(model, obs)
        gamma, xi = compute_responsibilities(
            fb.alpha, fb.beta, model.A, model.B, obs,
        )
        new_params = baum_welch_update(gamma, xi, obs, n_obs_symbols=3)

        # A rows sum to 1.
        np.testing.assert_allclose(new_params.A.sum(axis=1), 1.0, atol=1e-10)
        # B rows sum to 1.
        np.testing.assert_allclose(new_params.B.sum(axis=1), 1.0, atol=1e-10)
        # pi sums to 1.
        np.testing.assert_allclose(new_params.pi.sum(), 1.0, atol=1e-10)

    def test_shapes_preserved(self):
        model, obs = _make_model_and_obs()
        fb = run_forward_backward(model, obs)
        gamma, xi = compute_responsibilities(
            fb.alpha, fb.beta, model.A, model.B, obs,
        )
        new_params = baum_welch_update(gamma, xi, obs, n_obs_symbols=3)

        assert new_params.A.shape == (2, 2)
        assert new_params.B.shape == (2, 3)
        assert new_params.pi.shape == (2,)

    def test_one_step_improves_likelihood(self):
        """One EM step should not decrease the log-likelihood."""
        model, obs = _make_model_and_obs()

        # Before.
        fb_before = run_forward_backward(model, obs)
        ll_before = fb_before.log_likelihood

        # E-step.
        gamma, xi = compute_responsibilities(
            fb_before.alpha, fb_before.beta, model.A, model.B, obs,
        )
        # M-step.
        new_params = baum_welch_update(gamma, xi, obs, n_obs_symbols=3)
        model_after = HiddenMarkovModel(new_params)

        # After.
        fb_after = run_forward_backward(model_after, obs)
        ll_after = fb_after.log_likelihood

        assert ll_after >= ll_before - 1e-10, (
            f"EM step decreased LL: {ll_before:.6f} → {ll_after:.6f}"
        )
