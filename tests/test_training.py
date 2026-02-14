"""Tests for the full HMM training loop."""

import numpy as np
import pytest

from hmm_core.training.trainer import HMMTrainer
from hmm_core.training.training_result import TrainingResult


class TestHMMTrainer:
    """Integration tests for the Baum-Welch trainer."""

    def test_converges_on_simple_sequence(self):
        """Train on a simple repeating observation pattern."""
        obs = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.intp)
        trainer = HMMTrainer(n_states=2, n_obs_symbols=2, max_iterations=100, tolerance=1e-8, seed=42)
        result = trainer.fit(obs)

        assert isinstance(result, TrainingResult)
        assert result.n_iterations >= 1
        assert len(result.log_likelihood_history) == result.n_iterations
        assert len(result.parameter_history) == result.n_iterations

    def test_likelihood_monotonically_increases(self):
        """Log-likelihood should never decrease across EM iterations."""
        obs = np.array([0, 1, 2, 0, 1, 2, 0, 1, 0, 0, 2, 1], dtype=np.intp)
        trainer = HMMTrainer(n_states=2, n_obs_symbols=3, max_iterations=50, seed=123)
        result = trainer.fit(obs)

        history = result.log_likelihood_history
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1] - 1e-10, (
                f"LL decreased at iteration {i+1}: "
                f"{history[i-1]:.8f} → {history[i]:.8f}"
            )

    def test_final_params_stochastic(self):
        """Final trained parameters must remain valid stochastic matrices."""
        obs = np.array([0, 0, 1, 1, 2, 0, 1, 2], dtype=np.intp)
        trainer = HMMTrainer(n_states=3, n_obs_symbols=3, max_iterations=30, seed=7)
        result = trainer.fit(obs)

        p = result.model_params
        np.testing.assert_allclose(p.A.sum(axis=1), 1.0, atol=1e-8)
        np.testing.assert_allclose(p.B.sum(axis=1), 1.0, atol=1e-8)
        np.testing.assert_allclose(p.pi.sum(), 1.0, atol=1e-8)

    def test_result_repr(self):
        """TrainingResult should have a clean __repr__."""
        obs = np.array([0, 1, 0, 1], dtype=np.intp)
        trainer = HMMTrainer(n_states=2, n_obs_symbols=2, max_iterations=5, seed=0)
        result = trainer.fit(obs)
        r = repr(result)
        assert "converged=" in r
        assert "n_iterations=" in r

    def test_too_few_observations_raises(self):
        trainer = HMMTrainer(n_states=2, n_obs_symbols=2)
        with pytest.raises(ValueError, match="at least 2"):
            trainer.fit(np.array([0], dtype=np.intp))

    def test_obs_out_of_range_raises(self):
        trainer = HMMTrainer(n_states=2, n_obs_symbols=2)
        with pytest.raises(ValueError, match="exceeds"):
            trainer.fit(np.array([0, 1, 5], dtype=np.intp))
