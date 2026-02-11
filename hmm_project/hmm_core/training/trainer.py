"""
trainer.py
==========
Full EM training loop for Hidden Markov Models.

Orchestrates the Baum-Welch algorithm:
    1. Initialise λ = (A, B, π) — randomly or from user-supplied values.
    2. E-step: forward-backward → γ, ξ.
    3. M-step: re-estimate λ_new.
    4. Check convergence.
    5. Repeat until converged or max_iterations reached.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt

from hmm_core.inference.forward_backward import run_forward_backward
from hmm_core.inference.responsibilities import compute_responsibilities
from hmm_core.initialization.random_init import random_hmm_parameters
from hmm_core.model.hmm import HiddenMarkovModel
from hmm_core.model.parameters import HMMParameters
from hmm_core.optimization.baum_welch_step import baum_welch_update
from hmm_core.optimization.convergence import check_convergence
from hmm_core.training.training_result import TrainingResult

logger = logging.getLogger(__name__)


class HMMTrainer:
    """Baum-Welch (EM) trainer for discrete HMMs.

    Parameters
    ----------
    n_states : int
        Number of hidden states N.
    n_obs_symbols : int
        Number of distinct observation symbols M.
    max_iterations : int
        Hard cap on EM iterations.
    tolerance : float
        Convergence threshold on |ΔLL|.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_states: int,
        n_obs_symbols: int,
        max_iterations: int = 200,
        tolerance: float = 1e-6,
        seed: Optional[int] = None,
    ) -> None:
        if n_states < 1:
            raise ValueError(f"n_states must be ≥ 1, got {n_states}.")
        if n_obs_symbols < 1:
            raise ValueError(f"n_obs_symbols must be ≥ 1, got {n_obs_symbols}.")
        self.n_states = n_states
        self.n_obs_symbols = n_obs_symbols
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.seed = seed

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        observations: npt.NDArray[np.intp],
        initial_params: Optional[HMMParameters] = None,
    ) -> TrainingResult:
        """Run the full Baum-Welch EM loop.

        Parameters
        ----------
        observations : ndarray of int, shape (T,)
            Observation sequence (0-based integer codes).
        initial_params : HMMParameters or None
            Starting parameters.  If ``None``, random stochastic matrices
            are generated via Dirichlet sampling.

        Returns
        -------
        TrainingResult
            Final model, log-likelihood history, parameter snapshots,
            convergence flag, and iteration count.
        """
        observations = np.asarray(observations, dtype=np.intp)
        self._validate_observations(observations)

        # ----- Step 0: Initialisation -----
        if initial_params is None:
            params = random_hmm_parameters(
                self.n_states, self.n_obs_symbols, seed=self.seed,
            )
        else:
            params = initial_params

        model = HiddenMarkovModel(params)

        ll_history: list[float] = []
        param_history: list[HMMParameters] = []
        ll_old = -np.inf
        converged = False

        for iteration in range(1, self.max_iterations + 1):
            # ----- E-step -----
            fb = run_forward_backward(model, observations)
            ll_new = fb.log_likelihood

            ll_history.append(ll_new)
            param_history.append(model.params)

            logger.info(
                "Iteration %4d | LL = %+14.6f | ΔLL = %+.6e",
                iteration, ll_new, ll_new - ll_old,
            )

            # ----- Convergence check -----
            if iteration > 1 and check_convergence(
                ll_new, ll_old, self.tolerance, iteration, self.max_iterations,
            ):
                converged = True
                logger.info("Converged at iteration %d.", iteration)
                break

            # ----- Responsibilities -----
            gamma, xi = compute_responsibilities(
                fb.alpha, fb.beta, model.A, model.B, observations,
            )

            # ----- M-step -----
            new_params = baum_welch_update(
                gamma, xi, observations, self.n_obs_symbols,
            )
            model.params = new_params
            ll_old = ll_new

        return TrainingResult(
            model_params=model.params,
            log_likelihood_history=ll_history,
            parameter_history=param_history,
            converged=converged,
            n_iterations=len(ll_history),
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _validate_observations(
        self,
        observations: npt.NDArray[np.intp],
    ) -> None:
        """Sanity-check the observation sequence."""
        if observations.ndim != 1:
            raise ValueError(
                f"observations must be 1-D, got shape {observations.shape}."
            )
        if len(observations) < 2:
            raise ValueError("Need at least 2 observations for Baum-Welch.")
        if observations.min() < 0:
            raise ValueError("Observation indices must be non-negative.")
        if observations.max() >= self.n_obs_symbols:
            raise ValueError(
                f"Observation index {observations.max()} exceeds "
                f"n_obs_symbols={self.n_obs_symbols}."
            )
