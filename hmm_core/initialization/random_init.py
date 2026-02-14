"""
random_init.py
==============
Random stochastic initialization of HMM parameters using
Dirichlet sampling to ensure valid probability distributions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from hmm_core.model.parameters import HMMParameters


def random_hmm_parameters(
    n_states: int,
    n_obs_symbols: int,
    *,
    seed: Optional[int] = None,
    alpha_dirichlet: float = 1.0,
) -> HMMParameters:
    """Generate random stochastic HMM parameters.

    Parameters
    ----------
    n_states : int
        Number of hidden states N.
    n_obs_symbols : int
        Number of distinct observation symbols M.
    seed : int or None
        Random seed for reproducibility.
    alpha_dirichlet : float
        Concentration parameter for the Dirichlet distribution.
        1.0 → uniform prior (flat random);
        >1.0 → concentrates mass toward uniform distributions;
        <1.0 → concentrates mass toward sparse distributions.

    Returns
    -------
    HMMParameters
        Randomly initialised (A, B, π) with valid stochastic rows.
    """
    rng = np.random.default_rng(seed)
    alpha = np.full(n_states, alpha_dirichlet)

    # Transition matrix A: each row is a Dirichlet sample.
    A = rng.dirichlet(alpha, size=n_states)

    # Emission matrix B: each row is a Dirichlet sample over M symbols.
    alpha_obs = np.full(n_obs_symbols, alpha_dirichlet)
    B = rng.dirichlet(alpha_obs, size=n_states)

    # Initial distribution π.
    pi = rng.dirichlet(alpha)

    return HMMParameters(A=A, B=B, pi=pi)
