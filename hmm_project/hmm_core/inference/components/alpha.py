"""
alpha.py
========
Scaled forward recursion (α-pass).

Computes the *scaled* forward variable α̂_t(i) defined by:

    α̂_1(i) = π_i · B(i, O_1)       (then scaled by c_1)
    α̂_t(i) = [Σ_j α̂_{t-1}(j) · A(j,i)] · B(i, O_t)   (then scaled by c_t)

Scaling factors c_t are produced alongside the α matrix so that
downstream modules can compute the log-likelihood without overflow.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_alpha(
    A: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
    pi: npt.NDArray[np.float64],
    observations: npt.NDArray[np.intp],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Scaled forward algorithm.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        State transition matrix.
    B : ndarray, shape (N, M)
        Observation emission matrix.
    pi : ndarray, shape (N,)
        Initial state distribution.
    observations : ndarray of int, shape (T,)
        Observation sequence (integer-coded, 0-based).

    Returns
    -------
    alpha : ndarray, shape (T, N)
        Scaled forward variables.
    scaling_factors : ndarray, shape (T,)
        Scaling coefficients c_t (inverse of the normalization constant at
        each time step).
    """
    T = len(observations)
    N = A.shape[0]

    alpha = np.zeros((T, N), dtype=np.float64)
    scaling_factors = np.zeros(T, dtype=np.float64)

    # --- Initialization (t = 0) ---
    alpha[0] = pi * B[:, observations[0]]
    c0 = alpha[0].sum()
    if c0 == 0.0:
        # Guard: if the observation is impossible under every state,
        # fall back to uniform to avoid division by zero.
        c0 = 1.0
    scaling_factors[0] = 1.0 / c0
    alpha[0] *= scaling_factors[0]

    # --- Recursion (t = 1 … T-1) ---
    for t in range(1, T):
        # α_t(i) = [Σ_j α_{t-1}(j) · a_{ji}] · b_i(O_t)
        alpha[t] = (alpha[t - 1] @ A) * B[:, observations[t]]
        ct = alpha[t].sum()
        if ct == 0.0:
            ct = 1.0
        scaling_factors[t] = 1.0 / ct
        alpha[t] *= scaling_factors[t]

    return alpha, scaling_factors
