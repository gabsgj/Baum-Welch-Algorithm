"""
xi.py
=====
Transition responsibility computation.

Computes ξ_t(i, j) = P(q_t = S_i, q_{t+1} = S_j | O, λ):

    ξ_t(i,j) = α_t(i) · a_{ij} · b_j(O_{t+1}) · β_{t+1}(j)
               ──────────────────────────────────────────────
               Σ_i Σ_j α_t(i) · a_{ij} · b_j(O_{t+1}) · β_{t+1}(j)

Returns a (T-1) × N × N tensor.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_xi(
    alpha: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    A: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
    observations: npt.NDArray[np.intp],
) -> npt.NDArray[np.float64]:
    """Transition responsibilities ξ.

    Parameters
    ----------
    alpha : ndarray, shape (T, N)
        Scaled forward variables.
    beta : ndarray, shape (T, N)
        Scaled backward variables.
    A : ndarray, shape (N, N)
        State transition matrix.
    B : ndarray, shape (N, M)
        Observation emission matrix.
    observations : ndarray of int, shape (T,)
        Observation sequence (integer-coded, 0-based).

    Returns
    -------
    xi : ndarray, shape (T-1, N, N)
        ξ_t(i, j) for t = 0 … T-2.
    """
    T, N = alpha.shape
    xi = np.zeros((T - 1, N, N), dtype=np.float64)

    for t in range(T - 1):
        # Outer product: α_t(i) · a_{ij} · b_j(O_{t+1}) · β_{t+1}(j)
        # Shape: (N, N)
        numerator = (
            alpha[t, :, np.newaxis]
            * A
            * B[np.newaxis, :, observations[t + 1]]
            * beta[t + 1, np.newaxis, :]
        )
        denom = numerator.sum()
        if denom == 0.0:
            denom = 1.0
        xi[t] = numerator / denom

    return xi
