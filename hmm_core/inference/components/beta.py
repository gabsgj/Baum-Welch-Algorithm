"""
beta.py
=======
Scaled backward recursion (β-pass).

Computes the *scaled* backward variable β̂_t(i) defined by:

    β̂_T(i) = 1                         (then scaled by c_T)
    β̂_t(i) = Σ_j A(i,j) · B(j, O_{t+1}) · β̂_{t+1}(j)   (then scaled by c_{t})

Uses the *same* scaling factors c_t produced by the forward pass so
that the α·β product is correctly normalized.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_beta(
    A: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
    observations: npt.NDArray[np.intp],
    scaling_factors: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Scaled backward algorithm.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        State transition matrix.
    B : ndarray, shape (N, M)
        Observation emission matrix.
    observations : ndarray of int, shape (T,)
        Observation sequence (integer-coded, 0-based).
    scaling_factors : ndarray, shape (T,)
        Scaling coefficients from the forward pass.

    Returns
    -------
    beta : ndarray, shape (T, N)
        Scaled backward variables.
    """
    T = len(observations)
    N = A.shape[0]

    beta = np.zeros((T, N), dtype=np.float64)

    # --- Initialization (t = T-1, i.e. last time step) ---
    beta[T - 1] = scaling_factors[T - 1]  # β_T = c_T · 1

    # --- Recursion (t = T-2 … 0) ---
    for t in range(T - 2, -1, -1):
        # β_t(i) = Σ_j a_{ij} · b_j(O_{t+1}) · β_{t+1}(j)
        beta[t] = A @ (B[:, observations[t + 1]] * beta[t + 1])
        beta[t] *= scaling_factors[t]

    return beta
