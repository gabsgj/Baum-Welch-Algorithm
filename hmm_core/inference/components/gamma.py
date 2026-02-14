"""
gamma.py
========
State responsibility computation.

Computes γ_t(i) = P(q_t = S_i | O, λ) using scaled α and β:

    γ_t(i) = α_t(i) · β_t(i)  /  Σ_j α_t(j) · β_t(j)

Because the forward and backward variables share the same scaling
factors, the denominator simplifies and γ is already normalized
per time step.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_gamma(
    alpha: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Per-state responsibilities γ.

    Parameters
    ----------
    alpha : ndarray, shape (T, N)
        Scaled forward variables.
    beta : ndarray, shape (T, N)
        Scaled backward variables.

    Returns
    -------
    gamma : ndarray, shape (T, N)
        γ_t(i), the posterior probability of being in state i at time t.
    """
    numerator = alpha * beta
    denominator = numerator.sum(axis=1, keepdims=True)
    # Prevent division by zero.
    denominator = np.where(denominator == 0.0, 1.0, denominator)
    gamma = numerator / denominator
    return gamma
