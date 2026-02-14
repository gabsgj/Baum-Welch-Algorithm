"""
baum_welch_step.py
==================
Single M-step of the Baum-Welch (EM) algorithm.

Re-estimates λ = (A, B, π) from the E-step responsibilities γ and ξ
using the standard update equations:

    π̂_i   = γ_1(i)

    â_{ij} = Σ_{t=1}^{T-1} ξ_t(i,j)  /  Σ_{t=1}^{T-1} γ_t(i)

    b̂_j(k) = Σ_{t : O_t=k} γ_t(j)   /  Σ_t γ_t(j)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hmm_core.model.parameters import HMMParameters
from hmm_core.utils.normalization import normalize_rows, normalize_vector


def baum_welch_update(
    gamma: npt.NDArray[np.float64],
    xi: npt.NDArray[np.float64],
    observations: npt.NDArray[np.intp],
    n_obs_symbols: int,
) -> HMMParameters:
    """Compute updated HMM parameters from E-step quantities.

    Parameters
    ----------
    gamma : ndarray, shape (T, N)
        State responsibilities γ_t(i).
    xi : ndarray, shape (T-1, N, N)
        Transition responsibilities ξ_t(i, j).
    observations : ndarray of int, shape (T,)
        Observation sequence (0-based).
    n_obs_symbols : int
        Total number of distinct observation symbols M.

    Returns
    -------
    HMMParameters
        Updated (A_new, B_new, π_new).
    """
    T, N = gamma.shape

    # --- π_new: initial state distribution ---
    pi_new = gamma[0].copy()
    pi_new = normalize_vector(pi_new)

    # --- A_new: transition matrix ---
    # Numerator: Σ_{t=0}^{T-2} ξ_t(i,j)   →  shape (N, N)
    xi_sum = xi.sum(axis=0)
    # Denominator: Σ_{t=0}^{T-2} γ_t(i)    →  shape (N,)
    gamma_sum = gamma[:-1].sum(axis=0)
    # Avoid division by zero.
    gamma_sum_safe = np.where(gamma_sum == 0.0, 1.0, gamma_sum)
    A_new = xi_sum / gamma_sum_safe[:, np.newaxis]
    A_new = normalize_rows(A_new)

    # --- B_new: emission matrix ---
    B_new = np.zeros((N, n_obs_symbols), dtype=np.float64)
    gamma_full_sum = gamma.sum(axis=0)  # shape (N,)
    gamma_full_sum_safe = np.where(gamma_full_sum == 0.0, 1.0, gamma_full_sum)

    for k in range(n_obs_symbols):
        mask = (observations == k)  # shape (T,)
        # Σ_{t : O_t = k} γ_t(j) for each state j.
        B_new[:, k] = gamma[mask].sum(axis=0)

    B_new /= gamma_full_sum_safe[:, np.newaxis]
    B_new = normalize_rows(B_new)

    return HMMParameters(A=A_new, B=B_new, pi=pi_new)
