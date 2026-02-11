"""
responsibilities.py
===================
Coordinates γ and ξ computation from α and β.

This is the E-step's second half: given the forward-backward outputs,
compute the posterior state (γ) and transition (ξ) responsibilities
needed by the M-step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from hmm_core.inference.components.gamma import compute_gamma
from hmm_core.inference.components.xi import compute_xi

if TYPE_CHECKING:
    pass


def compute_responsibilities(
    alpha: npt.NDArray[np.float64],
    beta: npt.NDArray[np.float64],
    A: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
    observations: npt.NDArray[np.intp],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute γ and ξ from forward-backward outputs.

    Parameters
    ----------
    alpha : ndarray, shape (T, N)
    beta : ndarray, shape (T, N)
    A : ndarray, shape (N, N)
    B : ndarray, shape (N, M)
    observations : ndarray of int, shape (T,)

    Returns
    -------
    gamma : ndarray, shape (T, N)
    xi : ndarray, shape (T-1, N, N)
    """
    gamma = compute_gamma(alpha, beta)
    xi = compute_xi(alpha, beta, A, B, observations)
    return gamma, xi
