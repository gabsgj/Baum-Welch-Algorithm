"""
validation.py
=============
Shape, non-negativity, and stochasticity checks for HMM parameters.

Called during model construction to ensure A, B, π are valid before
any computation begins.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

_STOCHASTIC_TOL: float = 1e-6


def validate_hmm_parameters(
    A: npt.NDArray[np.float64],
    B: npt.NDArray[np.float64],
    pi: npt.NDArray[np.float64],
) -> None:
    """Validate shapes, non-negativity, and stochasticity of HMM parameters.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        State transition matrix.
    B : ndarray, shape (N, M)
        Observation emission matrix.
    pi : ndarray, shape (N,)
        Initial state distribution.

    Raises
    ------
    ValueError
        On any validation failure.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)

    # --- Dimensionality ---
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A.shape}.")
    if B.ndim != 2:
        raise ValueError(f"B must be 2-D, got shape {B.shape}.")
    if pi.ndim != 1:
        raise ValueError(f"pi must be 1-D, got shape {pi.shape}.")

    N = A.shape[0]

    # --- Shape consistency ---
    if A.shape != (N, N):
        raise ValueError(f"A must be square (N×N), got shape {A.shape}.")
    if B.shape[0] != N:
        raise ValueError(
            f"B rows ({B.shape[0]}) must equal number of states N={N}."
        )
    if pi.shape[0] != N:
        raise ValueError(
            f"pi length ({pi.shape[0]}) must equal number of states N={N}."
        )

    # --- Non-negativity ---
    if np.any(A < 0):
        raise ValueError("A contains negative entries.")
    if np.any(B < 0):
        raise ValueError("B contains negative entries.")
    if np.any(pi < 0):
        raise ValueError("pi contains negative entries.")

    # --- Stochasticity ---
    _check_row_stochastic(A, "A")
    _check_row_stochastic(B, "B")

    pi_sum = pi.sum()
    if abs(pi_sum - 1.0) > _STOCHASTIC_TOL:
        raise ValueError(
            f"pi must sum to 1.0, got {pi_sum:.8f}."
        )


def _check_row_stochastic(
    matrix: npt.NDArray[np.float64],
    name: str,
) -> None:
    """Assert every row of *matrix* sums to 1 within tolerance."""
    row_sums = matrix.sum(axis=1)
    bad = np.where(np.abs(row_sums - 1.0) > _STOCHASTIC_TOL)[0]
    if bad.size > 0:
        raise ValueError(
            f"{name} rows {bad.tolist()} do not sum to 1.0 "
            f"(sums: {row_sums[bad].tolist()})."
        )
