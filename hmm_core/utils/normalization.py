"""
normalization.py
================
Row and vector normalization utilities for stochastic matrices.

Ensures probability distributions sum to 1.0 with numerical floor clamping
to prevent exact-zero entries (which cause log(0) failures downstream).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Minimum probability floor to guarantee numerical stability.
_EPS: float = 1e-300


def normalize_vector(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalize a 1-D vector so its entries sum to 1.

    Parameters
    ----------
    vector : ndarray, shape (K,)
        Non-negative vector.

    Returns
    -------
    ndarray, shape (K,)
        Stochastic vector (sums to 1).

    Raises
    ------
    ValueError
        If all entries are zero or negative.
    """
    vector = np.asarray(vector, dtype=np.float64)
    total = vector.sum()
    if total <= 0.0:
        raise ValueError("Cannot normalize a zero or all-negative vector.")
    normalized = vector / total
    # Floor-clamp to prevent exact zeros.
    normalized = np.maximum(normalized, _EPS)
    normalized /= normalized.sum()
    return normalized


def normalize_rows(matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Normalize each row of a 2-D matrix to sum to 1.

    Parameters
    ----------
    matrix : ndarray, shape (R, C)
        Non-negative matrix.

    Returns
    -------
    ndarray, shape (R, C)
        Row-stochastic matrix.

    Raises
    ------
    ValueError
        If any row sums to zero.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got {matrix.ndim}-D.")
    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Cannot normalize: at least one row sums to zero.")
    normalized = matrix / row_sums
    normalized = np.maximum(normalized, _EPS)
    # Re-normalize after floor clamping.
    normalized /= normalized.sum(axis=1, keepdims=True)
    return normalized
