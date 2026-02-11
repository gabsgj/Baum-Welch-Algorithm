"""
parameters.py
=============
Immutable container for HMM parameters λ = (A, B, π).

* A — state transition matrix,    shape (N, N)
* B — observation emission matrix, shape (N, M)
* π — initial state distribution,  shape (N,)

Deep-copies arrays on construction so the caller's original data
is never mutated by downstream operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class HMMParameters:
    """Immutable snapshot of HMM parameters λ = (A, B, π).

    Attributes
    ----------
    A : ndarray, shape (N, N)
        Row-stochastic state transition matrix.
        A[i, j] = P(state_j at t+1 | state_i at t).
    B : ndarray, shape (N, M)
        Row-stochastic observation emission matrix.
        B[i, k] = P(obs_k | state_i).
    pi : ndarray, shape (N,)
        Initial state distribution.
        pi[i] = P(state_i at t=0).
    """

    A: npt.NDArray[np.float64] = field(repr=False)
    B: npt.NDArray[np.float64] = field(repr=False)
    pi: npt.NDArray[np.float64] = field(repr=False)

    def __post_init__(self) -> None:
        # Frozen dataclass — use object.__setattr__ for init-time deep copy.
        object.__setattr__(self, "A", np.array(self.A, dtype=np.float64, copy=True))
        object.__setattr__(self, "B", np.array(self.B, dtype=np.float64, copy=True))
        object.__setattr__(self, "pi", np.array(self.pi, dtype=np.float64, copy=True))

    # ----- Derived properties -----

    @property
    def n_states(self) -> int:
        """Number of hidden states N."""
        return int(self.A.shape[0])

    @property
    def n_observations(self) -> int:
        """Number of distinct observation symbols M."""
        return int(self.B.shape[1])

    # ----- Serialization helpers -----

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation (lists, not ndarrays)."""
        return {
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "pi": self.pi.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HMMParameters":
        """Reconstruct from a plain dict."""
        return cls(
            A=np.asarray(d["A"], dtype=np.float64),
            B=np.asarray(d["B"], dtype=np.float64),
            pi=np.asarray(d["pi"], dtype=np.float64),
        )

    def __repr__(self) -> str:
        return (
            f"HMMParameters(n_states={self.n_states}, "
            f"n_observations={self.n_observations})"
        )
