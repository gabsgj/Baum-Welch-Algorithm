"""
training_result.py
==================
Container for the complete output of an HMM training run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from hmm_core.model.parameters import HMMParameters


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Immutable record of a Baum-Welch training run.

    Attributes
    ----------
    model_params : HMMParameters
        Final trained parameters λ*.
    log_likelihood_history : list[float]
        Log P(O | λ) at each EM iteration (length = number of iterations).
    parameter_history : list[HMMParameters]
        Snapshot of (A, B, π) at each iteration (for trajectory plots).
    converged : bool
        ``True`` if the stopping tolerance was met before ``max_iterations``.
    n_iterations : int
        Total number of EM iterations executed.
    """

    model_params: HMMParameters
    log_likelihood_history: list[float] = field(repr=False)
    parameter_history: list[HMMParameters] = field(repr=False)
    converged: bool = True
    n_iterations: int = 0

    def __repr__(self) -> str:
        return (
            f"TrainingResult(converged={self.converged}, "
            f"n_iterations={self.n_iterations}, "
            f"final_ll={self.log_likelihood_history[-1]:.6f})"
        )
