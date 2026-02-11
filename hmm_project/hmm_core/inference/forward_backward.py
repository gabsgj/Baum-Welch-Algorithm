"""
forward_backward.py
===================
Coordinatesthe scaled forward and backward passes.

Produces a :class:`ForwardBackwardResult` containing α, β, scaling
factors, and the log-likelihood — everything the E-step needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from hmm_core.inference.components.alpha import compute_alpha
from hmm_core.inference.components.beta import compute_beta
from hmm_core.inference.scaling import compute_log_likelihood


@dataclass(frozen=True, slots=True)
class ForwardBackwardResult:
    """Container for forward-backward outputs.

    Attributes
    ----------
    alpha : ndarray, shape (T, N)
        Scaled forward variables.
    beta : ndarray, shape (T, N)
        Scaled backward variables.
    scaling_factors : ndarray, shape (T,)
        Scaling coefficients c_t.
    log_likelihood : float
        log P(O | λ).
    """

    alpha: npt.NDArray[np.float64]
    beta: npt.NDArray[np.float64]
    scaling_factors: npt.NDArray[np.float64]
    log_likelihood: float


def run_forward_backward(
    model: "HiddenMarkovModel",  # noqa: F821  (forward ref)
    observations: npt.NDArray[np.intp],
) -> ForwardBackwardResult:
    """Execute the scaled forward-backward algorithm.

    Parameters
    ----------
    model : HiddenMarkovModel
        HMM whose parameters define the computation.
    observations : ndarray of int, shape (T,)
        Observation sequence (0-based integer codes).

    Returns
    -------
    ForwardBackwardResult
    """
    from hmm_core.model.hmm import HiddenMarkovModel  # noqa: F811

    A = model.A
    B = model.B
    pi = model.pi

    alpha, scaling_factors = compute_alpha(A, B, pi, observations)
    beta = compute_beta(A, B, observations, scaling_factors)
    ll = compute_log_likelihood(scaling_factors)

    return ForwardBackwardResult(
        alpha=alpha,
        beta=beta,
        scaling_factors=scaling_factors,
        log_likelihood=ll,
    )
