"""
scaling.py
==========
Scaling-factor bookkeeping and log-likelihood computation.

The log-likelihood under the scaled forward algorithm is:

    log P(O | λ) = − Σ_{t=1}^{T} log(c_t)

where c_t = 1 / Σ_i α̂_t(i)  are the scaling coefficients produced
by :func:`hmm_core.inference.components.alpha.compute_alpha`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_log_likelihood(
    scaling_factors: npt.NDArray[np.float64],
) -> float:
    """Derive log P(O | λ) from the forward-pass scaling factors.

    Parameters
    ----------
    scaling_factors : ndarray, shape (T,)
        Scaling coefficients c_t from the forward pass.

    Returns
    -------
    float
        Log-likelihood of the observation sequence.
    """
    # c_t = 1 / Σ_i α_t(i)  →  log P(O|λ) = −Σ log(c_t)
    # Guard against log(0) with a tiny floor.
    safe = np.maximum(scaling_factors, 1e-300)
    return -np.sum(np.log(safe))
