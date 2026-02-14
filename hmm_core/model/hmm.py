"""
hmm.py
======
High-level Hidden Markov Model wrapper.

Wraps :class:`HMMParameters` with shape validation and provides
a convenience ``log_likelihood`` method that delegates to the
forward-backward engine.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from hmm_core.model.parameters import HMMParameters
from hmm_core.utils.validation import validate_hmm_parameters


class HiddenMarkovModel:
    """An HMM defined by λ = (A, B, π).

    Parameters
    ----------
    params : HMMParameters
        Validated parameter snapshot.

    Raises
    ------
    ValueError
        If `params` fails validation checks.
    """

    def __init__(self, params: HMMParameters) -> None:
        validate_hmm_parameters(params.A, params.B, params.pi)
        self._params = params

    # ----- Properties -----

    @property
    def params(self) -> HMMParameters:
        """Current parameter snapshot (immutable)."""
        return self._params

    @params.setter
    def params(self, new_params: HMMParameters) -> None:
        validate_hmm_parameters(new_params.A, new_params.B, new_params.pi)
        self._params = new_params

    @property
    def n_states(self) -> int:
        return self._params.n_states

    @property
    def n_observations(self) -> int:
        return self._params.n_observations

    @property
    def A(self) -> npt.NDArray[np.float64]:
        return self._params.A

    @property
    def B(self) -> npt.NDArray[np.float64]:
        return self._params.B

    @property
    def pi(self) -> npt.NDArray[np.float64]:
        return self._params.pi

    # ----- Convenience -----

    def log_likelihood(self, observations: npt.NDArray[np.intp]) -> float:
        """Compute log P(O | λ) using the scaled forward algorithm.

        Parameters
        ----------
        observations : ndarray of int, shape (T,)
            Observation sequence (integer-coded).

        Returns
        -------
        float
            Log-likelihood of the observation sequence under this model.
        """
        # Import here to avoid circular dependency at module level.
        from hmm_core.inference.forward_backward import run_forward_backward

        result = run_forward_backward(self, observations)
        return result.log_likelihood

    # ----- Factory -----

    @classmethod
    def from_arrays(
        cls,
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
        pi: npt.NDArray[np.float64],
    ) -> "HiddenMarkovModel":
        """Build an HMM directly from NumPy arrays."""
        return cls(HMMParameters(A=A, B=B, pi=pi))

    def __repr__(self) -> str:
        return (
            f"HiddenMarkovModel(n_states={self.n_states}, "
            f"n_observations={self.n_observations})"
        )
