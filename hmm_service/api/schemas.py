"""
schemas.py
==========
Request / response validation for the HMM training API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TrainRequest:
    """Validated training request payload.

    Attributes
    ----------
    observations : list[int]
        Integer-coded observation sequence (0-based).
    n_states : int
        Number of hidden states.
    n_obs_symbols : int
        Number of distinct observation symbols.
    max_iterations : int
        EM iteration cap.
    tolerance : float
        Convergence threshold.
    seed : int or None
        Random seed.
    """

    observations: list[int]
    n_states: int
    n_obs_symbols: int
    max_iterations: int = 200
    tolerance: float = 1e-6
    seed: Optional[int] = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TrainRequest":
        """Parse and validate a JSON payload.

        Raises
        ------
        ValueError
            On missing or invalid fields.
        """
        errors: list[str] = []

        observations = data.get("observations")
        if not isinstance(observations, list) or len(observations) < 2:
            errors.append("'observations' must be a list with ≥ 2 elements.")

        n_states = data.get("n_states")
        if not isinstance(n_states, int) or n_states < 1:
            errors.append("'n_states' must be a positive integer.")

        n_obs_symbols = data.get("n_obs_symbols")
        if not isinstance(n_obs_symbols, int) or n_obs_symbols < 1:
            errors.append("'n_obs_symbols' must be a positive integer.")

        if errors:
            raise ValueError("; ".join(errors))

        return cls(
            observations=observations,  # type: ignore[arg-type]
            n_states=n_states,  # type: ignore[arg-type]
            n_obs_symbols=n_obs_symbols,  # type: ignore[arg-type]
            max_iterations=int(data.get("max_iterations", 200)),
            tolerance=float(data.get("tolerance", 1e-6)),
            seed=data.get("seed"),
        )


@dataclass
class TrainResponse:
    """Serialisable training result.

    Attributes
    ----------
    model_id : str
        UUID key in the model store.
    converged : bool
    n_iterations : int
    final_log_likelihood : float
    A : list[list[float]]
    B : list[list[float]]
    pi : list[float]
    log_likelihood_history : list[float]
    plots : dict[str, str]
        Base-64-encoded plot images keyed by name.
    """

    model_id: str
    converged: bool
    n_iterations: int
    final_log_likelihood: float
    A: list[list[float]]
    B: list[list[float]]
    pi: list[float]
    log_likelihood_history: list[float] = field(repr=False)
    plots: dict[str, str] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "final_log_likelihood": self.final_log_likelihood,
            "A": self.A,
            "B": self.B,
            "pi": self.pi,
            "log_likelihood_history": self.log_likelihood_history,
            "plots": self.plots,
        }
