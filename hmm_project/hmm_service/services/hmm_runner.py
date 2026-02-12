"""
hmm_runner.py
=============
Orchestrates HMM training for both REST and WebSocket modes.

Contains ZERO Flask code — the ``on_iteration`` callback is a plain
callable injected by the service layer.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Callable, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server-side rendering.
import matplotlib.pyplot as plt
import numpy as np

from hmm_core.model.parameters import HMMParameters
from hmm_core.training.trainer import HMMTrainer
from hmm_core.training.training_result import TrainingResult
from hmm_visualization.heatmaps import plot_heatmap
from hmm_visualization.state_diagram import render_state_diagram
from hmm_visualization.training_plots import plot_log_likelihood


def _fig_to_base64(fig: plt.Figure, fmt: str = "png") -> str:
    """Render a Matplotlib figure to a base-64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ------------------------------------------------------------------ #
#  Synchronous (REST) training — kept for backward-compatibility      #
# ------------------------------------------------------------------ #

def run_training(
    observations: list[int],
    n_states: int,
    n_obs_symbols: int,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Train an HMM and return results + plot images (REST mode)."""
    obs_array = np.array(observations, dtype=np.intp)

    trainer = HMMTrainer(
        n_states=n_states,
        n_obs_symbols=n_obs_symbols,
        max_iterations=max_iterations,
        tolerance=tolerance,
        seed=seed,
    )
    result: TrainingResult = trainer.fit(obs_array)

    # ─── Generate plots ─── #
    plots: dict[str, str] = {}

    fig_ll = plot_log_likelihood(result.log_likelihood_history, show=False)
    plots["log_likelihood"] = _fig_to_base64(fig_ll)

    state_labels = [f"S{i}" for i in range(n_states)]
    fig_a = plot_heatmap(
        result.model_params.A,
        title="Transition Matrix A",
        row_labels=state_labels,
        col_labels=state_labels,
        show=False,
    )
    plots["heatmap_A"] = _fig_to_base64(fig_a)

    obs_labels = [f"O{k}" for k in range(n_obs_symbols)]
    fig_b = plot_heatmap(
        result.model_params.B,
        title="Emission Matrix B",
        row_labels=state_labels,
        col_labels=obs_labels,
        show=False,
    )
    plots["heatmap_B"] = _fig_to_base64(fig_b)

    try:
        dot = render_state_diagram(result.model_params.A, state_labels=state_labels)
        svg_bytes = dot.pipe(format="svg")
        plots["state_diagram"] = base64.b64encode(svg_bytes).decode("ascii")
    except Exception:
        plots["state_diagram"] = ""

    return {
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "final_log_likelihood": result.log_likelihood_history[-1],
        "A": result.model_params.A.tolist(),
        "B": result.model_params.B.tolist(),
        "pi": result.model_params.pi.tolist(),
        "log_likelihood_history": result.log_likelihood_history,
        "plots": plots,
    }


# ------------------------------------------------------------------ #
#  Live (WebSocket) training — emits per-iteration updates            #
# ------------------------------------------------------------------ #

def run_training_live(
    observations: list[int],
    n_states: int,
    n_obs_symbols: int,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
    seed: Optional[int] = None,
    init_params: Optional[dict[str, Any]] = None,
    on_iteration: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Train an HMM with per-iteration callback (WebSocket mode).

    No plots are generated — the client renders everything live.
    """
    obs_array = np.array(observations, dtype=np.intp)

    trainer = HMMTrainer(
        n_states=n_states,
        n_obs_symbols=n_obs_symbols,
        max_iterations=max_iterations,
        tolerance=tolerance,
        seed=seed,
    )
    hmm_init: Optional[HMMParameters] = None
    if init_params:
        try:
            hmm_init = HMMParameters(
                A=np.array(init_params['A'], dtype=np.float64),
                B=np.array(init_params['B'], dtype=np.float64),
                pi=np.array(init_params['pi'], dtype=np.float64)
            )
        except Exception as e:
            raise ValueError(f"Invalid initialization parameters: {e}")

    result: TrainingResult = trainer.fit(
        obs_array,
        initial_params=hmm_init,
        on_iteration=on_iteration,
    )

    return {
        "converged": result.converged,
        "n_iterations": result.n_iterations,
        "final_log_likelihood": result.log_likelihood_history[-1],
    }
