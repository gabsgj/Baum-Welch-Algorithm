"""
hmm_runner.py
=============
Orchestrates HMM training and visualisation generation.

Bridges ``hmm_core`` (pure engine) and ``hmm_visualization`` to produce
training results plus base-64-encoded plot images for the API layer.

Contains ZERO Flask code.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server-side rendering.
import matplotlib.pyplot as plt
import numpy as np

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


def run_training(
    observations: list[int],
    n_states: int,
    n_obs_symbols: int,
    max_iterations: int = 200,
    tolerance: float = 1e-6,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    """Train an HMM and return results + plot images.

    Returns
    -------
    dict
        Keys: ``converged``, ``n_iterations``, ``final_log_likelihood``,
        ``A``, ``B``, ``pi``, ``log_likelihood_history``, ``plots``.
    """
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

    # Log-likelihood curve.
    fig_ll = plot_log_likelihood(result.log_likelihood_history, show=False)
    plots["log_likelihood"] = _fig_to_base64(fig_ll)

    # Transition heatmap.
    state_labels = [f"S{i}" for i in range(n_states)]
    fig_a = plot_heatmap(
        result.model_params.A,
        title="Transition Matrix A",
        row_labels=state_labels,
        col_labels=state_labels,
        show=False,
    )
    plots["heatmap_A"] = _fig_to_base64(fig_a)

    # Emission heatmap.
    obs_labels = [f"O{k}" for k in range(n_obs_symbols)]
    fig_b = plot_heatmap(
        result.model_params.B,
        title="Emission Matrix B",
        row_labels=state_labels,
        col_labels=obs_labels,
        show=False,
    )
    plots["heatmap_B"] = _fig_to_base64(fig_b)

    # State diagram (SVG as base64).
    try:
        dot = render_state_diagram(
            result.model_params.A,
            state_labels=state_labels,
        )
        svg_bytes = dot.pipe(format="svg")
        plots["state_diagram"] = base64.b64encode(svg_bytes).decode("ascii")
    except Exception:
        # Graphviz binary might not be installed — degrade gracefully.
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
