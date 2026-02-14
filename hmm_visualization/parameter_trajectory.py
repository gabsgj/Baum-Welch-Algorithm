"""
parameter_trajectory.py
=======================
Subplot grids showing how each element of A, B, and π evolves across
EM iterations — helps diagnose training dynamics and identify
convergence behaviour per parameter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from hmm_visualization.styles import STATE_COLORS, hmm_style


def plot_parameter_evolution(
    parameter_history: Sequence,  # Sequence[HMMParameters]
    *,
    title: str = "Parameter Evolution",
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot how A, B, π change across EM iterations.

    Parameters
    ----------
    parameter_history : sequence of HMMParameters
        One snapshot per EM iteration.
    title : str
        Super-title for the figure.
    save_path : str or Path, optional
        If given, save figure to this path.
    show : bool
        If ``True``, display the plot interactively.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_iters = len(parameter_history)
    if n_iters == 0:
        raise ValueError("parameter_history is empty.")

    iters = np.arange(1, n_iters + 1)
    first = parameter_history[0]
    N = first.n_states
    M = first.n_observations

    with hmm_style():
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(title, fontsize=16, fontweight="bold")

        # ─── A (transition) ─── #
        ax_a = axes[0]
        for i in range(N):
            for j in range(N):
                vals = [p.A[i, j] for p in parameter_history]
                colour = STATE_COLORS[(i * N + j) % len(STATE_COLORS)]
                ax_a.plot(iters, vals, label=f"A[{i},{j}]", color=colour)
        ax_a.set_title("Transition Matrix A")
        ax_a.set_xlabel("Iteration")
        ax_a.set_ylabel("Probability")
        ax_a.legend(fontsize=7, ncol=2)

        # ─── B (emission) ─── #
        ax_b = axes[1]
        for i in range(N):
            for k in range(M):
                vals = [p.B[i, k] for p in parameter_history]
                colour = STATE_COLORS[(i * M + k) % len(STATE_COLORS)]
                ax_b.plot(iters, vals, label=f"B[{i},{k}]", color=colour)
        ax_b.set_title("Emission Matrix B")
        ax_b.set_xlabel("Iteration")
        ax_b.set_ylabel("Probability")
        ax_b.legend(fontsize=7, ncol=2)

        # ─── π (initial) ─── #
        ax_pi = axes[2]
        for i in range(N):
            vals = [p.pi[i] for p in parameter_history]
            ax_pi.plot(
                iters, vals,
                label=f"π[{i}]",
                color=STATE_COLORS[i % len(STATE_COLORS)],
            )
        ax_pi.set_title("Initial Distribution π")
        ax_pi.set_xlabel("Iteration")
        ax_pi.set_ylabel("Probability")
        ax_pi.legend(fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        if save_path is not None:
            fig.savefig(str(save_path))
        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig
