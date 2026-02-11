"""
training_plots.py
=================
Log-likelihood convergence curve.

Produces a scientific-quality line plot showing log P(O | λ) across
EM iterations with proper axis labels, gridlines, and optional SVG export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from hmm_visualization.styles import PALETTE, hmm_style


def plot_log_likelihood(
    history: Sequence[float],
    *,
    title: str = "Baum-Welch Convergence",
    save_path: Optional[str | Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot log-likelihood across EM iterations.

    Parameters
    ----------
    history : sequence of float
        Log-likelihood values, one per iteration.
    title : str
        Plot title.
    save_path : str or Path, optional
        If given, save figure to this path (SVG/PNG inferred from suffix).
    show : bool
        If ``True``, display the plot interactively.

    Returns
    -------
    matplotlib.figure.Figure
    """
    with hmm_style():
        fig, ax = plt.subplots()
        iterations = np.arange(1, len(history) + 1)

        ax.plot(
            iterations, history,
            marker="o", markersize=4,
            color=PALETTE["primary"],
            linewidth=2.0,
            label="log P(O | λ)",
        )

        ax.set_xlabel("EM Iteration")
        ax.set_ylabel("Log-Likelihood")
        ax.set_title(title)
        ax.legend(loc="lower right")

        # Highlight convergence point.
        ax.axhline(
            y=history[-1], linestyle=":", color=PALETTE["accent"],
            alpha=0.6, label=f"Final LL = {history[-1]:.4f}",
        )
        ax.legend(loc="lower right")

        if save_path is not None:
            fig.savefig(str(save_path))
        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig
