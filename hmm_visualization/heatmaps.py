"""
heatmaps.py
===========
Annotated heatmap visualization for HMM matrices (A, B, π).

Uses Seaborn with a perceptually uniform colourmap (``viridis`` by
default) and overlays probability values on each cell.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns

from hmm_visualization.styles import HEATMAP_CMAP, hmm_style


def plot_heatmap(
    matrix: npt.NDArray[np.float64],
    *,
    title: str = "Matrix Heatmap",
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
    save_path: Optional[str | Path] = None,
    show: bool = True,
    cmap: str = HEATMAP_CMAP,
    fmt: str = ".3f",
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Render an annotated heatmap for a probability matrix.

    Parameters
    ----------
    matrix : ndarray, shape (R, C)
        The matrix to visualize.
    title : str
        Plot title.
    row_labels, col_labels : sequence of str, optional
        Tick labels for rows and columns.
    save_path : str or Path, optional
        If given, save figure to this path (SVG/PNG inferred from suffix).
    show : bool
        If ``True``, display the plot interactively.
    cmap : str
        Matplotlib colourmap name.
    fmt : str
        Number format for annotations (e.g. ``".3f"``).
    figsize : tuple
        Figure dimensions.

    Returns
    -------
    matplotlib.figure.Figure
    """
    matrix = np.asarray(matrix, dtype=np.float64)

    with hmm_style():
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            matrix,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            linewidths=0.8,
            linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "Probability"},
            xticklabels=col_labels if col_labels else "auto",
            yticklabels=row_labels if row_labels else "auto",
            ax=ax,
            vmin=0.0,
            vmax=1.0,
            square=True,
        )

        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.tick_params(axis="both", labelsize=11)

        if save_path is not None:
            fig.savefig(str(save_path))
        if show:
            plt.show()
        else:
            plt.close(fig)

    return fig
