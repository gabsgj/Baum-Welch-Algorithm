"""
state_diagram.py
================
Graphviz-based state transition diagram renderer.

Produces a directed graph where:
* Nodes represent hidden states (circular layout).
* Directed edges represent transition probabilities.
* Edge widths are proportional to transition probability.
* Self-loops are clearly visible.
* Output format is SVG for publication-quality export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

try:
    import graphviz
except ImportError:  # pragma: no cover
    graphviz = None  # type: ignore[assignment]

from hmm_visualization.styles import STATE_COLORS


def render_state_diagram(
    A: npt.NDArray[np.float64],
    *,
    state_labels: Optional[Sequence[str]] = None,
    save_path: Optional[str | Path] = None,
    fmt: str = "svg",
    title: str = "HMM State Transition Diagram",
    prob_threshold: float = 0.01,
) -> "graphviz.Digraph":
    """Render the state transition diagram as a Graphviz digraph.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        Transition matrix.
    state_labels : sequence of str, optional
        Human-readable state names. Defaults to S0, S1, ….
    save_path : str or Path, optional
        File path (without extension) for rendering.
    fmt : str
        Output format: ``"svg"`` (default), ``"png"``, ``"pdf"``.
    title : str
        Graph title.
    prob_threshold : float
        Suppress edges with probability below this value.

    Returns
    -------
    graphviz.Digraph
    """
    if graphviz is None:
        raise ImportError(
            "The `graphviz` Python package is required. "
            "Install it with: pip install graphviz"
        )

    N = A.shape[0]
    if state_labels is None:
        state_labels = [f"S{i}" for i in range(N)]

    dot = graphviz.Digraph(
        name="HMM",
        comment=title,
        format=fmt,
        engine="circo",  # circular layout
    )

    # ─── Graph-level attributes ─── #
    dot.attr(
        rankdir="LR",
        bgcolor="#FAFAFA",
        label=title,
        labelloc="t",
        fontsize="18",
        fontname="Helvetica",
        pad="0.5",
    )

    # ─── Node style ─── #
    for i, label in enumerate(state_labels):
        color = STATE_COLORS[i % len(STATE_COLORS)]
        dot.node(
            str(i),
            label=label,
            shape="circle",
            style="filled",
            fillcolor=color,
            fontcolor="white",
            fontsize="14",
            fontname="Helvetica Bold",
            width="1.0",
            height="1.0",
            fixedsize="true",
        )

    # ─── Edges ─── #
    max_prob = A.max() if A.max() > 0 else 1.0
    for i in range(N):
        for j in range(N):
            p = A[i, j]
            if p < prob_threshold:
                continue
            # Width: 0.5 → 4.0 proportional to probability.
            penwidth = str(round(0.5 + 3.5 * (p / max_prob), 2))
            edge_label = f"{p:.3f}"
            edge_color = "#333333" if i != j else STATE_COLORS[i % len(STATE_COLORS)]
            dot.edge(
                str(i), str(j),
                label=edge_label,
                penwidth=penwidth,
                fontsize="10",
                fontname="Helvetica",
                color=edge_color,
                fontcolor=edge_color,
            )

    if save_path is not None:
        save_path = Path(save_path)
        dot.render(
            filename=save_path.stem,
            directory=str(save_path.parent),
            cleanup=True,
        )

    return dot
