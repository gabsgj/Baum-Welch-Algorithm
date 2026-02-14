"""
renderer.py
===========
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

from state_transition_diagrams.config import DiagramConfig


def render_state_diagram(
    A: npt.NDArray[np.float64],
    *,
    state_labels: Optional[Sequence[str]] = None,
    save_path: Optional[str | Path] = None,
    config: Optional[DiagramConfig] = None,
    fmt: Optional[str] = None,
    title: Optional[str] = None,
    prob_threshold: Optional[float] = None,
) -> "graphviz.Digraph":
    """Render the state transition diagram as a Graphviz digraph.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        Transition matrix.
    state_labels : sequence of str, optional
        Human-readable state names.  Defaults to S0, S1, ….
    save_path : str or Path, optional
        File path (without extension) for rendering.
    config : DiagramConfig, optional
        Full configuration object.  If ``None``, default config is used.
    fmt : str, optional
        Output format override (``"svg"``, ``"png"``, ``"pdf"``).
        Falls back to ``config.output_format``.
    title : str, optional
        Graph title override.  Falls back to ``config.title``.
    prob_threshold : float, optional
        Suppress edges with probability below this value.
        Falls back to ``config.prob_threshold``.

    Returns
    -------
    graphviz.Digraph
    """
    if graphviz is None:
        raise ImportError(
            "The `graphviz` Python package is required. "
            "Install it with: pip install graphviz"
        )

    if config is None:
        config = DiagramConfig()

    fmt = fmt or config.output_format
    title = title or config.title
    prob_threshold = prob_threshold if prob_threshold is not None else config.prob_threshold

    N = A.shape[0]
    if state_labels is None:
        state_labels = [f"S{i}" for i in range(N)]

    dot = graphviz.Digraph(
        name="StateDiagram",
        comment=title,
        format=fmt,
        engine=config.layout_engine,
    )

    # ─── Graph-level attributes ─── #
    dot.attr(
        rankdir="LR",
        bgcolor=config.background_color,
        label=title,
        labelloc="t",
        fontsize="18",
        fontname=config.font_family,
        pad="0.5",
    )

    # ─── Node style ─── #
    for i, label in enumerate(state_labels):
        color = config.state_colors[i % len(config.state_colors)]
        dot.node(
            str(i),
            label=label,
            shape=config.node_shape,
            style="filled",
            fillcolor=color,
            fontcolor="white",
            fontsize="14",
            fontname=f"{config.font_family} Bold",
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
            penwidth = str(round(0.5 + 3.5 * (p / max_prob), 2))
            edge_label = f"{p:.3f}"
            edge_color = (
                config.edge_color if i != j
                else config.state_colors[i % len(config.state_colors)]
            )
            dot.edge(
                str(i), str(j),
                label=edge_label,
                penwidth=penwidth,
                fontsize="10",
                fontname=config.font_family,
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
