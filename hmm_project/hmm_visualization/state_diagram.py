"""
state_diagram.py
================
Graphviz-based state transition diagram renderer.

This module delegates to the ``state_transition_diagrams`` library.
It is kept for backward compatibility so existing imports continue
to work unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt

from state_transition_diagrams import render_state_diagram as _render
from state_transition_diagrams.config import DiagramConfig
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
    config = DiagramConfig(state_colors=list(STATE_COLORS))
    return _render(
        A,
        state_labels=state_labels,
        save_path=save_path,
        config=config,
        fmt=fmt,
        title=title,
        prob_threshold=prob_threshold,
    )
