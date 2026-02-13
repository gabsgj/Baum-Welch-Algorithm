"""
state_transition_diagrams
=========================
A standalone, open-source library for creating interactive and static
state transition diagrams — commonly used for Hidden Markov Models,
finite-state machines, and other stateful systems.

Features
--------
* **Interactive (D3.js)** — Animated browser-based diagrams with replay
  controls, particle flow, inspector panel, and 3-tier layout.
* **Static (Graphviz)** — Publication-quality SVG / PNG / PDF diagrams
  rendered via Graphviz.
* **Flask integration** — Drop-in Flask blueprint that serves the JS/CSS
  assets so you can embed the interactive diagram in any Flask app.
* **Fully configurable** — Colours, layout, fonts, and animation
  parameters can be overridden via ``DiagramConfig``.

Quick start
-----------
>>> from state_transition_diagrams import render_state_diagram
>>> import numpy as np
>>> A = np.array([[0.7, 0.3], [0.4, 0.6]])
>>> dot = render_state_diagram(A, state_labels=["Rainy", "Sunny"])

Flask integration::

    from state_transition_diagrams import create_blueprint
    app.register_blueprint(create_blueprint(), url_prefix="/std")
"""

from __future__ import annotations

from state_transition_diagrams.config import DiagramConfig
from state_transition_diagrams.renderer import render_state_diagram
from state_transition_diagrams.flask_blueprint import create_blueprint

__all__ = [
    "DiagramConfig",
    "render_state_diagram",
    "create_blueprint",
]

__version__ = "1.0.0"
