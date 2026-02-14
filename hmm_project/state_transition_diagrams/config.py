"""
config.py
=========
Centralised configuration for state transition diagrams.

Users can create a ``DiagramConfig`` instance, override any settings,
and pass it to the renderer or Flask blueprint.

Example
-------
>>> from state_transition_diagrams import DiagramConfig
>>> cfg = DiagramConfig(
...     state_colors=["#E74C3C", "#3498DB", "#2ECC71"],
...     background_color="#FFFFFF",
...     font_family="Arial",
... )
>>> render_state_diagram(A, config=cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class DiagramConfig:
    """Configuration for both static (Graphviz) and interactive (D3) diagrams.

    All fields have sensible defaults.  Override only what you need.

    Attributes
    ----------
    state_colors : list[str]
        Hex colours for hidden-state nodes (cycled if fewer than N).
    observation_fill : str
        Background fill colour for observation nodes.
    observation_stroke : str
        Border colour for observation nodes.
    observation_text : str
        Text colour for observation labels.
    start_fill : str
        Background fill colour for the START node.
    start_stroke : str
        Border colour for the START node.
    start_text : str
        Text colour for the START node label.
    background_color : str
        Canvas / graph background colour.
    font_family : str
        Primary font family.
    mono_font_family : str
        Monospace font family (used for numeric labels).
    title : str
        Default graph title.
    prob_threshold : float
        Suppress edges whose probability is below this value.
    edge_color : str
        Default colour for inter-state transition edges.
    node_shape : str
        Graphviz node shape (``circle``, ``doublecircle``, etc.).
    node_radius : float
        Base radius for interactive diagram state nodes.
    layout_engine : str
        Graphviz layout engine (``circo``, ``dot``, ``neato``, …).
    output_format : str
        Default output format (``svg``, ``png``, ``pdf``).
    particles_enabled : bool
        Enable particle-flow animation in the interactive diagram.
    decongestion_enabled : bool
        Enable de-congestion filtering for transition/emission edges in the
        interactive diagram. When disabled, all edges are shown.
    animation_speed : float
        Playback speed multiplier for the interactive diagram.
    """

    # ── Colour palette ──────────────────────────────────────────── #
    state_colors: list[str] = field(default_factory=lambda: [
        "#2E86AB", "#A23B72", "#F18F01", "#2CA58D",
        "#E84855", "#6B4226", "#7768AE", "#1B998B",
    ])

    observation_fill: str = "#F1F5F9"
    observation_stroke: str = "#94A3B8"
    observation_text: str = "#334155"

    start_fill: str = "#F5F3FF"
    start_stroke: str = "#8B5CF6"
    start_text: str = "#5B21B6"

    background_color: str = "#FAFAFA"

    # ── Typography ──────────────────────────────────────────────── #
    font_family: str = "Helvetica"
    mono_font_family: str = "JetBrains Mono, monospace"

    # ── Graph settings ──────────────────────────────────────────── #
    title: str = "State Transition Diagram"
    prob_threshold: float = 0.01
    edge_color: str = "#333333"
    node_shape: str = "circle"
    node_radius: float = 40.0
    layout_engine: str = "circo"
    output_format: str = "svg"

    # ── Interactive diagram settings ────────────────────────────── #
    particles_enabled: bool = True
    decongestion_enabled: bool = False
    animation_speed: float = 1.0

    # ── Interactive diagram JS colour palette (richer) ──────────── #
    interactive_state_colors: list[dict] = field(default_factory=lambda: [
        {"base": "#F59E0B", "light": "#FDE68A", "dark": "#92400E", "grad": ["#FBBF24", "#D97706"]},
        {"base": "#3B82F6", "light": "#93C5FD", "dark": "#1E3A8A", "grad": ["#60A5FA", "#2563EB"]},
        {"base": "#10B981", "light": "#6EE7B7", "dark": "#064E3B", "grad": ["#34D399", "#059669"]},
        {"base": "#F43F5E", "light": "#FDA4AF", "dark": "#881337", "grad": ["#FB7185", "#E11D48"]},
        {"base": "#8B5CF6", "light": "#C4B5FD", "dark": "#4C1D95", "grad": ["#A78BFA", "#7C3AED"]},
        {"base": "#06B6D4", "light": "#67E8F9", "dark": "#155E75", "grad": ["#22D3EE", "#0891B2"]},
        {"base": "#EC4899", "light": "#F9A8D4", "dark": "#831843", "grad": ["#F472B6", "#DB2777"]},
        {"base": "#14B8A6", "light": "#5EEAD4", "dark": "#134E4A", "grad": ["#2DD4BF", "#0D9488"]},
        {"base": "#F97316", "light": "#FDBA74", "dark": "#7C2D12", "grad": ["#FB923C", "#EA580C"]},
        {"base": "#6366F1", "light": "#A5B4FC", "dark": "#3730A3", "grad": ["#818CF8", "#4F46E5"]},
        {"base": "#84CC16", "light": "#BEF264", "dark": "#3F6212", "grad": ["#A3E635", "#65A30D"]},
        {"base": "#E879F9", "light": "#F0ABFC", "dark": "#701A75", "grad": ["#D946EF", "#C026D3"]},
    ])

    def to_js_config(self) -> str:
        """Serialise the interactive-diagram settings to a JavaScript object literal.

        This is useful when you want to inject the configuration into an
        HTML template so the ``StateTransitionDiagram`` constructor can
        pick it up.

        Returns
        -------
        str
            A JavaScript object literal string.
        """
        import json
        return json.dumps({
            "stateColors": self.interactive_state_colors,
            "obsColor": {
                "fill": self.observation_fill,
                "stroke": self.observation_stroke,
                "dark": self.observation_text,
            },
            "piColor": {
                "fill": self.start_fill,
                "stroke": self.start_stroke,
                "dark": self.start_text,
            },
            "particlesEnabled": self.particles_enabled,
            "decongestionEnabled": self.decongestion_enabled,
            "animationSpeed": self.animation_speed,
            "fontFamily": self.font_family,
            "monoFontFamily": self.mono_font_family,
        }, indent=2)
