"""
styles.py
=========
Centralised style configuration for all HMM visualizations.

Provides a consistent color palette, typography, figure sizes, and a
context manager that temporarily activates a publication-quality
Matplotlib RC configuration.
"""

from __future__ import annotations

import contextlib
from typing import Generator

import matplotlib as mpl
import matplotlib.pyplot as plt

# ──────────────────────────── Colour Palette ─────────────────────────── #

PALETTE = {
    "primary":   "#2E86AB",   # Steel blue
    "secondary": "#A23B72",   # Deep magenta
    "accent":    "#F18F01",   # Warm amber
    "success":   "#2CA58D",   # Teal green
    "dark":      "#1B1B2F",   # Near black
    "light":     "#F5F5F5",   # Off white
    "grid":      "#CCCCCC",
}

STATE_COLORS: list[str] = [
    "#2E86AB", "#A23B72", "#F18F01", "#2CA58D",
    "#E84855", "#6B4226", "#7768AE", "#1B998B",
]

# ──────────────────────────── Defaults ───────────────────────────────── #

DEFAULT_FIGSIZE = (10, 6)
HEATMAP_CMAP = "viridis"
DPI = 150
FONT_FAMILY = "sans-serif"
FONT_SIZE = 12


# ──────────────────────────── RC Context ─────────────────────────────── #

_RC_PARAMS: dict = {
    "figure.figsize": DEFAULT_FIGSIZE,
    "figure.dpi": DPI,
    "font.family": FONT_FAMILY,
    "font.size": FONT_SIZE,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "grid.color": PALETTE["grid"],
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
    "savefig.bbox": "tight",
    "savefig.dpi": DPI,
}


@contextlib.contextmanager
def hmm_style() -> Generator[None, None, None]:
    """Context manager that temporarily applies HMM publication styles."""
    with mpl.rc_context(rc=_RC_PARAMS):
        yield
