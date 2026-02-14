"""
test_renderer.py
================
Tests for ``state_transition_diagrams.renderer.render_state_diagram``.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from state_transition_diagrams.config import DiagramConfig

# graphviz may not be installed — skip tests gracefully
graphviz = pytest.importorskip("graphviz")

from state_transition_diagrams.renderer import render_state_diagram

# The Python graphviz package can be installed while the system
# Graphviz binaries (dot, neato, …) are absent.  Tests that need to
# actually *render* to a file require the binary on PATH.
_HAS_GRAPHVIZ_BINARY = shutil.which("dot") is not None
needs_dot = pytest.mark.skipif(
    not _HAS_GRAPHVIZ_BINARY,
    reason="Graphviz 'dot' executable not found on PATH",
)


@pytest.fixture
def simple_A():
    return np.array([[0.7, 0.3], [0.4, 0.6]])


@pytest.fixture
def three_state_A():
    return np.array([
        [0.5, 0.3, 0.2],
        [0.1, 0.7, 0.2],
        [0.3, 0.3, 0.4],
    ])


class TestRenderBasic:
    """Basic rendering without saving."""

    def test_returns_digraph(self, simple_A):
        dot = render_state_diagram(simple_A)
        assert isinstance(dot, graphviz.Digraph)

    def test_graph_name(self, simple_A):
        dot = render_state_diagram(simple_A)
        assert dot.name == "StateDiagram"

    def test_default_format_svg(self, simple_A):
        dot = render_state_diagram(simple_A)
        assert dot.format == "svg"

    def test_default_engine_circo(self, simple_A):
        dot = render_state_diagram(simple_A)
        assert dot.engine == "circo"

    def test_default_labels(self, simple_A):
        dot = render_state_diagram(simple_A)
        source = dot.source
        assert "S0" in source
        assert "S1" in source

    def test_custom_labels(self, simple_A):
        dot = render_state_diagram(simple_A, state_labels=["Rainy", "Sunny"])
        source = dot.source
        assert "Rainy" in source
        assert "Sunny" in source

    def test_three_state(self, three_state_A):
        dot = render_state_diagram(three_state_A, state_labels=["A", "B", "C"])
        source = dot.source
        for label in ("A", "B", "C"):
            assert label in source


class TestRenderWithConfig:
    """Rendering with custom DiagramConfig."""

    def test_custom_format_png(self, simple_A):
        dot = render_state_diagram(simple_A, fmt="png")
        assert dot.format == "png"

    def test_custom_format_pdf(self, simple_A):
        dot = render_state_diagram(simple_A, fmt="pdf")
        assert dot.format == "pdf"

    def test_format_override_takes_precedence(self, simple_A):
        cfg = DiagramConfig(output_format="pdf")
        dot = render_state_diagram(simple_A, config=cfg, fmt="png")
        assert dot.format == "png"

    def test_config_layout_engine(self, simple_A):
        cfg = DiagramConfig(layout_engine="dot")
        dot = render_state_diagram(simple_A, config=cfg)
        assert dot.engine == "dot"

    def test_config_title(self, simple_A):
        dot = render_state_diagram(simple_A, title="My Custom Title")
        assert "My Custom Title" in dot.source

    def test_config_background_color(self, simple_A):
        cfg = DiagramConfig(background_color="#FFFFFF")
        dot = render_state_diagram(simple_A, config=cfg)
        assert "#FFFFFF" in dot.source

    def test_custom_state_colors(self, simple_A):
        cfg = DiagramConfig(state_colors=["#E74C3C", "#3498DB"])
        dot = render_state_diagram(simple_A, config=cfg)
        assert "#E74C3C" in dot.source
        assert "#3498DB" in dot.source

    def test_config_node_shape(self, simple_A):
        cfg = DiagramConfig(node_shape="doublecircle")
        dot = render_state_diagram(simple_A, config=cfg)
        assert "doublecircle" in dot.source

    def test_config_font_family(self, simple_A):
        cfg = DiagramConfig(font_family="Arial")
        dot = render_state_diagram(simple_A, config=cfg)
        assert "Arial" in dot.source


class TestRenderProbThreshold:
    """Edge suppression via prob_threshold."""

    def test_low_threshold_includes_all_edges(self, simple_A):
        dot = render_state_diagram(simple_A, prob_threshold=0.0)
        source = dot.source
        # 2×2 matrix with all values > 0 → 4 edges
        assert source.count("->") == 4

    def test_high_threshold_suppresses_edges(self):
        A = np.array([[0.95, 0.05], [0.02, 0.98]])
        dot = render_state_diagram(A, prob_threshold=0.1)
        source = dot.source
        # Only self-loops (0.95 and 0.98) should remain
        assert source.count("->") == 2

    def test_threshold_from_config(self):
        A = np.array([[0.95, 0.05], [0.02, 0.98]])
        cfg = DiagramConfig(prob_threshold=0.5)
        dot = render_state_diagram(A, config=cfg)
        source = dot.source
        assert source.count("->") == 2

    def test_threshold_param_overrides_config(self):
        A = np.array([[0.95, 0.05], [0.02, 0.98]])
        cfg = DiagramConfig(prob_threshold=0.001)
        dot = render_state_diagram(A, config=cfg, prob_threshold=0.5)
        source = dot.source
        assert source.count("->") == 2


@needs_dot
class TestRenderSaveToFile:
    """Rendering to disk."""

    def test_save_svg(self, simple_A):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_diagram"
            render_state_diagram(simple_A, save_path=save_path)
            assert (Path(tmpdir) / "test_diagram.svg").exists()

    def test_save_png(self, simple_A):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_diagram"
            render_state_diagram(simple_A, save_path=save_path, fmt="png")
            assert (Path(tmpdir) / "test_diagram.png").exists()

    def test_save_path_as_string(self, simple_A):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = str(Path(tmpdir) / "from_string")
            render_state_diagram(simple_A, save_path=save_path)
            assert Path(save_path + ".svg").exists()


class TestRenderEdgeCases:
    """Edge cases and larger matrices."""

    def test_single_state(self):
        A = np.array([[1.0]])
        dot = render_state_diagram(A)
        assert "S0" in dot.source

    def test_large_matrix(self):
        rng = np.random.default_rng(42)
        N = 10
        A = rng.dirichlet(np.ones(N), size=N)
        dot = render_state_diagram(A)
        for i in range(N):
            assert f"S{i}" in dot.source

    def test_sparse_matrix(self):
        A = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        dot = render_state_diagram(A)
        source = dot.source
        # Only 3 self-loops
        assert source.count("->") == 3

    def test_zero_matrix(self):
        A = np.zeros((3, 3))
        dot = render_state_diagram(A, prob_threshold=0.0)
        # No edges at all when all probabilities are 0 and max_prob fallback is 1
        assert isinstance(dot, graphviz.Digraph)

    def test_labels_cycling_when_fewer_colours(self):
        A = np.eye(5)
        cfg = DiagramConfig(state_colors=["#AA0000", "#00BB00"])
        dot = render_state_diagram(A, config=cfg)
        assert isinstance(dot, graphviz.Digraph)
