"""
test_package.py
===============
Tests for package-level imports and metadata.
"""

from __future__ import annotations

import pytest


class TestPackageImports:
    """Verify top-level imports work."""

    def test_import_package(self):
        import state_transition_diagrams
        assert hasattr(state_transition_diagrams, "__version__")

    def test_import_diagram_config(self):
        from state_transition_diagrams import DiagramConfig
        assert DiagramConfig is not None

    def test_import_render_state_diagram(self):
        from state_transition_diagrams import render_state_diagram
        assert callable(render_state_diagram)

    def test_import_create_blueprint(self):
        from state_transition_diagrams import create_blueprint
        assert callable(create_blueprint)

    def test_version_string(self):
        from state_transition_diagrams import __version__
        assert isinstance(__version__, str)
        parts = __version__.split(".")
        assert len(parts) >= 2

    def test_all_exports(self):
        from state_transition_diagrams import __all__
        assert "DiagramConfig" in __all__
        assert "render_state_diagram" in __all__
        assert "create_blueprint" in __all__


class TestDiagramConfigIsDataclass:
    """Verify DiagramConfig behaves like a dataclass."""

    def test_is_dataclass(self):
        from dataclasses import is_dataclass
        from state_transition_diagrams import DiagramConfig
        assert is_dataclass(DiagramConfig)

    def test_equality(self):
        from state_transition_diagrams import DiagramConfig
        a = DiagramConfig()
        b = DiagramConfig()
        assert a == b

    def test_inequality_on_override(self):
        from state_transition_diagrams import DiagramConfig
        a = DiagramConfig()
        b = DiagramConfig(title="Different")
        assert a != b

    def test_repr_contains_class_name(self):
        from state_transition_diagrams import DiagramConfig
        r = repr(DiagramConfig())
        assert "DiagramConfig" in r
