"""
test_config.py
==============
Tests for ``state_transition_diagrams.config.DiagramConfig``.
"""

from __future__ import annotations

import json
import pytest

from state_transition_diagrams.config import DiagramConfig


class TestDiagramConfigDefaults:
    """Verify all default values are sensible."""

    def test_default_construction(self):
        cfg = DiagramConfig()
        assert isinstance(cfg, DiagramConfig)

    def test_state_colors_default_length(self):
        cfg = DiagramConfig()
        assert len(cfg.state_colors) == 8

    def test_state_colors_are_hex(self):
        cfg = DiagramConfig()
        for color in cfg.state_colors:
            assert color.startswith("#"), f"Expected hex colour, got {color!r}"

    def test_interactive_state_colors_length(self):
        cfg = DiagramConfig()
        assert len(cfg.interactive_state_colors) == 12

    def test_interactive_state_colors_structure(self):
        cfg = DiagramConfig()
        for entry in cfg.interactive_state_colors:
            assert "base" in entry
            assert "light" in entry
            assert "dark" in entry
            assert "grad" in entry
            assert isinstance(entry["grad"], list)
            assert len(entry["grad"]) == 2

    def test_default_observation_colors(self):
        cfg = DiagramConfig()
        assert cfg.observation_fill == "#F1F5F9"
        assert cfg.observation_stroke == "#94A3B8"
        assert cfg.observation_text == "#334155"

    def test_default_start_colors(self):
        cfg = DiagramConfig()
        assert cfg.start_fill == "#F5F3FF"
        assert cfg.start_stroke == "#8B5CF6"
        assert cfg.start_text == "#5B21B6"

    def test_default_background(self):
        assert DiagramConfig().background_color == "#FAFAFA"

    def test_default_typography(self):
        cfg = DiagramConfig()
        assert cfg.font_family == "Helvetica"
        assert "monospace" in cfg.mono_font_family

    def test_default_graph_settings(self):
        cfg = DiagramConfig()
        assert cfg.title == "State Transition Diagram"
        assert cfg.prob_threshold == 0.01
        assert cfg.edge_color == "#333333"
        assert cfg.node_shape == "circle"
        assert cfg.node_radius == 40.0
        assert cfg.layout_engine == "circo"
        assert cfg.output_format == "svg"

    def test_default_interactive_settings(self):
        cfg = DiagramConfig()
        assert cfg.particles_enabled is True
        assert cfg.decongestion_enabled is False
        assert cfg.animation_speed == 1.0


class TestDiagramConfigOverrides:
    """Verify config can be customised."""

    def test_override_state_colors(self):
        cfg = DiagramConfig(state_colors=["#AA0000", "#00BB00"])
        assert cfg.state_colors == ["#AA0000", "#00BB00"]

    def test_override_title(self):
        cfg = DiagramConfig(title="My Custom Title")
        assert cfg.title == "My Custom Title"

    def test_override_layout_engine(self):
        cfg = DiagramConfig(layout_engine="dot")
        assert cfg.layout_engine == "dot"

    def test_override_particles(self):
        cfg = DiagramConfig(particles_enabled=False)
        assert cfg.particles_enabled is False

    def test_override_decongestion(self):
        cfg = DiagramConfig(decongestion_enabled=True)
        assert cfg.decongestion_enabled is True

    def test_override_animation_speed(self):
        cfg = DiagramConfig(animation_speed=2.5)
        assert cfg.animation_speed == 2.5

    def test_override_prob_threshold(self):
        cfg = DiagramConfig(prob_threshold=0.1)
        assert cfg.prob_threshold == 0.1

    def test_override_output_format(self):
        cfg = DiagramConfig(output_format="png")
        assert cfg.output_format == "png"


class TestDiagramConfigToJsConfig:
    """Verify ``to_js_config()`` produces valid JSON."""

    def test_returns_valid_json(self):
        cfg = DiagramConfig()
        raw = cfg.to_js_config()
        data = json.loads(raw)
        assert isinstance(data, dict)

    def test_contains_state_colors(self):
        data = json.loads(DiagramConfig().to_js_config())
        assert "stateColors" in data
        assert len(data["stateColors"]) == 12

    def test_contains_obs_color(self):
        data = json.loads(DiagramConfig().to_js_config())
        obs = data["obsColor"]
        assert obs["fill"] == "#F1F5F9"
        assert obs["stroke"] == "#94A3B8"
        assert obs["dark"] == "#334155"

    def test_contains_pi_color(self):
        data = json.loads(DiagramConfig().to_js_config())
        pi = data["piColor"]
        assert pi["fill"] == "#F5F3FF"
        assert pi["stroke"] == "#8B5CF6"
        assert pi["dark"] == "#5B21B6"

    def test_contains_particles_enabled(self):
        data = json.loads(DiagramConfig().to_js_config())
        assert data["particlesEnabled"] is True

    def test_contains_decongestion_enabled(self):
        data = json.loads(DiagramConfig().to_js_config())
        assert data["decongestionEnabled"] is False

    def test_contains_animation_speed(self):
        data = json.loads(DiagramConfig().to_js_config())
        assert data["animationSpeed"] == 1.0

    def test_contains_font_family(self):
        data = json.loads(DiagramConfig().to_js_config())
        assert "fontFamily" in data
        assert "monoFontFamily" in data

    def test_custom_config_reflected_in_js(self):
        cfg = DiagramConfig(
            particles_enabled=False,
            decongestion_enabled=True,
            animation_speed=3.0,
        )
        data = json.loads(cfg.to_js_config())
        assert data["particlesEnabled"] is False
        assert data["decongestionEnabled"] is True
        assert data["animationSpeed"] == 3.0

    def test_custom_observation_colors_in_js(self):
        cfg = DiagramConfig(
            observation_fill="#FF0000",
            observation_stroke="#00FF00",
            observation_text="#0000FF",
        )
        data = json.loads(cfg.to_js_config())
        assert data["obsColor"]["fill"] == "#FF0000"
        assert data["obsColor"]["stroke"] == "#00FF00"
        assert data["obsColor"]["dark"] == "#0000FF"
