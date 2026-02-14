"""
test_demo_app.py
================
Tests for the standalone demo Flask application.

These tests verify that the demo app is completely self-contained
and all its endpoints work correctly.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

# Ensure the library root is importable
_lib_root = Path(__file__).resolve().parent.parent
if str(_lib_root.parent) not in sys.path:
    sys.path.insert(0, str(_lib_root.parent))

flask = pytest.importorskip("flask")

from demo.app import app as demo_app


@pytest.fixture
def client():
    demo_app.config["TESTING"] = True
    return demo_app.test_client()


class TestDemoIndex:
    """Test the main demo page."""

    def test_index_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_index_contains_html(self, client):
        data = client.get("/").get_data(as_text=True)
        assert "<!DOCTYPE html>" in data

    def test_index_loads_library_css(self, client):
        data = client.get("/").get_data(as_text=True)
        assert "/std/css/state_diagram.css" in data

    def test_index_loads_library_js(self, client):
        data = client.get("/").get_data(as_text=True)
        assert "/std/js/state_diagram.js" in data

    def test_index_loads_d3(self, client):
        data = client.get("/").get_data(as_text=True)
        assert "d3.v7" in data or "d3.js" in data

    def test_index_contains_js_config(self, client):
        data = client.get("/").get_data(as_text=True)
        assert "STD_SERVER_CONFIG" in data

    def test_index_contains_all_features(self, client):
        data = client.get("/").get_data(as_text=True)
        features = [
            "feedIteration", "onComplete", "play()", "pause()",
            "stepForward", "stepBack", "goFirst", "goLast",
            "seekTo", "setSpeed", "toggleParticles", "toggleDecongestion",
            "toggle3D", "saveSVG", "savePNG", "wireControls", "reset",
            "DiagramConfig", "render_state_diagram", "create_blueprint",
            "HMMDiagram",
        ]
        for feat in features:
            assert feat in data, f"Feature '{feat}' not found in demo page"


class TestDemoLibraryAssets:
    """Verify library assets are served via the blueprint."""

    def test_library_js(self, client):
        resp = client.get("/std/js/state_diagram.js")
        assert resp.status_code == 200
        assert "StateTransitionDiagram" in resp.get_data(as_text=True)

    def test_library_css(self, client):
        resp = client.get("/std/css/state_diagram.css")
        assert resp.status_code == 200
        assert ".std-canvas" in resp.get_data(as_text=True)


class TestSimulateEndpoint:
    """Test the /api/simulate endpoint."""

    def test_simulate_default(self, client):
        resp = client.post("/api/simulate", json={})
        assert resp.status_code == 200
        data = resp.get_json()
        assert "iterations" in data
        assert "seed" in data
        assert len(data["iterations"]) > 0

    def test_simulate_custom_params(self, client):
        resp = client.post("/api/simulate", json={
            "n_states": 4,
            "n_observations": 3,
            "n_iterations": 10,
            "seed": 42,
        })
        data = resp.get_json()
        assert len(data["iterations"]) == 10
        assert data["seed"] == 42

    def test_simulate_iteration_structure(self, client):
        resp = client.post("/api/simulate", json={"n_states": 2, "n_observations": 3, "n_iterations": 1, "seed": 99})
        data = resp.get_json()
        it = data["iterations"][0]
        assert "A" in it
        assert "B" in it
        assert "pi" in it
        assert "iteration" in it
        assert "log_likelihood" in it

    def test_simulate_matrix_dimensions(self, client):
        resp = client.post("/api/simulate", json={"n_states": 3, "n_observations": 5, "n_iterations": 1, "seed": 7})
        it = resp.get_json()["iterations"][0]
        A = it["A"]
        B = it["B"]
        pi = it["pi"]
        assert len(A) == 3
        assert len(A[0]) == 3
        assert len(B) == 3
        assert len(B[0]) == 5
        assert len(pi) == 3

    def test_simulate_clamped_states(self, client):
        # n_states capped at 12
        resp = client.post("/api/simulate", json={"n_states": 50, "n_iterations": 1})
        it = resp.get_json()["iterations"][0]
        assert len(it["A"]) <= 12

    def test_simulate_clamped_iterations(self, client):
        resp = client.post("/api/simulate", json={"n_iterations": 999})
        data = resp.get_json()
        assert len(data["iterations"]) <= 200

    def test_simulate_row_normalisation(self, client):
        """Transition matrix rows should approximately sum to 1."""
        resp = client.post("/api/simulate", json={"n_states": 3, "n_iterations": 1, "seed": 123})
        A = resp.get_json()["iterations"][0]["A"]
        for row in A:
            assert abs(sum(row) - 1.0) < 0.01


class TestConfigEndpoint:
    """Test the /api/config endpoint."""

    def test_config_200(self, client):
        resp = client.get("/api/config")
        assert resp.status_code == 200

    def test_config_structure(self, client):
        data = client.get("/api/config").get_json()
        expected_keys = [
            "state_colors", "interactive_state_colors",
            "observation_fill", "observation_stroke", "observation_text",
            "start_fill", "start_stroke", "start_text",
            "background_color", "font_family", "mono_font_family",
            "title", "prob_threshold", "edge_color",
            "node_shape", "node_radius", "layout_engine", "output_format",
            "particles_enabled", "decongestion_enabled", "animation_speed",
            "js_config",
        ]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

    def test_config_js_config_subobject(self, client):
        data = client.get("/api/config").get_json()
        js = data["js_config"]
        assert "stateColors" in js
        assert "obsColor" in js
        assert "piColor" in js
        assert "particlesEnabled" in js


class TestRenderStaticEndpoint:
    """Test the /api/render_static endpoint (requires graphviz binary)."""

    @pytest.fixture(autouse=True)
    def check_graphviz(self):
        pytest.importorskip("graphviz")
        if not shutil.which("dot"):
            pytest.skip("Graphviz 'dot' executable not found on PATH")

    def test_render_svg(self, client):
        resp = client.post("/api/render_static", json={
            "A": [[0.7, 0.3], [0.4, 0.6]],
            "state_labels": ["Rainy", "Sunny"],
            "format": "svg",
        })
        assert resp.status_code == 200
        assert "svg" in resp.content_type

    def test_render_default_matrix(self, client):
        resp = client.post("/api/render_static", json={})
        assert resp.status_code == 200
