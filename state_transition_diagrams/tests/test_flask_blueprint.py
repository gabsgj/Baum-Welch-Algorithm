"""
test_flask_blueprint.py
=======================
Tests for ``state_transition_diagrams.flask_blueprint.create_blueprint``.
"""

from __future__ import annotations

import os

import pytest

flask = pytest.importorskip("flask")
from flask import Flask

from state_transition_diagrams import create_blueprint


@pytest.fixture
def app():
    """Create a minimal Flask app with the library blueprint registered."""
    application = Flask(__name__)
    application.register_blueprint(create_blueprint(), url_prefix="/std")
    application.config["TESTING"] = True
    return application


@pytest.fixture
def client(app):
    return app.test_client()


class TestBlueprintRegistration:
    """Blueprint registration and basic serving."""

    def test_blueprint_registers(self, app):
        assert "state_transition_diagrams" in app.blueprints

    def test_custom_name(self):
        application = Flask(__name__)
        application.register_blueprint(
            create_blueprint(name="my_diagrams"), url_prefix="/diag"
        )
        assert "my_diagrams" in application.blueprints

    def test_custom_url_prefix(self):
        application = Flask(__name__)
        application.register_blueprint(
            create_blueprint(url_prefix="/custom"), url_prefix="/custom"
        )
        client = application.test_client()
        resp = client.get("/custom/js/state_diagram.js")
        assert resp.status_code == 200


class TestStaticAssetServing:
    """Verify that static JS and CSS files are served correctly."""

    def test_js_file_served(self, client):
        resp = client.get("/std/js/state_diagram.js")
        assert resp.status_code == 200
        data = resp.get_data(as_text=True)
        assert "StateTransitionDiagram" in data

    def test_css_file_served(self, client):
        resp = client.get("/std/css/state_diagram.css")
        assert resp.status_code == 200
        data = resp.get_data(as_text=True)
        assert ".std-canvas" in data

    def test_js_content_type(self, client):
        resp = client.get("/std/js/state_diagram.js")
        ct = resp.content_type
        assert "javascript" in ct or "text" in ct

    def test_css_content_type(self, client):
        resp = client.get("/std/css/state_diagram.css")
        ct = resp.content_type
        assert "css" in ct or "text" in ct

    def test_nonexistent_file_404(self, client):
        resp = client.get("/std/js/nonexistent.js")
        assert resp.status_code == 404


class TestStaticFilesExist:
    """Verify the static directory contains expected files."""

    def test_static_dir_exists(self):
        static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
        assert os.path.isdir(static_dir)

    def test_js_file_on_disk(self):
        js_path = os.path.join(
            os.path.dirname(__file__), "..", "static", "js", "state_diagram.js"
        )
        assert os.path.isfile(js_path)

    def test_css_file_on_disk(self):
        css_path = os.path.join(
            os.path.dirname(__file__), "..", "static", "css", "state_diagram.css"
        )
        assert os.path.isfile(css_path)


class TestJsFileContents:
    """Verify the JS file contains all expected public API methods."""

    @pytest.fixture(autouse=True)
    def load_js(self):
        js_path = os.path.join(
            os.path.dirname(__file__), "..", "static", "js", "state_diagram.js"
        )
        with open(js_path, "r", encoding="utf-8") as f:
            self.js_source = f.read()

    def test_class_declaration(self):
        assert "class StateTransitionDiagram" in self.js_source

    def test_hmm_alias(self):
        assert "HMMDiagram" in self.js_source

    def test_feed_iteration(self):
        assert "feedIteration" in self.js_source

    def test_on_complete(self):
        assert "onComplete" in self.js_source

    def test_play(self):
        assert "play()" in self.js_source or "play(" in self.js_source

    def test_pause(self):
        assert "pause()" in self.js_source or "pause(" in self.js_source

    def test_step_forward(self):
        assert "stepForward" in self.js_source

    def test_step_back(self):
        assert "stepBack" in self.js_source

    def test_go_first(self):
        assert "goFirst" in self.js_source

    def test_go_last(self):
        assert "goLast" in self.js_source

    def test_seek_to(self):
        assert "seekTo" in self.js_source

    def test_set_speed(self):
        assert "setSpeed" in self.js_source

    def test_toggle_particles(self):
        assert "toggleParticles" in self.js_source

    def test_toggle_decongestion(self):
        assert "toggleDecongestion" in self.js_source

    def test_toggle_3d(self):
        assert "toggle3D" in self.js_source

    def test_wire_controls(self):
        assert "wireControls" in self.js_source

    def test_reset(self):
        assert "reset()" in self.js_source or "reset(" in self.js_source

    def test_save_svg(self):
        assert "saveSVG" in self.js_source

    def test_save_png(self):
        assert "savePNG" in self.js_source

    def test_render_inspector(self):
        assert "_renderInspector" in self.js_source

    def test_fullscreen_handler(self):
        assert "_handleFullscreenChange" in self.js_source


class TestCssFileContents:
    """Verify the CSS file contains all expected class selectors."""

    @pytest.fixture(autouse=True)
    def load_css(self):
        css_path = os.path.join(
            os.path.dirname(__file__), "..", "static", "css", "state_diagram.css"
        )
        with open(css_path, "r", encoding="utf-8") as f:
            self.css_source = f.read()

    def test_std_canvas(self):
        assert ".std-canvas" in self.css_source

    def test_std_controls(self):
        assert ".std-controls" in self.css_source

    def test_std_ctrl_btn(self):
        assert ".std-ctrl-btn" in self.css_source

    def test_std_speed_select(self):
        assert ".std-speed-select" in self.css_source

    def test_std_timeline(self):
        assert ".std-timeline" in self.css_source

    def test_std_iter_label(self):
        assert ".std-iter-label" in self.css_source

    def test_std_inspector(self):
        assert ".std-inspector" in self.css_source

    def test_std_mode_label(self):
        assert ".std-mode-label" in self.css_source

    def test_hmm_compat_canvas(self):
        assert ".hmm-canvas" in self.css_source

    def test_hmm_compat_controls(self):
        assert ".hmm-controls" in self.css_source

    def test_view_3d(self):
        assert ".view-3d" in self.css_source

    def test_fullscreen(self):
        assert "fullscreen" in self.css_source

    def test_mobile_responsive(self):
        assert "@media" in self.css_source
