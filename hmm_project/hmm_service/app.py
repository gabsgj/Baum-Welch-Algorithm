"""
app.py
======
Flask application factory.

Serves the single-page UI at ``/`` and mounts the ``/api`` blueprint.
"""

from __future__ import annotations

from flask import Flask, render_template

from hmm_service.api.routes import api_bp


def create_app() -> Flask:
    """Build and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["JSON_SORT_KEYS"] = False

    app.register_blueprint(api_bp)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app


# ─── Dev entry-point ─── #
if __name__ == "__main__":
    application = create_app()
    application.run(debug=True, port=5000)
