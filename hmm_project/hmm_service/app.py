"""
app.py
======
Flask + SocketIO application factory.

Serves the single-page dashboard at ``/`` and mounts the ``/api``
REST blueprint.  The ``socketio`` instance is exported so that
``routes.py`` can register WebSocket event handlers.
"""

from __future__ import annotations

from flask import Flask, render_template
from flask_socketio import SocketIO

from hmm_service.api.routes import api_bp, register_socket_events
from state_transition_diagrams import create_blueprint as std_blueprint

# SocketIO instance — shared with the routes module.
socketio = SocketIO(cors_allowed_origins="*", async_mode="eventlet")


def create_app() -> Flask:
    """Build and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["JSON_SORT_KEYS"] = False

    app.register_blueprint(api_bp)
    app.register_blueprint(std_blueprint(), url_prefix="/std")
    socketio.init_app(app)
    register_socket_events(socketio)

    @app.route("/")
    def index():
        return render_template("index.html")

    return app


# ─── Dev entry-point ─── #
if __name__ == "__main__":
    application = create_app()
    socketio.run(application, debug=True, host="0.0.0.0", port=5000)
