"""
flask_blueprint.py
==================
Flask blueprint that serves the interactive state transition diagram
static assets (JS + CSS).

Usage
-----
::

    from state_transition_diagrams import create_blueprint

    # Register with url_prefix so assets are available at /std/...
    app.register_blueprint(create_blueprint(), url_prefix="/std")

Then in your HTML template::

    <link rel="stylesheet" href="/std/css/state_diagram.css">
    <script src="/std/js/state_diagram.js"></script>
"""

from __future__ import annotations

import os

from flask import Blueprint


def create_blueprint(
    name: str = "state_transition_diagrams",
    url_prefix: str = "/std",
) -> Blueprint:
    """Create a Flask blueprint that serves the library's static files.

    Parameters
    ----------
    name : str
        Blueprint name (must be unique within the app).
    url_prefix : str
        URL prefix under which the static files are served.

    Returns
    -------
    flask.Blueprint
    """
    static_dir = os.path.join(os.path.dirname(__file__), "static")

    bp = Blueprint(
        name,
        __name__,
        static_folder=static_dir,
        static_url_path="",
        url_prefix=url_prefix,
    )
    return bp
