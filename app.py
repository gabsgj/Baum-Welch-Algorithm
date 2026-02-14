"""
app.py — Production entry point for Zeabur / Gunicorn.

Zeabur auto-detects this file and uses Gunicorn to serve it.
The ``app`` variable is the WSGI application object.
The ``socketio`` instance wraps it for WebSocket support.
"""

import os
from hmm_service.app import create_app, socketio

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
