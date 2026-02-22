"""
app.py — Production entry point (Zeabur / Docker / local).

The committed Dockerfile runs ``python app.py`` directly, which starts
the eventlet-based Socket.IO server without Gunicorn.  The ``PORT``
environment variable is respected (Zeabur injects it); defaults to 5000
for local development.
"""

import os
from hmm_service.app import create_app, socketio

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
