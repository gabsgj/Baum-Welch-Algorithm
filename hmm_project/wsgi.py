"""
wsgi.py
=======
WSGI application entry point for production deployment (Zeabur, Heroku, etc.).

Gunicorn will import and run this 'application' object.
"""

import os
from hmm_service.app import create_app, socketio

# Create the Flask app
app = create_app()

if __name__ == "__main__":
    # This is used only when running directly (not recommended for production)
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
