"""
routes.py
=========
Flask API blueprint + SocketIO event handlers for HMM training.

REST endpoints
--------------
POST /api/train
    Synchronous training (kept for backward-compatibility).

WebSocket events
----------------
Client → Server:  "start_training"   { n_states, observations, max_iterations, tolerance }
Server → Client:  "training_update"  { iteration, log_likelihood, A, B, pi, converged }
Server → Client:  "training_complete" { converged, n_iterations, final_ll }
Server → Client:  "training_error"   { error }
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from flask import Blueprint, jsonify, request
from flask_socketio import SocketIO, emit

from hmm_service.api.schemas import TrainRequest
from hmm_service.services.hmm_runner import run_training, run_training_live
from hmm_service.services.model_store import store

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")


# ------------------------------------------------------------------ #
#  REST endpoint (backward-compatible)                                #
# ------------------------------------------------------------------ #

@api_bp.route("/train", methods=["POST"])
def train():
    """Train an HMM via Baum-Welch and return results + plots."""
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    try:
        req = TrainRequest.from_json(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422

    try:
        result = run_training(
            observations=req.observations,
            n_states=req.n_states,
            n_obs_symbols=req.n_obs_symbols,
            max_iterations=req.max_iterations,
            tolerance=req.tolerance,
            seed=req.seed,
        )
    except Exception as exc:
        return jsonify({"error": f"Training failed: {exc}"}), 500

    model_id = store.save(result)
    return jsonify({"model_id": model_id, **result}), 200


# ------------------------------------------------------------------ #
#  WebSocket event handlers                                           #
# ------------------------------------------------------------------ #

def register_socket_events(sio: SocketIO) -> None:
    """Attach SocketIO event handlers to the given instance."""

    @sio.on("start_training")
    def handle_start_training(data: dict[str, Any]) -> None:
        """Handle a training request over WebSocket."""
        logger.info("WebSocket start_training received: %s", data)

        # Capture the session id NOW, while we're still in the event context.
        # The background task runs in a separate greenlet and has no request context.
        sid = request.sid

        try:
            observations = data.get("observations", [])
            if isinstance(observations, str):
                observations = [int(x.strip()) for x in observations.split(",") if x.strip()]

            n_states = int(data.get("n_states", 2))
            max_iterations = int(data.get("max_iterations", 200))
            tolerance = float(data.get("tolerance", 1e-6))
            init_params = data.get("init_params")

            obs_array = np.array(observations, dtype=np.intp)
            n_obs_symbols = int(obs_array.max()) + 1 if len(obs_array) > 0 else 1

        except Exception as exc:
            emit("training_error", {"error": f"Invalid input: {exc}"})
            return

        def run_training_task() -> None:
            """Background greenlet: runs training and emits updates to the client."""

            def on_iteration(update: dict[str, Any]) -> None:
                sio.emit("training_update", update, to=sid)
                sio.sleep(0)  # yield so eventlet flushes the message

            try:
                result = run_training_live(
                    observations=observations,
                    n_states=n_states,
                    n_obs_symbols=n_obs_symbols,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    init_params=init_params,
                    on_iteration=on_iteration,
                )
                sio.emit("training_complete", {
                    "converged": result["converged"],
                    "n_iterations": result["n_iterations"],
                    "final_log_likelihood": result["final_log_likelihood"],
                }, to=sid)
            except Exception as exc:
                logger.exception("Training failed")
                sio.emit("training_error", {"error": str(exc)}, to=sid)

        sio.start_background_task(run_training_task)
