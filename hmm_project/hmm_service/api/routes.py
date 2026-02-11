"""
routes.py
=========
Flask API blueprint for HMM training.

Endpoints
---------
POST /api/train
    Accept observation sequence + config, return trained parameters + plots.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from hmm_service.api.schemas import TrainRequest, TrainResponse
from hmm_service.services.hmm_runner import run_training
from hmm_service.services.model_store import store

api_bp = Blueprint("api", __name__, url_prefix="/api")


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

    response = TrainResponse(
        model_id=model_id,
        converged=result["converged"],
        n_iterations=result["n_iterations"],
        final_log_likelihood=result["final_log_likelihood"],
        A=result["A"],
        B=result["B"],
        pi=result["pi"],
        log_likelihood_history=result["log_likelihood_history"],
        plots=result["plots"],
    )
    return jsonify(response.to_dict()), 200
