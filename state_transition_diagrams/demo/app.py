"""
demo/app.py
===========
Standalone demo Flask application for the ``state_transition_diagrams`` library.

This app is completely independent from any external project and showcases
**every** feature of the library:

* Interactive D3.js diagram with all controls
* Static Graphviz rendering (SVG / PNG)
* Flask blueprint asset serving
* DiagramConfig customisation + ``to_js_config()``
* Simulated Baum-Welch training iterations via a REST endpoint

Run
---
::

    cd state_transition_diagrams
    pip install -e ".[all]"
    python -m demo.app          # or: python demo/app.py
    # Open http://127.0.0.1:5050

"""

from __future__ import annotations

import json
import os
import sys
import random
import tempfile
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

# ---------------------------------------------------------------------------
# Ensure the library is importable even when running from the demo/ folder
# ---------------------------------------------------------------------------
_lib_root = Path(__file__).resolve().parent.parent
if str(_lib_root.parent) not in sys.path:
    sys.path.insert(0, str(_lib_root.parent))

from state_transition_diagrams import create_blueprint, DiagramConfig, render_state_diagram

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)

# Register the library blueprint so assets are at /std/css/… and /std/js/…
app.register_blueprint(create_blueprint(), url_prefix="/std")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise a matrix so each row sums to 1."""
    sums = matrix.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    return matrix / sums


def _random_transition_matrix(n: int, *, rng: np.random.Generator) -> np.ndarray:
    raw = rng.dirichlet(np.ones(n), size=n)
    return raw


def _random_emission_matrix(n: int, m: int, *, rng: np.random.Generator) -> np.ndarray:
    raw = rng.dirichlet(np.ones(m), size=n)
    return raw


def _random_initial(n: int, *, rng: np.random.Generator) -> np.ndarray:
    return rng.dirichlet(np.ones(n))


def _simulate_training(
    n_states: int,
    n_obs: int,
    n_iterations: int,
    seed: int | None = None,
) -> list[dict]:
    """Simulate Baum-Welch–style training iterations (synthetic convergence)."""
    rng = np.random.default_rng(seed)

    # Target (converged) parameters
    A_target = _random_transition_matrix(n_states, rng=rng)
    B_target = _random_emission_matrix(n_states, n_obs, rng=rng)
    pi_target = _random_initial(n_states, rng=rng)

    # Starting parameters (far from target)
    A = _random_transition_matrix(n_states, rng=rng)
    B = _random_emission_matrix(n_states, n_obs, rng=rng)
    pi = _random_initial(n_states, rng=rng)

    iterations = []
    ll = -500.0 - rng.random() * 200

    for i in range(1, n_iterations + 1):
        # Blend toward target (simulating convergence)
        alpha = 1.0 - np.exp(-0.12 * i)
        A = _normalise_rows((1 - alpha) * A + alpha * A_target + rng.normal(0, 0.005 / i, A.shape).clip(-0.05, 0.05))
        B = _normalise_rows((1 - alpha) * B + alpha * B_target + rng.normal(0, 0.005 / i, B.shape).clip(-0.05, 0.05))
        pi_raw = (1 - alpha) * pi + alpha * pi_target + rng.normal(0, 0.005 / i, pi.shape).clip(-0.05, 0.05)
        pi_raw = np.clip(pi_raw, 1e-10, None)
        pi = pi_raw / pi_raw.sum()

        ll += rng.uniform(1.0, 15.0) * np.exp(-0.05 * i)

        iterations.append({
            "A": A.tolist(),
            "B": B.tolist(),
            "pi": pi.tolist(),
            "iteration": i,
            "log_likelihood": round(float(ll), 4),
        })

    return iterations


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the main demo page."""
    config = DiagramConfig(
        particles_enabled=True,
        decongestion_enabled=False,
        animation_speed=1.0,
    )
    return render_template("demo.html", js_config=config.to_js_config())


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Generate simulated training iterations and return as JSON.

    Request body (JSON)::

        {
            "n_states": 3,
            "n_observations": 4,
            "n_iterations": 30,
            "seed": 42          // optional
        }
    """
    data = request.get_json(silent=True) or {}
    n_states = int(data.get("n_states", 3))
    n_obs = int(data.get("n_observations", 4))
    n_iter = int(data.get("n_iterations", 30))
    seed_val = data.get("seed")
    seed = int(seed_val) if seed_val is not None else random.randint(0, 2**31)

    n_states = max(2, min(n_states, 12))
    n_obs = max(2, min(n_obs, 12))
    n_iter = max(1, min(n_iter, 200))

    iterations = _simulate_training(n_states, n_obs, n_iter, seed=seed)
    return jsonify({"iterations": iterations, "seed": seed})


@app.route("/api/render_static", methods=["POST"])
def render_static():
    """Render a static Graphviz diagram and return the file.

    Request body (JSON)::

        {
            "A": [[0.7, 0.3], [0.4, 0.6]],
            "state_labels": ["Rainy", "Sunny"],
            "format": "svg",
            "title": "Weather HMM"
        }
    """
    data = request.get_json(silent=True) or {}
    A = np.array(data.get("A", [[0.7, 0.3], [0.4, 0.6]]))
    labels = data.get("state_labels")
    fmt = data.get("format", "svg")
    title = data.get("title", "State Transition Diagram")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "diagram")
        cfg = DiagramConfig(title=title, output_format=fmt)
        render_state_diagram(A, state_labels=labels, save_path=save_path, config=cfg, fmt=fmt)

        out_file = save_path + "." + fmt
        mime = {
            "svg": "image/svg+xml",
            "png": "image/png",
            "pdf": "application/pdf",
        }.get(fmt, "application/octet-stream")

        return send_file(out_file, mimetype=mime, as_attachment=True, download_name=f"diagram.{fmt}")


@app.route("/api/config", methods=["GET"])
def get_config():
    """Return the default DiagramConfig as JSON (for inspection)."""
    cfg = DiagramConfig()
    return jsonify({
        "state_colors": cfg.state_colors,
        "interactive_state_colors": cfg.interactive_state_colors,
        "observation_fill": cfg.observation_fill,
        "observation_stroke": cfg.observation_stroke,
        "observation_text": cfg.observation_text,
        "start_fill": cfg.start_fill,
        "start_stroke": cfg.start_stroke,
        "start_text": cfg.start_text,
        "background_color": cfg.background_color,
        "font_family": cfg.font_family,
        "mono_font_family": cfg.mono_font_family,
        "title": cfg.title,
        "prob_threshold": cfg.prob_threshold,
        "edge_color": cfg.edge_color,
        "node_shape": cfg.node_shape,
        "node_radius": cfg.node_radius,
        "layout_engine": cfg.layout_engine,
        "output_format": cfg.output_format,
        "particles_enabled": cfg.particles_enabled,
        "decongestion_enabled": cfg.decongestion_enabled,
        "animation_speed": cfg.animation_speed,
        "js_config": json.loads(cfg.to_js_config()),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  ✦  State Transition Diagrams — Demo App")
    print("     http://127.0.0.1:5050\n")
    app.run(host="127.0.0.1", port=5050, debug=True)
