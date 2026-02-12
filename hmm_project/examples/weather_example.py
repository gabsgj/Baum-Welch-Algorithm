#!/usr/bin/env python3
"""
weather_example.py
==================
Classic Rainy / Sunny HMM demonstration.

Hidden states:  Rainy (0), Sunny (1)
Observations:   Walk (0), Shop (1), Clean (2)

This script:
    1. Generates a synthetic observation sequence.
    2. Trains an HMM from random initialisation using Baum-Welch.
    3. Prints the learned parameters.
    4. Generates all visualisations (likelihood curve, heatmaps,
       parameter trajectory, and state diagram).

Usage
-----
    cd hmm_project
    python examples/weather_example.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# Ensure hmm_project is on the path when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hmm_core.training.trainer import HMMTrainer
from hmm_visualization.training_plots import plot_log_likelihood
from hmm_visualization.parameter_trajectory import plot_parameter_evolution
from hmm_visualization.heatmaps import plot_heatmap
from hmm_visualization.state_diagram import render_state_diagram

# ─── Logging ─── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    # ────────────────────────────────────────────────────────────────── #
    #  Ground-truth model (for generating observations)                 #
    # ────────────────────────────────────────────────────────────────── #
    #
    #   States:   Rainy=0, Sunny=1
    #   Obs:      Walk=0,  Shop=1,  Clean=2
    #
    #   True A =  [[0.7, 0.3],          True B = [[0.1, 0.4, 0.5],
    #              [0.4, 0.6]]                     [0.6, 0.3, 0.1]]
    #
    #   True π =  [0.6, 0.4]
    #

    A_true = np.array([[0.7, 0.3],
                       [0.4, 0.6]])
    B_true = np.array([[0.1, 0.4, 0.5],
                       [0.6, 0.3, 0.1]])
    pi_true = np.array([0.6, 0.4])

    state_labels = ["Rainy", "Sunny"]
    obs_labels = ["Walk", "Shop", "Clean"]

    # ─── Generate synthetic observations ─── #
    rng = np.random.default_rng(seed=2024)
    T = 1000  # observation sequence length (Increased for robustness)

    states = np.zeros(T, dtype=np.intp)
    observations = np.zeros(T, dtype=np.intp)

    states[0] = rng.choice(2, p=pi_true)
    observations[0] = rng.choice(3, p=B_true[states[0]])

    for t in range(1, T):
        states[t] = rng.choice(2, p=A_true[states[t - 1]])
        observations[t] = rng.choice(3, p=B_true[states[t]])

    logger.info("Generated %d observations from ground-truth Weather HMM.", T)
    logger.info("First 20 observations: %s", observations[:20].tolist())

    # ────────────────────────────────────────────────────────────────── #
    #  Train from random initialisation                                 #
    # ────────────────────────────────────────────────────────────────── #
    trainer = HMMTrainer(
        n_states=2,
        n_obs_symbols=3,
        max_iterations=200,
        tolerance=1e-8,
        seed=6,  # Found to converge to ground truth for this problem
    )
    result = trainer.fit(observations)

    logger.info("Training complete.")
    logger.info("  Converged:   %s", result.converged)
    logger.info("  Iterations:  %d", result.n_iterations)
    logger.info("  Final LL:    %.6f", result.log_likelihood_history[-1])

    # ─── Print learned parameters ─── #
    print("\n" + "=" * 60)
    print("  LEARNED HMM PARAMETERS (Weather Example)")
    print("=" * 60)
    print(f"\n  Transition matrix A (rows = {state_labels}):")
    for i, label in enumerate(state_labels):
        row = "  ".join(f"{v:.4f}" for v in result.model_params.A[i])
        print(f"    {label:>6s} | {row}")

    print(f"\n  Emission matrix B (cols = {obs_labels}):")
    for i, label in enumerate(state_labels):
        row = "  ".join(f"{v:.4f}" for v in result.model_params.B[i])
        print(f"    {label:>6s} | {row}")

    print("\n  Initial distribution pi:")
    for i, label in enumerate(state_labels):
        print(f"    {label:>6s} | {result.model_params.pi[i]:.4f}")

    print(f"\n  Ground truth A:\n    {A_true}")
    print(f"  Ground truth B:\n    {B_true}")
    print(f"  Ground truth pi: {pi_true}")
    print("=" * 60)

    # ────────────────────────────────────────────────────────────────── #
    #  Visualisations                                                   #
    # ────────────────────────────────────────────────────────────────── #
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)

    # 1. Log-likelihood curve.
    plot_log_likelihood(
        result.log_likelihood_history,
        title="Weather HMM — Baum-Welch Convergence",
        save_path=out_dir / "convergence.svg",
        show=False,
    )
    logger.info("Saved convergence plot → %s", out_dir / "convergence.svg")

    # 2. Parameter trajectory.
    plot_parameter_evolution(
        result.parameter_history,
        title="Weather HMM — Parameter Evolution",
        save_path=out_dir / "parameter_trajectory.svg",
        show=False,
    )
    logger.info("Saved parameter trajectory → %s", out_dir / "parameter_trajectory.svg")

    # 3. Heatmaps.
    plot_heatmap(
        result.model_params.A,
        title="Learned Transition Matrix A",
        row_labels=state_labels,
        col_labels=state_labels,
        save_path=out_dir / "heatmap_A.svg",
        show=False,
    )
    plot_heatmap(
        result.model_params.B,
        title="Learned Emission Matrix B",
        row_labels=state_labels,
        col_labels=obs_labels,
        save_path=out_dir / "heatmap_B.svg",
        show=False,
    )
    logger.info("Saved heatmaps → %s", out_dir)

    # 4. State diagram.
    try:
        render_state_diagram(
            result.model_params.A,
            state_labels=state_labels,
            save_path=out_dir / "state_diagram",
            fmt="svg",
            title="Learned Weather HMM",
        )
        logger.info("Saved state diagram → %s", out_dir / "state_diagram.svg")
    except Exception as exc:
        logger.warning(
            "Graphviz state diagram skipped: %s. "
            "Install the Graphviz system binary to render state diagrams.", exc
        )

    print(f"\n  All visualizations saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
