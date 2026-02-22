# HMM Baum-Welch Algorithm

> A production-grade Hidden Markov Model engine with real-time interactive web dashboard

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Flask](https://img.shields.io/badge/flask-3.0%2B-lightgrey.svg)](https://flask.palletsprojects.com/)
[![NumPy](https://img.shields.io/badge/numpy-1.24%2B-013243.svg)](https://numpy.org/)

A complete, from-scratch implementation of **Hidden Markov Model (HMM)** training using the **Baum-Welch algorithm** (Expectation-Maximization for HMMs). The project spans three layers:

- **`hmm_core`** — Pure-Python/NumPy mathematical engine: scaled forward-backward inference, responsibility computation ($\gamma$, $\xi$), Baum-Welch re-estimation, and convergence monitoring. Every intermediate variable is inspectable for educational use.
- **`hmm_service`** — Full-stack Flask + Socket.IO web application providing a single-page dashboard with real-time per-iteration training updates streamed over WebSocket.
- **`hmm_visualization`** — Matplotlib/Seaborn diagnostic plots (convergence curves, matrix heatmaps, parameter trajectories) and Graphviz state transition diagrams.

The codebase is designed to be both **educational** — walk through the Baum-Welch pipeline step by step with inspectable $\alpha$, $\beta$, $\gamma$, $\xi$ arrays — and **practical** — deploy as a web service or embed the core engine into your own Python projects.

### Key Features

| Feature | Description |
|---|---|
| **Real-time training** | Per-iteration parameter updates streamed via WebSocket; watch $A$, $B$, $\pi$ evolve live |
| **Animated state diagram** | D3.js-powered interactive diagram with particle flow animations, drag-to-rearrange nodes, replay controls, speed slider, and fullscreen mode |
| **Interactive heatmaps** | Plotly.js heatmaps for transition ($A$) and emission ($B$) matrices with hover detail |
| **Convergence plotting** | Log-likelihood curve rendered with Plotly.js, auto-scaled axes |
| **Programmatic API** | `HMMTrainer` class for scripting, Jupyter notebooks, and batch experiments |
| **REST + WebSocket API** | `POST /api/train` for synchronous training; `start_training` WebSocket event for streaming |
| **Numerical stability** | Per-timestep scaling prevents underflow on long sequences |
| **One-command deploy** | `Dockerfile` + `Procfile` + `zbpack.json` configure Zeabur; starts directly via eventlet — no Gunicorn required |
| **Modular architecture** | Each mathematical operation lives in its own module; easy to extend or swap components |

---

## Live Demo

**https://bwa.gabrieljames.me**

---

## Screenshots
![text](docs/screenshots/1.png)
![text](docs/screenshots/2.png)
![text](docs/screenshots/3.png)
![text](docs/screenshots/4.png)
![text](docs/screenshots/5.png)
![text](docs/screenshots/6.png)
![text](docs/screenshots/7.png)
![text](docs/screenshots/8.png)
![text](docs/screenshots/9.png)
![text](docs/screenshots/10.png)
![text](docs/screenshots/11.png)
![text](docs/screenshots/12.png)
![text](docs/screenshots/13.png)
![text](docs/screenshots/14.png)
![text](docs/screenshots/15.png)
![text](docs/screenshots/16.png)
![text](docs/screenshots/17.png)

---

## Table of Contents

- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Programmatic Usage](#programmatic-usage)
- [Web Dashboard Guide](#web-dashboard-guide)
- [API Reference](#api-reference)
- [Weather Example Walkthrough](#weather-example-walkthrough)
- [Theory Background](#theory-background)
- [Numerical Stability](#numerical-stability)
- [Math-to-Code Map](#math-to-code-map)
- [Data Preparation](#data-preparation)
- [Convergence and Model Selection](#convergence-and-model-selection)
- [Practical Applications](#practical-applications)
- [Testing](#testing)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Reproducibility Checklist](#reproducibility-checklist)
- [Contributing](#contributing)
- [License](#license)

---

## Tech Stack

| Layer | Technologies | Role |
|---|---|---|
| **Core engine** | Python 3.11+, NumPy ≥1.24 | Forward-backward algorithm, Baum-Welch re-estimation, scaling |
| **Web framework** | Flask ≥3.0, Flask-SocketIO ≥5.3, eventlet ≥0.35 | HTTP server, WebSocket transport, async worker |
| **Frontend** | Tailwind CSS 3.4, Plotly.js 2.27, D3.js v7, Socket.IO 4.7 | Dashboard styling, interactive charts, live diagrams |
| **Visualization** | Matplotlib ≥3.7, Seaborn ≥0.13, Graphviz ≥0.20 | Static plots, heatmaps, state transition diagrams |
| **Deployment** | Docker, Zeabur, eventlet | Container-based deploy; eventlet serves HTTP + WebSocket directly — no Gunicorn |
| **Dev tools** | pytest ≥7.0, pytest-cov, Node.js/npm (Tailwind rebuilds) | Testing and CSS build pipeline |

---

## Repository Structure

```text
Baum-Welch-Algorithm/
├── app.py                        # Production entry point — runs socketio.run() directly via eventlet
├── Dockerfile                    # Container definition used by Zeabur (python:3.11-slim, pip install -e .)
├── Procfile                      # Heroku-style fallback: web: python app.py
├── zbpack.json                   # Zeabur build hints (Python 3.11, start_command)
├── pyproject.toml                # PEP 621 packaging, dependencies, and tool config
├── requirements.txt              # Runtime dependencies (pip install -r)
├── package.json                  # npm scripts for Tailwind CSS build
├── tailwind.config.js            # Tailwind content paths and theme extensions
├── README.md                     # This file
│
├── hmm_core/                     # Pure-Python/NumPy mathematical engine
│   ├── __init__.py
│   ├── inference/                # E-step: forward-backward pass
│   │   ├── forward_backward.py   # Orchestrates scaled forward + backward + log-likelihood
│   │   ├── responsibilities.py   # Computes γ and ξ from α, β
│   │   ├── scaling.py            # Per-timestep scaling utilities
│   │   └── components/           # Individual computation modules
│   │       ├── alpha.py          # Forward variable α_t(i)
│   │       ├── beta.py           # Backward variable β_t(i)
│   │       ├── gamma.py          # State responsibility γ_t(i)
│   │       └── xi.py             # Transition responsibility ξ_t(i,j)
│   ├── initialization/
│   │   └── random_init.py        # Dirichlet-based random A, B, π initialization
│   ├── model/
│   │   ├── hmm.py                # HiddenMarkovModel wrapper dataclass
│   │   └── parameters.py         # HMMParameters(A, B, pi) container
│   ├── optimization/
│   │   ├── baum_welch_step.py    # M-step: re-estimate A, B, π from γ, ξ
│   │   └── convergence.py        # |ΔLL| convergence check
│   ├── training/
│   │   ├── trainer.py            # HMMTrainer — full EM loop with callbacks
│   │   └── training_result.py    # TrainingResult dataclass (params, history, metadata)
│   └── utils/
│       ├── normalization.py      # Row-stochastic normalization helpers
│       └── validation.py         # Input validation (dimensions, non-negativity, etc.)
│
├── hmm_service/                  # Flask + Socket.IO web application
│   ├── __init__.py
│   ├── app.py                    # Application factory: create_app() + SocketIO init
│   ├── api/
│   │   ├── routes.py             # REST blueprint (/api/train) + WebSocket event handlers
│   │   └── schemas.py            # TrainRequest / TrainResponse validation dataclasses
│   ├── services/
│   │   ├── hmm_runner.py         # Training orchestration for REST and WebSocket modes
│   │   └── model_store.py        # In-memory model store keyed by UUID
│   ├── templates/
│   │   └── index.html            # Single-page dashboard (1600+ lines)
│   └── static/
│       ├── style.css             # Base custom styles
│       ├── css/
│       │   ├── tailwind.input.css    # Tailwind directives (input)
│       │   ├── tailwind.generated.css # Compiled Tailwind output (committed)
│       │   └── hmm_diagram.css       # Diagram-specific styles (fullscreen, controls)
│       └── js/
│           └── hmm_diagram.js        # Legacy diagram JS (kept for compat)
│
├── hmm_visualization/            # Matplotlib/Seaborn diagnostic plots
│   ├── heatmaps.py               # A/B matrix heatmaps (plot_heatmap)
│   ├── parameter_trajectory.py   # Parameter evolution over iterations
│   ├── state_diagram.py          # Graphviz-rendered state transition diagrams
│   ├── styles.py                 # Shared Matplotlib style configuration
│   └── training_plots.py         # Log-likelihood convergence curve
│
├── state_transition_diagrams/    # D3.js interactive state diagram library
│   ├── __init__.py               # Flask blueprint factory (create_blueprint)
│   ├── config.py                 # Default diagram configuration
│   ├── renderer.py               # Server-side rendering utilities
│   ├── flask_blueprint.py        # Blueprint registration helpers
│   └── static/
│       ├── css/state_diagram.css # Diagram CSS with fullscreen support
│       └── js/state_diagram.js   # StateTransitionDiagram class (D3.js, ~940 lines)
│
├── examples/
│   ├── weather_example.py        # Classic Rainy/Sunny HMM demo script
│   └── output/                   # Generated plots from running the example
│
├── tests/
│   ├── test_baum_welch.py        # Baum-Welch re-estimation tests
│   ├── test_forward_backward.py  # Forward-backward numerical correctness
│   ├── test_normalization.py     # Row-stochastic normalization tests
│   ├── test_training.py          # End-to-end training pipeline tests
│   └── audit_hmm.py             # Diagnostic audit script
│
└── docs/
    └── screenshots/              # Screenshots for documentation
```

---

## Setup

### Prerequisites

| Requirement | Purpose | Required? |
|---|---|---|
| **Python 3.11+** | Runtime for core engine and web server | Yes |
| **pip** | Package installation | Yes |
| **[Graphviz](https://graphviz.org/download/)** system binary | Rendering static state transition diagrams (SVG/PDF) | Optional |
| **Node.js + npm** | Rebuilding Tailwind CSS after template changes | Optional |

### Installation

**1. Clone the repository:**

```bash
git clone https://github.com/<your-username>/Baum-Welch-Algorithm.git
cd Baum-Welch-Algorithm
```

**2. Create and activate a virtual environment:**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3. Install the package (editable mode, recommended):**

```bash
pip install -e .
```

This installs `hmm_core`, `hmm_service`, `hmm_visualization`, and `state_transition_diagrams` as linked packages so code changes take effect immediately.

**Alternatively**, install only the runtime dependencies:

```bash
pip install -r requirements.txt
```

**4. (Optional) Install dev dependencies for testing:**

```bash
pip install -e ".[dev]"
```

**5. (Optional) Rebuild Tailwind CSS** (only needed if you modify HTML templates or JS files):

```bash
npm install
npx tailwindcss -c tailwind.config.js -i ./hmm_service/static/css/tailwind.input.css -o ./hmm_service/static/css/tailwind.generated.css --minify
```

### Verifying the Installation

```bash
# Run the test suite
pytest

# Check that the core engine works
python -c "from hmm_core.training.trainer import HMMTrainer; print('OK')"

# Check that the web app starts
python app.py
# → Open http://127.0.0.1:5000/
```

---

## Quick Start

### Run the Web Dashboard

```bash
python app.py
```

Open **http://127.0.0.1:5000/** in your browser. The dashboard lets you:

1. Enter an observation sequence (comma-separated integers, e.g. `0, 1, 1, 0, 2, 1, 0`)
2. Set the number of hidden states ($N$)
3. Configure max iterations and convergence tolerance
4. Optionally provide initial parameters or use random initialization
5. Click **Train** — watch the model learn in real time:
   - Log-likelihood curve updates after each iteration
   - Transition and emission heatmaps refresh live
   - Animated state transition diagram shows particle flow proportional to transition probabilities
6. Use the **fullscreen** button on the state diagram for a detailed view
7. Export the diagram as PNG or SVG

### Run the Weather Example

```bash
python examples/weather_example.py
```

This script demonstrates the full pipeline:

1. Defines a ground-truth 2-state weather HMM (Rainy/Sunny with Walk/Shop/Clean observations)
2. Generates 1000 synthetic observations from the true model
3. Trains a new HMM from random initialization using Baum-Welch
4. Prints learned vs. true parameters for comparison
5. Saves all visualizations to `examples/output/`:

| Output File | Description |
|---|---|
| `convergence.svg` | Log-likelihood curve showing EM convergence |
| `parameter_trajectory.svg` | How each parameter in $A$, $B$, $\pi$ evolved over iterations |
| `heatmap_A.svg` | Learned transition matrix as a color-coded heatmap |
| `heatmap_B.svg` | Learned emission matrix as a color-coded heatmap |
| `state_diagram.svg` | Graphviz-rendered state transition diagram (requires Graphviz) |

---

## Programmatic Usage

### Basic Training

```python
import numpy as np
from hmm_core.training.trainer import HMMTrainer

# Integer-coded observation sequence (symbols in [0, M-1])
observations = np.array([0, 1, 1, 0, 1, 2, 0, 1], dtype=np.intp)

trainer = HMMTrainer(
    n_states=2,         # Number of hidden states (N)
    n_obs_symbols=3,    # Number of observation symbols (M)
    max_iterations=200, # EM iteration cap
    tolerance=1e-6,     # Convergence threshold on |ΔLL|
    seed=42,            # Random seed for reproducibility
)
result = trainer.fit(observations)

print("Converged:", result.converged)
print("Iterations:", result.n_iterations)
print("Final log-likelihood:", result.log_likelihood_history[-1])
print("Transition matrix A:\n", result.model_params.A)
print("Emission matrix B:\n", result.model_params.B)
print("Initial distribution pi:", result.model_params.pi)
```

### Accessing Training History

```python
# Full log-likelihood history (one value per iteration)
ll_history = result.log_likelihood_history  # list[float]

# Parameter snapshots at each iteration
param_history = result.parameter_history    # list[HMMParameters]

# Check the model at iteration 10
params_at_10 = param_history[10]
print("A at iter 10:\n", params_at_10.A)
print("B at iter 10:\n", params_at_10.B)
```

### Generating Visualizations Programmatically

```python
from hmm_visualization.training_plots import plot_log_likelihood
from hmm_visualization.heatmaps import plot_heatmap
from hmm_visualization.parameter_trajectory import plot_parameter_evolution
from hmm_visualization.state_diagram import render_state_diagram

# Convergence plot
plot_log_likelihood(
    result.log_likelihood_history,
    title="My HMM — Convergence",
    save_path="convergence.svg",
    show=False,
)

# Heatmaps
plot_heatmap(result.model_params.A, title="Transition Matrix A",
             row_labels=["S0", "S1"], col_labels=["S0", "S1"],
             save_path="heatmap_A.svg", show=False)

plot_heatmap(result.model_params.B, title="Emission Matrix B",
             row_labels=["S0", "S1"], col_labels=["O0", "O1", "O2"],
             save_path="heatmap_B.svg", show=False)

# Parameter evolution
plot_parameter_evolution(result.parameter_history, save_path="trajectory.svg", show=False)

# Graphviz state diagram (requires Graphviz system binary)
render_state_diagram(result.model_params.A, state_labels=["S0", "S1"],
                     save_path="diagram", fmt="svg")
```

### With a Per-Iteration Callback

```python
def on_iteration(update: dict) -> None:
    print(f"  Iter {update['iteration']}: LL = {update['log_likelihood']:.6f}")

result = trainer.fit(observations, on_iteration=on_iteration)
```

---

## Web Dashboard Guide

The single-page dashboard at `http://127.0.0.1:5000/` combines all training and visualization features:

### Input Panel

- **Observation Sequence** — Comma-separated integers (e.g. `0, 1, 0, 2, 1`). Each integer represents a discrete observation symbol.
- **Number of States** — The number of hidden states $N$ to learn.
- **Max Iterations** — Upper bound on EM iterations (default: 200).
- **Tolerance** — Convergence threshold; training stops when $|\Delta \text{LL}| < \text{tolerance}$ (default: $10^{-6}$).
- **Initial Parameters** (optional) — Supply custom $A$, $B$, $\pi$ matrices as JSON, or leave blank for random Dirichlet initialization.

### Live Training Display

Once training starts, the dashboard updates in real time via WebSocket:

- **Log-Likelihood Chart** — Plotly.js line chart that grows with each iteration. Should show monotonic increase converging to a plateau.
- **Transition Matrix Heatmap** — Color-coded $N \times N$ matrix showing $a_{ij}$. Strong diagonal indicates "sticky" states.
- **Emission Matrix Heatmap** — Color-coded $N \times M$ matrix showing $b_i(o)$. Distinct row patterns indicate well-separated states.
- **State Transition Diagram** — Interactive D3.js diagram with:
  - Animated particles flowing along edges proportional to transition probabilities
  - Self-loop arcs for same-state transitions
  - Edge thickness proportional to $a_{ij}$
  - Drag nodes to rearrange layout
  - Replay controls: play/pause, speed slider, step forward/back
  - Fullscreen mode for detailed inspection
  - Export as PNG or SVG

### Training Summary

After convergence (or reaching max iterations), a summary panel shows:

- Whether the model converged
- Total iterations run
- Final log-likelihood value
- Learned $A$, $B$, $\pi$ matrices

---

## API Reference

### REST Endpoint

#### `POST /api/train`

Submit a training configuration and receive the fully trained model.

**Request body** (JSON):

```json
{
  "observations": [0, 1, 1, 0, 1, 2, 0, 1],
  "n_states": 2,
  "n_obs_symbols": 3,
  "max_iterations": 200,
  "tolerance": 1e-6,
  "seed": 42
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `observations` | `int[]` | Yes | — | Integer-coded observation sequence (≥ 2 elements) |
| `n_states` | `int` | Yes | — | Number of hidden states ($N \geq 1$) |
| `n_obs_symbols` | `int` | Yes | — | Number of observation symbols ($M \geq 1$) |
| `max_iterations` | `int` | No | 200 | EM iteration cap |
| `tolerance` | `float` | No | 1e-6 | Convergence threshold |
| `seed` | `int\|null` | No | null | Random seed for reproducibility |

**Response** (200 OK):

```json
{
  "model_id": "uuid-string",
  "converged": true,
  "n_iterations": 47,
  "final_log_likelihood": -1234.567,
  "A": [[0.7, 0.3], [0.4, 0.6]],
  "B": [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]],
  "pi": [0.6, 0.4],
  "log_likelihood_history": [-2000.0, -1800.5, "..."],
  "plots": { "convergence": "base64...", "heatmap_A": "base64..." }
}
```

**Error responses:**

| Status | Cause |
|---|---|
| 400 | Request body is not valid JSON |
| 422 | Missing or invalid fields (details in `error` key) |
| 500 | Training failure (numerical or runtime error) |

### WebSocket Events

Connect via Socket.IO to `http://127.0.0.1:5000/`.

#### Client → Server

| Event | Payload | Description |
|---|---|---|
| `start_training` | `{ observations, n_states, max_iterations, tolerance, init_params? }` | Begin streaming training. `observations` can be a list of ints or a comma-separated string. `init_params` is optional JSON with `A`, `B`, `pi`. |

#### Server → Client

| Event | Payload | Description |
|---|---|---|
| `training_update` | `{ iteration, log_likelihood, A, B, pi, converged }` | Emitted after each EM iteration with current parameters |
| `training_complete` | `{ converged, n_iterations, final_log_likelihood }` | Emitted once when training finishes |
| `training_error` | `{ error }` | Emitted if training fails; contains error message string |

**Example (JavaScript):**

```javascript
const socket = io("http://127.0.0.1:5000");

socket.emit("start_training", {
  observations: [0, 1, 1, 0, 1, 2, 0, 1],
  n_states: 2,
  max_iterations: 100,
  tolerance: 1e-6,
});

socket.on("training_update", (data) => {
  console.log(`Iter ${data.iteration}: LL = ${data.log_likelihood}`);
});

socket.on("training_complete", (data) => {
  console.log("Done!", data);
});

socket.on("training_error", (data) => {
  console.error("Error:", data.error);
});
```

---

## Weather Example Walkthrough

The `examples/weather_example.py` script implements the classic ["Umbrella World"](https://en.wikipedia.org/wiki/Hidden_Markov_model#Weather_guessing_game) HMM exercise:

**Ground-truth model:**

| | Rainy → Rainy | Rainy → Sunny |
|---|---|---|
| **Transition $A$** | 0.7 | 0.3 |

| | Sunny → Rainy | Sunny → Sunny |
|---|---|---|
| | 0.4 | 0.6 |

| | Walk | Shop | Clean |
|---|---|---|---|
| **Emission $B$ (Rainy)** | 0.1 | 0.4 | 0.5 |
| **Emission $B$ (Sunny)** | 0.6 | 0.3 | 0.1 |

**Initial distribution**: $\pi = [0.6, 0.4]$

The script generates $T = 1000$ synthetic observations, then trains a new HMM from scratch using `HMMTrainer` with `seed=6`. After convergence, the learned parameters closely match the ground truth, demonstrating that Baum-Welch successfully recovers the hidden model structure from observations alone.

**Running it:**

```bash
cd Baum-Welch-Algorithm
python examples/weather_example.py
```

**Expected console output:**

```text
Generated 1000 observations from ground-truth Weather HMM.
First 20 observations: [1, 2, 0, 1, ...]
Training complete.
  Converged:   True
  Iterations:  78
  Final LL:    -1028.123456
============================================================
  LEARNED HMM PARAMETERS (Weather Example)
============================================================
  Transition matrix A:
     Rainy | 0.70xx  0.29xx
     Sunny | 0.39xx  0.60xx
  ...
  All visualizations saved to: examples/output/
```

---

## Theory Background

The implementation follows standard HMM theory (see Rabiner 1989). A summary of the key formulations follows.

### HMM Definition

An HMM is parameterized by $\lambda = (A, B, \pi)$:

| Symbol | Name | Definition |
|---|---|---|
| $Q = \{1, 2, \dots, N\}$ | Hidden states | The unobservable states of the system |
| $\mathcal{O} = \{O_1, O_2, \dots, O_M\}$ | Observation symbols | The discrete outputs the system can emit |
| $\pi_i = P(q_1 = i)$ | Initial distribution | Probability of starting in state $i$ |
| $a_{ij} = P(q_{t+1}=j \mid q_t=i)$ | Transition probabilities | $N \times N$ row-stochastic matrix $A$ |
| $b_i(o) = P(O_t=o \mid q_t=i)$ | Emission probabilities | $N \times M$ row-stochastic matrix $B$ |

The three canonical HMM problems:

1. **Evaluation** — Given $\lambda$ and observation sequence $O$, compute $P(O \mid \lambda)$. Solved by the forward algorithm.
2. **Decoding** — Find the most likely hidden state sequence. Solved by the Viterbi algorithm (not implemented in this project).
3. **Learning** — Find $\lambda$ that maximizes $P(O \mid \lambda)$. Solved by the Baum-Welch algorithm (this project's focus).

### Forward Variables ($\alpha$)

$$\alpha_t(i) = P(O_1, \dots, O_t,\; q_t=i \mid \lambda)$$

- **Initialization**: $\alpha_1(i) = \pi_i \, b_i(O_1)$
- **Recursion**: $\alpha_{t+1}(j) = \left(\sum_{i=1}^N \alpha_t(i)\,a_{ij}\right) b_j(O_{t+1})$
- **Termination**: $P(O \mid \lambda) = \sum_{i=1}^N \alpha_T(i)$

Implemented in `hmm_core/inference/components/alpha.py`.

### Backward Variables ($\beta$)

$$\beta_t(i) = P(O_{t+1}, \dots, O_T \mid q_t=i, \lambda)$$

- **Initialization**: $\beta_T(i) = 1$
- **Recursion**: $\beta_t(i) = \sum_{j=1}^N a_{ij}\,b_j(O_{t+1})\,\beta_{t+1}(j)$

Implemented in `hmm_core/inference/components/beta.py`.

### Responsibilities ($\gamma$ and $\xi$)

$$\gamma_t(i) = P(q_t = i \mid O, \lambda) = \frac{\alpha_t(i)\,\beta_t(i)}{P(O \mid \lambda)}$$

$$\xi_t(i,j) = P(q_t = i, q_{t+1} = j \mid O, \lambda) = \frac{\alpha_t(i)\,a_{ij}\,b_j(O_{t+1})\,\beta_{t+1}(j)}{P(O \mid \lambda)}$$

The relationship: $\gamma_t(i) = \sum_{j=1}^N \xi_t(i,j)$.

Implemented in `hmm_core/inference/responsibilities.py` with components in `gamma.py` and `xi.py`.

### M-Step (Baum-Welch Re-estimation)

$$\pi_i^{\text{new}} = \gamma_1(i)$$

$$a_{ij}^{\text{new}} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$$

$$b_i^{\text{new}}(o) = \frac{\sum_{t:\,O_t=o} \gamma_t(i)}{\sum_{t=1}^T \gamma_t(i)}$$

Implemented in `hmm_core/optimization/baum_welch_step.py`.

### The EM Loop

Each iteration of the Baum-Welch algorithm:

1. **E-step**: Run forward-backward, compute $\gamma$ and $\xi$ given current $\lambda$
2. **M-step**: Normalize expected counts to get new $(A, B, \pi)$
3. **Convergence check**: If $|\text{LL}_{\text{new}} - \text{LL}_{\text{old}}| < \varepsilon$, stop

```text
Initialize A, B, pi (random Dirichlet or user-supplied)
repeat:
    alpha, scales = forward(A, B, pi, O)
    beta          = backward(A, B, O, scales)
    gamma, xi     = responsibilities(alpha, beta, A, B, O)
    A, B, pi      = m_step(gamma, xi, O)
    ll_new        = log_likelihood(scales)
until |ll_new - ll_old| < tolerance  or  iteration >= max_iterations
```

**Key guarantee**: The log-likelihood is monotonically non-decreasing at each iteration. Any decrease indicates a numerical bug.

---

## Numerical Stability

Raw forward-backward probabilities underflow to zero for long sequences (products of many small numbers). This project uses **per-timestep scaling** (Rabiner's method):

1. At each timestep $t$, normalize $\alpha_t$ to sum to 1, recording the scaling constant:

$$c_t = \frac{1}{\sum_{i=1}^N \hat{\alpha}_t(i)}$$

2. Work with scaled values $\hat{\alpha}_t(i) = c_t \cdot \alpha_t(i)$ throughout

3. Apply the same scaling factors in the backward pass for consistency

4. Recover the total log-likelihood without ever computing the raw probability:

$$\log P(O \mid \lambda) = -\sum_{t=1}^T \log c_t$$

This approach is implemented across `hmm_core/inference/scaling.py`, `alpha.py`, `beta.py`, and `forward_backward.py`.

---

## Math-to-Code Map

| Mathematical Concept | Source Module | Key Function/Class |
|---|---|---|
| Forward $\alpha_t(i)$ | `hmm_core/inference/components/alpha.py` | `compute_alpha()` |
| Backward $\beta_t(i)$ | `hmm_core/inference/components/beta.py` | `compute_beta()` |
| State responsibility $\gamma_t(i)$ | `hmm_core/inference/components/gamma.py` | `compute_gamma()` |
| Transition responsibility $\xi_t(i,j)$ | `hmm_core/inference/components/xi.py` | `compute_xi()` |
| Combined $\gamma, \xi$ | `hmm_core/inference/responsibilities.py` | `compute_responsibilities()` |
| Scaled forward-backward + log-likelihood | `hmm_core/inference/forward_backward.py` | `run_forward_backward()` |
| Baum-Welch re-estimation ($A, B, \pi$) | `hmm_core/optimization/baum_welch_step.py` | `baum_welch_update()` |
| Convergence check ($\|\Delta\text{LL}\| < \varepsilon$) | `hmm_core/optimization/convergence.py` | `check_convergence()` |
| Full EM training loop | `hmm_core/training/trainer.py` | `HMMTrainer.fit()` |
| Parameter container | `hmm_core/model/parameters.py` | `HMMParameters` |
| Model wrapper | `hmm_core/model/hmm.py` | `HiddenMarkovModel` |
| Random initialization | `hmm_core/initialization/random_init.py` | `random_hmm_parameters()` |
| Row-stochastic normalization | `hmm_core/utils/normalization.py` | `normalize_rows()` |
| Input validation | `hmm_core/utils/validation.py` | `validate_*()` |

**Suggested reading order for learners:**

1. Read the [Theory Background](#theory-background) section above
2. `hmm_core/inference/components/alpha.py` → `beta.py` — understand the forward and backward passes
3. `hmm_core/inference/responsibilities.py` — see how $\gamma$ and $\xi$ are derived
4. `hmm_core/optimization/baum_welch_step.py` — the M-step that closes the EM loop
5. `hmm_core/training/trainer.py` — the full training pipeline that ties everything together

---

## Data Preparation

The Baum-Welch algorithm operates on **discrete observation symbols**. Your observations must be encoded as contiguous integers in $[0, M-1]$.

### Encoding Steps

1. **Build a vocabulary** from your raw data (e.g. categories, binned values, event types)
2. **Map each unique value** to an integer starting at 0
3. **Encode** the observation sequence as a NumPy integer array with `dtype=np.intp`
4. **Set $M$** to the total number of unique symbols (i.e., `max(observations) + 1`)
5. **Keep the reverse map** to interpret learned emission probabilities

### Example

```python
import numpy as np

# Raw categorical data
raw = ["sunny", "cloudy", "rainy", "sunny", "cloudy"]

# Build vocabulary
vocab = {label: idx for idx, label in enumerate(sorted(set(raw)))}
# {'cloudy': 0, 'rainy': 1, 'sunny': 2}

# Encode
observations = np.array([vocab[x] for x in raw], dtype=np.intp)
# array([2, 0, 1, 2, 0])

n_obs_symbols = len(vocab)  # M = 3
```

### Common Pitfalls

| Pitfall | Consequence | Fix |
|---|---|---|
| Non-contiguous labels (e.g. `[0, 2, 5]`) | Index-out-of-bounds in $B$ matrix | Re-map to contiguous `[0, 1, 2]` |
| Negative indices | Array indexing errors | Ensure all values $\geq 0$ |
| `n_obs_symbols` too small | Missing columns in $B$; crash or silent data loss | Set $M = \max(O) + 1$ |
| Mixed types (floats + ints) | Unexpected rounding | Cast to `np.intp` explicitly |
| Single observation ($T = 1$) | No transitions to learn | Need $T \geq 2$ |

---

## Convergence and Model Selection

### Tips for Successful Training

- **Multiple random restarts**: Run 5–20 seeds and keep the model with highest final log-likelihood. EM finds local optima, not global ones.
- **Sufficient data**: Short sequences ($T < 50$) cause noisy parameter estimates. Use $T \geq 200$ when possible.
- **Monitor monotonicity**: Log-likelihood must never decrease. If it does, there's a numerical bug.
- **Tolerance range**: $10^{-6}$ to $10^{-8}$ works well in practice. Lower values give more precision at the cost of more iterations.

### Choosing the Number of States ($N$)

There is no closed-form solution for optimal $N$. Practical approaches:

1. **Domain knowledge** — If you know there are 2 weather regimes (dry/wet), set $N = 2$
2. **Incremental $N$** — Train with $N = 2, 3, 4, \dots$ and compare final log-likelihoods
3. **Held-out validation** — Split data into train/test; evaluate test log-likelihood to detect overfitting
4. **Examine learned parameters** — If two states have nearly identical emission distributions, $N$ is too high

### Reading the Diagnostics

| Plot | Healthy Pattern | Problematic Pattern |
|---|---|---|
| **Log-likelihood curve** | Monotonic increase, then smooth plateau | Oscillation, sudden drops, or no plateau |
| **Transition heatmap ($A$)** | Strong diagonal = "sticky" states that persist | Uniform rows = no temporal structure learned |
| **Emission heatmap ($B$)** | Clear row-wise peaks = states specialize in different symbols | Uniform rows = states are indistinguishable |
| **Parameter trajectory** | Parameters stabilize after initial movement | Parameters keep oscillating = not converged |

---

## Practical Applications

| Domain | Hidden States | Observations | Use Case |
|---|---|---|---|
| **Weather modeling** | Atmospheric regimes (dry, wet) | Daily event types (walk, shop, clean) | Classic HMM teaching example |
| **Finance** | Market conditions (bull, bear, sideways) | Discretized returns or indicator levels | Regime detection and switching models |
| **User behavior analytics** | User intent states (browsing, comparing, buying) | Clickstream event types | Conversion funnel analysis |
| **Predictive maintenance** | Machine health (normal, degrading, faulty) | Binned sensor readings | Failure prediction and scheduling |
| **Healthcare** | Disease progression stages | Diagnosis codes, treatment events | Clinical pathway modeling |
| **Natural language processing** | Part-of-speech tags, named entity categories | Word/token observations | Sequence labeling and structure discovery |
| **Bioinformatics** | Gene regions (exon, intron, intergenic) | DNA base sequences | Gene finding and sequence annotation |
| **Speech recognition** | Phoneme states | Acoustic feature vectors (after discretization) | Sub-word unit modeling |

---

## Testing

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=hmm_core --cov-report=term-missing

# Run a specific test file
pytest tests/test_baum_welch.py -v
```

### What the Tests Cover

| Test File | Coverage |
|---|---|
| `test_forward_backward.py` | Forward/backward variable computation, scaling correctness, log-likelihood consistency |
| `test_normalization.py` | Row-stochastic normalization edge cases (zeros, near-zeros, large values) |
| `test_baum_welch.py` | M-step re-estimation formulas, parameter update correctness |
| `test_training.py` | End-to-end training: convergence on known models, reproducibility with seeds, callback invocation |

---

## Deployment

The project ships three deployment files that work together to ensure Zeabur always starts the app correctly — with eventlet serving WebSocket traffic directly and no Gunicorn in the chain.

### Files

| File | Purpose |
|---|---|
| `Dockerfile` | Primary container definition. Zeabur detects this and uses it instead of auto-generating one. Installs the package with `pip install -e .` and runs `python app.py`. |
| `Procfile` | Fallback for Heroku-style platforms: `web: python app.py`. Ignored by Zeabur when a `Dockerfile` is present. |
| `zbpack.json` | Zeabur build hints: Python version, entry file, start command. Ignored when a `Dockerfile` is present but kept for reference. |

### Why no Gunicorn?

Gunicorn manages worker processes and restarts them after errors, which is fine for plain WSGI apps. But Socket.IO requires a **single persistent async worker** — when Gunicorn kills a worker (on timeout, error, or WebSocket disconnect), all active Socket.IO sessions are invalidated. The eventlet server started by `socketio.run()` handles HTTP and WebSocket on the same loop without any worker process management.

### Deploying to Zeabur

1. Push the repository (including `Dockerfile`) to your Git provider.
2. In Zeabur, create a new service → **Deploy from Git** → select the repo.
3. Zeabur detects `Dockerfile`, builds the image, and runs `python app.py`.
4. The app binds to `$PORT` (injected by Zeabur, typically 8080).
5. No environment variables need to be set manually.

### Local production simulation

```bash
# Exactly mimics the Zeabur container
docker build -t hmm-bwa .
docker run -p 8080:8080 -e PORT=8080 hmm-bwa
# → Open http://localhost:8080/
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| `WinError 10048` / "Address already in use" | Port 5000 is occupied by another process | Kill the process (see below), then restart |
| Flask-SocketIO fails to start | Missing `eventlet` package | Run `pip install eventlet` |
| WebSocket not connecting | Proxy stripping upgrade headers, or app started with Gunicorn instead of eventlet | Ensure the app is started with `python app.py` (eventlet); do **not** use Gunicorn — it tears down the WebSocket connection on worker exit |
| State diagrams don't render (Graphviz) | Graphviz system binary not installed | Install from [graphviz.org](https://graphviz.org/download/) and add to `PATH` |
| `ModuleNotFoundError` | Running from wrong directory or not installed | Run from project root; ensure `pip install -e .` was done |
| Heatmaps / plots not appearing | Matplotlib backend issue in server mode | The server uses `Agg` backend automatically; check logs for errors |
| Training never converges | Tolerance too low or insufficient data | Increase `max_iterations`, raise `tolerance`, or use more observations |
| 404 on `/` | Root directory misconfigured in deployment | Ensure Zeabur root directory is empty (repository root) |
| Build fails on Zeabur | Incomplete `requirements.txt` | Verify all deps resolve locally with `pip install -r requirements.txt` |
| Live training not updating on Zeabur | Zeabur auto-detected Flask and injected Gunicorn, overriding the start command | Ensure `Dockerfile` exists at repo root — Zeabur always prefers it over auto-detection |

**Freeing port 5000 on Windows:**

```powershell
Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
```

**Freeing port 5000 on macOS/Linux:**

```bash
lsof -ti:5000 | xargs kill -9
```

---

## Reproducibility Checklist

When running experiments, record these details to ensure reproducible results:

- [ ] Random seed passed to `HMMTrainer(seed=...)`
- [ ] Number of states $N$ and observation symbols $M$
- [ ] Observation sequence length $T$ and the sequence itself (or its generation method)
- [ ] Convergence tolerance and max iterations
- [ ] Final $(A, B, \pi)$ matrices saved to file
- [ ] Log-likelihood history saved for convergence analysis
- [ ] All visualization plots saved with timestamps
- [ ] Observation encoding/vocabulary map stored alongside results

---

## Contributing

Contributions are welcome. To get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Make your changes and add tests
5. Run the test suite: `pytest`
6. Submit a pull request

Please follow the existing code style: type hints on all function signatures, docstrings in NumPy/Sphinx format, and one module per mathematical concept.

---

## License

MIT

