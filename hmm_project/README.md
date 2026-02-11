# HMM Baum-Welch Engine

A **production-grade, modular Hidden Markov Model engine** implementing the
Baum–Welch (Expectation-Maximisation) algorithm in Python.

---

## Architecture

```
hmm_project/
│
├── hmm_core/                  # Pure computational engine (ZERO Flask / plotting)
│   ├── model/
│   │   ├── parameters.py      # Immutable λ = (A, B, π) container
│   │   └── hmm.py             # HMM wrapper with validation
│   ├── inference/
│   │   ├── components/
│   │   │   ├── alpha.py        # Scaled forward recursion
│   │   │   ├── beta.py         # Scaled backward recursion
│   │   │   ├── gamma.py        # State responsibilities γ
│   │   │   └── xi.py           # Transition responsibilities ξ
│   │   ├── scaling.py          # Log-likelihood from scaling factors
│   │   ├── forward_backward.py # Coordinates α + β + scaling
│   │   └── responsibilities.py # Coordinates γ + ξ
│   ├── optimization/
│   │   ├── baum_welch_step.py  # Single EM M-step update
│   │   └── convergence.py      # Stopping criterion
│   ├── training/
│   │   ├── trainer.py          # Full EM loop
│   │   └── training_result.py  # Result container
│   ├── initialization/
│   │   └── random_init.py      # Dirichlet-sampled stochastic init
│   └── utils/
│       ├── normalization.py    # Row / vector normalization
│       └── validation.py       # Shape / stochasticity checks
│
├── hmm_visualization/          # Independent visualisation (no Flask)
│   ├── styles.py               # Colour palette & RC params
│   ├── training_plots.py       # Log-likelihood convergence curve
│   ├── parameter_trajectory.py # A, B, π element evolution plots
│   ├── state_diagram.py        # Graphviz state transition diagram (SVG)
│   └── heatmaps.py             # Seaborn annotated heatmaps
│
├── hmm_service/                # Flask web UI (completely separate)
│   ├── app.py                  # App factory
│   ├── api/
│   │   ├── routes.py           # POST /api/train endpoint
│   │   └── schemas.py          # Request / response validation
│   ├── services/
│   │   ├── hmm_runner.py       # Orchestrates training + viz
│   │   └── model_store.py      # In-memory UUID-keyed store
│   ├── templates/
│   │   └── index.html          # Single-page dark-themed UI
│   └── static/
│
├── tests/                      # Pytest suite
├── examples/
│   └── weather_example.py      # Rainy / Sunny demo script
├── pyproject.toml
└── README.md
```

---

## Quick Start

```bash
cd hmm_project

# Install (editable) with dev dependencies
pip install -e ".[dev]"

# Run the tests
pytest tests/ -v

# Run the weather example
python examples/weather_example.py

# Launch the Flask UI
python -m hmm_service.app
# → Open http://localhost:5000
```

---

## Mathematical Background

### Hidden Markov Models

An HMM is defined by the tuple **λ = (A, B, π)** where:

| Symbol | Name | Shape | Definition |
|--------|------|-------|------------|
| **A** | State transition matrix | N × N | A\[i,j\] = P(q_{t+1} = S_j \| q_t = S_i) |
| **B** | Observation emission matrix | N × M | B\[i,k\] = P(O_t = v_k \| q_t = S_i) |
| **π** | Initial state distribution | N | π\[i\] = P(q_1 = S_i) |

### The Forward Algorithm (α-pass)

The forward variable α_t(i) = P(O_1, …, O_t, q_t = S_i | λ):

```
α_1(i)  =  π_i · b_i(O_1)
α_t(i)  =  [Σ_j α_{t-1}(j) · a_{ji}] · b_i(O_t)     for t = 2, …, T
```

The observation likelihood: **P(O | λ) = Σ_i α_T(i)**.

### The Backward Algorithm (β-pass)

The backward variable β_t(i) = P(O_{t+1}, …, O_T | q_t = S_i, λ):

```
β_T(i)  =  1
β_t(i)  =  Σ_j a_{ij} · b_j(O_{t+1}) · β_{t+1}(j)    for t = T-1, …, 1
```

### Scaling for Numerical Stability

Raw α values underflow for long sequences. We use scaling factors:

```
c_t  =  1 / Σ_i α̂_t(i)
α̂_t  ←  c_t · α̂_t
```

The log-likelihood is recovered as:  **log P(O | λ) = −Σ_t log(c_t)**.

The same scaling factors are applied to β for consistency.

### State Responsibilities (γ)

```
γ_t(i) = P(q_t = S_i | O, λ) = α_t(i) · β_t(i) / Σ_j α_t(j) · β_t(j)
```

### Transition Responsibilities (ξ)

```
ξ_t(i,j) = P(q_t = S_i, q_{t+1} = S_j | O, λ)
          = α_t(i) · a_{ij} · b_j(O_{t+1}) · β_{t+1}(j) / normaliser
```

### Baum-Welch Re-estimation (EM)

The EM algorithm iterates:

1. **E-step**: Compute γ and ξ using forward-backward.
2. **M-step**: Update parameters:

```
π̂_i    =  γ_1(i)

â_{ij} =  Σ_{t=1}^{T-1} ξ_t(i,j)  /  Σ_{t=1}^{T-1} γ_t(i)

b̂_j(k) =  Σ_{t: O_t=k} γ_t(j)    /  Σ_t γ_t(j)
```

3. **Convergence**: Stop when |ΔLL| < tolerance.

**Property**: Each EM iteration is guaranteed to increase (or maintain)
the log-likelihood — the algorithm never decreases P(O | λ).

### How States Evolve During Training

Initially, with random parameters, all states behave similarly — they emit
each observation with roughly equal probability. As EM progresses:

- **Specialisation**: States differentiate — each begins to "own" a
  distinct subset of the observation symbols.
- **Transition sharpening**: The transition matrix develops clear
  patterns, reflecting the temporal structure of the data.
- **π stabilisation**: The initial distribution settles to reflect
  which state the sequence most likely starts in.

### The Weather Example (Rainy / Sunny)

The classic textbook example uses:

- **2 hidden states**: Rainy, Sunny
- **3 observations**: Walk, Shop, Clean

The intuition is that weather (hidden) influences a person's activity
(observed). Baum-Welch discovers these latent weather patterns from
the activity sequence alone.

### Practical Applications

| Domain | Hidden States | Observations |
|--------|--------------|--------------|
| Speech recognition | Phonemes | Acoustic features |
| Bioinformatics | Gene regions | DNA bases |
| Finance | Market regimes | Returns/volumes |
| NLP | POS tags | Words |
| Robotics | Locations | Sensor readings |

---

## Extending the Engine

The modular architecture allows:

- **Gaussian HMM** — replace `B` with continuous emission densities by
  swapping `components/alpha.py` and `components/beta.py`.
- **Log-space arithmetic** — replace `scaling.py` with a log-domain
  implementation.
- **FastAPI** — swap `hmm_service` for a FastAPI frontend; `hmm_core`
  and `hmm_visualization` require zero changes.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Core engine | Python 3.11, NumPy |
| Web UI | Flask |
| State diagrams | Graphviz (Python + system binary) |
| Plots | Matplotlib, Seaborn |
| Testing | Pytest |

---

## License

MIT
