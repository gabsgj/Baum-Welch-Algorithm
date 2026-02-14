# HMM Baum–Welch Project (Discrete Hidden Markov Models)

This project is a full, production-oriented implementation of **Hidden Markov Model (HMM)** training using the **Baum–Welch algorithm** (the EM algorithm for HMMs), with:

- a clean, modular Python core (`hmm_core`) for inference and optimization,
- a visualization layer (`hmm_visualization`) for diagnostics,
- a Flask + Socket.IO web dashboard (`hmm_service`) for interactive training.

It is designed to be both **educational** (you can inspect every step of the math) and **practical** (you can train, monitor convergence, and deploy as a web service).

---

## Live Demo

- Production deployment: **https://bwa.gabrieljames.me**

---

## Tech Stack

### Core modeling and training

- Python 3.11+
- NumPy (numerical computation)
- Custom HMM engine (`hmm_core`) implementing:
  - scaled forward-backward inference,
  - Baum–Welch EM optimization,
  - convergence checks and parameter validation.

### Web application

- Flask (web server + routing)
- Flask-SocketIO + eventlet (realtime iteration updates)
- HTML/CSS/JavaScript dashboard (`hmm_service`)
- Plotly.js + D3.js for interactive visualizations

### Visualization and diagrams

- Matplotlib + Seaborn (plots and heatmaps)
- Graphviz (state transition diagrams)

### Deployment and packaging

- Gunicorn (production server)
- `pyproject.toml` + `requirements.txt`
- Zeabur-compatible entry setup (`app.py`, `zbpack.json`)

---

## Practical Applications

This project can be used as both a learning framework and a practical sequence-modeling engine in domains where system states are hidden but outputs are observed.

### 1) Weather and environmental pattern modeling

- Infer hidden weather regimes (e.g., dry/wet, stable/unstable) from observed event sequences.
- Use transition matrix \(A\) to understand regime persistence and switching behavior.

### 2) Finance and market regime detection

- Model latent market conditions (bull, bear, sideways) from discretized price/volume behavior.
- Track transition probabilities to estimate how likely the market is to remain in or switch regimes.

### 3) User behavior analytics

- Learn hidden user intent states from clickstream or interaction sequences.
- Use emission distributions \(B\) to interpret which actions are characteristic of each latent behavior mode.

### 4) Predictive maintenance and IoT monitoring

- Represent hidden machine health states (normal, degraded, critical) from sensor-event symbols.
- Detect abnormal transition patterns early and support preventive intervention.

### 5) Healthcare and patient-journey modeling

- Model latent clinical progression stages from observed diagnosis/treatment events.
- Support retrospective analysis of pathway transitions and care sequence dynamics.

### 6) NLP and sequence labeling (discrete setting)

- Apply to token-category sequences where tags/states are hidden.
- Useful as a transparent baseline for segmentation, tagging, or simple structure discovery tasks.

### 7) Education and research use

- Demonstrate the full Baum–Welch pipeline with inspectable intermediate variables \(\alpha, \beta, \gamma, \xi\).
- Compare convergence behavior across sequence lengths, initialization seeds, and model sizes.

### Why this repository is useful in practice

- **Interactive dashboard**: inspect training dynamics iteration-by-iteration.
- **Programmatic API**: embed training directly into experiments or production scripts.
- **Visualization toolkit**: generate interpretable artifacts for reports and presentations.
- **Deployment-ready setup**: host and share model behavior through a web UI.

---

## Screenshots

Add your final UI screenshots in this section when publishing. Suggested files:

- `docs/screenshots/dashboard-overview.png`
- `docs/screenshots/training-in-progress.png`
- `docs/screenshots/convergence-chart.png`
- `docs/screenshots/state-diagram.png`

Recommended README image blocks:

```markdown
![Dashboard Overview](docs/screenshots/dashboard-overview.png)
![Training In Progress](docs/screenshots/training-in-progress.png)
![Convergence Chart](docs/screenshots/convergence-chart.png)
![State Diagram](docs/screenshots/state-diagram.png)
```

---

## 1) Theory Background (from HMM_v3)

The project follows the theory in `HMM_v3.pdf`, which explains Baum–Welch in an intuitive but mathematically rigorous way.

### 1.1 HMM definition

An HMM is defined by the parameter tuple:

\[
\lambda = (A, B, \pi)
\]

Where:

- Hidden state set: \(Q = \{1,2,\dots,N\}\)
- Observation symbol set: \(\mathcal{O} = \{O_1, O_2, \dots, O_M\}\)
- Initial distribution: \(\pi_i = P(q_1 = i)\)
- Transition probabilities: \(a_{ij} = P(q_{t+1}=j \mid q_t=i)\)
- Emission probabilities: \(b_i(o) = P(O_t=o \mid q_t=i)\)

For an observation sequence \(O=(O_1, O_2, \dots, O_T)\), we want parameters that maximize \(P(O\mid\lambda)\).

---

### 1.2 Forward variables (prefix evidence)

\[
\alpha_t(i) = P(O_1, O_2, \dots, O_t, q_t=i \mid \lambda)
\]

Recurrence:

- Initialization:
  \[
  \alpha_1(i) = \pi_i\,b_i(O_1)
  \]
- Recursion:
  \[
  \alpha_{t+1}(j) = \left(\sum_{i=1}^N \alpha_t(i)a_{ij}\right)b_j(O_{t+1})
  \]

---

### 1.3 Backward variables (suffix evidence)

\[
\beta_t(i) = P(O_{t+1}, O_{t+2}, \dots, O_T \mid q_t=i, \lambda)
\]

Recurrence:

- Initialization:
  \[
  \beta_T(i)=1
  \]
- Recursion:
  \[
  \beta_t(i)=\sum_{j=1}^N a_{ij} b_j(O_{t+1})\beta_{t+1}(j)
  \]

---

### 1.4 Sequence likelihood

\[
P(O\mid\lambda)=\sum_{i=1}^N \alpha_T(i)
\]

In practice, long sequences can underflow numerically, so scaled recursions are used (implemented in this repository).

---

### 1.5 Responsibilities (soft assignments)

State occupancy responsibility:

\[
\gamma_t(i)=P(q_t=i\mid O,\lambda)=\frac{\alpha_t(i)\beta_t(i)}{P(O\mid\lambda)}
\]

Transition responsibility:

\[
\xi_t(i,j)=\frac{\alpha_t(i)a_{ij}b_j(O_{t+1})\beta_{t+1}(j)}{P(O\mid\lambda)}
\]

Interpretation:

- \(\gamma_t(i)\): “How much of time step \(t\) belongs to state \(i\)?”
- \(\xi_t(i,j)\): “How much expected flow goes from \(i\to j\) at time \(t\)?”

---

### 1.6 M-step updates (Baum–Welch)

Given responsibilities, re-estimate parameters:

- Initial distribution:
  \[
  \pi_i^{new}=\gamma_1(i)
  \]

- Transition matrix:
  \[
  a_{ij}^{new}=\frac{\sum_{t=1}^{T-1}\xi_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)}
  \]

- Emission matrix (discrete symbols):
  \[
  b_i^{new}(o)=\frac{\sum_{t:O_t=o}\gamma_t(i)}{\sum_{t=1}^{T}\gamma_t(i)}
  \]

This is exactly “normalized expected counts.”

---

### 1.7 EM loop intuition

Each iteration does:

1. **E-step**: run forward-backward and compute \(\gamma,\xi\)
2. **M-step**: normalize expected counts to get new \((A,B,\pi)\)
3. Check convergence via likelihood improvement

A key point emphasized in `HMM_v3`: observations do not directly overwrite parameters; they **reshape probability flow**, and that flow updates parameters.

---

### 1.8 Why longer sequences help

As shown in the detailed short-vs-long sequence examples in `HMM_v3`:

- short sequences can cause sharp, unstable parameter jumps,
- longer sequences provide more expected counts,
- variance reduces through averaging,
- updates become smoother and converge more reliably.

---

## 2) Repository Structure

```text
Baum-Welch-Algorithm/
  app.py                          # Root production entry (Zeabur/Gunicorn)
  pyproject.toml                  # Packaging + dependencies
  requirements.txt                # Runtime dependencies
  DEPLOYMENT.md                   # Deployment guide

  hmm_core/                       # Mathematical engine
    inference/                    # forward/backward, responsibilities, scaling
    initialization/               # random parameter initialization
    model/                        # HMM and parameter classes
    optimization/                 # Baum-Welch update + convergence checks
    training/                     # training loop and result containers
    utils/                        # validation/normalization helpers

  hmm_service/                    # Flask + Socket.IO app
    api/                          # REST + WebSocket routes/schemas
    templates/                    # Dashboard HTML
    static/                       # CSS/JS assets

  hmm_visualization/              # Diagnostic plots and diagrams
  state_transition_diagrams/      # Diagram rendering components
  examples/                       # runnable example scripts
  tests/                          # unit tests

  HMM_v3_extracted.txt            # extracted text from HMM_v3.pdf
  _extract_pdf.py                 # helper script used for PDF text extraction
```

---

## 3) Setup

### 3.1 Requirements

- Python 3.11+
- (Optional) Graphviz system binary for diagram rendering
- (Optional) Node.js + npm for Tailwind CSS rebuilds used by web UI

### 3.2 Install

```bash
cd Baum-Welch-Algorithm
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Alternative:

```bash
pip install -r requirements.txt
```

---

## 4) Running the project

### 4.1 Run web service (recommended)

```bash
cd Baum-Welch-Algorithm
python app.py
```

or

```bash
python -m hmm_service.app
```

Then open:

- `http://127.0.0.1:5000/`

### 4.2 Run training example

```bash
cd Baum-Welch-Algorithm
python examples/weather_example.py
```

This script trains a weather HMM and saves convergence/parameter visualizations under `examples/output/`.

---

## Practical Examples

### Example 1: End-to-end weather model training

Run:

```bash
cd Baum-Welch-Algorithm
python examples/weather_example.py
```

What you get:

- trained transition matrix \(A\), emission matrix \(B\), and \(\pi\),
- per-iteration log-likelihood history,
- saved plots:
  - convergence curve,
  - parameter trajectory,
  - heatmaps,
  - state diagram.

### Example 2: Web dashboard training session

1. Start server:

```bash
cd Baum-Welch-Algorithm
python app.py
```

2. Open `http://127.0.0.1:5000/`
3. Enter sequence/configuration in the UI
4. Watch live per-iteration updates (Socket.IO)
5. Inspect final learned parameters and plots

### Example 3: Programmatic integration

Use `HMMTrainer` directly in your own scripts or notebooks for custom workflows.

```python
import numpy as np
from hmm_core.training.trainer import HMMTrainer

observations = np.array([0, 1, 2, 1, 0, 2, 2, 1, 0], dtype=np.intp)
trainer = HMMTrainer(n_states=3, n_obs_symbols=3, max_iterations=150, tolerance=1e-6, seed=7)
result = trainer.fit(observations)

print("Converged:", result.converged)
print("Iterations:", result.n_iterations)
print("Final LL:", result.log_likelihood_history[-1])
```

---

## 5) Core API usage (programmatic)

Minimal training usage:

```python
import numpy as np
from hmm_core.training.trainer import HMMTrainer

observations = np.array([0, 1, 1, 0, 1, 2, 0, 1], dtype=np.intp)
trainer = HMMTrainer(
    n_states=2,
    n_obs_symbols=3,
    max_iterations=200,
    tolerance=1e-6,
    seed=42,
)
result = trainer.fit(observations)

print(result.converged)
print(result.log_likelihood_history[-1])
print(result.model_params.A)
print(result.model_params.B)
print(result.model_params.pi)
```

---

## 6) Web API and socket events

### REST endpoint

- `POST /api/train`
  - accepts training config and observation sequence,
  - returns trained model summary.

### WebSocket events

Client → Server:

- `start_training`

Server → Client:

- `training_update`
- `training_complete`
- `training_error`

This allows real-time convergence plotting in the dashboard.

---

## 7) Visual outputs

The repository includes utilities for:

- log-likelihood convergence curves,
- parameter evolution over iterations,
- matrix heatmaps for \(A\) and \(B\),
- state transition diagrams.

These are useful both for debugging and for understanding EM behavior.

---

## 8) How to extract and use theory from HMM_v3.pdf

You asked for practical measures to take data from the PDF. The safest workflow is:

### 8.1 Keep the source immutable

- Keep `HMM_v3.pdf` as the canonical source in project root.
- Write extracted/processed text to separate files (`HMM_v3_extracted.txt`).

### 8.2 Use scripted extraction (already included)

`_extract_pdf.py` currently:

- reads `HMM_v3.pdf`,
- extracts text page by page using `pypdf` (fallback to `PyPDF2`),
- writes tagged output with `===== PAGE X =====` markers.

Run it:

```bash
cd Baum-Welch-Algorithm
python _extract_pdf.py
```

### 8.3 Post-process quality checks

After extraction, verify:

- equation lines were not broken unexpectedly,
- symbols (\(\alpha, \beta, \gamma, \xi\), subscripts, fractions) remained readable,
- table structures and headers are preserved,
- section order matches PDF.

A practical routine:

1. Compare random pages in PDF vs extracted text.
2. Normalize spacing and hyphenation only (avoid semantic edits).
3. Keep a “raw extract” and a “cleaned extract” separately.

### 8.4 Build a theory dataset for documentation/code comments

Create a structured notes file from extracted text with fields:

- section title,
- formula(s),
- semantic interpretation,
- implementation mapping (`module/function`).

Example mapping in this project:

- Forward recursion → `hmm_core/inference/components/alpha.py`
- Backward recursion → `hmm_core/inference/components/beta.py`
- Responsibilities \(\gamma,\xi\) → `hmm_core/inference/responsibilities.py`
- M-step updates → `hmm_core/optimization/baum_welch_step.py`

### 8.5 Version control recommendations

- Commit extraction script and extracted text.
- If PDF changes, regenerate extraction in a single commit.
- Use commit messages like: `docs: refresh HMM_v3 extraction and update theory notes`.

### 8.6 Optional improvements (recommended)

If you want higher-fidelity math extraction, consider:

- adding `pdfplumber` for layout-aware extraction,
- preserving equation blocks separately,
- storing formulas in a structured Markdown glossary.

---

## 9) Practical convergence guidance

For real datasets:

- Start with multiple random seeds and compare final log-likelihood.
- Use enough sequence length; very short sequences may overfit and oscillate.
- Monitor monotonic likelihood growth; large drops indicate numerical/implementation issues.
- Keep tolerance moderate (`1e-6` to `1e-8` depending on scale).

---

## 10) Testing

```bash
cd Baum-Welch-Algorithm
pytest
```

Tests cover core numerical routines and training behavior.

---

## 11) Deployment

For hosted deployment steps, see `DEPLOYMENT.md`.

`app.py` at root is prepared as an entry point for Gunicorn/Eventlet-based deployment (e.g., Zeabur).

---

## 12) Final note

This codebase is intentionally structured so that each mathematical concept from the Baum–Welch derivation has a clear implementation location. If you are reading this as a learner, follow this sequence:

1. Theory section in this README,
2. `hmm_core/inference` (forward/backward),
3. `hmm_core/inference/responsibilities.py`,
4. `hmm_core/optimization/baum_welch_step.py`,
5. `hmm_core/training/trainer.py`.

That path mirrors the EM derivation and makes the algorithm much easier to internalize.

---

## 13) Full EM training procedure (step-by-step)

The training loop implemented in this project follows the canonical EM structure for discrete HMMs.

### 13.1 Inputs

- Observation sequence \(O = (O_1, \dots, O_T)\), integer-encoded
- Number of states \(N\)
- Number of symbols \(M\)
- Initial parameters \(\lambda^{(0)} = (A^{(0)}, B^{(0)}, \pi^{(0)})\)

### 13.2 Iterative updates

At iteration \(k\):

1. Run forward recursion to obtain \(\alpha_t^{(k)}(i)\)
2. Run backward recursion to obtain \(\beta_t^{(k)}(i)\)
3. Compute responsibilities \(\gamma_t^{(k)}(i)\), \(\xi_t^{(k)}(i,j)\)
4. Re-estimate parameters to get \(\lambda^{(k+1)}\)
5. Compute log-likelihood improvement:

\[
\Delta \mathcal{L}^{(k)} = \log P(O\mid\lambda^{(k+1)}) - \log P(O\mid\lambda^{(k)})
\]

6. Stop if \(|\Delta \mathcal{L}^{(k)}| < \varepsilon\) or max iterations reached

### 13.3 Pseudocode

```text
Initialize A, B, pi
repeat:
    alpha, scales = forward(A, B, pi, O)
    beta         = backward(A, B, O, scales)
    gamma, xi    = responsibilities(alpha, beta, A, B, O)
    A, B, pi     = m_step(gamma, xi, O)
    ll_new       = log_likelihood(scales)
until convergence
```

---

## 14) Numerical stability and scaling

For long sequences, raw forward/backward probabilities can underflow to zero. This project uses scaling in inference to maintain stable values.

### 14.1 Why scaling is needed

Forward values multiply many probabilities in \([0,1]\), so magnitude can decay as \(\mathcal{O}(c^T)\) for \(c<1\). Even when mathematically correct, floating-point representation becomes unreliable.

### 14.2 Scaled forward-backward idea

- Compute per-time normalization constants \(c_t\)
- Normalize \(\alpha_t\) at each step
- Reuse consistent scaling in backward pass
- Compute total log-likelihood as:

\[
\log P(O\mid\lambda) = -\sum_{t=1}^{T}\log c_t
\]

This preserves relative structure while avoiding vanishing magnitudes.

---

## 15) Math-to-code map (important for study)

This is the practical bridge between the PDF theory and this codebase.

### 15.1 Core recurrences

- Forward \(\alpha\): `hmm_core/inference/components/alpha.py`
- Backward \(\beta\): `hmm_core/inference/components/beta.py`
- State occupancy \(\gamma\) and transition flow \(\xi\): `hmm_core/inference/responsibilities.py`
- Scaled orchestration + log-likelihood: `hmm_core/inference/forward_backward.py`

### 15.2 Parameter learning

- Baum–Welch re-estimation formulas: `hmm_core/optimization/baum_welch_step.py`
- Convergence criteria: `hmm_core/optimization/convergence.py`
- Full EM loop and callback streaming: `hmm_core/training/trainer.py`

### 15.3 Model and validation layer

- Parameter container and shape-safe model wrapper:
  - `hmm_core/model/parameters.py`
  - `hmm_core/model/hmm.py`
- Probability normalization and input checks:
  - `hmm_core/utils/normalization.py`
  - `hmm_core/utils/validation.py`

---

## 16) Data preparation guide (before training)

Baum–Welch in this project assumes **discrete symbols**. If your source data is text, events, or sensor categories, map each unique item to an integer in \([0, M-1]\).

### 16.1 Recommended preprocessing steps

1. Build vocabulary / category map
2. Encode observations into contiguous integer IDs
3. Validate all IDs are non-negative and within range
4. Keep a reverse map for interpretation in outputs

### 16.2 Common mistakes

- Non-contiguous labels (e.g., 1, 5, 9) without remapping
- Negative indices
- Mixed-type arrays (`str` + `int`)
- Declaring too-small \(M\) compared to actual max observation index

---

## 17) Practical model selection strategy

Since HMM training is non-convex, local optima are expected.

### 17.1 Multi-seed protocol

Run training with multiple random seeds and compare:

- final log-likelihood,
- convergence speed,
- parameter interpretability.

Keep the best run by likelihood (or by downstream metric if task-specific labels exist).

### 17.2 Choosing number of hidden states

Increase \(N\) gradually and monitor:

- improvement in fit quality,
- stability of learned transitions,
- whether extra states are meaningfully distinct.

When possible, compare held-out log-likelihood to avoid overfitting.

---

## 18) Diagnostics and what they mean

### 18.1 Log-likelihood curve

- Healthy training: monotonic increase then flattening
- Sudden oscillation or drop: possible numerical issue or invalid update
- Very early plateau: initialization or model-capacity mismatch

### 18.2 Transition heatmap (A)

- Strong diagonal: sticky regimes (slow state switching)
- Off-diagonal dominance: frequent alternation

### 18.3 Emission heatmap (B)

- Clear row-wise peaks: states specialize in distinct symbols
- Uniform rows: states not well separated

---

## 19) Advanced extraction measures for HMM_v3.pdf

If you want a durable, research-grade extraction pipeline (not just quick text dump), use this workflow.

### 19.1 3-layer extraction strategy

1. **Raw extraction** (already implemented): page text as-is
2. **Clean extraction**: fix spacing/hyphenation, preserve meaning
3. **Structured theory index**: formulas + definitions + implementation pointers

### 19.2 Suggested file set

- `HMM_v3_extracted_raw.txt`
- `HMM_v3_extracted_clean.txt`
- `docs/theory_index.md`
- `docs/formula_glossary.md`

### 19.3 Formula preservation checks

For each key formula block, manually verify against PDF:

- \(\alpha\) initialization and recursion
- \(\beta\) initialization and recursion
- \(\gamma\), \(\xi\) definitions
- re-estimation equations for \(A,B,\pi\)

If plain extraction garbles equations, add one of:

- page-level OCR fallback,
- equation snippets inserted manually into glossary,
- Markdown LaTeX blocks checked against source pages.

### 19.4 Change-management policy

When PDF is updated:

1. Regenerate raw extraction
2. Rebuild clean extraction
3. Diff formula glossary
4. Update README theory sections if needed
5. Commit all in one documentation change set

---

## 20) Troubleshooting (including local server start failures)

### 20.1 `WinError 10048` / port already in use

Cause: another process is bound to `5000`.

Fix (PowerShell):

```powershell
Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
```

Then restart service.

### 20.2 Flask-SocketIO/eventlet startup issues

- Ensure dependencies are installed in the active environment:
  - `flask`
  - `flask-socketio`
  - `eventlet`
- Prefer launching from the project root where imports resolve correctly.

### 20.3 Diagram output missing

If state-diagram rendering fails, install Graphviz system binaries and verify they are on `PATH`.

---

## 21) Reproducibility checklist

To make runs reproducible and audit-friendly:

- Fix random seed in trainer initialization
- Record \(N, M, T\), tolerance, and max iterations
- Save final \((A,B,\pi)\) with timestamp
- Save likelihood history and plots per run
- Keep observation encoding map with the experiment

This checklist is especially important when comparing model variants or reporting experimental results.

---

