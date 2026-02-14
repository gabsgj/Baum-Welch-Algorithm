# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

_(Nothing yet.)_

---

## [1.0.0] — 2026-02-14

### Added

- **Interactive D3.js diagram** (`StateTransitionDiagram` class)
  - 3-tier layout: Start (π) → Hidden States → Observations
  - Particle flow animation on transition arcs
  - De-congestion toggle: filtered mode vs all-transitions mode
  - 3D perspective view toggle
  - Inspector panel (click a state node)
  - Replay controls: play, pause, step, seek, speed
  - Timeline scrubbing with pointer/touch support
  - Fullscreen mode
  - SVG and PNG export (`saveSVG()`, `savePNG()`)
  - `HMMDiagram` backward-compatibility alias
- **Static Graphviz renderer** (`render_state_diagram()`)
  - Configurable output format: SVG, PNG, PDF
  - Configurable layout engines: circo, dot, neato, fdp
  - Probability threshold for edge suppression
  - Custom state labels and colours
- **`DiagramConfig` dataclass**
  - Centralised configuration for both static and interactive diagrams
  - `to_js_config()` serialisation for template injection
- **Flask blueprint** (`create_blueprint()`)
  - Serves JS/CSS static assets under a configurable URL prefix
- **Demo application** (`demo/`)
  - Self-contained Flask app exercising all library features
  - Simulated training iterations via `/api/simulate`
  - Server-side Graphviz rendering via `/api/render_static`
  - Config inspector via `/api/config`
- **Test suite** (`tests/`)
  - Package imports and metadata (`test_package.py`)
  - DiagramConfig defaults, overrides, JS serialisation (`test_config.py`)
  - Graphviz rendering and file output (`test_renderer.py`)
  - Flask blueprint and asset verification (`test_flask_blueprint.py`)
  - Demo app endpoint coverage (`test_demo_app.py`)
- **Documentation**
  - Comprehensive `README.md` with configuration reference and API tables
  - `CONTRIBUTING.md` with dev setup, testing, and release instructions
  - MIT `LICENSE`
