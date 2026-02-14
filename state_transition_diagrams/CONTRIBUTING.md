# Contributing to State Transition Diagrams

Thank you for considering contributing! This guide covers the development workflow.

---

## Development Setup

```bash
# Clone and enter the project
git clone https://github.com/gabsgj/state-transition-diagrams.git
cd state-transition-diagrams

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows PowerShell

# Install in editable mode with all extras
pip install -e ".[all,dev,demo]"
```

---

## Running the Demo

```bash
python -m demo.app
# Open http://127.0.0.1:5050
```

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=state_transition_diagrams --cov-report=term-missing

# Specific test file
pytest tests/test_config.py -v
```

> **Note:** Tests in `test_renderer.py` require the `graphviz` Python package
> *and* the [Graphviz system binary](https://graphviz.org/download/). They are
> automatically skipped if either is missing.

---

## Project Layout

| Path | Purpose |
|---|---|
| `__init__.py` | Package exports (`DiagramConfig`, `render_state_diagram`, `create_blueprint`) |
| `config.py` | `DiagramConfig` dataclass — all configuration in one place |
| `renderer.py` | Graphviz-based static diagram renderer |
| `flask_blueprint.py` | Flask blueprint for serving JS/CSS assets |
| `static/js/state_diagram.js` | Interactive D3.js diagram (the main JS library) |
| `static/css/state_diagram.css` | Styles for the interactive diagram |
| `demo/` | Self-contained Flask demo app |
| `tests/` | pytest test suite |

---

## Coding Standards

- **Python:** Follow PEP 8.  Use type hints where practical.
- **JavaScript:** Use `class` syntax. Prefix private methods with `_`.
- **CSS:** Use the `std-` prefix for all new class names. Keep backward-compatible `hmm-` aliases.
- **Commits:** Use [Conventional Commits](https://www.conventionalcommits.org/) format.

---

## Pull Request Checklist

1. [ ] All existing tests pass (`pytest tests/ -v`)
2. [ ] New features include corresponding tests
3. [ ] Demo app updated if the public API changed
4. [ ] `CHANGELOG.md` updated under the **Unreleased** section
5. [ ] No breaking changes to the public API without discussion

---

## Releasing

1. Update version in `__init__.py` and `pyproject.toml`
2. Update `CHANGELOG.md` — move Unreleased items under a versioned heading
3. Tag the commit: `git tag v1.x.x`
4. Build: `python -m build`
5. Upload: `twine upload dist/*`

---

## License

By contributing you agree that your contributions will be licensed under the [MIT License](LICENSE).
