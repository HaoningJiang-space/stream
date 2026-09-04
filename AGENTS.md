# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `stream/`. Major areas include workload parsing and IR (`stream/workload/`, `stream/parser/`, `stream/ir/`), optimization (`stream/opt/`), cost modeling (`stream/cost_model/`), hardware descriptions (`stream/hardware/` and `stream/inputs/`), and AIE compilation (`stream/compiler/`). Runnable examples and analysis utilities belong in `scripts/`. Tests are grouped under `tests/unit/`, `tests/integration/`, `tests/compiler/`, and `tests/rewrites/`; reusable inputs belong in `tests/fixtures/`. Documentation sources and images live in `docs/source/`.

## Build, Test, and Development Commands

Use Python 3.12 or newer and develop in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

- `pytest tests/ -m "not slow"` runs the normal local suite.
- `pytest tests/ -m "slow" -k "not gurobi"` runs solver-heavy integration coverage without licensed Gurobi.
- `ruff check .` and `ruff format --check .` reproduce CI style checks.
- `just --list` shows supported workflows; for example, `just matrix` exercises all generic hardware/workload combinations.
- `cd docs && pip install -r requirements.txt && mkdocs build --strict` validates documentation.

## Coding Style & Naming Conventions

Ruff enforces formatting, imports, and lint rules with a 120-character line limit. Use four-space indentation, absolute imports, `snake_case` for modules/functions, and `PascalCase` for classes; pipeline stage classes end in `Stage`. Prefer Python 3.12 syntax such as `X | None` and `list[X]`. Add type hints to public APIs and write Google-style docstrings. Keep imports ordered standard library, third-party, then internal.

## Testing Guidelines

Use pytest. Name files `test_<behavior>.py` and test functions `test_<expected_behavior>`. Put focused logic tests in `tests/unit/` and end-to-end solver paths in `tests/integration/`; mark expensive cases with `@pytest.mark.slow`. Add fixtures under `tests/fixtures/` rather than embedding large generated data. There is no numeric coverage threshold, but every behavior change should include a regression test.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative summaries such as `Fix dma task release ...`, often followed by a PR number. Keep each commit focused; reserve `bump version X -> Y` for automated releases. PRs should explain the problem and solution, link relevant issues, list validation commands, and update documentation for public behavior. Include generated-output excerpts or screenshots when visualization, tracing, or documentation changes are user-visible.
