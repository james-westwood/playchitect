# Testing

Playchitect maintains a comprehensive test suite with >85% coverage target for core modules.

## Running Tests

Run the full suite:
```bash
uv run pytest -v
```

### Coverage Reports

Generate an HTML coverage report:
```bash
uv run pytest --cov=playchitect --cov-report=html
```

Open `htmlcov/index.html` in your browser to view line-by-line coverage.

### Test Categories

*   **Unit Tests**: `tests/unit/` - Test individual classes and functions in isolation.
*   **Integration Tests**: `tests/integration/` - Verify CLI commands and end-to-end functionality.
*   **GUI Tests**: `tests/gui/` - Headless tests for GTK widgets (using mocks).
*   **Benchmarks**: `tests/benchmarks/` - Performance regression testing.

Run specific test types:
```bash
uv run pytest tests/unit/
uv run pytest tests/gui/
```

### Benchmarks

Run benchmarks only (skip functional tests):
```bash
uv run pytest tests/benchmarks/ --benchmark-only
```

## Pre-commit Hooks

We use `pre-commit` to ensure code quality. This runs automatically before every commit, but you can also run it manually:

```bash
uv run pre-commit run --all-files
```

Checks include:
*   **ruff**: Linting and formatting
*   **ty**: Static type checking
*   **pytest**: Quick unit tests
*   **cli-smoke-test**: Basic integration verification

## Writing Tests

*   Use `tmp_path` fixture for file I/O tests.
*   Mock external dependencies (e.g., `librosa`, `Gtk`) where appropriate.
*   Ensure tests verify behavior, not just execution.
