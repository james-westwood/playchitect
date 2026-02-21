# Getting Started

Thank you for your interest in contributing to Playchitect! This guide will help you set up your development environment.

## Prerequisites

*   **Python**: 3.13+
*   **Package Manager**: `uv` (recommended) or `pip`
*   **System Dependencies**: `python3-gobject`, `gtk4` (for GUI development)
*   **Documentation**: Node.js 20+ (for building this site)

## 1. Fork and Clone

Fork the repository on GitHub and clone your fork locally:

```bash
git clone https://github.com/YOUR_USERNAME/playchitect
cd playchitect
```

## 2. Development Setup

Create a virtual environment. We use `--system-site-packages` to allow access to the system-installed GTK4 bindings.

```bash
uv venv --python /usr/bin/python3 --system-site-packages
uv pip install -e ".[dev]"
```

## 3. Pre-commit Hooks

Install the pre-commit hooks to ensure code quality (linting, formatting, type checking) before every commit.

```bash
uv run pre-commit install
```

## 4. Running Tests

Run the full test suite to make sure everything is working:

```bash
uv run pytest -v
```

## Workflow

*   **Never commit to `main`**. Always create a feature branch.
*   **Write tests first** (TDD is encouraged).
*   **Run the review script**: We use an AI-assisted review process. Before opening a PR, run:
    ```bash
    ./scripts/review_pr.sh
    ```
    This will generate a review report using Gemini (if configured) or print a checklist for you to follow.
