# PyPI Publishing Guide

This guide explains how to publish Playchitect to PyPI using the automated GitHub Actions workflow.

## 1. Configure Trusted Publishers

Playchitect uses OIDC-based trusted publishing, which means no long-lived API tokens are needed. You must configure this once on PyPI and TestPyPI.

Follow the [official PyPI documentation on trusted publishers](https://docs.pypi.org/trusted-publishers/) to set this up.

### On PyPI (pypi.org)
- **PyPI project name**: `playchitect`
- **Owner**: `james-westwood`
- **GitHub Repository**: `james-westwood/playchitect`
- **Workflow name**: `publish.yml`
- **Environment**: `pypi`

### On TestPyPI (test.pypi.org)
- **PyPI project name**: `playchitect`
- **Owner**: `james-westwood`
- **GitHub Repository**: `james-westwood/playchitect`
- **Workflow name**: `publish.yml`
- **Environment**: `testpypi`

## 2. Triggering a Release

To trigger the publishing workflow:

1. Update the version number in `pyproject.toml` (if not already done).
2. Push the change to `main`.
3. Create a new GitHub Release on the repository.
4. When the release is published, the `Publish to PyPI` workflow will automatically:
   - Build the distribution using `uv build`.
   - Publish to TestPyPI.
   - Publish to PyPI.

## 3. End User Installation

### Via uv (Recommended)
Users can install Playchitect as a tool using `uv`:

```bash
uv tool install playchitect
```

To include the GUI:
```bash
uv tool install "playchitect[gui]"
```

### GTK4 System Dependency
The GUI requires GTK4 and PyGObject. These must be installed via the OS package manager, as they cannot be reliably built from PyPI.

- **Fedora**: `sudo dnf install python3-gobject gtk4`
- **Ubuntu**: `sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0`

When installing into a virtual environment (including via `uv`), the environment must be created with access to system site packages to find the `gi` module.
