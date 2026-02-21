# Releasing

Playchitect uses GitHub Actions for automated releases to PyPI and GitHub Releases (Flatpak).

## Versioning

We follow [Semantic Versioning](https://semver.org/).

1.  Update the version number in `pyproject.toml`.
2.  Update `CHANGELOG.md` with release notes.

## Release Process

1.  **Pull Request**: Changes are merged into `main` via a Pull Request.
2.  **Tagging**: Create a git tag for the new version (e.g., `v1.2.0`).
3.  **Push Tag**: Push the tag to GitHub:
    ```bash
    git tag v1.2.0
    git push origin v1.2.0
    ```
4.  **GitHub Actions**: The `.github/workflows/publish.yml` and `.github/workflows/flatpak.yml` workflows trigger automatically.

### PyPI Publish
*   Builds source distribution (sdist) and wheel.
*   Publishes to PyPI via Trusted Publishing (OIDC).

### Flatpak Release
*   Builds the Flatpak bundle using `flatpak-builder`.
*   Uploads the `.flatpak` file to the GitHub Release associated with the tag.

## Post-Release

*   Verify installation via `pip install playchitect`.
*   Verify Flatpak installation.
*   Announce the release!
