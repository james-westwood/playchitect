#!/usr/bin/env python3
"""Generate Flatpak pip sources for offline builds.

This script uses `pip download` to fetch wheels and source tarballs for all
runtime dependencies defined in pyproject.toml, then outputs a Flatpak module
JSON file with explicit download URLs and SHA256 hashes.

The generated JSON replaces the network-using pip install in the Flatpak
manifest — required for Flathub submissions (no network during build).

Usage:
    uv run python scripts/generate_flatpak_sources.py
    # → writes packaging/flatpak/python-deps.json

Prerequisites:
    pip install flatpak-pip-generator
    # or: clone https://github.com/flatpak/flatpak-builder-tools

The recommended tool is flatpak-pip-generator from flatpak-builder-tools.
This script is a thin wrapper that calls it with the right arguments.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
OUTPUT = REPO_ROOT / "packaging" / "flatpak" / "python-deps.json"

# Runtime dependencies from pyproject.toml — keep in sync manually.
DEPENDENCIES = [
    "librosa>=0.10.0",
    "mutagen>=1.47.0",
    "scikit-learn>=1.3.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "click>=8.1.0",
    "pyyaml>=6.0",
]


def main() -> int:
    print("Generating Flatpak pip sources...")
    print(f"Output: {OUTPUT}")

    # Prefer the installed flatpak-pip-generator command
    cmd = ["flatpak-pip-generator", "--output", str(OUTPUT), "--runtime", "org.gnome.Platform//49"]
    cmd += DEPENDENCIES

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\nDone. Generated: {OUTPUT}")
        print("\nNext steps:")
        print("  1. Review the generated JSON.")
        print("  2. Replace the python-deps module in the manifest with:")
        print("       !include python-deps.json")
        print("  3. Remove 'build-options: env: PIP_DISABLE_PIP_VERSION_CHECK'.")
        print("  4. Test: flatpak-builder --user --install --force-clean build-dir \\")
        print("            packaging/flatpak/com.github.jameswestwood.Playchitect.yml")
        return result.returncode
    except FileNotFoundError:
        print(
            "Error: flatpak-pip-generator not found.\n"
            "Install it with:\n"
            "  pip install flatpak-pip-generator\n"
            "Or clone flatpak-builder-tools:\n"
            "  https://github.com/flatpak/flatpak-builder-tools\n",
            file=sys.stderr,
        )
        return 1
    except subprocess.CalledProcessError as e:
        print(f"flatpak-pip-generator failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode


if __name__ == "__main__":
    sys.exit(main())
