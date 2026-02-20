#!/usr/bin/env python3
"""Generate PNG icon sizes and .ico file from source JPEG logo.

Usage:
    uv run python scripts/generate_icons.py          # generate all icons
    uv run python scripts/generate_icons.py --check  # verify all paths exist
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).parent.parent
SOURCE_IMAGE = REPO_ROOT / "img" / "playchitect_logo.jpg"
APP_ID = "com.github.jameswestwood.Playchitect"
PNG_SIZES = (16, 22, 24, 32, 48, 64, 128, 256, 512)
ICO_SIZES = (16, 32, 48, 64, 128, 256)
ICONS_DIR = REPO_ROOT / "data" / "icons"
ICO_PATH = REPO_ROOT / "data" / "icons" / "playchitect.ico"


def png_path(size: int) -> Path:
    return ICONS_DIR / "hicolor" / f"{size}x{size}" / "apps" / f"{APP_ID}.png"


def generate() -> None:
    if not SOURCE_IMAGE.exists():
        print(f"ERROR: source image not found: {SOURCE_IMAGE}", file=sys.stderr)
        sys.exit(1)

    with Image.open(SOURCE_IMAGE) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGBA")

        for size in PNG_SIZES:
            dest = png_path(size)
            dest.parent.mkdir(parents=True, exist_ok=True)
            resized = img.resize((size, size), Image.LANCZOS)
            resized.save(dest, format="PNG")
            print(f"  wrote {dest.relative_to(REPO_ROOT)}")

        ICO_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Pillow's ICO plugin resizes from the base image when given `sizes`
        img.save(
            ICO_PATH,
            format="ICO",
            sizes=[(s, s) for s in ICO_SIZES],
        )
        print(f"  wrote {ICO_PATH.relative_to(REPO_ROOT)}")

    print("Icon generation complete.")


def check() -> None:
    missing: list[Path] = []
    for size in PNG_SIZES:
        p = png_path(size)
        if not p.exists():
            missing.append(p)
    if not ICO_PATH.exists():
        missing.append(ICO_PATH)

    if missing:
        print("MISSING icon files:", file=sys.stderr)
        for p in missing:
            print(f"  {p.relative_to(REPO_ROOT)}", file=sys.stderr)
        sys.exit(1)

    print("All icon files present.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify all icon files exist without regenerating",
    )
    args = parser.parse_args()

    if args.check:
        check()
    else:
        generate()


if __name__ == "__main__":
    main()
