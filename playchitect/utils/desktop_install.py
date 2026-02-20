"""Cross-platform desktop integration installer for Playchitect.

Copies the .desktop file, AppStream metainfo, and icon PNGs to the appropriate
XDG locations and refreshes the desktop and icon caches.

Usage (after ``pip install -e .`` or via uv):
    playchitect-install-desktop             # per-user install (~/.local/share)
    playchitect-install-desktop --system-wide  # system-wide (/usr/local/share)
    playchitect-install-desktop --uninstall    # remove per-user files
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from pathlib import Path

import click

APP_ID = "com.github.jameswestwood.Playchitect"
PNG_SIZES = (16, 22, 24, 32, 48, 64, 128, 256, 512)

# Source tree layout: this file lives at playchitect/utils/desktop_install.py,
# so three parents up is the repo root.
_DATA_DIR = Path(__file__).parent.parent.parent / "data"

log = logging.getLogger(__name__)


def _run(cmd: list[str]) -> None:
    """Run a post-install helper command; log but never abort on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.info("Command %s exited %d: %s", cmd, result.returncode, result.stderr.strip())


def install(system_wide: bool = False) -> None:
    """Install desktop integration files.

    Args:
        system_wide: If True, installs to /usr/local/share (requires write
            permission); otherwise installs to ~/.local/share.
    """
    if platform.system() != "Linux":
        click.echo(f"Desktop integration not supported on {platform.system()}")
        return

    base = Path("/usr/local/share") if system_wide else Path.home() / ".local" / "share"

    # 1. .desktop file
    app_dir = base / "applications"
    app_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_DATA_DIR / "playchitect.desktop", app_dir / "playchitect.desktop")
    click.echo(f"  installed {app_dir / 'playchitect.desktop'}")

    # 2. AppStream metainfo
    metainfo_dir = base / "metainfo"
    metainfo_dir.mkdir(parents=True, exist_ok=True)
    appdata_src = _DATA_DIR / f"{APP_ID}.appdata.xml"
    shutil.copy2(appdata_src, metainfo_dir / f"{APP_ID}.appdata.xml")
    click.echo(f"  installed {metainfo_dir / f'{APP_ID}.appdata.xml'}")

    # 3. Icon PNGs
    for size in PNG_SIZES:
        src = _DATA_DIR / "icons" / "hicolor" / f"{size}x{size}" / "apps" / f"{APP_ID}.png"
        if not src.exists():
            log.warning("Icon not found, skipping: %s", src)
            continue
        dest_dir = base / "icons" / "hicolor" / f"{size}x{size}" / "apps"
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / f"{APP_ID}.png")
        click.echo(f"  installed {dest_dir / f'{APP_ID}.png'}")

    # 4. Refresh desktop database
    _run(["update-desktop-database", str(app_dir)])

    # 5. Refresh icon cache
    icon_base = str(base / "icons" / "hicolor")
    _run(["gtk-update-icon-cache", "--force", icon_base])

    click.echo("Desktop integration installed.")


def uninstall(system_wide: bool = False) -> None:
    """Remove desktop integration files.

    Args:
        system_wide: If True, removes from /usr/local/share; otherwise from
            ~/.local/share.
    """
    if platform.system() != "Linux":
        click.echo(f"Desktop integration not supported on {platform.system()}")
        return

    base = Path("/usr/local/share") if system_wide else Path.home() / ".local" / "share"

    files_to_remove: list[Path] = [
        base / "applications" / "playchitect.desktop",
        base / "metainfo" / f"{APP_ID}.appdata.xml",
    ]
    for size in PNG_SIZES:
        files_to_remove.append(
            base / "icons" / "hicolor" / f"{size}x{size}" / "apps" / f"{APP_ID}.png"
        )

    for path in files_to_remove:
        if path.exists():
            path.unlink()
            click.echo(f"  removed {path}")
        else:
            log.debug("Already absent: %s", path)

    click.echo("Desktop integration removed.")


@click.command()
@click.option(
    "--system-wide",
    is_flag=True,
    help="Install to /usr/local/share instead of ~/.local/share",
)
@click.option("--uninstall", "do_uninstall", is_flag=True, help="Remove installed files")
def main(system_wide: bool, do_uninstall: bool) -> None:
    """Install or uninstall Playchitect desktop integration."""
    logging.basicConfig(level=logging.INFO)
    if do_uninstall:
        uninstall(system_wide=system_wide)
    else:
        install(system_wide=system_wide)
