from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PreviewResult:
    """Result of a track preview operation."""

    success: bool
    launcher: str  # 'sushi', 'xdg-open', or 'none'
    filepath: Path | None
    error: str | None = None  # set on failure


class TrackPreviewer:
    """
    Service for previewing audio tracks using system tools.
    Supports GNOME Sushi and xdg-open.
    """

    def __init__(self, prefer_sushi: bool = True):
        """
        Initialize the track previewer.

        Args:
            prefer_sushi: Whether to prefer GNOME Sushi over xdg-open.
        """
        self.prefer_sushi = prefer_sushi

    @staticmethod
    def sushi_available() -> bool:
        """
        Check if GNOME Sushi is available on the system.

        Returns:
            True if 'sushi' is on PATH, False otherwise.
        """
        return shutil.which("sushi") is not None

    @staticmethod
    def xdg_open_available() -> bool:
        """
        Check if xdg-open is available on the system.

        Returns:
            True if 'xdg-open' is on PATH, False otherwise.
        """
        return shutil.which("xdg-open") is not None

    def launcher_name(self) -> str:
        """
        Determine the name of the launcher that will be used.

        Returns:
            'sushi', 'xdg-open', or 'none' based on availability and preference.
        """
        if self.prefer_sushi and self.sushi_available():
            return "sushi"
        if self.xdg_open_available():
            return "xdg-open"
        return "none"

    def preview(self, filepath: Path) -> PreviewResult:
        """
        Launch a preview for a single file.

        Args:
            filepath: The path to the file to preview.

        Returns:
            A PreviewResult indicating the outcome of the operation.
        """
        launcher = self.launcher_name()

        if launcher == "none":
            return PreviewResult(
                success=False,
                launcher="none",
                filepath=filepath,
                error="No preview launcher available (sushi or xdg-open not found)",
            )

        try:
            # Use Popen to launch the previewer without blocking
            subprocess.Popen(
                [launcher, str(filepath)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return PreviewResult(success=True, launcher=launcher, filepath=filepath)
        except OSError as e:
            return PreviewResult(
                success=False,
                launcher=launcher,
                filepath=filepath,
                error=f"Failed to launch {launcher}: {e}",
            )

    def preview_first(self, filepaths: list[Path]) -> PreviewResult:
        """
        Preview the first file in a list of file paths.

        Args:
            filepaths: A list of file paths.

        Returns:
            A PreviewResult for the first file, or a failure if the list is empty.
        """
        if not filepaths:
            return PreviewResult(
                success=False,
                launcher="none",
                filepath=None,
                error="No files provided for preview",
            )

        return self.preview(filepaths[0])
