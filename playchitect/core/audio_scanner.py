"""
Audio file scanner for discovering music files in directories.

Supports common audio formats: MP3, FLAC, WAV, OGG, M4A, AAC, WMA, AIFF, APE, OPUS
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioScanner:
    """Scans directories for audio files with supported formats."""

    # Supported audio file extensions
    SUPPORTED_EXTENSIONS: set[str] = {
        ".mp3",
        ".flac",
        ".wav",
        ".ogg",
        ".m4a",
        ".aac",
        ".wma",
        ".aiff",
        ".aif",
        ".ape",
        ".opus",
    }

    def __init__(self, follow_symlinks: bool = False):
        """
        Initialize audio scanner.

        Args:
            follow_symlinks: Whether to follow symbolic links during directory traversal
        """
        self.follow_symlinks = follow_symlinks

    def scan(self, directory: Path) -> list[Path]:
        """
        Scan directory recursively for audio files.

        Args:
            directory: Root directory to scan

        Returns:
            List of Path objects for discovered audio files

        Raises:
            FileNotFoundError: If directory does not exist
            PermissionError: If directory cannot be accessed
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory does not exist: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory}")

        logger.info(f"Scanning directory: {directory}")
        audio_files: list[Path] = []

        try:
            for file_path in directory.rglob("*"):
                # Skip if it's a symlink and we're not following them
                if file_path.is_symlink() and not self.follow_symlinks:
                    logger.debug(f"Skipping symlink: {file_path}")
                    continue

                # Check if it's a file with supported extension
                if file_path.is_file() and self._is_supported_format(file_path):
                    audio_files.append(file_path)
                    logger.debug(f"Found audio file: {file_path}")

        except PermissionError as e:
            logger.error(f"Permission denied accessing: {e.filename}")
            raise

        logger.info(f"Found {len(audio_files)} audio files")
        return audio_files

    def _is_supported_format(self, file_path: Path) -> bool:
        """
        Check if file has a supported audio format extension.

        Args:
            file_path: Path to check

        Returns:
            True if file extension is supported
        """
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def scan_multiple(self, directories: list[Path]) -> list[Path]:
        """
        Scan multiple directories for audio files.

        Args:
            directories: List of directories to scan

        Returns:
            Combined list of audio files from all directories

        Raises:
            FileNotFoundError: If any directory does not exist
        """
        all_files: list[Path] = []

        for directory in directories:
            files = self.scan(directory)
            all_files.extend(files)

        return all_files

    @classmethod
    def get_supported_extensions(cls) -> set[str]:
        """
        Get set of supported audio file extensions.

        Returns:
            Set of supported extensions (including leading dot)
        """
        return cls.SUPPORTED_EXTENSIONS.copy()
