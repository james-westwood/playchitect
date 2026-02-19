"""
Unit tests for audio_scanner module.
"""

from pathlib import Path

import pytest

from playchitect.core.audio_scanner import AudioScanner


class TestAudioScanner:
    """Test AudioScanner class."""

    def test_supported_extensions(self):
        """Test that supported extensions are defined correctly."""
        extensions = AudioScanner.get_supported_extensions()

        assert ".mp3" in extensions
        assert ".flac" in extensions
        assert ".wav" in extensions
        assert ".ogg" in extensions
        assert ".m4a" in extensions
        assert ".opus" in extensions
        assert ".aiff" in extensions

    def test_scanner_initialization(self):
        """Test scanner initialization."""
        scanner = AudioScanner(follow_symlinks=False)
        assert scanner.follow_symlinks is False

        scanner_with_symlinks = AudioScanner(follow_symlinks=True)
        assert scanner_with_symlinks.follow_symlinks is True

    def test_scan_nonexistent_directory(self):
        """Test scanning nonexistent directory raises error."""
        scanner = AudioScanner()
        nonexistent_path = Path("/nonexistent/path/to/music")

        with pytest.raises(FileNotFoundError):
            scanner.scan(nonexistent_path)

    def test_scan_file_instead_of_directory(self, tmp_path):
        """Test scanning a file instead of directory raises error."""
        scanner = AudioScanner()
        test_file = tmp_path / "test.txt"
        test_file.write_text("not a directory")

        with pytest.raises(NotADirectoryError):
            scanner.scan(test_file)

    def test_scan_empty_directory(self, tmp_path):
        """Test scanning empty directory returns empty list."""
        scanner = AudioScanner()
        result = scanner.scan(tmp_path)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_scan_directory_with_audio_files(self, tmp_path):
        """Test scanning directory with audio files."""
        scanner = AudioScanner()

        # Create test audio files
        (tmp_path / "track1.mp3").touch()
        (tmp_path / "track2.flac").touch()
        (tmp_path / "track3.wav").touch()
        (tmp_path / "readme.txt").touch()  # Non-audio file

        result = scanner.scan(tmp_path)

        assert len(result) == 3
        assert all(isinstance(p, Path) for p in result)

        # Check file names (order may vary)
        filenames = {p.name for p in result}
        assert filenames == {"track1.mp3", "track2.flac", "track3.wav"}

    def test_scan_nested_directories(self, tmp_path):
        """Test scanning nested directories."""
        scanner = AudioScanner()

        # Create nested structure
        subdir1 = tmp_path / "techno"
        subdir2 = tmp_path / "house"
        subdir1.mkdir()
        subdir2.mkdir()

        (subdir1 / "track1.mp3").touch()
        (subdir1 / "track2.flac").touch()
        (subdir2 / "track3.wav").touch()

        result = scanner.scan(tmp_path)

        assert len(result) == 3

    def test_scan_with_various_extensions(self, tmp_path):
        """Test scanning with various audio format extensions."""
        scanner = AudioScanner()

        # Create files with different extensions
        extensions = [".mp3", ".flac", ".wav", ".ogg", ".m4a", ".opus", ".aiff"]
        for i, ext in enumerate(extensions):
            (tmp_path / f"track{i}{ext}").touch()

        result = scanner.scan(tmp_path)

        assert len(result) == len(extensions)

    def test_scan_case_insensitive_extensions(self, tmp_path):
        """Test that extensions are case-insensitive."""
        scanner = AudioScanner()

        (tmp_path / "track1.MP3").touch()
        (tmp_path / "track2.FlAc").touch()
        (tmp_path / "track3.WAV").touch()

        result = scanner.scan(tmp_path)

        assert len(result) == 3

    def test_scan_ignores_non_audio_files(self, tmp_path):
        """Test that non-audio files are ignored."""
        scanner = AudioScanner()

        (tmp_path / "track.mp3").touch()
        (tmp_path / "image.jpg").touch()
        (tmp_path / "document.pdf").touch()
        (tmp_path / "script.py").touch()

        result = scanner.scan(tmp_path)

        assert len(result) == 1
        assert result[0].name == "track.mp3"

    def test_scan_multiple_directories(self, tmp_path):
        """Test scanning multiple directories."""
        scanner = AudioScanner()

        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "track1.mp3").touch()
        (dir2 / "track2.flac").touch()

        result = scanner.scan_multiple([dir1, dir2])

        assert len(result) == 2

    def test_scan_with_symlinks(self, tmp_path):
        """Test scanning with symlinks (when follow_symlinks is enabled)."""
        # Create original file
        original_dir = tmp_path / "original"
        original_dir.mkdir()
        original_file = original_dir / "track.mp3"
        original_file.touch()

        # Create symlink
        link_dir = tmp_path / "links"
        link_dir.mkdir()
        link_file = link_dir / "linked_track.mp3"

        try:
            link_file.symlink_to(original_file)
        except OSError:
            pytest.skip("Symlinks not supported on this platform")

        # Scan without following symlinks
        scanner_no_follow = AudioScanner(follow_symlinks=False)
        result_no_follow = scanner_no_follow.scan(link_dir)
        assert len(result_no_follow) == 0

        # Scan with following symlinks
        scanner_follow = AudioScanner(follow_symlinks=True)
        result_follow = scanner_follow.scan(link_dir)
        assert len(result_follow) == 1

    def test_is_supported_format_private_method(self):
        """Test _is_supported_format private method."""
        scanner = AudioScanner()

        assert scanner._is_supported_format(Path("track.mp3")) is True
        assert scanner._is_supported_format(Path("track.flac")) is True
        assert scanner._is_supported_format(Path("track.txt")) is False
        assert scanner._is_supported_format(Path("track.jpg")) is False
