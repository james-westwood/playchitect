from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from playchitect.core.track_previewer import PreviewResult, TrackPreviewer


class TestTrackPreviewer:
    """Unit tests for the TrackPreviewer service."""

    def test_sushi_available_true(self):
        """Test sushi_available returns True when sushi is on PATH."""
        with patch("shutil.which", return_value="/usr/bin/sushi"):
            assert TrackPreviewer.sushi_available() is True

    def test_sushi_available_false(self):
        """Test sushi_available returns False when sushi is not on PATH."""
        with patch("shutil.which", return_value=None):
            assert TrackPreviewer.sushi_available() is False

    def test_xdg_open_available_true(self):
        """Test xdg_open_available returns True when xdg-open is on PATH."""
        with patch("shutil.which", return_value="/usr/bin/xdg-open"):
            assert TrackPreviewer.xdg_open_available() is True

    def test_xdg_open_available_false(self):
        """Test xdg_open_available returns False when xdg-open is not on PATH."""
        with patch("shutil.which", return_value=None):
            assert TrackPreviewer.xdg_open_available() is False

    @pytest.mark.parametrize(
        "prefer_sushi, sushi_avail, xdg_avail, expected",
        [
            (True, True, True, "sushi"),
            (True, True, False, "sushi"),
            (True, False, True, "xdg-open"),
            (True, False, False, "none"),
            (False, True, True, "xdg-open"),
            # Even if sushi is available, if we don't prefer it and xdg-open is missing
            (False, True, False, "none"),
            (False, False, True, "xdg-open"),
            (False, False, False, "none"),
        ],
    )
    def test_launcher_name(self, prefer_sushi, sushi_avail, xdg_avail, expected):
        """Test launcher_name returns the correct launcher based on availability and preference."""
        previewer = TrackPreviewer(prefer_sushi=prefer_sushi)
        with (
            patch.object(TrackPreviewer, "sushi_available", return_value=sushi_avail),
            patch.object(TrackPreviewer, "xdg_open_available", return_value=xdg_avail),
        ):
            assert previewer.launcher_name() == expected

    def test_preview_success_sushi(self):
        """Test successful preview using sushi."""
        previewer = TrackPreviewer(prefer_sushi=True)
        test_path = Path("/tmp/test.mp3")

        with (
            patch.object(previewer, "launcher_name", return_value="sushi"),
            patch("subprocess.Popen") as mock_popen,
        ):
            result = previewer.preview(test_path)

            assert result.success is True
            assert result.launcher == "sushi"
            assert result.filepath == test_path
            mock_popen.assert_called_once()
            args, kwargs = mock_popen.call_args
            assert args[0] == ["sushi", str(test_path)]

    def test_preview_success_xdg_open(self):
        """Test successful preview using xdg-open."""
        previewer = TrackPreviewer(prefer_sushi=False)
        test_path = Path("/tmp/test.mp3")

        with (
            patch.object(previewer, "launcher_name", return_value="xdg-open"),
            patch("subprocess.Popen") as mock_popen,
        ):
            result = previewer.preview(test_path)

            assert result.success is True
            assert result.launcher == "xdg-open"
            assert result.filepath == test_path
            mock_popen.assert_called_once_with(
                ["xdg-open", str(test_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def test_preview_none_available(self):
        """Test preview when no launcher is available."""
        previewer = TrackPreviewer()
        test_path = Path("/tmp/test.mp3")

        with patch.object(previewer, "launcher_name", return_value="none"):
            result = previewer.preview(test_path)

            assert result.success is False
            assert result.launcher == "none"
            assert result.filepath == test_path
            assert result.error is not None
            assert "No preview launcher available" in result.error

    def test_preview_os_error(self):
        """Test preview handles OSError from subprocess.Popen."""
        previewer = TrackPreviewer()
        test_path = Path("/tmp/test.mp3")

        with (
            patch.object(previewer, "launcher_name", return_value="sushi"),
            patch("subprocess.Popen", side_effect=OSError("Permission denied")),
        ):
            result = previewer.preview(test_path)

            assert result.success is False
            assert result.launcher == "sushi"
            assert result.error is not None
            assert "Failed to launch sushi" in result.error
            assert "Permission denied" in result.error

    def test_preview_first_success(self):
        """Test preview_first with a non-empty list."""
        previewer = TrackPreviewer()
        test_paths = [Path("/tmp/test1.mp3"), Path("/tmp/test2.mp3")]

        with patch.object(
            previewer, "preview", return_value=PreviewResult(True, "sushi", test_paths[0])
        ) as mock_preview:
            result = previewer.preview_first(test_paths)

            assert result.success is True
            assert result.filepath == test_paths[0]
            mock_preview.assert_called_once_with(test_paths[0])

    def test_preview_first_empty_list(self):
        """Test preview_first with an empty list."""
        previewer = TrackPreviewer()
        result = previewer.preview_first([])

        assert result.success is False
        assert result.launcher == "none"
        assert result.filepath is None
        assert result.error is not None
        assert "No files provided" in result.error
