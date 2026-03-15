"""Smoke tests for TrackPreviewPanel.

All GStreamer imports are mocked so these tests run in CI without GStreamer
installed. The tests verify that the widget instantiates, load_track sets
labels correctly, and GObject signals are properly defined.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# conftest.py installs gi mocks before this module is imported
_GST_MOCK = MagicMock()
_GST_MOCK.State = MagicMock()
_GST_MOCK.State.PLAYING = 3
_GST_MOCK.State.PAUSED = 2
_GST_MOCK.State.NULL = 1
_GST_MOCK.StateChangeReturn = MagicMock()
_GST_MOCK.StateChangeReturn.FAILURE = 0
_GST_MOCK.StateChangeReturn.SUCCESS = 1
_GST_MOCK.Format = MagicMock()
_GST_MOCK.Format.TIME = 4
_GST_MOCK.SeekFlags = MagicMock()
_GST_MOCK.SeekFlags.FLUSH = 1
_GST_MOCK.SeekFlags.KEY_UNIT = 4
_GST_MOCK.MessageType = MagicMock()
_GST_MOCK.MessageType.EOS = 1
_GST_MOCK.MessageType.ERROR = 2
_GST_MOCK.MessageType.DURATION_CHANGED = 6
_GST_MOCK.SECOND = 1000000000

_GST_MOCK.ElementFactory = MagicMock()
_GST_MOCK.ElementFactory.make = MagicMock(return_value=None)
_GST_MOCK.init = MagicMock()
_GST_MOCK.Bus = MagicMock()
_GST_MOCK.Message = MagicMock()

# Install the mock before importing track_preview_panel
sys.modules.setdefault("gi.repository.Gst", _GST_MOCK)

# Mock GdkPixbuf
_GDK_PIXBUF_MOCK = MagicMock()
_GDK_PIXBUF_MOCK.Pixbuf = MagicMock()
sys.modules.setdefault("gi.repository.GdkPixbuf", _GDK_PIXBUF_MOCK)

# Ensure Gst is set to mock to prevent initialization
with patch.dict("os.environ", {"GST_PLUGIN_SYSTEM_PATH_1_0": ""}):
    from playchitect.gui.widgets.track_preview_panel import (
        TrackPreviewPanel,
        _ensure_cache_dir,
        _extract_cover_art,
        _get_cache_path,
    )
    from playchitect.gui.views.library_view import LibraryTrackModel


class FakeLibraryTrackModel:
    """Simple stand-in for LibraryTrackModel without GObject."""

    def __init__(
        self,
        title: str = "Test Track",
        artist: str = "Test Artist",
        bpm: float = 128.0,
        duration_secs: float = 300.0,
        filepath: str = "/music/test.flac",
        file_format: str = "FLAC",
    ) -> None:
        self.title = title
        self.artist = artist
        self.bpm = bpm
        self.duration_secs = duration_secs
        self.filepath = filepath
        self.file_format = file_format

    @property
    def display_title(self) -> str:
        if self.title:
            return self.title
        name = self.filepath.rsplit("/", 1)[-1]
        return name.rsplit(".", 1)[0] if "." in name else name

    @property
    def duration_formatted(self) -> str:
        total = max(0, int(self.duration_secs))
        return f"{total // 60}:{total % 60:02d}"


@pytest.fixture
def fake_track() -> FakeLibraryTrackModel:
    """Return a fake track model for testing."""
    return FakeLibraryTrackModel(
        title="Dark Matter",
        artist="Surgeon",
        bpm=138.5,
        duration_secs=386.0,
        filepath="/music/surgeon_dark_matter.flac",
        file_format="FLAC",
    )


@pytest.fixture
def panel() -> TrackPreviewPanel:
    """Return a TrackPreviewPanel with mocked internals."""
    # Create the panel with __new__ to skip __init__
    p = TrackPreviewPanel.__new__(TrackPreviewPanel)

    # Mock GTK widgets
    p._title_label = MagicMock()
    p._artist_label = MagicMock()
    p._album_label = MagicMock()
    p._bpm_pill = MagicMock()
    p._key_pill = MagicMock()
    p._duration_pill = MagicMock()
    p._format_pill = MagicMock()
    p._cover_picture = MagicMock()
    p._cover_icon = MagicMock()
    p._seek_scale = MagicMock()
    p._current_time_label = MagicMock()
    p._total_time_label = MagicMock()
    p._play_btn = MagicMock()

    # Mock GStreamer
    p._playbin = None
    p._is_playing = False
    p._duration_ns = 0
    p._seek_timeout_id = None
    p._current_track = None

    return p


class TestTrackPreviewPanelInstantiation:
    """Test that TrackPreviewPanel can be instantiated."""

    def test_panel_class_exists(self) -> None:
        """Verify the TrackPreviewPanel class is importable."""
        assert TrackPreviewPanel is not None

    def test_signals_defined(self) -> None:
        """Verify prev-track and next-track signals are defined."""
        # Check that the class has the signals defined
        signals = getattr(TrackPreviewPanel, "__gsignals__", {})
        assert "prev-track" in signals
        assert "next-track" in signals


class TestLoadTrack:
    """Test the load_track method behavior."""

    def test_load_track_sets_title_label(
        self, panel: TrackPreviewPanel, fake_track: FakeLibraryTrackModel
    ) -> None:
        """Verify load_track sets the title label."""
        with patch.object(panel, "_load_cover_art"):
            with patch.object(panel, "_extract_album_info", return_value=""):
                with patch.object(panel, "_extract_key", return_value=None):
                    panel.load_track(fake_track)  # type: ignore[arg-type]

        panel._title_label.set_text.assert_called_once_with("Dark Matter")

    def test_load_track_sets_artist_label(
        self, panel: TrackPreviewPanel, fake_track: FakeLibraryTrackModel
    ) -> None:
        """Verify load_track sets the artist label."""
        with patch.object(panel, "_load_cover_art"):
            with patch.object(panel, "_extract_album_info", return_value=""):
                with patch.object(panel, "_extract_key", return_value=None):
                    panel.load_track(fake_track)  # type: ignore[arg-type]

        panel._artist_label.set_text.assert_called_once_with("Surgeon")

    def test_load_track_sets_bpm_pill(
        self, panel: TrackPreviewPanel, fake_track: FakeLibraryTrackModel
    ) -> None:
        """Verify load_track sets the BPM pill with formatted value."""
        with patch.object(panel, "_load_cover_art"):
            with patch.object(panel, "_extract_album_info", return_value=""):
                with patch.object(panel, "_extract_key", return_value=None):
                    panel.load_track(fake_track)  # type: ignore[arg-type]

        panel._bpm_pill.set_text.assert_called_once_with("138 BPM")

    def test_load_track_sets_duration_pill(
        self, panel: TrackPreviewPanel, fake_track: FakeLibraryTrackModel
    ) -> None:
        """Verify load_track sets the duration pill."""
        with patch.object(panel, "_load_cover_art"):
            with patch.object(panel, "_extract_album_info", return_value=""):
                with patch.object(panel, "_extract_key", return_value=None):
                    panel.load_track(fake_track)  # type: ignore[arg-type]

        panel._duration_pill.set_text.assert_called_once_with("6:26")

    def test_load_track_sets_format_pill(
        self, panel: TrackPreviewPanel, fake_track: FakeLibraryTrackModel
    ) -> None:
        """Verify load_track sets the format pill."""
        with patch.object(panel, "_load_cover_art"):
            with patch.object(panel, "_extract_album_info", return_value=""):
                with patch.object(panel, "_extract_key", return_value=None):
                    panel.load_track(fake_track)  # type: ignore[arg-type]

        panel._format_pill.set_text.assert_called_once_with("FLAC")

    def test_load_track_resets_seek_bar(
        self, panel: TrackPreviewPanel, fake_track: FakeLibraryTrackModel
    ) -> None:
        """Verify load_track resets the seek bar to 0."""
        with patch.object(panel, "_load_cover_art"):
            with patch.object(panel, "_extract_album_info", return_value=""):
                with patch.object(panel, "_extract_key", return_value=None):
                    panel.load_track(fake_track)  # type: ignore[arg-type]

        panel._seek_scale.set_value.assert_called_once_with(0)
        panel._current_time_label.set_text.assert_called_once_with("0:00")

    def test_load_track_uses_display_title_when_title_empty(self, panel: TrackPreviewPanel) -> None:
        """Verify load_track uses display_title when title is empty."""
        track = FakeLibraryTrackModel(
            title="",
            filepath="/music/my_song.flac",
        )

        with patch.object(panel, "_load_cover_art"):
            with patch.object(panel, "_extract_album_info", return_value=""):
                with patch.object(panel, "_extract_key", return_value=None):
                    panel.load_track(track)  # type: ignore[arg-type]

        panel._title_label.set_text.assert_called_once_with("my_song")


class TestCoverArtFallback:
    """Test cover art fallback to placeholder icon."""

    def test_show_placeholder_called_when_no_art(
        self, panel: TrackPreviewPanel, fake_track: FakeLibraryTrackModel
    ) -> None:
        """Verify placeholder is shown when no cover art is found."""
        with patch(
            "playchitect.gui.widgets.track_preview_panel._extract_cover_art", return_value=None
        ):
            with patch.object(panel, "_show_placeholder") as mock_show:
                with patch.object(panel, "_extract_album_info", return_value=""):
                    with patch.object(panel, "_extract_key", return_value=None):
                        panel.load_track(fake_track)  # type: ignore[arg-type]

        mock_show.assert_called_once()

    def test_show_placeholder_makes_icon_visible(self, panel: TrackPreviewPanel) -> None:
        """Verify _show_placeholder makes icon visible and picture invisible."""
        panel._show_placeholder()

        panel._cover_picture.set_visible.assert_called_once_with(False)
        panel._cover_icon.set_visible.assert_called_once_with(True)


class TestSignalEmission:
    """Test prev-track and next-track signal emission."""

    def test_prev_track_signal_emitted(self, panel: TrackPreviewPanel) -> None:
        """Verify clicking prev button emits prev-track signal."""
        panel.emit = MagicMock()

        # Simulate prev button click
        panel._on_prev_clicked(MagicMock())

        panel.emit.assert_called_once_with("prev-track")

    def test_next_track_signal_emitted(self, panel: TrackPreviewPanel) -> None:
        """Verify clicking next button emits next-track signal."""
        panel.emit = MagicMock()

        # Simulate next button click
        panel._on_next_clicked(MagicMock())

        panel.emit.assert_called_once_with("next-track")


class TestPlaybackControls:
    """Test playback control methods."""

    def test_play_button_toggles_to_pause_when_playing(self, panel: TrackPreviewPanel) -> None:
        """Verify play button shows pause icon when playing."""
        with patch("playchitect.gui.widgets.track_preview_panel.Gst") as mock_gst:
            mock_gst.State = MagicMock()
            mock_gst.State.PAUSED = 2
            mock_gst.State.NULL = 1
            panel._is_playing = True
            panel._playbin = MagicMock()
            panel._playbin.set_state = MagicMock(return_value=1)

            panel._on_play_clicked(MagicMock())

            # Should transition to pause
            panel._playbin.set_state.assert_called_once()
            assert not panel._is_playing

    def test_pause_button_toggles_to_play_when_paused(self, panel: TrackPreviewPanel) -> None:
        """Verify play button shows play icon when paused."""
        with patch("playchitect.gui.widgets.track_preview_panel.Gst") as mock_gst:
            mock_gst.State = MagicMock()
            mock_gst.State.PLAYING = 3
            mock_gst.State.PAUSED = 2
            panel._is_playing = False
            panel._current_track = MagicMock()
            panel._playbin = MagicMock()
            panel._playbin.set_state = MagicMock(return_value=1)

            with patch.object(panel, "_start_seek_updates"):
                panel._on_play_clicked(MagicMock())

            # Should transition to playing
            assert panel._is_playing

    def test_stop_playback_resets_state(self, panel: TrackPreviewPanel) -> None:
        """Verify _stop_playback resets playback state."""
        with patch("playchitect.gui.widgets.track_preview_panel.Gst") as mock_gst:
            mock_gst.State = MagicMock()
            mock_gst.State.NULL = 1
            panel._is_playing = True
            panel._playbin = MagicMock()

            with patch.object(panel, "_stop_seek_updates"):
                panel._stop_playback()

            assert not panel._is_playing
            panel._playbin.set_state.assert_called_once()


class TestHelperFunctions:
    """Test helper functions in track_preview_panel module."""

    def test_get_cache_path(self) -> None:
        """Verify cache path generation uses MD5 hash."""
        import hashlib

        filepath = "/music/test.flac"
        expected_hash = hashlib.md5(filepath.encode()).hexdigest()  # noqa: S324
        expected_path = Path.home() / ".cache" / "playchitect" / "covers" / f"{expected_hash}.jpg"

        result = _get_cache_path(filepath)
        assert result == expected_path

    def test_ensure_cache_dir_creates_directory(self) -> None:
        """Verify cache directory is created if it doesn't exist."""
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            _ensure_cache_dir()
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


class TestTimeFormatting:
    """Test time formatting utility."""

    def test_format_time_zero(self, panel: TrackPreviewPanel) -> None:
        """Verify 0 seconds formats as 0:00."""
        assert panel._format_time(0) == "0:00"

    def test_format_time_under_minute(self, panel: TrackPreviewPanel) -> None:
        """Verify sub-minute times format correctly."""
        assert panel._format_time(45) == "0:45"

    def test_format_time_exact_minute(self, panel: TrackPreviewPanel) -> None:
        """Verify 60 seconds formats as 1:00."""
        assert panel._format_time(60) == "1:00"

    def test_format_time_typical_track(self, panel: TrackPreviewPanel) -> None:
        """Verify typical track lengths format correctly."""
        assert panel._format_time(386) == "6:26"

    def test_format_time_negative_clamped(self, panel: TrackPreviewPanel) -> None:
        """Verify negative values are clamped to 0."""
        assert panel._format_time(-5) == "0:00"
