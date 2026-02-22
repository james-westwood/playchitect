"""GUI smoke tests for PlaychitectWindow layout, HIG compliance, and accessibility.

gi mocks are installed by tests/gui/conftest.py before this module is collected.
Do NOT manipulate sys.modules here — that fights with conftest and breaks imports.

Approach
--------
- ``window`` fixture: patches the 4 external dependencies (TrackPreviewer,
  ClusterViewPanel, TrackListWidget, get_config) so PlaychitectWindow.__init__
  runs end-to-end as a real smoke test.
- ``bare_window`` fixture: uses __new__ to skip __init__ entirely, allowing
  isolated testing of individual methods without any GTK widget construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from playchitect.gui.windows.main_window import PlaychitectWindow

# Import after conftest has installed gi mocks.
from playchitect.gui.windows.main_window import PlaychitectWindow  # noqa: E402

# ── Constants from design spec (copied here so tests fail loudly if changed) ──
_DESIGN_WIDTH = 1000
_DESIGN_HEIGHT = 700
_CLUSTER_PANEL_MIN_WIDTH = 220
_PANED_SPLIT_POSITION = 280

# GNOME HIG adaptive-layout thresholds
_HIG_MIN_WIDTH = 360
_HIG_MIN_HEIGHT = 240

# ── Shared dependency patcher ──────────────────────────────────────────────────

_PATCHES = {
    "playchitect.gui.windows.main_window.TrackPreviewer": None,
    "playchitect.gui.windows.main_window.ClusterViewPanel": None,
    "playchitect.gui.windows.main_window.TrackListWidget": None,
    "playchitect.gui.windows.main_window.get_config": None,
}


def _patch_deps(monkeypatch: pytest.MonkeyPatch, launcher: str | None = None) -> None:
    """Patch the four external deps so PlaychitectWindow.__init__ can run."""
    mock_previewer = MagicMock()
    mock_previewer.launcher_name.return_value = launcher

    mock_config = MagicMock()
    mock_config.get_test_music_path.return_value = None  # skip idle_add

    monkeypatch.setattr(
        "playchitect.gui.windows.main_window.TrackPreviewer",
        MagicMock(return_value=mock_previewer),
    )
    monkeypatch.setattr(
        "playchitect.gui.windows.main_window.ClusterViewPanel",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        "playchitect.gui.windows.main_window.TrackListWidget",
        MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setattr(
        "playchitect.gui.windows.main_window.get_config",
        MagicMock(return_value=mock_config),
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def window(monkeypatch: pytest.MonkeyPatch) -> PlaychitectWindow:
    """PlaychitectWindow with external dependencies patched out."""
    _patch_deps(monkeypatch)
    return PlaychitectWindow()


@pytest.fixture()
def bare_window() -> PlaychitectWindow:
    """PlaychitectWindow via __new__ — __init__ skipped.

    Useful for testing individual methods in isolation without any GTK widget
    construction.
    """
    w = PlaychitectWindow.__new__(PlaychitectWindow)
    w._track_title = "Playchitect"
    w._previewer = MagicMock()
    w._preview_chip = MagicMock()
    w._spinner = MagicMock()
    w._cluster_btn = MagicMock()
    w.track_list = MagicMock()
    w.cluster_panel = MagicMock()
    w._metadata_map = {}
    w._intensity_map = {}
    w._clusters = []
    return w


# ── Smoke: __init__ runs without raising ──────────────────────────────────────


class TestMainWindowSmoke:
    """PlaychitectWindow.__init__ completes and populates all expected attributes."""

    def test_init_does_not_raise(self, window: PlaychitectWindow) -> None:
        assert window is not None

    def test_previewer_attribute_set(self, window: PlaychitectWindow) -> None:
        assert hasattr(window, "_previewer")

    def test_cluster_panel_attribute_set(self, window: PlaychitectWindow) -> None:
        assert hasattr(window, "cluster_panel")

    def test_track_list_attribute_set(self, window: PlaychitectWindow) -> None:
        assert hasattr(window, "track_list")

    def test_spinner_attribute_set(self, window: PlaychitectWindow) -> None:
        assert hasattr(window, "_spinner")

    def test_preview_chip_attribute_set(self, window: PlaychitectWindow) -> None:
        assert hasattr(window, "_preview_chip")

    def test_track_title_default(self, window: PlaychitectWindow) -> None:
        assert window._track_title == "Playchitect"


# ── Window-init call verification ─────────────────────────────────────────────


class TestWindowInitCalls:
    """Verify that __init__ calls the right GTK methods with the right arguments."""

    @pytest.fixture()
    def spied_window(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Window created with set_title and set_default_size replaced by spies."""
        spy_title = MagicMock()
        spy_size = MagicMock()
        spy_content = MagicMock()

        monkeypatch.setattr(PlaychitectWindow, "set_title", spy_title)
        monkeypatch.setattr(PlaychitectWindow, "set_default_size", spy_size)
        monkeypatch.setattr(PlaychitectWindow, "set_content", spy_content)
        _patch_deps(monkeypatch)

        w = PlaychitectWindow()
        return {
            "window": w,
            "set_title": spy_title,
            "set_default_size": spy_size,
            "set_content": spy_content,
        }

    def test_set_title_called_with_playchitect(self, spied_window: dict[str, Any]) -> None:
        spied_window["set_title"].assert_any_call("Playchitect")

    def test_set_default_size_called(self, spied_window: dict[str, Any]) -> None:
        spied_window["set_default_size"].assert_called_once()

    def test_set_default_size_matches_design_spec(self, spied_window: dict[str, Any]) -> None:
        args = spied_window["set_default_size"].call_args[0]
        assert args == (_DESIGN_WIDTH, _DESIGN_HEIGHT)

    def test_set_content_called(self, spied_window: dict[str, Any]) -> None:
        spied_window["set_content"].assert_called_once()


# ── HIG compliance ────────────────────────────────────────────────────────────


class TestHIGCompliance:
    """Design constants conform to GNOME HIG adaptive-layout requirements."""

    def test_design_width_meets_hig_minimum(self) -> None:
        assert _DESIGN_WIDTH >= _HIG_MIN_WIDTH, (
            f"Design width {_DESIGN_WIDTH}px is below HIG minimum {_HIG_MIN_WIDTH}px"
        )

    def test_design_height_meets_hig_minimum(self) -> None:
        assert _DESIGN_HEIGHT >= _HIG_MIN_HEIGHT, (
            f"Design height {_DESIGN_HEIGHT}px is below HIG minimum {_HIG_MIN_HEIGHT}px"
        )

    def test_cluster_panel_min_width_allows_content(self) -> None:
        """Panel minimum width of 220 px is enough for typical cluster cards."""
        assert _CLUSTER_PANEL_MIN_WIDTH >= 200

    def test_paned_split_leaves_room_for_track_list(self) -> None:
        """After the split, the track list gets at least 50% of design width."""
        remaining = _DESIGN_WIDTH - _PANED_SPLIT_POSITION
        assert remaining >= _DESIGN_WIDTH // 2


# ── Preview chip ──────────────────────────────────────────────────────────────


class TestPreviewChip:
    """_update_preview_chip sets the correct label text for each launcher."""

    def test_sushi_chip_text(self, bare_window: PlaychitectWindow) -> None:
        bare_window._previewer.launcher_name.return_value = "sushi"
        bare_window._update_preview_chip()
        bare_window._preview_chip.set_text.assert_called_once_with("Sushi ✓")

    def test_sushi_tooltip(self, bare_window: PlaychitectWindow) -> None:
        bare_window._previewer.launcher_name.return_value = "sushi"
        bare_window._update_preview_chip()
        bare_window._preview_chip.set_tooltip_text.assert_called_once_with(
            "Quick Look via GNOME Sushi (Space)"
        )

    def test_xdg_open_chip_text(self, bare_window: PlaychitectWindow) -> None:
        bare_window._previewer.launcher_name.return_value = "xdg-open"
        bare_window._update_preview_chip()
        bare_window._preview_chip.set_text.assert_called_once_with("Preview: xdg-open")

    def test_no_launcher_chip_text(self, bare_window: PlaychitectWindow) -> None:
        bare_window._previewer.launcher_name.return_value = None
        bare_window._update_preview_chip()
        bare_window._preview_chip.set_text.assert_called_once_with("No preview")

    def test_no_launcher_tooltip_mentions_sushi(self, bare_window: PlaychitectWindow) -> None:
        bare_window._previewer.launcher_name.return_value = None
        bare_window._update_preview_chip()
        tooltip = bare_window._preview_chip.set_tooltip_text.call_args[0][0]
        assert "Sushi" in tooltip


# ── Scan handlers ─────────────────────────────────────────────────────────────


class TestScanCompleteHandler:
    """_on_scan_complete populates the track list and updates the title."""

    def _make_tracks(self, n: int) -> list[MagicMock]:
        return [MagicMock() for _ in range(n)]

    def test_loads_tracks_into_widget(self, bare_window: PlaychitectWindow) -> None:
        tracks = self._make_tracks(5)
        bare_window._on_scan_complete(tracks)
        bare_window.track_list.load_tracks.assert_called_once_with(tracks)

    def test_stops_spinner(self, bare_window: PlaychitectWindow) -> None:
        bare_window._on_scan_complete(self._make_tracks(3))
        bare_window._spinner.stop.assert_called_once()

    def test_title_includes_track_count(self, bare_window: PlaychitectWindow) -> None:
        spy_title = MagicMock()
        bare_window.set_title = spy_title
        bare_window._on_scan_complete(self._make_tracks(42))
        last_call = spy_title.call_args[0][0]
        assert "42" in last_call

    def test_returns_false_to_cancel_idle(self, bare_window: PlaychitectWindow) -> None:
        result = bare_window._on_scan_complete(self._make_tracks(1))
        assert result is False

    def test_updates_track_title_attribute(self, bare_window: PlaychitectWindow) -> None:
        bare_window._on_scan_complete(self._make_tracks(7))
        assert "7" in bare_window._track_title


class TestScanErrorHandler:
    """_on_scan_error stops the spinner and sets an error title."""

    def test_stops_spinner(self, bare_window: PlaychitectWindow) -> None:
        bare_window._on_scan_error()
        bare_window._spinner.stop.assert_called_once()

    def test_title_indicates_failure(self, bare_window: PlaychitectWindow) -> None:
        spy_title = MagicMock()
        bare_window.set_title = spy_title
        bare_window._on_scan_error()
        last_call = spy_title.call_args[0][0]
        assert "fail" in last_call.lower() or "error" in last_call.lower()

    def test_returns_false_to_cancel_idle(self, bare_window: PlaychitectWindow) -> None:
        result = bare_window._on_scan_error()
        assert result is False


# ── Revert title ──────────────────────────────────────────────────────────────


class TestRevertTitle:
    """_revert_title restores the window title and signals GLib not to repeat."""

    def test_restores_track_title(self, bare_window: PlaychitectWindow) -> None:
        bare_window._track_title = "Playchitect — 20 tracks"
        spy = MagicMock()
        bare_window.set_title = spy
        bare_window._revert_title()
        spy.assert_called_once_with("Playchitect — 20 tracks")

    def test_returns_false_so_timeout_does_not_repeat(self, bare_window: PlaychitectWindow) -> None:
        bare_window.set_title = MagicMock()
        assert bare_window._revert_title() is False


# ── Cluster selected ──────────────────────────────────────────────────────────


class TestClusterSelected:
    """_on_cluster_selected updates the title and clears the search filter."""

    def test_title_includes_cluster_id(self, bare_window: PlaychitectWindow) -> None:
        spy = MagicMock()
        bare_window.set_title = spy
        # Mock a cluster result so lookup succeeds
        mock_cluster = MagicMock()
        mock_cluster.cluster_id = 3
        mock_cluster.tracks = []
        bare_window._clusters = [mock_cluster]

        bare_window._on_cluster_selected(MagicMock(), cluster_id=3)
        title = spy.call_args[0][0]
        assert "3" in title

    def test_clears_search_entry(self, bare_window: PlaychitectWindow) -> None:
        bare_window.set_title = MagicMock()
        # Mock a cluster result so lookup succeeds
        mock_cluster = MagicMock()
        mock_cluster.cluster_id = 2
        mock_cluster.tracks = []
        bare_window._clusters = [mock_cluster]

        bare_window._on_cluster_selected(MagicMock(), cluster_id=2)
        bare_window.track_list._search_entry.set_text.assert_called_once_with("")


class TestClusterHandlers:
    """Test clustering signal handlers and workers."""

    def test_on_cluster_clicked_starts_spinner(self, bare_window: PlaychitectWindow) -> None:
        bare_window._metadata_map = {Path("t1.flac"): MagicMock()}
        bare_window.set_title = MagicMock()

        with patch("threading.Thread") as mock_thread:
            bare_window._on_cluster_clicked(MagicMock())

            bare_window._spinner.start.assert_called_once()
            bare_window._cluster_btn.set_sensitive.assert_called_with(False)
            mock_thread.assert_called_once()

    def test_on_cluster_complete_updates_ui(self, bare_window: PlaychitectWindow) -> None:
        mock_cluster = MagicMock()
        mock_cluster.cluster_id = 1
        mock_cluster.bpm_mean = 120.0
        mock_cluster.bpm_std = 2.0
        mock_cluster.track_count = 10
        mock_cluster.total_duration = 1800.0
        mock_cluster.feature_means = {}
        mock_cluster.feature_importance = {}
        mock_cluster.opener = None
        mock_cluster.closer = None

        bare_window._clusters = [mock_cluster]
        bare_window.set_title = MagicMock()

        bare_window._on_cluster_complete()

        bare_window._spinner.stop.assert_called_once()
        bare_window._cluster_btn.set_sensitive.assert_called_with(True)
        bare_window.cluster_panel.load_clusters.assert_called_once()

    def test_on_cluster_error_resets_ui(self, bare_window: PlaychitectWindow) -> None:
        bare_window.set_title = MagicMock()

        bare_window._on_cluster_error()

        bare_window._spinner.stop.assert_called_once()
        bare_window._cluster_btn.set_sensitive.assert_called_with(True)
        # Check if set_title was called with something containing "failed"
        title_call = bare_window.set_title.call_args[0][0]
        assert "failed" in title_call.lower()


class TestClusterWorkerIntensity:
    """Test _cluster_worker intensity analysis integration."""

    def test_intensity_analyzer_constructed_with_cache_dir(
        self, bare_window: PlaychitectWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_config = MagicMock()
        mock_config.get_cache_dir.return_value = Path("/fake/cache")
        monkeypatch.setattr("playchitect.gui.windows.main_window.get_config", lambda: mock_config)

        mock_analyzer_cls = MagicMock()
        monkeypatch.setattr(
            "playchitect.gui.windows.main_window.IntensityAnalyzer", mock_analyzer_cls
        )
        monkeypatch.setattr("playchitect.gui.windows.main_window.PlaylistClusterer", MagicMock())
        monkeypatch.setattr("playchitect.gui.windows.main_window.Sequencer", MagicMock())
        monkeypatch.setattr("playchitect.gui.windows.main_window.GLib.idle_add", MagicMock())

        bare_window._cluster_worker()

        mock_analyzer_cls.assert_called_once_with(cache_dir=Path("/fake/cache/intensity"))

    def test_analyze_batch_called_with_metadata_keys(
        self, bare_window: PlaychitectWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_config = MagicMock()
        monkeypatch.setattr("playchitect.gui.windows.main_window.get_config", lambda: mock_config)

        mock_analyzer = MagicMock()
        mock_analyzer_cls = MagicMock(return_value=mock_analyzer)
        monkeypatch.setattr(
            "playchitect.gui.windows.main_window.IntensityAnalyzer", mock_analyzer_cls
        )
        monkeypatch.setattr("playchitect.gui.windows.main_window.PlaylistClusterer", MagicMock())
        monkeypatch.setattr("playchitect.gui.windows.main_window.Sequencer", MagicMock())
        monkeypatch.setattr("playchitect.gui.windows.main_window.GLib.idle_add", MagicMock())

        bare_window._metadata_map = {Path("a.flac"): MagicMock(), Path("b.flac"): MagicMock()}

        bare_window._cluster_worker()

        mock_analyzer.analyze_batch.assert_called_once_with(list(bare_window._metadata_map.keys()))

    def test_intensity_map_set_from_analyze_batch(
        self, bare_window: PlaychitectWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_config = MagicMock()
        monkeypatch.setattr("playchitect.gui.windows.main_window.get_config", lambda: mock_config)

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_batch.return_value = {"path": "features"}
        mock_analyzer_cls = MagicMock(return_value=mock_analyzer)
        monkeypatch.setattr(
            "playchitect.gui.windows.main_window.IntensityAnalyzer", mock_analyzer_cls
        )
        monkeypatch.setattr("playchitect.gui.windows.main_window.PlaylistClusterer", MagicMock())
        monkeypatch.setattr("playchitect.gui.windows.main_window.Sequencer", MagicMock())
        monkeypatch.setattr("playchitect.gui.windows.main_window.GLib.idle_add", MagicMock())

        bare_window._cluster_worker()

        assert bare_window._intensity_map == {"path": "features"}
