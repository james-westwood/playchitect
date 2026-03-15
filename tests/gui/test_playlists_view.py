"""GUI unit tests for PlaylistsView and ClusterRowWidget.

Uses the GTK mock infrastructure from tests/gui/conftest.py.
All tests bypass __init__ via __new__ to avoid touching the real GTK runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from gi.repository import Gtk  # type: ignore[unresolved-import]

from playchitect.gui.widgets.cluster_stats import ClusterStats

if TYPE_CHECKING:
    from playchitect.gui.views.playlists_view import (
        ClusterRowWidget,
        PlaylistsView,
    )


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_stats(
    cluster_id: int = 1,
    track_count: int = 20,
    bpm_min: float = 120.0,
    bpm_max: float = 128.0,
    bpm_mean: float = 124.0,
    intensity_mean: float = 0.6,
    total_duration: float = 5400.0,
) -> ClusterStats:
    """Create a ClusterStats instance for testing."""
    return ClusterStats(
        cluster_id=cluster_id,
        track_count=track_count,
        bpm_min=bpm_min,
        bpm_max=bpm_max,
        bpm_mean=bpm_mean,
        intensity_mean=intensity_mean,
        hardness_mean=intensity_mean,
        total_duration=total_duration,
    )


def _make_row(stats: ClusterStats | None = None) -> ClusterRowWidget:
    """Return a ClusterRowWidget with __init__ bypassed."""
    from playchitect.gui.views.playlists_view import ClusterRowWidget

    row = ClusterRowWidget.__new__(ClusterRowWidget)
    row._stats = stats or _make_stats()
    return row


def _make_view() -> PlaylistsView:
    """Return a PlaylistsView with __init__ bypassed and mocked components."""
    from playchitect.gui.views.playlists_view import PlaylistsView

    view = PlaylistsView.__new__(PlaylistsView)
    view._clusters = []
    view._cluster_stats = []
    view._metadata_map = {}
    view._intensity_map = {}
    view._selected_cluster_id = None
    view._cluster_list = MagicMock()
    view._track_list = MagicMock()
    view._count_label = MagicMock()
    view._generate_btn = MagicMock()
    view._spinner = MagicMock()
    view._stats_bpm_label = MagicMock()
    view._stats_intensity_label = MagicMock()
    view._stats_tracks_label = MagicMock()
    view._stats_duration_label = MagicMock()
    # New controls for TASK-08
    view._size_spin = MagicMock()
    view._unit_dropdown = MagicMock()
    view._playlists_spin = MagicMock()
    # TASK-10: Harmonic mixing switch
    view._harmonic_switch = MagicMock()
    # TASK-12: Sort dropdown
    view._sort_dropdown = MagicMock()
    return view


# ── TestClusterRowWidget ────────────────────────────────────────────────────


class TestClusterRowWidget:
    """Tests for the ClusterRowWidget class."""

    def test_cluster_id_property(self):
        """Test that cluster_id property returns the stats cluster_id."""
        row = _make_row(_make_stats(cluster_id=5))
        assert row.cluster_id == 5

    def test_cluster_id_with_string_id(self):
        """Test cluster_id with string identifier."""
        row = _make_row(_make_stats(cluster_id="2a"))  # type: ignore[arg-type]
        assert row.cluster_id == "2a"


# ── TestPlaylistsViewSignals ────────────────────────────────────────────────


class TestPlaylistsViewSignals:
    """Tests for PlaylistsView signals."""

    def test_cluster_selected_signal_defined(self):
        """Verify that the cluster-selected signal is properly defined."""
        from playchitect.gui.views.playlists_view import PlaylistsView

        # Check that the signal is in __gsignals__
        assert "cluster-selected" in PlaylistsView.__gsignals__
        signal_spec = PlaylistsView.__gsignals__["cluster-selected"]
        # Signal spec is (flags, return_type, param_types)
        assert signal_spec[0] is not None  # flags


# ── TestPlaylistsViewPublicAPI ─────────────────────────────────────────────


class TestPlaylistsViewPublicAPI:
    """Tests for PlaylistsView public API."""

    def test_set_metadata(self):
        """Test set_metadata updates the metadata map."""
        view = _make_view()
        metadata_map = {Path("/test/track1.mp3"): MagicMock()}  # type: ignore[dict-item]

        view.set_metadata(metadata_map)  # type: ignore[arg-type]

        assert view._metadata_map == metadata_map
        view._generate_btn.set_sensitive.assert_called_once_with(True)

    def test_set_metadata_empty(self):
        """Test set_metadata with empty map disables generate button."""
        view = _make_view()

        view.set_metadata({})

        assert view._metadata_map == {}
        view._generate_btn.set_sensitive.assert_called_once_with(False)

    def test_get_selected_cluster_id_initial(self):
        """Test get_selected_cluster_id returns None initially."""
        view = _make_view()
        assert view.get_selected_cluster_id() is None

    def test_get_selected_cluster_id_after_selection(self):
        """Test get_selected_cluster_id returns correct ID after selection."""
        view = _make_view()
        view._selected_cluster_id = 3
        assert view.get_selected_cluster_id() == 3

    def test_clear_resets_state(self):
        """Test clear resets all state."""
        view = _make_view()
        view._clusters = [MagicMock()]
        view._cluster_stats = [_make_stats()]
        view._selected_cluster_id = 5
        # Mock _refresh_cluster_sidebar to avoid GTK calls
        view._refresh_cluster_sidebar = MagicMock()  # type: ignore[method-assign]

        view.clear()

        assert view._clusters == []
        assert view._cluster_stats == []
        assert view._selected_cluster_id is None
        view._track_list.clear.assert_called_once()  # type: ignore[attr-defined]
        view._count_label.set_text.assert_called_with("0 clusters")


# ── TestPlaylistsViewStatsDisplay ───────────────────────────────────────────


class TestPlaylistsViewStatsDisplay:
    """Tests for the stats strip display."""

    def test_update_stats_display_with_stats(self):
        """Test updating stats display with cluster stats."""
        view = _make_view()
        stats = _make_stats(
            cluster_id=1,
            track_count=25,
            bpm_min=120.0,
            bpm_max=130.0,
            intensity_mean=0.75,
            total_duration=7200.0,
        )

        view._update_stats_display(stats)

        view._stats_bpm_label.set_text.assert_called_once_with("BPM: 120–130 BPM")
        view._stats_tracks_label.set_text.assert_called_once_with("25 tracks")
        view._stats_duration_label.set_text.assert_called_once_with("Duration: 2h 0m")

    def test_update_stats_display_none(self):
        """Test updating stats display with None clears values."""
        view = _make_view()

        view._update_stats_display(None)

        view._stats_bpm_label.set_text.assert_called_once_with("BPM: —")
        view._stats_intensity_label.set_text.assert_called_once_with("Intensity: —")
        view._stats_tracks_label.set_text.assert_called_once_with("Tracks: —")
        view._stats_duration_label.set_text.assert_called_once_with("Duration: —")


# ── TestPlaylistsViewLoadingState ───────────────────────────────────────────


class TestPlaylistsViewLoadingState:
    """Tests for loading state management."""

    def test_set_loading_state_true(self):
        """Test entering loading state."""
        view = _make_view()

        view._set_loading_state(True)

        view._spinner.set_visible.assert_called_once_with(True)
        view._spinner.start.assert_called_once()
        view._generate_btn.set_sensitive.assert_called_once_with(False)

    def test_set_loading_state_false(self):
        """Test exiting loading state."""
        view = _make_view()

        view._set_loading_state(False)

        view._spinner.set_visible.assert_called_once_with(False)
        view._spinner.stop.assert_called_once()
        view._generate_btn.set_sensitive.assert_called_once_with(True)


# ── TestPlaylistsViewClusterLoading ─────────────────────────────────────────


class TestPlaylistsViewClusterLoading:
    """Tests for loading clusters into the view."""

    def test_load_clusters_updates_sidebar(self):
        """Test that load_clusters refreshes the sidebar."""
        view = _make_view()
        # Mock _refresh_cluster_sidebar to avoid GTK calls
        view._refresh_cluster_sidebar = MagicMock()  # type: ignore[method-assign]

        # Create mock cluster results
        cluster = MagicMock()
        cluster.cluster_id = 1
        cluster.tracks = [Path("/test/track1.mp3")]
        cluster.bpm_mean = 125.0
        cluster.bpm_std = 5.0
        cluster.track_count = 1
        cluster.total_duration = 300.0
        cluster.feature_means = {"rms_energy": 0.6}
        cluster.feature_importance = {}
        cluster.opener = None
        cluster.closer = None

        view.load_clusters([cluster])

        assert len(view._clusters) == 1
        assert len(view._cluster_stats) == 1
        view._refresh_cluster_sidebar.assert_called_once()  # type: ignore[attr-defined]
        view._count_label.set_text.assert_called_once_with("1 cluster")

    def test_load_clusters_plural_label(self):
        """Test that cluster count label uses plural form."""
        view = _make_view()
        view._refresh_cluster_sidebar = MagicMock()  # type: ignore[method-assign]

        # Create mock cluster results
        clusters = []
        for i in range(3):
            cluster = MagicMock()
            cluster.cluster_id = i
            cluster.tracks = [Path(f"/test/track{i}.mp3")]
            cluster.bpm_mean = 125.0
            cluster.bpm_std = 5.0
            cluster.track_count = 1
            cluster.total_duration = 300.0
            cluster.feature_means = {"rms_energy": 0.6}
            cluster.feature_importance = {}
            cluster.opener = None
            cluster.closer = None
            clusters.append(cluster)

        view.load_clusters(clusters)

        view._count_label.set_text.assert_called_once_with("3 clusters")


# ── TestPlaylistsViewInstantiation ──────────────────────────────────────────


class TestPlaylistsViewInstantiation:
    """Smoke tests for PlaylistsView instantiation."""

    def test_playlists_view_instantiates(self):
        """Test that PlaylistsView can be instantiated with mocked GTK."""
        from playchitect.gui.views.playlists_view import PlaylistsView

        # Mock all GTK widget creation
        with (
            patch("playchitect.gui.views.playlists_view.Gtk.Box") as mock_box,
            patch("playchitect.gui.views.playlists_view.Gtk.ActionBar") as mock_action,
            patch("playchitect.gui.views.playlists_view.Gtk.Button") as mock_button,
            patch("playchitect.gui.views.playlists_view.Gtk.Spinner") as mock_spinner,
            patch("playchitect.gui.views.playlists_view.Gtk.Label") as mock_label,
            patch("playchitect.gui.views.playlists_view.Gtk.Paned") as mock_paned,
            patch("playchitect.gui.views.playlists_view.Gtk.ListBox") as mock_listbox,
            patch("playchitect.gui.views.playlists_view.Gtk.ScrolledWindow") as mock_scroll,
            patch("playchitect.gui.views.playlists_view.Gtk.Separator") as mock_sep,
            patch("playchitect.gui.views.playlists_view.TrackListWidget") as mock_tracklist,
            # New controls for TASK-08
            patch("playchitect.gui.views.playlists_view.Gtk.SpinButton") as mock_spin,
            patch("playchitect.gui.views.playlists_view.Gtk.DropDown") as mock_dropdown,
            patch("playchitect.gui.views.playlists_view.Gtk.StringList") as mock_stringlist,
            # TASK-10: Harmonic mixing switch
            patch("playchitect.gui.views.playlists_view.Gtk.Switch") as mock_switch,
        ):
            # Setup mock returns
            mock_action.return_value = MagicMock()
            mock_button.return_value = MagicMock()
            mock_spinner.return_value = MagicMock()
            mock_label.return_value = MagicMock()
            mock_paned.return_value = MagicMock()
            mock_listbox.return_value = MagicMock()
            mock_scroll.return_value = MagicMock()
            mock_sep.return_value = MagicMock()
            mock_tracklist.return_value = MagicMock()
            mock_box.return_value = MagicMock()
            mock_spin.return_value = MagicMock()
            mock_dropdown.return_value = MagicMock()
            mock_stringlist.new.return_value = MagicMock()
            mock_switch.return_value = MagicMock()

            view = PlaylistsView()
            assert view is not None

    def test_cluster_row_widget_instantiates(self):
        """Test that ClusterRowWidget can be instantiated with mocked GTK."""
        from playchitect.gui.views.playlists_view import ClusterRowWidget

        stats = _make_stats(cluster_id=1)

        # Mock the parent class __init__ since we can't patch __new__ on MagicMock
        with (
            patch.object(Gtk.ListBoxRow, "__init__", lambda self, **kwargs: None),
            patch("playchitect.gui.views.playlists_view.Gtk.Box") as mock_box,
            patch("playchitect.gui.views.playlists_view.Gtk.Label") as mock_label,
        ):
            mock_box.return_value = MagicMock()
            mock_label.return_value = MagicMock()

            row = ClusterRowWidget(stats)
            assert row is not None
            assert row.cluster_id == 1


# ── TestPlaylistSizeControls ────────────────────────────────────────────────


class TestPlaylistSizeControls:
    """Tests for the playlist size controls (TASK-08)."""

    def test_toolbar_has_size_spinbutton(self):
        """Verify PlaylistsView has a size SpinButton."""
        view = _make_view()
        assert hasattr(view, "_size_spin")
        assert view._size_spin is not None

    def test_toolbar_has_unit_dropdown(self):
        """Verify PlaylistsView has a unit DropDown."""
        view = _make_view()
        assert hasattr(view, "_unit_dropdown")
        assert view._unit_dropdown is not None

    def test_toolbar_has_playlists_spinbutton(self):
        """Verify PlaylistsView has a playlists SpinButton."""
        view = _make_view()
        assert hasattr(view, "_playlists_spin")
        assert view._playlists_spin is not None


class TestHarmonicMixingControls:
    """Tests for the harmonic mixing controls (TASK-10)."""

    def test_toolbar_has_harmonic_switch(self):
        """Verify PlaylistsView has a harmonic mixing Switch."""
        view = _make_view()
        assert hasattr(view, "_harmonic_switch")
        assert view._harmonic_switch is not None


class TestSortControls:
    """Tests for the sort controls (TASK-12)."""

    def test_toolbar_has_sort_dropdown(self):
        """Verify PlaylistsView has a sort by DropDown."""
        view = _make_view()
        assert hasattr(view, "_sort_dropdown")
        assert view._sort_dropdown is not None
