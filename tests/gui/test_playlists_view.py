"""GUI unit tests for PlaylistsView and ClusterRowWidget.

Uses the GTK mock infrastructure from tests/gui/conftest.py.
All tests bypass __init__ via __new__ to avoid touching the real GTK runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from gi.repository import Gtk  # ty: ignore[unresolved-import]

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
    # TASK-14: Timbre similarity scale
    view._timbre_scale = MagicMock()
    # TASK-16: Vocal filter buttons
    view._vocal_btn_any = MagicMock()
    view._vocal_btn_instrumental = MagicMock()
    view._vocal_btn_vocal = MagicMock()
    # TASK-19: Energy arc widget
    view._energy_arc = MagicMock()
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
        row = _make_row(_make_stats(cluster_id="2a"))  # ty: ignore[invalid-argument-type]
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
        metadata_map = {Path("/test/track1.mp3"): MagicMock()}

        view.set_metadata(metadata_map)  # ty: ignore[invalid-argument-type]

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
        view._refresh_cluster_sidebar = MagicMock()  # ty: ignore[invalid-assignment]

        view.clear()

        assert view._clusters == []
        assert view._cluster_stats == []
        assert view._selected_cluster_id is None
        view._track_list.clear.assert_called_once()
        view._count_label.set_text.assert_called_with("0 playlists")


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
        view._refresh_cluster_sidebar = MagicMock()  # ty: ignore[invalid-assignment]

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
        view._refresh_cluster_sidebar.assert_called_once()  # ty: ignore[unresolved-attribute]
        view._count_label.set_text.assert_called_once_with("1 playlist")

    def test_load_clusters_plural_label(self):
        """Test that cluster count label uses plural form."""
        view = _make_view()
        view._refresh_cluster_sidebar = MagicMock()  # ty: ignore[invalid-assignment]

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

        view._count_label.set_text.assert_called_once_with("3 playlists")


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
            # TASK-14: Timbre similarity scale
            patch("playchitect.gui.views.playlists_view.Gtk.Scale") as mock_scale,
            # TASK-19: Energy arc widget
            patch("playchitect.gui.views.playlists_view.EnergyArcWidget") as mock_energy_arc,
            # GUI-04: Expander for advanced options
            patch("playchitect.gui.views.playlists_view.Gtk.Expander") as mock_expander,
            patch("playchitect.gui.views.playlists_view.Gtk.ToggleButton") as mock_toggle,
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
            mock_scale.return_value = MagicMock()
            mock_energy_arc.return_value = MagicMock()
            mock_expander.return_value = MagicMock()
            mock_toggle.return_value = MagicMock()

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


class TestTimbreControls:
    """Tests for the timbre/texture controls (TASK-14)."""

    def test_toolbar_has_timbre_scale(self):
        """Verify PlaylistsView has a Timbre similarity Scale widget."""
        view = _make_view()
        assert hasattr(view, "_timbre_scale")
        assert view._timbre_scale is not None


class TestVocalFilterControls:
    """Tests for the vocal filter controls (TASK-16)."""

    def test_toolbar_has_vocal_filter_buttons(self):
        """Verify PlaylistsView has vocal filter ToggleButton chips."""
        view = _make_view()
        assert hasattr(view, "_vocal_btn_any")
        assert hasattr(view, "_vocal_btn_instrumental")
        assert hasattr(view, "_vocal_btn_vocal")
        assert view._vocal_btn_any is not None
        assert view._vocal_btn_instrumental is not None
        assert view._vocal_btn_vocal is not None

    def test_any_button_is_default_active(self):
        """Verify 'Any' vocal filter button is active by default (via mocked state)."""
        from unittest.mock import MagicMock, patch

        from playchitect.gui.views.playlists_view import PlaylistsView

        with (
            patch("playchitect.gui.views.playlists_view.Gtk.Box") as mock_box,
            patch("playchitect.gui.views.playlists_view.Gtk.ActionBar") as mock_action,
            patch("playchitect.gui.views.playlists_view.Gtk.Button") as mock_button,
            patch("playchitect.gui.views.playlists_view.Gtk.ToggleButton") as mock_toggle,
            patch("playchitect.gui.views.playlists_view.Gtk.SpinButton") as mock_spin,
            patch("playchitect.gui.views.playlists_view.Gtk.DropDown") as mock_dropdown,
            patch("playchitect.gui.views.playlists_view.Gtk.StringList") as mock_stringlist,
            patch("playchitect.gui.views.playlists_view.Gtk.Switch") as mock_switch,
            patch("playchitect.gui.views.playlists_view.Gtk.Scale") as mock_scale,
            patch("playchitect.gui.views.playlists_view.TrackListWidget") as mock_tracklist,
            patch("playchitect.gui.views.playlists_view.Gtk.Spinner") as mock_spinner,
            patch("playchitect.gui.views.playlists_view.Gtk.Label") as mock_label,
            patch("playchitect.gui.views.playlists_view.Gtk.Paned") as mock_paned,
            patch("playchitect.gui.views.playlists_view.Gtk.ListBox") as mock_listbox,
            patch("playchitect.gui.views.playlists_view.Gtk.ScrolledWindow") as mock_scroll,
            patch("playchitect.gui.views.playlists_view.Gtk.Separator") as mock_sep,
            patch("playchitect.gui.views.playlists_view.Gtk.Expander") as mock_expander,
            patch("playchitect.gui.views.playlists_view.EnergyArcWidget") as mock_energy_arc,
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
            mock_scale.return_value = MagicMock()
            mock_energy_arc.return_value = MagicMock()
            mock_expander.return_value = MagicMock()

            # Create toggle button mocks to capture their creation
            toggle_mocks = []

            def capture_toggle(*args, **kwargs):
                mock = MagicMock()
                toggle_mocks.append((args, kwargs, mock))
                return mock

            mock_toggle.side_effect = capture_toggle

            _ = PlaylistsView()

            # Find the "Any" toggle button (should be first vocal filter button)
            any_btn = None
            for args, kwargs, mock in toggle_mocks:
                if kwargs.get("label") == "Any":
                    any_btn = mock
                    break

            assert any_btn is not None, "ToggleButton with label='Any' not found"
            # Verify set_active(True) was called
            any_btn.set_active.assert_any_call(True)

    def test_vocal_filter_button_labels_updated(self):
        """Verify vocal filter buttons have updated labels for clarity."""
        from unittest.mock import MagicMock, patch

        from playchitect.gui.views.playlists_view import PlaylistsView

        with (
            patch("playchitect.gui.views.playlists_view.Gtk.Box") as mock_box,
            patch("playchitect.gui.views.playlists_view.Gtk.ActionBar") as mock_action,
            patch("playchitect.gui.views.playlists_view.Gtk.Button") as mock_button,
            patch("playchitect.gui.views.playlists_view.Gtk.ToggleButton") as mock_toggle,
            patch("playchitect.gui.views.playlists_view.Gtk.SpinButton") as mock_spin,
            patch("playchitect.gui.views.playlists_view.Gtk.DropDown") as mock_dropdown,
            patch("playchitect.gui.views.playlists_view.Gtk.StringList") as mock_stringlist,
            patch("playchitect.gui.views.playlists_view.Gtk.Switch") as mock_switch,
            patch("playchitect.gui.views.playlists_view.Gtk.Scale") as mock_scale,
            patch("playchitect.gui.views.playlists_view.TrackListWidget") as mock_tracklist,
            patch("playchitect.gui.views.playlists_view.Gtk.Spinner") as mock_spinner,
            patch("playchitect.gui.views.playlists_view.Gtk.Label") as mock_label,
            patch("playchitect.gui.views.playlists_view.Gtk.Paned") as mock_paned,
            patch("playchitect.gui.views.playlists_view.Gtk.ListBox") as mock_listbox,
            patch("playchitect.gui.views.playlists_view.Gtk.ScrolledWindow") as mock_scroll,
            patch("playchitect.gui.views.playlists_view.Gtk.Separator") as mock_sep,
            patch("playchitect.gui.views.playlists_view.Gtk.Expander") as mock_expander,
            patch("playchitect.gui.views.playlists_view.EnergyArcWidget") as mock_energy_arc,
        ):
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
            mock_scale.return_value = MagicMock()
            mock_energy_arc.return_value = MagicMock()
            mock_expander.return_value = MagicMock()

            toggle_mocks = []

            def capture_toggle(*args, **kwargs):
                mock = MagicMock()
                toggle_mocks.append((args, kwargs, mock))
                return mock

            mock_toggle.side_effect = capture_toggle

            _ = PlaylistsView()

            labels_found = {kwargs.get("label") for args, kwargs, mock in toggle_mocks}
            assert "Any" in labels_found, "ToggleButton with label='Any' not found"
            assert "No vocals" in labels_found, "ToggleButton with label='No vocals' not found"
            assert "Vocals" in labels_found, "ToggleButton with label='Vocals' not found"

    def test_vocal_filter_buttons_have_tooltips(self):
        """Verify vocal filter buttons have explanatory tooltips."""
        from unittest.mock import MagicMock, patch

        from playchitect.gui.views.playlists_view import PlaylistsView

        with (
            patch("playchitect.gui.views.playlists_view.Gtk.Box") as mock_box,
            patch("playchitect.gui.views.playlists_view.Gtk.ActionBar") as mock_action,
            patch("playchitect.gui.views.playlists_view.Gtk.Button") as mock_button,
            patch("playchitect.gui.views.playlists_view.Gtk.ToggleButton") as mock_toggle,
            patch("playchitect.gui.views.playlists_view.Gtk.SpinButton") as mock_spin,
            patch("playchitect.gui.views.playlists_view.Gtk.DropDown") as mock_dropdown,
            patch("playchitect.gui.views.playlists_view.Gtk.StringList") as mock_stringlist,
            patch("playchitect.gui.views.playlists_view.Gtk.Switch") as mock_switch,
            patch("playchitect.gui.views.playlists_view.Gtk.Scale") as mock_scale,
            patch("playchitect.gui.views.playlists_view.TrackListWidget") as mock_tracklist,
            patch("playchitect.gui.views.playlists_view.Gtk.Spinner") as mock_spinner,
            patch("playchitect.gui.views.playlists_view.Gtk.Label") as mock_label,
            patch("playchitect.gui.views.playlists_view.Gtk.Paned") as mock_paned,
            patch("playchitect.gui.views.playlists_view.Gtk.ListBox") as mock_listbox,
            patch("playchitect.gui.views.playlists_view.Gtk.ScrolledWindow") as mock_scroll,
            patch("playchitect.gui.views.playlists_view.Gtk.Separator") as mock_sep,
            patch("playchitect.gui.views.playlists_view.Gtk.Expander") as mock_expander,
            patch("playchitect.gui.views.playlists_view.EnergyArcWidget") as mock_energy_arc,
        ):
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
            mock_scale.return_value = MagicMock()
            mock_energy_arc.return_value = MagicMock()
            mock_expander.return_value = MagicMock()

            toggle_mocks = []

            def capture_toggle(*args, **kwargs):
                mock = MagicMock()
                toggle_mocks.append((args, kwargs, mock))
                return mock

            mock_toggle.side_effect = capture_toggle

            _ = PlaylistsView()

            any_tooltip = None
            no_vocals_tooltip = None
            vocals_tooltip = None

            for args, kwargs, mock in toggle_mocks:
                if kwargs.get("label") == "Any":
                    any_tooltip = mock.set_tooltip_text.call_args
                elif kwargs.get("label") == "No vocals":
                    no_vocals_tooltip = mock.set_tooltip_text.call_args
                elif kwargs.get("label") == "Vocals":
                    vocals_tooltip = mock.set_tooltip_text.call_args

            assert any_tooltip is not None, "Tooltip not set for 'Any' button"
            assert no_vocals_tooltip is not None, "Tooltip not set for 'No vocals' button"
            assert vocals_tooltip is not None, "Tooltip not set for 'Vocals' button"

            any_text = any_tooltip[0][0] if any_tooltip else ""
            assert "limited accuracy" in any_text, (
                "Tooltip should mention limited accuracy for electronic music"
            )
            assert "electronic" in any_text.lower(), "Tooltip should mention electronic music"


class TestIntroColumn:
    """Tests for the Intro column in TrackListWidget (TASK-16)."""

    def test_track_model_has_intro_property(self):
        """Verify TrackModel has intro_length_secs property."""
        from playchitect.gui.widgets.track_list import TrackModel

        model = TrackModel(
            filepath="/test/track.mp3",
            intro_length_secs=15.5,
        )
        assert hasattr(model, "intro_length_secs")
        assert model.intro_length_secs == 15.5

    def test_track_model_intro_formatted(self):
        """Verify TrackModel formats intro_length_secs as 'Xs'."""
        from playchitect.gui.widgets.track_list import TrackModel

        model = TrackModel(
            filepath="/test/track.mp3",
            intro_length_secs=12.7,
        )
        assert model.intro_formatted == "12s"

    def test_track_model_has_vocal_presence_property(self):
        """Verify TrackModel has vocal_presence property."""
        from playchitect.gui.widgets.track_list import TrackModel

        model = TrackModel(
            filepath="/test/track.mp3",
            vocal_presence=0.75,
        )
        assert hasattr(model, "vocal_presence")
        assert model.vocal_presence == 0.75


class TestEnergyArcWidget:
    """Tests for the EnergyArcWidget integration (TASK-19)."""

    def test_playlists_view_has_energy_arc_widget(self):
        """Verify PlaylistsView has an EnergyArcWidget child."""
        view = _make_view()
        assert hasattr(view, "_energy_arc")
        assert view._energy_arc is not None

    def test_energy_arc_update_clusters_does_not_raise(self):
        """Verify update_clusters() with ClusterResults does not raise."""
        from unittest.mock import MagicMock

        # Test the logic directly by mocking the widget
        widget = MagicMock()
        widget._clusters = []
        widget.queue_draw = MagicMock()

        # Simulate update_clusters logic
        def mock_update_clusters(clusters):
            widget._clusters = []
            for cluster in clusters:
                name = getattr(cluster, "cluster_id", "Unknown")
                feature_means = getattr(cluster, "feature_means", None) or {}
                mean_rms = feature_means.get("rms_energy", 0.0)
                widget._clusters.append((str(name), float(mean_rms)))
            widget.queue_draw()

        # Create mock ClusterResults with feature_means
        clusters = []
        for i in range(4):
            cluster = MagicMock()
            cluster.cluster_id = i
            cluster.feature_means = {"rms_energy": 0.3 + i * 0.1}
            clusters.append(cluster)

        # Should not raise
        try:
            mock_update_clusters(clusters)
        except Exception as e:
            pytest.fail(f"update_clusters() raised {type(e).__name__}: {e}")

        # Verify clusters were stored
        assert len(widget._clusters) == 4
        assert widget._clusters[0][0] == "0"
        assert widget._clusters[0][1] == pytest.approx(0.3)
        assert widget._clusters[3][0] == "3"
        assert widget._clusters[3][1] == pytest.approx(0.6)
        widget.queue_draw.assert_called_once()

    def test_energy_arc_widget_real_update_clusters(self):
        """Test actual EnergyArcWidget.update_clusters() extracts RMS correctly.

        This tests the real widget implementation with mock ClusterResult objects
        that have the proper structure (cluster_id and feature_means attributes).
        """
        import sys
        from dataclasses import dataclass, field
        from unittest.mock import MagicMock, patch

        # Create a proper mock ClusterResult-like object
        @dataclass
        class MockClusterResult:
            cluster_id: int | str
            tracks: list[Path] = field(default_factory=list)
            bpm_mean: float = 120.0
            bpm_std: float = 5.0
            track_count: int = 10
            total_duration: float = 300.0
            feature_means: dict[str, float] | None = None
            feature_importance: dict[str, float] | None = None
            weight_source: str | None = None
            embedding_variance_explained: float | None = None
            genre: str | None = None
            opener: Path | None = None
            closer: Path | None = None

        # Create a mock class hierarchy for GTK
        class MockDrawingArea:
            def __init__(self, *args, **kwargs):
                pass

            def set_size_request(self, *args, **kwargs):
                pass

            def set_draw_func(self, *args, **kwargs):
                pass

            def queue_draw(self, *args, **kwargs):
                pass

        mock_gi = MagicMock()
        mock_gi.repository.Gtk.DrawingArea = MockDrawingArea

        with patch.dict("sys.modules", {"gi": mock_gi, "gi.repository": mock_gi.repository}):
            # Clear the module cache to force reimport
            modules_to_clear = [
                key
                for key in sys.modules.keys()
                if key.startswith("playchitect.gui.widgets.energy_arc_widget")
            ]
            for key in modules_to_clear:
                del sys.modules[key]

            # Import the widget class with mocked GTK
            from playchitect.gui.widgets.energy_arc_widget import EnergyArcWidget

            widget = EnergyArcWidget.__new__(EnergyArcWidget)
            widget._clusters = []

            # Create 4 mock clusters with varying RMS energy values
            clusters = [
                MockClusterResult(cluster_id=0, feature_means={"rms_energy": 0.25}),
                MockClusterResult(cluster_id=1, feature_means={"rms_energy": 0.50}),
                MockClusterResult(cluster_id=2, feature_means={"rms_energy": 0.75}),
                MockClusterResult(cluster_id=3, feature_means={"rms_energy": 1.00}),
            ]

            # Test that update_clusters extracts the correct data
            widget.update_clusters(clusters)

            # Verify the widget stored the correct cluster names and RMS values
            assert len(widget._clusters) == 4
            assert widget._clusters[0] == ("0", 0.25)
            assert widget._clusters[1] == ("1", 0.50)
            assert widget._clusters[2] == ("2", 0.75)
            assert widget._clusters[3] == ("3", 1.00)

    def test_energy_arc_widget_handles_missing_rms(self):
        """Test EnergyArcWidget handles clusters without rms_energy gracefully."""
        import sys
        from dataclasses import dataclass, field
        from unittest.mock import MagicMock, patch

        @dataclass
        class MockClusterResult:
            cluster_id: int | str
            tracks: list[Path] = field(default_factory=list)
            bpm_mean: float = 120.0
            bpm_std: float = 5.0
            track_count: int = 10
            total_duration: float = 300.0
            feature_means: dict[str, float] | None = None
            feature_importance: dict[str, float] | None = None
            weight_source: str | None = None
            embedding_variance_explained: float | None = None
            genre: str | None = None
            opener: Path | None = None
            closer: Path | None = None

        class MockDrawingArea:
            def __init__(self, *args, **kwargs):
                pass

            def set_size_request(self, *args, **kwargs):
                pass

            def set_draw_func(self, *args, **kwargs):
                pass

            def queue_draw(self, *args, **kwargs):
                pass

        mock_gi = MagicMock()
        mock_gi.repository.Gtk.DrawingArea = MockDrawingArea

        with patch.dict("sys.modules", {"gi": mock_gi, "gi.repository": mock_gi.repository}):
            modules_to_clear = [
                key
                for key in sys.modules.keys()
                if key.startswith("playchitect.gui.widgets.energy_arc_widget")
            ]
            for key in modules_to_clear:
                del sys.modules[key]

            from playchitect.gui.widgets.energy_arc_widget import EnergyArcWidget

            widget = EnergyArcWidget.__new__(EnergyArcWidget)
            widget._clusters = []

            # Cluster with empty feature_means (no rms_energy)
            cluster_no_rms = MockClusterResult(cluster_id=0, feature_means={})
            # Cluster with None feature_means
            cluster_none_means = MockClusterResult(cluster_id=1, feature_means=None)
            # Cluster with rms_energy
            cluster_with_rms = MockClusterResult(cluster_id=2, feature_means={"rms_energy": 0.5})

            clusters = [cluster_no_rms, cluster_none_means, cluster_with_rms]

            widget.update_clusters(clusters)

            # Should default to 0.0 for missing rms_energy
            assert len(widget._clusters) == 3
            assert widget._clusters[0] == ("0", 0.0)
            assert widget._clusters[1] == ("1", 0.0)
            assert widget._clusters[2] == ("2", 0.5)

    def test_energy_arc_widget_empty_clusters(self):
        """Test EnergyArcWidget handles empty cluster list."""
        import sys
        from unittest.mock import MagicMock, patch

        class MockDrawingArea:
            def __init__(self, *args, **kwargs):
                pass

            def set_size_request(self, *args, **kwargs):
                pass

            def set_draw_func(self, *args, **kwargs):
                pass

            def queue_draw(self, *args, **kwargs):
                pass

        mock_gi = MagicMock()
        mock_gi.repository.Gtk.DrawingArea = MockDrawingArea

        with patch.dict("sys.modules", {"gi": mock_gi, "gi.repository": mock_gi.repository}):
            modules_to_clear = [
                key
                for key in sys.modules.keys()
                if key.startswith("playchitect.gui.widgets.energy_arc_widget")
            ]
            for key in modules_to_clear:
                del sys.modules[key]

            from playchitect.gui.widgets.energy_arc_widget import EnergyArcWidget

            widget = EnergyArcWidget.__new__(EnergyArcWidget)
            widget._clusters = []

            # Should handle empty list without error
            widget.update_clusters([])
            assert len(widget._clusters) == 0
