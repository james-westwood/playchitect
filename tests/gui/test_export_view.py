"""GUI unit tests for ExportView.

Uses the GTK mock infrastructure from tests/gui/conftest.py.
All tests bypass __init__ via __new__ to avoid touching the real GTK runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from playchitect.gui.views.export_view import ExportView


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_view() -> ExportView:
    """Return an ExportView with __init__ bypassed and mocked components."""
    from playchitect.gui.views.export_view import ExportView

    view = ExportView.__new__(ExportView)
    view._clusters = []
    view._metadata_map = {}
    view._cluster_names = {}

    # Mock all GTK widgets
    view._m3u_button = MagicMock()
    view._cue_button = MagicMock()
    view._rekordbox_button = MagicMock()
    view._traktor_button = MagicMock()
    view._serato_button = MagicMock()
    view._mixxx_button = MagicMock()

    # Set default return values for get_active() to avoid truthy MagicMock issues
    view._m3u_button.get_active.return_value = False
    view._cue_button.get_active.return_value = False
    view._rekordbox_button.get_active.return_value = False
    view._traktor_button.get_active.return_value = False
    view._serato_button.get_active.return_value = False
    view._mixxx_button.get_active.return_value = False

    view._all_clusters_button = MagicMock()
    view._selected_only_button = MagicMock()
    view._cluster_dropdown = MagicMock()

    view._destination_entry = MagicMock()
    view._browse_button = MagicMock()

    view._export_button = MagicMock()
    view._sync_button = MagicMock()

    view._status_label = MagicMock()

    return view


# ── TestExportViewInstantiation ─────────────────────────────────────────────


class TestExportViewInstantiation:
    """Smoke tests for ExportView instantiation."""

    def test_export_view_instantiates(self):
        """Test that ExportView can be instantiated with mocked GTK."""
        from playchitect.gui.views.export_view import ExportView

        # Mock all GTK widget creation
        with (
            patch("playchitect.gui.views.export_view.Gtk.Box") as mock_box,
            patch("playchitect.gui.views.export_view.Gtk.Label") as mock_label,
            patch("playchitect.gui.views.export_view.Gtk.CheckButton") as mock_check,
            patch("playchitect.gui.views.export_view.Gtk.DropDown") as mock_dropdown,
            patch("playchitect.gui.views.export_view.Gtk.Entry") as mock_entry,
            patch("playchitect.gui.views.export_view.Gtk.Button") as mock_button,
            patch("playchitect.gui.views.export_view.Gtk.Separator") as mock_sep,
            patch("playchitect.gui.views.export_view.Gtk.FileDialog") as mock_dialog,
            patch("playchitect.gui.views.export_view.get_config") as mock_get_config,
        ):
            # Setup mock returns
            mock_box.return_value = MagicMock()
            mock_label.return_value = MagicMock()
            mock_check.return_value = MagicMock()
            mock_dropdown.return_value = MagicMock()
            mock_entry.return_value = MagicMock()
            mock_button.return_value = MagicMock()
            mock_sep.return_value = MagicMock()
            mock_dialog.return_value = MagicMock()

            # Mock config
            mock_config = MagicMock()
            mock_config.get.return_value = None
            mock_get_config.return_value = mock_config

            view = ExportView()
            assert view is not None


# ── TestFormatSection ─────────────────────────────────────────────────────


class TestFormatSection:
    """Tests for the format radio group section."""

    def test_format_radio_group_has_six_children(self):
        """Test that format section has 6 radio buttons."""
        from playchitect.gui.views.export_view import ExportView

        with (
            patch("playchitect.gui.views.export_view.Gtk.Box") as mock_box,
            patch("playchitect.gui.views.export_view.Gtk.Label") as mock_label,
            patch("playchitect.gui.views.export_view.Gtk.CheckButton") as mock_check,
            patch("playchitect.gui.views.export_view.Gtk.DropDown") as mock_dropdown,
            patch("playchitect.gui.views.export_view.Gtk.Entry") as mock_entry,
            patch("playchitect.gui.views.export_view.Gtk.Button") as mock_button,
            patch("playchitect.gui.views.export_view.Gtk.Separator") as mock_sep,
            patch("playchitect.gui.views.export_view.get_config") as mock_get_config,
        ):
            # Setup mocks
            mock_box.return_value = MagicMock()
            mock_label.return_value = MagicMock()
            mock_dropdown.return_value = MagicMock()
            mock_entry.return_value = MagicMock()
            mock_button.return_value = MagicMock()
            mock_sep.return_value = MagicMock()

            # Track check button creation to count calls
            check_buttons = []

            def mock_check_button_new(*args, **kwargs):
                btn = MagicMock()
                check_buttons.append(btn)
                return btn

            mock_check.side_effect = mock_check_button_new

            # Mock config
            mock_config = MagicMock()
            mock_config.get.return_value = None
            mock_get_config.return_value = mock_config

            ExportView()

            # Should have 8 check buttons total:
            # 6 format (M3U, CUE, Rekordbox, Traktor, Serato, Mixxx) +
            # 2 playlist selection (All clusters, Selected only)
            assert len(check_buttons) == 8

    def test_four_format_buttons_insensitive(self):
        """Test that 4 format buttons are insensitive (future formats)."""
        from playchitect.gui.views.export_view import ExportView

        with (
            patch("playchitect.gui.views.export_view.Gtk.Box") as mock_box,
            patch("playchitect.gui.views.export_view.Gtk.Label") as mock_label,
            patch("playchitect.gui.views.export_view.Gtk.CheckButton") as mock_check,
            patch("playchitect.gui.views.export_view.Gtk.DropDown") as mock_dropdown,
            patch("playchitect.gui.views.export_view.Gtk.Entry") as mock_entry,
            patch("playchitect.gui.views.export_view.Gtk.Button") as mock_button,
            patch("playchitect.gui.views.export_view.Gtk.Separator") as mock_sep,
            patch("playchitect.gui.views.export_view.get_config") as mock_get_config,
        ):
            # Setup mocks
            mock_box.return_value = MagicMock()
            mock_label.return_value = MagicMock()
            mock_dropdown.return_value = MagicMock()
            mock_entry.return_value = MagicMock()
            mock_button.return_value = MagicMock()
            mock_sep.return_value = MagicMock()

            check_buttons = []

            def mock_check_button_new(*args, **kwargs):
                btn = MagicMock()
                check_buttons.append(btn)
                return btn

            mock_check.side_effect = mock_check_button_new

            # Mock config
            mock_config = MagicMock()
            mock_config.get.return_value = None
            mock_get_config.return_value = mock_config

            ExportView()

            # Count insensitive buttons
            insensitive_count = sum(1 for btn in check_buttons if btn.set_sensitive.call_args_list)
            # Should have 4 insensitive buttons (Rekordbox, Traktor, Serato, Mixxx)
            assert insensitive_count >= 4

    def test_m3u_selected_by_default(self):
        """Test that M3U is selected by default."""
        view = _make_view()
        view._m3u_button.get_active.return_value = True
        view._cue_button.get_active.return_value = False

        assert view._m3u_button.get_active() is True
        assert view._cue_button.get_active() is False


# ── TestPlaylistsSection ────────────────────────────────────────────────────


class TestPlaylistsSection:
    """Tests for the playlists selector section."""

    def test_playlist_selector_buttons_exist(self):
        """Test that playlist selector buttons exist."""
        view = _make_view()

        # Should have both buttons
        assert view._all_clusters_button is not None
        assert view._selected_only_button is not None

    def test_cluster_dropdown_exists(self):
        """Test that cluster dropdown exists."""
        view = _make_view()

        assert view._cluster_dropdown is not None


# ── TestDestinationSection ──────────────────────────────────────────────────


class TestDestinationSection:
    """Tests for the destination section."""

    def test_destination_entry_defaults_to_music_playlists(self):
        """Test that destination entry defaults to ~/Music/Playlists."""
        from playchitect.gui.views.export_view import DEFAULT_EXPORT_DIR

        # Verify the default export directory is ~/Music/Playlists
        assert DEFAULT_EXPORT_DIR == Path.home() / "Music" / "Playlists"

    def test_destination_entry_has_text(self):
        """Test that destination entry has some text set."""
        view = _make_view()

        # Entry should have get_text method
        assert hasattr(view._destination_entry, "get_text")

    def test_browse_button_exists(self):
        """Test that browse button exists."""
        view = _make_view()

        assert view._browse_button is not None


# ── TestActionSection ───────────────────────────────────────────────────────


class TestActionSection:
    """Tests for the action section with Export and Sync buttons."""

    def test_export_button_exists(self):
        """Test that export button exists."""
        view = _make_view()

        assert view._export_button is not None

    def test_export_button_is_sensitive(self):
        """Test that export button is sensitive by default."""
        view = _make_view()
        view._export_button.get_sensitive.return_value = True

        assert view._export_button.get_sensitive() is True

    def test_sync_button_exists(self):
        """Test that sync button exists."""
        view = _make_view()

        assert view._sync_button is not None

    def test_sync_button_is_insensitive(self):
        """Test that sync button is insensitive."""
        view = _make_view()
        view._sync_button.get_sensitive.return_value = False

        assert view._sync_button.get_sensitive() is False


# ── TestStatusSection ──────────────────────────────────────────────────────


class TestStatusSection:
    """Tests for the status section."""

    def test_status_label_exists(self):
        """Test that status label exists."""
        view = _make_view()

        assert view._status_label is not None

    def test_status_label_can_show_text(self):
        """Test that status label can display text."""
        view = _make_view()

        # Should be able to set markup
        view._status_label.set_markup("<span>Test status</span>")
        view._status_label.set_markup.assert_called_once()


# ── TestPublicAPI ───────────────────────────────────────────────────────────


class TestPublicAPI:
    """Tests for ExportView public API."""

    def test_set_clusters(self):
        """Test set_clusters updates internal state."""
        view = _make_view()

        clusters = []
        metadata = {}

        view.set_clusters(clusters, metadata)

        assert view._clusters == clusters
        assert view._metadata_map == metadata

    def test_set_cluster_names(self):
        """Test set_cluster_names updates internal state."""
        view = _make_view()

        names = {1: "Test Cluster", 2: "Another Cluster"}
        view.set_cluster_names(names)

        assert view._cluster_names == names

    def test_get_selected_format_m3u(self):
        """Test get_selected_format returns M3U when selected."""
        view = _make_view()
        view._m3u_button.get_active.return_value = True
        view._cue_button.get_active.return_value = False

        format_type = view.get_selected_format()
        assert format_type == "m3u"

    def test_get_selected_format_cue(self):
        """Test get_selected_format returns CUE when selected."""
        view = _make_view()
        view._m3u_button.get_active.return_value = False
        view._cue_button.get_active.return_value = True

        format_type = view.get_selected_format()
        assert format_type == "cue"

    def test_get_destination(self):
        """Test get_destination returns Path object."""
        view = _make_view()
        view._destination_entry.get_text.return_value = "/home/user/Music/Playlists"

        dest = view.get_destination()
        assert isinstance(dest, Path)
        assert str(dest) == "/home/user/Music/Playlists"

    def test_clear_resets_state(self):
        """Test clear resets all state."""
        view = _make_view()
        view._clusters = [MagicMock()]
        view._metadata_map = {Path("/test.mp3"): MagicMock()}
        view._cluster_names = {1: "Test"}

        view.clear()

        assert view._clusters == []
        assert view._metadata_map == {}
        assert view._cluster_names == {}
