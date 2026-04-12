"""Export view for playlist export functionality.

Provides a user interface for exporting playlists to various formats
including M3U and CUE sheets, with support for selecting clusters
and configuring the export destination.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import (  # type: ignore[unresolved-import]  # noqa: E402
    Adw,
    Gio,
    GLib,
    Gtk,
)

from playchitect.core.export import CUEExporter, M3UExporter  # noqa: E402
from playchitect.core.exporters import (  # noqa: E402
    RekordboxXMLExporter,
    TraktorNMLExporter,
    sync_all_playlists,
)
from playchitect.utils.config import get_config  # noqa: E402

if TYPE_CHECKING:
    from playchitect.core.clustering import ClusterResult
    from playchitect.core.intensity_analyzer import IntensityFeatures
    from playchitect.core.metadata_extractor import TrackMetadata

logger = logging.getLogger(__name__)

# Format identifiers
FORMAT_M3U = "m3u"
FORMAT_CUE = "cue"
FORMAT_REKORDBOX = "rekordbox"
FORMAT_TRAKTOR = "traktor"
FORMAT_SERATO = "serato"
FORMAT_MIXXX = "mixxx"

# Default export directory
DEFAULT_EXPORT_DIR = Path.home() / "Music" / "Playlists"


class ExportView(Gtk.Box):
    """Export view for playlist export functionality.

    Features:
    - Format selection (M3U, CUE, and future DJ software formats)
    - Playlist selection (all or selected only)
    - Destination directory selection with persistence
    - Export action with background threading
    - Export status display
    """

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.set_margin_top(12)
        self.set_margin_bottom(12)
        self.set_margin_start(12)
        self.set_margin_end(12)

        # State
        self._playlists: list[ClusterResult] = []
        self._metadata_map: dict[Path, TrackMetadata] = {}
        self._features_map: dict[Path, IntensityFeatures] = {}
        self._playlist_names: dict[int | str, str] = {}

        # Build UI sections
        self._build_format_section()
        self.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))
        self._build_playlists_section()
        self.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))
        self._build_destination_section()
        self.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))
        self._build_action_section()
        self._build_status_section()

    def _build_format_section(self) -> None:
        """Build the format radio group section."""
        format_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        format_box.set_margin_start(4)
        format_box.set_margin_end(4)

        # Section label
        format_label = Gtk.Label()
        format_label.set_xalign(0.0)
        format_label.set_markup("<b>Format</b>")
        format_box.append(format_label)

        # Radio buttons container
        radio_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        radio_box.set_margin_start(8)

        # M3U playlist (enabled, selected by default)
        self._m3u_button = Gtk.CheckButton(label="M3U playlist")
        self._m3u_button.set_active(True)
        radio_box.append(self._m3u_button)

        # CUE sheet (enabled)
        self._cue_button = Gtk.CheckButton(label="CUE sheet")
        self._cue_button.set_group(self._m3u_button)
        radio_box.append(self._cue_button)

        # Rekordbox XML (enabled)
        self._rekordbox_button = Gtk.CheckButton(label="Rekordbox XML")
        self._rekordbox_button.set_group(self._m3u_button)
        self._rekordbox_button.set_sensitive(True)
        radio_box.append(self._rekordbox_button)

        # Traktor NML (enabled)
        self._traktor_button = Gtk.CheckButton(label="Traktor NML")
        self._traktor_button.set_group(self._m3u_button)
        self._traktor_button.set_sensitive(True)
        radio_box.append(self._traktor_button)

        # Serato crates (disabled)
        self._serato_button = Gtk.CheckButton(label="Serato crates")
        self._serato_button.set_group(self._m3u_button)
        self._serato_button.set_sensitive(False)
        self._serato_button.set_tooltip_text("Coming in a future release")
        radio_box.append(self._serato_button)

        # Mixxx crate (disabled)
        self._mixxx_button = Gtk.CheckButton(label="Mixxx crate")
        self._mixxx_button.set_group(self._m3u_button)
        self._mixxx_button.set_sensitive(False)
        self._mixxx_button.set_tooltip_text("Coming in a future release")
        radio_box.append(self._mixxx_button)

        format_box.append(radio_box)
        self.append(format_box)

    def _build_playlists_section(self) -> None:
        """Build the playlists selector section."""
        playlists_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        playlists_box.set_margin_start(4)
        playlists_box.set_margin_end(4)

        # Section label
        playlists_label = Gtk.Label()
        playlists_label.set_xalign(0.0)
        playlists_label.set_markup("<b>Playlists</b>")
        playlists_box.append(playlists_label)

        # Radio buttons container
        radio_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        radio_box.set_margin_start(8)

        # All playlists (default)
        self._all_playlists_button = Gtk.CheckButton(label="All playlists")
        self._all_playlists_button.set_active(True)
        self._all_playlists_button.connect("toggled", self._on_playlist_selection_changed)
        radio_box.append(self._all_playlists_button)

        # Selected only
        self._selected_only_button = Gtk.CheckButton(label="Selected only")
        self._selected_only_button.set_group(self._all_playlists_button)
        self._selected_only_button.connect("toggled", self._on_playlist_selection_changed)
        radio_box.append(self._selected_only_button)

        playlists_box.append(radio_box)

        # Playlist dropdown (insensitive unless 'Selected only' is active)
        dropdown_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        dropdown_box.set_margin_start(8)
        dropdown_box.set_margin_top(4)

        dropdown_label = Gtk.Label(label="Playlist:")
        dropdown_box.append(dropdown_label)

        self._playlist_dropdown = Gtk.DropDown()
        self._playlist_dropdown.set_sensitive(False)
        self._playlist_dropdown.set_hexpand(True)
        dropdown_box.append(self._playlist_dropdown)

        playlists_box.append(dropdown_box)
        self.append(playlists_box)

    def _build_destination_section(self) -> None:
        """Build the destination section with entry and browse button."""
        dest_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        dest_box.set_margin_start(4)
        dest_box.set_margin_end(4)

        # Section label
        dest_label = Gtk.Label()
        dest_label.set_xalign(0.0)
        dest_label.set_markup("<b>Destination</b>")
        dest_box.append(dest_label)

        # Entry and browse button
        entry_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        entry_box.set_margin_start(8)

        # Load saved destination or use default
        config = get_config()
        saved_dest = config.get("export.last_destination", None)
        default_path = str(saved_dest) if saved_dest else str(DEFAULT_EXPORT_DIR)

        self._destination_entry = Gtk.Entry()
        self._destination_entry.set_text(default_path)
        self._destination_entry.set_hexpand(True)
        entry_box.append(self._destination_entry)

        self._browse_button = Gtk.Button(label="Browse…")
        self._browse_button.connect("clicked", self._on_browse_clicked)
        entry_box.append(self._browse_button)

        dest_box.append(entry_box)
        self.append(dest_box)

    def _build_action_section(self) -> None:
        """Build the action row with Export and Sync buttons."""
        action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        action_box.set_margin_start(4)
        action_box.set_margin_end(4)
        action_box.set_margin_top(8)

        # Export button (suggested-action style)
        self._export_button = Gtk.Button(label="Export")
        self._export_button.add_css_class("suggested-action")
        self._export_button.connect("clicked", self._on_export_clicked)
        action_box.append(self._export_button)

        # Sync with Mixxx button (enabled - requires config)
        self._sync_button = Gtk.Button(label="↺ Sync with Mixxx")
        self._sync_button.set_sensitive(True)
        self._sync_button.set_tooltip_text("Sync playlists as crates to Mixxx database")
        self._sync_button.connect("clicked", self._on_sync_mixxx_clicked)
        action_box.append(self._sync_button)

        # Spacer
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        action_box.append(spacer)

        self.append(action_box)

    def _build_status_section(self) -> None:
        """Build the status section showing last export result."""
        status_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        status_box.set_margin_start(4)
        status_box.set_margin_end(4)
        status_box.set_margin_top(8)

        self._status_label = Gtk.Label()
        self._status_label.set_xalign(0.0)
        self._status_label.set_markup("<span foreground='gray'>No export performed yet</span>")
        status_box.append(self._status_label)

        self.append(status_box)

    def _on_playlist_selection_changed(self, button: Gtk.CheckButton) -> None:
        """Handle playlist selection radio button toggle."""
        # Enable dropdown only when "Selected only" is active
        selected_only_active = self._selected_only_button.get_active()
        self._playlist_dropdown.set_sensitive(selected_only_active)

    def _on_browse_clicked(self, _button: Gtk.Button) -> None:
        """Handle browse button click - open folder dialog."""
        dialog = Gtk.FileDialog()
        dialog.set_title("Select Export Destination")

        # Set initial folder from current entry text if valid
        current_path = Path(self._destination_entry.get_text())
        if current_path.exists():
            dialog.set_initial_folder(Gio.File.new_for_path(str(current_path)))
        else:
            dialog.set_initial_folder(Gio.File.new_for_path(str(Path.home())))

        dialog.select_folder(
            self.get_ancestor(Gtk.Window),
            None,  # cancellable
            self._on_folder_selected,
            None,  # user_data
        )

    def _on_folder_selected(
        self,
        dialog: Gtk.FileDialog,
        result: Gio.AsyncResult,
        _user_data: object,
    ) -> None:
        """Handle folder selection from dialog."""
        try:
            folder = dialog.select_folder_finish(result)
            if folder is not None:
                path = Path(folder.get_path())
                self._destination_entry.set_text(str(path))

                # Persist to config
                config = get_config()
                config.set("export.last_destination", str(path))
                config.save()
        except Exception as e:
            logger.debug("Folder selection cancelled or failed: %s", e)

    def _on_export_clicked(self, _button: Gtk.Button) -> None:
        """Handle export button click - run export in background thread."""
        destination = Path(self._destination_entry.get_text())
        destination.mkdir(parents=True, exist_ok=True)

        # Determine format
        selected_format = self._get_selected_format()

        # Determine playlists to export
        if self._selected_only_button.get_active():
            # Export selected playlist only
            selected_item = self._playlist_dropdown.get_selected_item()
            if selected_item is None:
                self._show_status("No playlist selected", error=True)
                return
            playlist_id = self._get_playlist_id_from_dropdown_index(
                self._playlist_dropdown.get_selected()
            )
            playlists_to_export = [c for c in self._playlists if c.cluster_id == playlist_id]
        else:
            # Export all playlists
            playlists_to_export = self._playlists

        if not playlists_to_export:
            self._show_status("No playlists available to export", error=True)
            return

        # Update UI state
        self._export_button.set_sensitive(False)
        self._show_status("Exporting…", error=False)

        # Run export in background thread
        thread = threading.Thread(
            target=self._export_worker,
            args=(playlists_to_export, destination, selected_format),
            daemon=True,
        )
        thread.start()

    def _get_selected_format(self) -> str:
        """Determine the selected export format."""
        if self._cue_button.get_active():
            return FORMAT_CUE
        if self._rekordbox_button.get_active():
            return FORMAT_REKORDBOX
        if self._traktor_button.get_active():
            return FORMAT_TRAKTOR
        return FORMAT_M3U

    def _export_worker(
        self,
        playlists: list[ClusterResult],
        destination: Path,
        format_type: str,
    ) -> None:
        """Background worker for export operation."""
        try:
            if format_type == FORMAT_M3U:
                exporter = M3UExporter(destination)
                paths = exporter.export_clusters(playlists, self._metadata_map)
            elif format_type == FORMAT_CUE:
                exporter = CUEExporter(destination)
                paths = exporter.export_clusters(playlists, self._metadata_map)
            elif format_type == FORMAT_REKORDBOX:
                exporter = RekordboxXMLExporter(destination)
                paths = exporter.export_clusters(playlists, self._metadata_map, self._features_map)
            elif format_type == FORMAT_TRAKTOR:
                exporter = TraktorNMLExporter(destination)
                paths = exporter.export_clusters(playlists, self._metadata_map, self._features_map)
            else:
                raise ValueError(f"Unsupported format: {format_type}")

            GLib.idle_add(self._on_export_complete, paths)
        except Exception as e:
            logger.exception("Export failed")
            GLib.idle_add(self._on_export_error, str(e))

    def _on_export_complete(self, paths: list[Path]) -> bool:
        """Handle successful export completion."""
        self._export_button.set_sensitive(True)
        self._show_status(f"Exported {len(paths)} playlist(s) successfully", error=False)

        # Show toast
        self._show_toast(f"Exported {len(paths)} playlist(s)")
        return False

    def _on_export_error(self, error_message: str) -> bool:
        """Handle export error."""
        self._export_button.set_sensitive(True)
        self._show_status(f"Export failed: {error_message}", error=True)
        return False

    def _show_status(self, message: str, error: bool = False) -> None:
        """Update the status label."""
        if error:
            self._status_label.set_markup(f"<span foreground='red'>{message}</span>")
        else:
            self._status_label.set_markup(f"<span foreground='gray'>{message}</span>")

    def _show_toast(self, message: str) -> None:
        """Show a toast notification via the ancestor ToastOverlay."""
        toast_overlay = self.get_ancestor(Adw.ToastOverlay)
        if toast_overlay is not None:
            toast = Adw.Toast.new(message)
            toast_overlay.add_toast(toast)

    def _on_sync_mixxx_clicked(self, _button: Gtk.Button) -> None:
        """Handle Sync with Mixxx button click."""
        # Check if Mixxx DB path is configured
        config = get_config()
        mixxx_db_path = config.get("mixxx_db_path")

        if not mixxx_db_path:
            # Show alert dialog asking user to configure path
            dialog = Adw.AlertDialog.new(
                "Mixxx Database Not Configured",
                "Please set the Mixxx database path in Preferences to enable synchronization.",
            )
            dialog.add_response("ok", "OK")
            dialog.set_default_response("ok")
            dialog.present(self.get_ancestor(Gtk.Window))
            return

        db_path = Path(mixxx_db_path)
        if not db_path.exists():
            dialog = Adw.AlertDialog.new(
                "Mixxx Database Not Found",
                f"The configured Mixxx database was not found:\n{db_path}\n\n"
                "Please check the path in Preferences.",
            )
            dialog.add_response("ok", "OK")
            dialog.set_default_response("ok")
            dialog.present(self.get_ancestor(Gtk.Window))
            return

        # Run sync in background thread
        self._sync_button.set_sensitive(False)
        self._show_status("Syncing with Mixxx…", error=False)

        thread = threading.Thread(
            target=self._sync_mixxx_worker,
            args=(db_path,),
            daemon=True,
        )
        thread.start()

    def _sync_mixxx_worker(self, db_path: Path) -> None:
        """Background worker for Mixxx sync operation."""
        try:
            results = sync_all_playlists(db_path, self._playlists)
            total_tracks = sum(results.values())
            total_crates = len(results)
            GLib.idle_add(
                self._on_sync_complete, f"Synced {total_tracks} tracks to {total_crates} crates"
            )
        except Exception as e:
            logger.exception("Mixxx sync failed")
            GLib.idle_add(self._on_sync_error, str(e))

    def _on_sync_complete(self, message: str) -> bool:
        """Handle successful Mixxx sync completion."""
        self._sync_button.set_sensitive(True)
        self._show_status(message, error=False)
        self._show_toast(message)
        return False

    def _on_sync_error(self, error_message: str) -> bool:
        """Handle Mixxx sync error."""
        self._sync_button.set_sensitive(True)
        self._show_status(f"Sync failed: {error_message}", error=True)
        return False

    def _get_playlist_id_from_dropdown_index(self, index: int) -> int | str:
        """Get playlist ID from dropdown index."""
        if 0 <= index < len(self._playlists):
            return self._playlists[index].cluster_id
        return -1

    def _update_playlist_dropdown(self) -> None:
        """Update the playlist dropdown with current playlist names."""
        if not self._playlists:
            self._playlist_dropdown.set_model(Gtk.StringList.new([]))
            return

        # Build playlist name list
        playlist_names: list[str] = []
        for playlist in self._playlists:
            playlist_id = playlist.cluster_id
            # Use custom name if available, otherwise generate default
            if playlist_id in self._playlist_names:
                name = self._playlist_names[playlist_id]
            else:
                bpm_label = (
                    f"{int(playlist.bpm_mean)}-{int(playlist.bpm_mean + playlist.bpm_std)}bpm"
                )
                name = f"Playlist {playlist_id} [{bpm_label}]"
            playlist_names.append(name)

        self._playlist_dropdown.set_model(Gtk.StringList.new(playlist_names))

    # ── Public API ───────────────────────────────────────────────────────────

    def set_clusters(
        self,
        playlists: list[ClusterResult],
        metadata_map: dict[Path, TrackMetadata],
        features_map: dict[Path, IntensityFeatures] | None = None,
    ) -> None:
        """Set the playlists and metadata for export.

        Args:
            playlists: List of ClusterResult objects to export
            metadata_map: Mapping of track paths to metadata
            features_map: Optional mapping of track paths to intensity features
        """
        self._playlists = playlists
        self._metadata_map = metadata_map
        self._features_map = features_map or {}
        self._update_playlist_dropdown()

    def set_cluster_names(self, names: dict[int | str, str]) -> None:
        """Set custom names for playlists.

        Args:
            names: Mapping of playlist cluster_id to display name
        """
        self._playlist_names = names
        self._update_playlist_dropdown()

    def get_selected_format(self) -> str:
        """Return the currently selected export format."""
        return self._get_selected_format()

    def get_destination(self) -> Path:
        """Return the current export destination path."""
        return Path(self._destination_entry.get_text())

    def clear(self) -> None:
        """Clear all state and reset to defaults."""
        self._playlists = []
        self._metadata_map = {}
        self._features_map = {}
        self._playlist_names = {}
        self._update_playlist_dropdown()
        self._show_status("No export performed yet", error=False)
