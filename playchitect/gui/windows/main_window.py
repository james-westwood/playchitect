from __future__ import annotations

import logging
import threading
from pathlib import Path

import gi

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")

from gi.repository import Adw, Gio, GLib, Gtk  # type: ignore[unresolved-import]  # noqa: E402

from playchitect.core.arc_sequencer import BUILTIN_PRESETS, apply_arc  # noqa: E402
from playchitect.core.audio_scanner import AudioScanner  # noqa: E402
from playchitect.core.clustering import ClusterResult, PlaylistClusterer  # noqa: E402
from playchitect.core.intensity_analyzer import IntensityAnalyzer, IntensityFeatures  # noqa: E402
from playchitect.core.metadata_extractor import MetadataExtractor, TrackMetadata  # noqa: E402
from playchitect.core.naming.playlist_namer import PlaylistNamer  # noqa: E402
from playchitect.core.play_history import PlayHistory  # noqa: E402
from playchitect.core.sequencer import Sequencer, sequence_fresh  # noqa: E402
from playchitect.core.track_previewer import TrackPreviewer  # noqa: E402
from playchitect.gui.preferences_window import PreferencesWindow  # noqa: E402
from playchitect.gui.views import (  # noqa: E402
    ExportView,
    LibraryView,
    PlaylistsView,
    SetBuilderView,
)
from playchitect.gui.views.library_view import LibraryTrackModel  # noqa: E402
from playchitect.gui.widgets.track_list import TrackModel  # noqa: E402
from playchitect.gui.widgets.track_preview_panel import TrackPreviewPanel  # noqa: E402
from playchitect.utils.config import get_config  # noqa: E402

logger = logging.getLogger(__name__)

# How long (ms) the "Previewing…" title stays before reverting.
_PREVIEW_TITLE_TIMEOUT_MS: int = 3000

# Sidebar navigation row indices
_NAV_LIBRARY = 0
_NAV_PLAYLISTS = 1
_NAV_SET_BUILDER = 2
_NAV_EXPORT = 3


class PlaychitectWindow(Adw.ApplicationWindow):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)

        self.set_title("Playchitect")
        self.set_default_size(1000, 700)

        # ── Preview service ───────────────────────────────────────────────────
        self._previewer = TrackPreviewer()
        self._track_title = "Playchitect"  # restored after preview flash

        # ── Playlist naming service ───────────────────────────────────────────
        self._playlist_namer = PlaylistNamer()
        self._cluster_names: dict[int | str, str] = {}

        # ── State ─────────────────────────────────────────────────────────────
        self._metadata_map: dict[Path, TrackMetadata] = {}
        self._intensity_map: dict[Path, IntensityFeatures] = {}
        self._clusters: list[ClusterResult] = []
        self._original_clusters: list[ClusterResult] = []  # For arc reapplication
        self._play_history = PlayHistory()
        self._prefer_fresh: bool = False

        # ── Header bar ────────────────────────────────────────────────────────
        header = Adw.HeaderBar()
        self._spinner = Gtk.Spinner()
        header.pack_end(self._spinner)

        # Preview availability chip (right side of header)
        self._preview_chip = Gtk.Label()
        self._preview_chip.add_css_class("caption")
        self._update_preview_chip()
        header.pack_start(self._preview_chip)

        # Arc selector DropDown
        arc_label = Gtk.Label(label="Arc:")
        arc_label.set_margin_start(8)
        header.pack_start(arc_label)

        # Create string list for dropdown: "None" + preset names
        arc_names = ["None"] + [p.name for p in BUILTIN_PRESETS]
        arc_model = Gtk.StringList.new(arc_names)
        self._arc_dropdown = Gtk.DropDown(model=arc_model)
        self._arc_dropdown.set_selected(0)  # Default to "None"
        self._arc_dropdown.set_sensitive(False)  # Enabled after clustering
        self._arc_dropdown.connect("notify::selected", self._on_arc_selected)
        header.pack_start(self._arc_dropdown)

        # Prefer fresh tracks switch
        fresh_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        fresh_box.set_margin_start(12)
        fresh_label = Gtk.Label(label="Fresh:")
        fresh_box.append(fresh_label)

        self._fresh_switch = Gtk.Switch()
        self._fresh_switch.set_tooltip_text("Prioritize tracks not recently played")
        self._fresh_switch.connect("notify::active", self._on_fresh_switch_toggled)
        fresh_box.append(self._fresh_switch)

        header.pack_start(fresh_box)
        # ── Clustering Options ────────────────────────────────────────────────
        options_btn = Gtk.MenuButton(icon_name="emblem-system-symbolic")
        options_btn.set_tooltip_text("Clustering Options")

        popover_content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        popover_content.set_margin_top(6)
        popover_content.set_margin_bottom(6)
        popover_content.set_margin_start(6)
        popover_content.set_margin_end(6)

        group = Adw.PreferencesGroup(title="Playlist Size")
        row = Adw.ActionRow(title="Target")

        suffix = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)

        self._target_spin = Gtk.SpinButton.new_with_range(1, 500, 1)
        suffix.append(self._target_spin)

        self._target_unit = Gtk.DropDown.new_from_strings(["Tracks", "Minutes"])
        self._target_unit.connect("notify::selected-item", self._on_unit_changed)
        suffix.append(self._target_unit)

        row.add_suffix(suffix)
        group.add(row)
        popover_content.append(group)

        popover = Gtk.Popover()
        popover.set_child(popover_content)
        options_btn.set_popover(popover)
        header.pack_end(options_btn)

        # Menu button (primary menu)
        self._menu_button = Gtk.MenuButton()
        self._menu_button.set_icon_name("open-menu-symbolic")
        self._menu_button.set_tooltip_text("Menu")
        self._build_menu()
        header.pack_end(self._menu_button)

        # ── Main content: OverlaySplitView ─────────────────────────────────────
        # Sidebar (navigation)
        sidebar_content = self._build_sidebar()

        # Main content stack
        self._view_stack = self._build_view_stack()

        # OverlaySplitView with sidebar
        self._split_view = Adw.OverlaySplitView()
        self._split_view.set_sidebar_width_fraction(0.18)
        self._split_view.set_sidebar(sidebar_content)
        self._split_view.set_content(self._view_stack)

        # ── ToolbarView wrapper ────────────────────────────────────────────────
        toolbar_view = Adw.ToolbarView()
        toolbar_view.add_top_bar(header)
        toolbar_view.set_content(self._split_view)

        self.set_content(toolbar_view)

        # Load config defaults
        config = get_config()

        # Load target size
        def_tracks = config.get("default_target_tracks")
        def_duration = config.get("default_target_duration")

        if def_duration:
            self._target_unit.set_selected(1)  # Minutes
            self._target_spin.set_range(5, 300)
            self._target_spin.set_value(def_duration)
        else:
            self._target_unit.set_selected(0)  # Tracks
            self._target_spin.set_range(1, 500)
            self._target_spin.set_value(def_tracks or 25)

        # Load default music directory after the window is shown
        music_path = config.get_test_music_path()
        if music_path and music_path.is_dir():
            GLib.idle_add(self._start_scan, music_path)

    def _on_unit_changed(self, dropdown: Gtk.DropDown, _pspec: object) -> None:
        """Update spinner range when unit changes."""
        is_minutes = dropdown.get_selected() == 1
        if is_minutes:
            self._target_spin.set_range(5, 300)  # 5 mins to 5 hours
            # If value is huge (from tracks mode), clamp it logic?
            # Or just let set_range handle it.
            # Ideally we might want to convert, but simple switching is safer.
            current = self._target_spin.get_value()
            if current < 5:
                self._target_spin.set_value(60)  # Default to 1 hour
        else:
            self._target_spin.set_range(1, 500)  # 1 to 500 tracks
            current = self._target_spin.get_value()
            if current > 500:
                self._target_spin.set_value(25)  # Default to 25 tracks

    def _build_menu(self) -> None:
        """Build the primary menu with app actions."""
        menu = Gio.Menu()

        # Open Folder
        menu.append("Open Folder", "app.open-folder")

        menu.append_section(None, Gio.Menu())  # Separator

        # Preferences section
        pref_section = Gio.Menu()
        pref_section.append("Preferences", "app.preferences")
        pref_section.append("Keyboard Shortcuts", "app.keyboard-shortcuts")
        pref_section.append("About Playchitect", "app.about")
        menu.append_section(None, pref_section)

        self._menu_button.set_menu_model(menu)

    def _build_sidebar(self) -> Gtk.Widget:
        """Build the navigation sidebar with four views."""
        sidebar_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        sidebar_box.set_margin_top(6)
        sidebar_box.set_margin_bottom(6)
        sidebar_box.set_margin_start(6)
        sidebar_box.set_margin_end(6)

        # ListBox for navigation
        self._nav_list = Gtk.ListBox()
        self._nav_list.add_css_class("navigation-sidebar")
        self._nav_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self._nav_list.connect("row-selected", self._on_nav_row_selected)

        # Navigation items: (icon-name, label)
        nav_items = [
            ("folder-music-symbolic", "Library"),
            ("media-playlist-symbolic", "Playlists"),
            ("view-list-symbolic", "Set Builder"),
            ("document-send-symbolic", "Export"),
        ]

        for icon_name, label_text in nav_items:
            row = Gtk.ListBoxRow()
            row_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
            row_box.set_margin_start(12)
            row_box.set_margin_end(12)
            row_box.set_margin_top(8)
            row_box.set_margin_bottom(8)

            icon = Gtk.Image.new_from_icon_name(icon_name)
            icon.set_pixel_size(16)
            row_box.append(icon)

            label = Gtk.Label(label=label_text)
            label.set_xalign(0.0)
            label.set_hexpand(True)
            row_box.append(label)

            row.set_child(row_box)
            self._nav_list.append(row)

        # Select first row by default
        self._nav_list.select_row(self._nav_list.get_row_at_index(0))

        sidebar_box.append(self._nav_list)

        return sidebar_box

    def _build_view_stack(self) -> Adw.ViewStack:
        """Build the main content view stack with four pages."""
        stack = Adw.ViewStack()

        # Library view with preview panel
        self._library_view = LibraryView()
        self._library_view.connect("scan-complete", self._on_library_scan_complete)
        self._library_view.connect("track-selected", self._on_library_track_selected)
        self._library_view.connect("preview-toggled", self._on_preview_toggled)

        # Create track preview panel (hidden by default)
        self._track_preview = TrackPreviewPanel()
        self._track_preview.set_visible(False)
        self._track_preview.connect("prev-track", self._on_preview_prev)
        self._track_preview.connect("next-track", self._on_preview_next)

        # Library page with split pane
        library_paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        library_paned.set_start_child(self._library_view)
        library_paned.set_end_child(self._track_preview)
        library_paned.set_shrink_end_child(False)
        library_paned.set_position(600)
        stack.add_titled(library_paned, "library", "Library")

        # Playlists view (using new PlaylistsView)
        self._playlists_view = PlaylistsView()
        self._playlists_view.connect("cluster-selected", self._on_playlists_cluster_selected)
        stack.add_titled(self._playlists_view, "playlists", "Playlists")

        # Set Builder view
        self._set_builder_view = SetBuilderView()
        self._set_builder_view.load_library(self._metadata_map, self._intensity_map)
        stack.add_titled(self._set_builder_view, "set-builder", "Set Builder")

        # Export view
        self._export_view = ExportView()
        stack.add_titled(self._export_view, "export", "Export")

        return stack

    def _on_playlists_cluster_selected(self, _view: PlaylistsView, cluster_id: object) -> None:
        """Handle cluster selection from playlists view."""
        logger.debug("Playlist cluster selected: %s", cluster_id)
        # The playlists view already loads tracks internally
        # Just update the title
        self.set_title(f"Playchitect — Cluster {cluster_id}")

    def _on_nav_row_selected(self, _listbox: Gtk.ListBox, row: Gtk.ListBoxRow | None) -> None:
        """Handle navigation row selection to switch view stack pages."""
        if row is None:
            return

        index = row.get_index()
        page_names = ["library", "playlists", "set-builder", "export"]

        if 0 <= index < len(page_names) and hasattr(self, "_view_stack"):
            self._view_stack.set_visible_child_name(page_names[index])

    def show_preferences(self) -> None:
        """Show the preferences window."""
        prefs = PreferencesWindow()
        prefs.set_transient_for(self)
        prefs.present()

        # ── Preview chip ──────────────────────────────────────────────────────────

    def _update_preview_chip(self) -> None:
        """Set the header chip text and style to reflect preview availability."""
        launcher = self._previewer.launcher_name()
        if launcher == "sushi":
            self._preview_chip.set_text("Sushi ✓")
            self._preview_chip.set_tooltip_text("Quick Look via GNOME Sushi (Space)")
        elif launcher == "xdg-open":
            self._preview_chip.set_text("Preview: xdg-open")
            self._preview_chip.set_tooltip_text("Quick Look via xdg-open (Space)")
        else:
            self._preview_chip.set_text("No preview")
            self._preview_chip.set_tooltip_text("Install GNOME Sushi for Quick Look support")

    def _revert_title(self) -> bool:
        self.set_title(self._track_title)
        return False  # don't repeat

    # ── Scan ──────────────────────────────────────────────────────────────────

    def _start_scan(self, music_path: Path) -> bool:
        """Kick off a background scan; return False so GLib.idle_add doesn't repeat."""
        self._spinner.start()
        self.set_title("Playchitect — scanning…")
        threading.Thread(target=self._scan_worker, args=(music_path,), daemon=True).start()
        return False

    def _scan_worker(self, music_path: Path) -> None:
        """Run in a background thread: scan files and extract metadata."""
        try:
            scanner = AudioScanner()
            audio_files = scanner.scan(music_path)
            logger.info("Found %d audio files in %s", len(audio_files), music_path)

            extractor = MetadataExtractor()
            self._metadata_map = extractor.extract_batch(audio_files)

            tracks = [
                TrackModel(
                    filepath=str(meta.filepath),
                    title=meta.title or "",
                    artist=meta.artist or "",
                    bpm=meta.bpm or 0.0,
                    intensity=0.0,
                    hardness=0.0,
                    duration=meta.duration or 0.0,
                    audio_format=meta.filepath.suffix,
                    mood="",
                )
                for meta in self._metadata_map.values()
                if meta is not None
            ]
            tracks.sort(key=lambda t: t.title.lower() or t.filepath)

            GLib.idle_add(self._on_scan_complete, tracks)
        except Exception:
            logger.exception("Error scanning %s", music_path)
            GLib.idle_add(self._on_scan_error)

    def _on_scan_complete(self, tracks: list[TrackModel]) -> bool:
        """Handle scan completion - pass metadata to playlists view."""
        self._spinner.stop()
        self._track_title = f"Playchitect — {len(tracks)} tracks"
        self.set_title(self._track_title)
        # Pass metadata to playlists view for clustering
        self._playlists_view.set_metadata(self._metadata_map)
        return False

    def _on_scan_error(self) -> bool:
        self._spinner.stop()
        self._track_title = "Playchitect — scan failed"
        self.set_title(self._track_title)
        return False

    def _on_fresh_switch_toggled(self, switch: Gtk.Switch, _param: object) -> None:
        """Handle toggling of the 'Prefer fresh tracks' switch."""
        self._prefer_fresh = switch.get_active()
        logger.debug("Prefer fresh tracks: %s", self._prefer_fresh)

    def _on_cluster_clicked(self, _btn: Gtk.Button) -> None:
        """Perform clustering and sequencing on the scanned tracks."""
        if not self._metadata_map:
            return

        self._spinner.start()
        self.set_title("Playchitect — analysing & clustering…")

        # Read UI state (main thread)
        target_val = self._target_spin.get_value()
        is_minutes = self._target_unit.get_selected() == 1

        # Persist to config
        config = get_config()
        if is_minutes:
            config.set("default_target_duration", target_val)
            config.set("default_target_tracks", None)
        else:
            config.set("default_target_tracks", int(target_val))
            config.set("default_target_duration", None)
        config.save()

        # Perform in a thread to keep UI responsive
        threading.Thread(
            target=self._cluster_worker, args=(target_val, is_minutes), daemon=True
        ).start()

    def _cluster_worker(self, target_val: float, is_minutes: bool) -> None:
        """Background worker for clustering."""
        try:
            config = get_config()
            int_analyzer = IntensityAnalyzer(cache_dir=config.get_cache_dir() / "intensity")
            self._intensity_map = int_analyzer.analyze_batch(list(self._metadata_map.keys()))

            # Configure clusterer based on UI selection
            if is_minutes:
                clusterer = PlaylistClusterer(target_duration_per_playlist=target_val)
            else:
                clusterer = PlaylistClusterer(target_tracks_per_playlist=int(target_val))

            self._clusters = clusterer.cluster_by_features(self._metadata_map, self._intensity_map)

            # Store original clusters for arc reapplication
            self._original_clusters = list(self._clusters)

            # Sequence each cluster based on preference
            if self._prefer_fresh:
                # Use freshness-aware sequencing
                for cluster in self._clusters:
                    cluster.tracks = sequence_fresh(
                        cluster.tracks, self._intensity_map, self._play_history
                    )
            else:
                # Use default energy ramp sequencing
                sequencer = Sequencer()
                for cluster in self._clusters:
                    cluster.tracks = sequencer.sequence(
                        cluster, self._metadata_map, self._intensity_map, mode="ramp"
                    )

            # Generate intelligent names for all clusters
            self._cluster_names = self._playlist_namer.name_all_clusters(
                self._clusters, self._intensity_map, self._metadata_map
            )
            logger.info("Generated cluster names: %s", self._cluster_names)

            GLib.idle_add(self._on_cluster_complete)
        except Exception:
            logger.exception("Error during clustering")
            GLib.idle_add(self._on_cluster_error)

    def _on_cluster_complete(self) -> bool:
        """Update UI with clustering results."""
        self._spinner.stop()

        # Enable arc dropdown and reset to "None"
        self._arc_dropdown.set_sensitive(True)
        self._arc_dropdown.set_selected(0)

        # Load clusters into playlists view
        self._playlists_view.load_clusters(self._clusters)

        # Load clusters into set builder view
        self._set_builder_view.load_clusters(
            self._clusters, self._metadata_map, self._intensity_map
        )

        # Load clusters into export view
        self._export_view.set_clusters(self._clusters, self._metadata_map)
        self._export_view.set_cluster_names(self._cluster_names)

        self._track_title = f"Playchitect — {len(self._clusters)} clusters"
        self.set_title(self._track_title)
        return False

    def _on_cluster_error(self) -> bool:
        self._spinner.stop()
        self.set_title("Playchitect — clustering failed")
        return False

    def _on_arc_selected(self, dropdown: Gtk.DropDown, _param: object) -> None:
        """Handle arc preset selection from dropdown."""
        if not self._original_clusters:
            return

        selected_index = dropdown.get_selected()
        preset_name = "Original"
        if selected_index == 0:
            # "None" selected - restore original cluster order
            self._clusters = list(self._original_clusters)
        else:
            # Apply arc preset (index 0 is "None", so preset is at index-1)
            preset = BUILTIN_PRESETS[selected_index - 1]
            preset_name = preset.name
            self._clusters = apply_arc(self._original_clusters, preset)

        # Update UI with reordered clusters via playlists view
        self._playlists_view.load_clusters(self._clusters)

        # Also update export view with new clusters
        self._export_view.set_clusters(self._clusters, self._metadata_map)

        self._track_title = f"Playchitect — {len(self._clusters)} clusters ({preset_name})"
        self.set_title(self._track_title)

    def record_exported_tracks(self, track_paths: list[Path]) -> None:
        """Record exported tracks to play history.

        Args:
            track_paths: List of paths that were exported.
        """
        for path in track_paths:
            self._play_history.record(path)
        self._play_history.save()
        logger.info("Recorded %d tracks to play history", len(track_paths))

    def _on_library_scan_complete(self, view: LibraryView, store: Gio.ListStore) -> None:
        """Handle library view scan completion."""
        count = store.get_n_items()
        self._track_title = f"Playchitect — {count} tracks"
        self.set_title(self._track_title)

    def _on_library_track_selected(self, view: LibraryView, track: LibraryTrackModel) -> None:
        """Handle track selection in library view."""
        logger.debug("Track selected: %s", track.display_title)
        # Load track into preview panel if visible
        if self._track_preview.get_visible():
            self._track_preview.load_track(track)

    def _on_preview_toggled(self, _view: LibraryView, active: bool) -> None:
        """Handle preview panel toggle button."""
        self._track_preview.set_visible(active)
        if active:
            # Load current selection if panel is being shown
            track = self._library_view.get_selected_track()
            if track is not None:
                self._track_preview.load_track(track)
        else:
            # Stop playback when hiding panel
            self._track_preview._stop_playback()

    def _on_preview_prev(self, _panel: TrackPreviewPanel) -> None:
        """Handle previous track request from preview panel."""
        logger.debug("Previous track requested from preview panel")
        # TODO: Implement track navigation in library view

    def _on_preview_next(self, _panel: TrackPreviewPanel) -> None:
        """Handle next track request from preview panel."""
        logger.debug("Next track requested from preview panel")
        # TODO: Implement track navigation in library view
