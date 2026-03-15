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
from playchitect.gui.views import LibraryView  # noqa: E402
from playchitect.gui.widgets.cluster_stats import ClusterStats  # noqa: E402
from playchitect.gui.widgets.cluster_view import ClusterViewPanel  # noqa: E402
from playchitect.gui.widgets.track_list import TrackListWidget, TrackModel  # noqa: E402
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

        # Menu button (primary menu)
        self._menu_button = Adw.MenuButton()
        self._menu_button.set_icon_name("open-menu-symbolic")
        self._menu_button.set_tooltip_text("Menu")
        self._build_menu()
        header.pack_end(self._menu_button)

        # ── Main content: OverlaySplitView ─────────────────────────────────────
        # Sidebar (navigation)
        sidebar_content = self._build_sidebar()

        # Main content stack
        self._view_stack = self._build_view_stack()

        # Create cluster panel and track list (for backward compatibility)
        self._build_cluster_panel_and_track_list()

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

        # Load default music directory after the window is shown
        config = get_config()
        music_path = config.get_test_music_path()
        if music_path and music_path.is_dir():
            GLib.idle_add(self._start_scan, music_path)

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

        # Library view (using new LibraryView)
        self._library_view = LibraryView()
        self._library_view.connect("scan-complete", self._on_library_scan_complete)
        self._library_view.connect("track-selected", self._on_library_track_selected)
        stack.add_titled(self._library_view, "library", "Library")

        # Playlists view (stub)
        playlists_page = Gtk.Label(label="Playlists — coming soon")
        playlists_page.set_vexpand(True)
        stack.add_titled(playlists_page, "playlists", "Playlists")

        # Set Builder view (stub)
        set_builder_page = Gtk.Label(label="Set Builder — coming soon")
        set_builder_page.set_vexpand(True)
        stack.add_titled(set_builder_page, "set-builder", "Set Builder")

        # Export view (stub)
        export_page = Gtk.Label(label="Export — coming soon")
        export_page.set_vexpand(True)
        stack.add_titled(export_page, "export", "Export")

        return stack

    def _build_cluster_panel_and_track_list(self) -> tuple[Gtk.Widget, Gtk.Widget]:
        """Build the cluster panel and track list widgets.

        Returns:
            Tuple of (cluster_panel, track_list)
        """
        # Create paned layout for cluster panel and track list
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.set_position(280)
        paned.set_shrink_start_child(False)
        paned.set_shrink_end_child(False)
        paned.set_vexpand(True)

        self.cluster_panel = ClusterViewPanel()
        self.cluster_panel.set_size_request(220, -1)
        self.cluster_panel.connect("cluster-selected", self._on_cluster_selected)
        paned.set_start_child(self.cluster_panel)

        self.track_list = TrackListWidget()
        self.track_list.connect("preview-requested", self._on_preview_requested)
        paned.set_end_child(self.track_list)

        return paned

    def _build_library_page(self) -> Gtk.Widget:
        """Build the library page containing the scan/cluster UI."""
        page_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        page_box.set_vexpand(True)

        # Cluster button (moved from header to library page)
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        button_box.set_margin_start(12)
        button_box.set_margin_end(12)
        button_box.set_margin_top(12)
        button_box.set_margin_bottom(6)

        self._cluster_btn = Gtk.Button(label="Cluster")
        self._cluster_btn.add_css_class("suggested-action")
        self._cluster_btn.connect("clicked", self._on_cluster_clicked)
        self._cluster_btn.set_sensitive(False)
        button_box.append(self._cluster_btn)

        page_box.append(button_box)

        # Split pane: cluster panel (left) + track list (right)
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.set_position(280)
        paned.set_shrink_start_child(False)
        paned.set_shrink_end_child(False)
        paned.set_vexpand(True)

        self.cluster_panel = ClusterViewPanel()
        self.cluster_panel.set_size_request(220, -1)
        self.cluster_panel.connect("cluster-selected", self._on_cluster_selected)
        paned.set_start_child(self.cluster_panel)

        self.track_list = TrackListWidget()
        self.track_list.connect("preview-requested", self._on_preview_requested)
        paned.set_end_child(self.track_list)

        page_box.append(paned)

        return page_box

    def _on_nav_row_selected(self, _listbox: Gtk.ListBox, row: Gtk.ListBoxRow | None) -> None:
        """Handle navigation row selection to switch view stack pages."""
        if row is None:
            return

        index = row.get_index()
        page_names = ["library", "playlists", "set-builder", "export"]

        if 0 <= index < len(page_names):
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

    # ── Preview handler ───────────────────────────────────────────────────────

    def _on_preview_requested(self, _widget: TrackListWidget) -> None:
        """Handle spacebar / Quick Look — preview first selected track."""
        paths = [Path(p) for p in self.track_list.get_selected_paths()]
        result = self._previewer.preview_first(paths)

        if result.success and result.filepath is not None:
            name = result.filepath.stem
            self.set_title(f"Playchitect — Previewing: {name}")
            GLib.timeout_add(_PREVIEW_TITLE_TIMEOUT_MS, self._revert_title)
        else:
            logger.warning("Preview failed: %s", result.error)

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
        self.track_list.load_tracks(tracks)
        self._spinner.stop()
        self._cluster_btn.set_sensitive(True)
        self._track_title = f"Playchitect — {len(tracks)} tracks"
        self.set_title(self._track_title)
        return False

    def _on_scan_error(self) -> bool:
        self._spinner.stop()
        self._track_title = "Playchitect — scan failed"
        self.set_title(self._track_title)
        return False

    def _on_cluster_clicked(self, _btn: Gtk.Button) -> None:
        """Perform clustering and sequencing on the scanned tracks."""
        if not self._metadata_map:
            return

        self._spinner.start()
        self._cluster_btn.set_sensitive(False)
        self.set_title("Playchitect — analysing & clustering…")

        # Perform in a thread to keep UI responsive
        threading.Thread(target=self._cluster_worker, daemon=True).start()

    def _on_fresh_switch_toggled(self, switch: Gtk.Switch, _param: object) -> None:
        """Handle toggling of the 'Prefer fresh tracks' switch."""
        self._prefer_fresh = switch.get_active()
        logger.debug("Prefer fresh tracks: %s", self._prefer_fresh)

    def _cluster_worker(self) -> None:
        """Background worker for clustering."""
        try:
            config = get_config()
            int_analyzer = IntensityAnalyzer(cache_dir=config.get_cache_dir() / "intensity")
            self._intensity_map = int_analyzer.analyze_batch(list(self._metadata_map.keys()))

            clusterer = PlaylistClusterer(target_tracks_per_playlist=20)
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
        self._cluster_btn.set_sensitive(True)

        # Enable arc dropdown and reset to "None"
        self._arc_dropdown.set_sensitive(True)
        self._arc_dropdown.set_selected(0)

        stats = ClusterStats.from_results(self._clusters)
        self.cluster_panel.load_clusters(stats)

        # Apply generated names to cluster cards
        if self._cluster_names:
            self.cluster_panel.set_cluster_names(self._cluster_names)

        self._track_title = f"Playchitect — {len(self._clusters)} clusters"
        self.set_title(self._track_title)
        return False

    def _on_cluster_error(self) -> bool:
        self._spinner.stop()
        self._cluster_btn.set_sensitive(True)
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

        # Update UI with reordered clusters
        stats = ClusterStats.from_results(self._clusters)
        self.cluster_panel.load_clusters(stats)

        # Re-apply cluster names after arc selection
        if self._cluster_names:
            # Build updated names dict for the new cluster order
            updated_names = {}
            for cluster in self._clusters:
                # Try to find original name by cluster_id
                if cluster.cluster_id in self._cluster_names:
                    updated_names[cluster.cluster_id] = self._cluster_names[cluster.cluster_id]
                else:
                    updated_names[cluster.cluster_id] = f"Cluster {cluster.cluster_id}"
            self.cluster_panel.set_cluster_names(updated_names)

        self._track_title = f"Playchitect — {len(self._clusters)} clusters ({preset_name})"
        self.set_title(self._track_title)

    def _on_cluster_selected(self, _panel: ClusterViewPanel, cluster_id: object) -> None:
        """Filter the track list to show only tracks in the selected cluster, in sequenced order."""
        # Find the cluster result
        cluster = next((c for c in self._clusters if str(c.cluster_id) == str(cluster_id)), None)
        if not cluster:
            return

        # Map paths to TrackModel objects
        # We'll re-extract from the original full list if needed, or maintain a lookup.
        # For MVP, we'll just rebuild models for the cluster tracks in sequence.
        cluster_tracks = []
        for path in cluster.tracks:
            meta = self._metadata_map.get(path)
            if not meta:
                continue

            intensity = self._intensity_map.get(path)
            model = TrackModel(
                filepath=str(path),
                title=meta.title or "",
                artist=meta.artist or "",
                bpm=meta.bpm or 0.0,
                intensity=intensity.rms_energy if intensity else 0.0,
                hardness=intensity.hardness if intensity else 0.0,
                cluster=cluster.cluster_id if isinstance(cluster.cluster_id, int) else -1,
                duration=meta.duration or 0.0,
                audio_format=path.suffix,
                mood=intensity.mood_label if intensity else "",
            )
            cluster_tracks.append(model)

        self.track_list.load_tracks(cluster_tracks)
        self.track_list._search_entry.set_text("")
        self.set_title(f"Playchitect — Cluster {cluster_id}")

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
        self._cluster_btn.set_sensitive(count > 0)

    def _on_library_track_selected(self, view: LibraryView, track: object) -> None:
        """Handle track selection in library view."""
        # TODO: Implement track preview or details view
        logger.debug("Track selected: %s", getattr(track, "display_title", track))
