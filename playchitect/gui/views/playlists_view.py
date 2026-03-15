"""Playlists view for managing and browsing generated clusters/playlists.

Architecture:
    PlaylistsView (Gtk.Box, vertical)
    └── Gtk.ActionBar (toolbar)
        ├── [Generate Playlists] button
        ├── Gtk.Spinner
        └── cluster count label
    └── Gtk.Paned (horizontal)
        ├── Left: Gtk.ListBox (cluster sidebar)
        │   └── ClusterRowWidget rows
        └── Right: Gtk.Box (vertical)
            ├── Gtk.ColumnView (tracks)
            └── ClusterStats widget (stats strip)

Signals:
    cluster-selected(int|str): emitted when a cluster row is clicked
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import (  # type: ignore[unresolved-import]  # noqa: E402
    GLib,
    GObject,
    Gtk,
)

from playchitect.core.clustering import ClusterResult, PlaylistClusterer  # noqa: E402
from playchitect.core.intensity_analyzer import IntensityAnalyzer  # noqa: E402
from playchitect.core.metadata_extractor import TrackMetadata  # noqa: E402
from playchitect.gui.widgets.cluster_stats import ClusterStats  # noqa: E402
from playchitect.gui.widgets.track_list import TrackListWidget, TrackModel  # noqa: E402
from playchitect.utils.config import get_config  # noqa: E402

logger = logging.getLogger(__name__)

# Sidebar width in pixels
_SIDEBAR_WIDTH: int = 250

# Bar characters for energy indicator
_BAR_FULL: str = "█"
_BAR_EMPTY: str = "░"
_BAR_WIDTH: int = 10


class ClusterRowWidget(Gtk.ListBoxRow):
    """A single row in the cluster sidebar showing cluster summary.

    Shows cluster index label, track count badge, and energy indicator bar.
    """

    def __init__(self, stats: ClusterStats) -> None:
        super().__init__()
        self._stats = stats
        self._build()

    def _build(self) -> None:
        """Build the row content."""
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(8)
        box.set_margin_bottom(8)

        # Cluster index label
        self._index_label = Gtk.Label(label=self._stats.cluster_label)
        self._index_label.set_xalign(0.0)
        self._index_label.set_width_chars(12)
        box.append(self._index_label)

        # Track count badge
        count_badge = Gtk.Label(label=self._stats.track_count_str)
        count_badge.add_css_class("caption")
        count_badge.add_css_class("dim-label")
        box.append(count_badge)

        # Spacer
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        box.append(spacer)

        # Energy indicator bar
        energy_bar = Gtk.Label()
        energy_bar.set_markup(f"<tt>{self._stats.intensity_bars}</tt>")
        energy_bar.set_tooltip_text(
            f"Intensity: {self._stats.intensity_mean:.2f} ({self._stats.intensity_label})"
        )
        box.append(energy_bar)

        self.set_child(box)

    @property
    def cluster_id(self) -> int | str:
        """Return the cluster ID."""
        return self._stats.cluster_id

    def update_stats(self, stats: ClusterStats) -> None:
        """Update the row with new stats."""
        self._stats = stats
        self._index_label.set_text(stats.cluster_label)
        # Rebuild the row
        child = self.get_child()
        if child is not None:
            self.set_child(None)
        self._build()


class PlaylistsView(Gtk.Box):
    """Main playlists view with cluster sidebar and track list.

    Signals:
        cluster-selected(GObject.TYPE_PYOBJECT): cluster_id (int|str)
    """

    __gsignals__ = {
        "cluster-selected": (GObject.SignalFlags.RUN_FIRST, None, (GObject.TYPE_PYOBJECT,)),
    }

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # State
        self._clusters: list[ClusterResult] = []
        self._cluster_stats: list[ClusterStats] = []
        self._metadata_map: dict[Path, TrackMetadata] = {}
        self._intensity_map: dict[Path, Any] = {}
        self._selected_cluster_id: int | str | None = None

        # Build UI
        self._build_toolbar()
        self.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))
        self._build_content()

    def _build_toolbar(self) -> None:
        """Build the top ActionBar toolbar."""
        self._action_bar = Gtk.ActionBar()

        # Left: Generate Playlists button (primary style)
        self._generate_btn = Gtk.Button(label="Generate Playlists")
        self._generate_btn.add_css_class("suggested-action")
        self._generate_btn.connect("clicked", self._on_generate_clicked)
        self._action_bar.pack_start(self._generate_btn)

        # Left: Playlist size controls box
        controls_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        controls_box.set_margin_start(12)

        # Size value SpinButton with label
        size_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        size_label = Gtk.Label(label="Size:")
        size_box.append(size_label)

        self._size_spin = Gtk.SpinButton()
        self._size_spin.set_range(5, 200)
        self._size_spin.set_increments(1, 10)
        self._size_spin.set_value(20)
        self._size_spin.set_snap_to_ticks(True)
        self._size_spin.set_numeric(True)
        self._size_spin.set_width_chars(4)
        size_box.append(self._size_spin)
        controls_box.append(size_box)

        # Unit DropDown (tracks/minutes)
        unit_model = Gtk.StringList.new(["tracks", "minutes"])
        self._unit_dropdown = Gtk.DropDown(model=unit_model)
        self._unit_dropdown.set_selected(0)  # Default to "tracks"
        self._unit_dropdown.connect("notify::selected", self._on_unit_changed)
        controls_box.append(self._unit_dropdown)

        # Number of playlists SpinButton with label
        playlists_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        playlists_label = Gtk.Label(label="Playlists:")
        playlists_box.append(playlists_label)

        self._playlists_spin = Gtk.SpinButton()
        self._playlists_spin.set_range(0, 20)
        self._playlists_spin.set_increments(1, 5)
        self._playlists_spin.set_value(0)
        self._playlists_spin.set_snap_to_ticks(True)
        self._playlists_spin.set_numeric(True)
        self._playlists_spin.set_width_chars(3)
        self._playlists_spin.set_tooltip_text("0 = auto (elbow/silhouette)")
        playlists_box.append(self._playlists_spin)
        controls_box.append(playlists_box)

        self._action_bar.pack_start(controls_box)

        # Center: Harmonic mixing control
        harmonic_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        harmonic_box.set_margin_start(12)
        harmonic_label = Gtk.Label(label="Harmonic mixing")
        harmonic_box.append(harmonic_label)

        self._harmonic_switch = Gtk.Switch()
        self._harmonic_switch.set_valign(Gtk.Align.CENTER)
        self._harmonic_switch.set_tooltip_text("Enable harmonic key sequencing")
        self._harmonic_switch.connect("notify::active", self._on_harmonic_switch_toggled)
        harmonic_box.append(self._harmonic_switch)

        self._action_bar.pack_start(harmonic_box)

        # Right: Spinner (hidden while idle)
        self._spinner = Gtk.Spinner()
        self._spinner.set_visible(False)
        self._action_bar.pack_end(self._spinner)

        # Right: Cluster count label
        self._count_label = Gtk.Label(label="0 clusters")
        self._count_label.set_margin_end(12)
        self._count_label.add_css_class("caption")
        self._count_label.add_css_class("dim-label")
        self._action_bar.pack_end(self._count_label)

        self.append(self._action_bar)

    def _build_content(self) -> None:
        """Build the main content area with paned layout."""
        self._paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self._paned.set_position(_SIDEBAR_WIDTH)
        self._paned.set_shrink_start_child(False)
        self._paned.set_shrink_end_child(False)
        self._paned.set_vexpand(True)

        # Left: Cluster sidebar
        self._build_cluster_sidebar()

        # Right: Track list + stats
        self._build_right_content()

        self.append(self._paned)

    def _build_cluster_sidebar(self) -> None:
        """Build the left cluster sidebar with ListBox."""
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.set_size_request(_SIDEBAR_WIDTH, -1)

        self._cluster_list = Gtk.ListBox()
        self._cluster_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self._cluster_list.connect("row-selected", self._on_cluster_row_selected)

        # Placeholder for empty state
        self._cluster_placeholder = Gtk.Label(
            label="No clusters yet.\nClick 'Generate Playlists' to analyze tracks."
        )
        self._cluster_placeholder.set_justify(Gtk.Justification.CENTER)
        self._cluster_placeholder.add_css_class("dim-label")
        self._cluster_placeholder.set_vexpand(True)
        self._cluster_placeholder.set_valign(Gtk.Align.CENTER)
        self._cluster_list.set_placeholder(self._cluster_placeholder)

        scroll.set_child(self._cluster_list)
        self._paned.set_start_child(scroll)

    def _build_right_content(self) -> None:
        """Build the right pane with track ColumnView and stats strip."""
        right_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # Track list (ColumnView)
        self._track_list = TrackListWidget()
        self._track_list.set_vexpand(True)
        right_box.append(self._track_list)

        # Stats strip
        self._stats_separator = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        right_box.append(self._stats_separator)

        self._stats_widget = self._create_stats_widget()
        right_box.append(self._stats_widget)

        self._paned.set_end_child(right_box)

    def _create_stats_widget(self) -> Gtk.Box:
        """Create the stats strip widget (reuses ClusterStats data)."""
        stats_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        stats_box.set_margin_start(12)
        stats_box.set_margin_end(12)
        stats_box.set_margin_top(8)
        stats_box.set_margin_bottom(8)

        # Stats labels
        self._stats_bpm_label = Gtk.Label(label="BPM: —")
        self._stats_bpm_label.add_css_class("caption")
        stats_box.append(self._stats_bpm_label)

        self._stats_intensity_label = Gtk.Label(label="Intensity: —")
        self._stats_intensity_label.add_css_class("caption")
        stats_box.append(self._stats_intensity_label)

        self._stats_tracks_label = Gtk.Label(label="Tracks: —")
        self._stats_tracks_label.add_css_class("caption")
        stats_box.append(self._stats_tracks_label)

        self._stats_duration_label = Gtk.Label(label="Duration: —")
        self._stats_duration_label.add_css_class("caption")
        stats_box.append(self._stats_duration_label)

        # Spacer
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        stats_box.append(spacer)

        return stats_box

    def _update_stats_display(self, stats: ClusterStats | None) -> None:
        """Update the stats strip with cluster stats."""
        if stats is None:
            self._stats_bpm_label.set_text("BPM: —")
            self._stats_intensity_label.set_text("Intensity: —")
            self._stats_tracks_label.set_text("Tracks: —")
            self._stats_duration_label.set_text("Duration: —")
            return

        self._stats_bpm_label.set_text(f"BPM: {stats.bpm_range_str}")
        self._stats_intensity_label.set_text(
            f"Intensity: {stats.intensity_label} ({stats.intensity_mean:.2f})"
        )
        self._stats_tracks_label.set_text(stats.track_count_str)
        self._stats_duration_label.set_text(f"Duration: {stats.duration_str}")

    # ── Signal Handlers ──────────────────────────────────────────────────────

    def _on_cluster_row_selected(self, _listbox: Gtk.ListBox, row: Gtk.ListBoxRow | None) -> None:
        """Handle cluster row selection."""
        if row is None:
            return

        if isinstance(row, ClusterRowWidget):
            self._selected_cluster_id = row.cluster_id
            self.emit("cluster-selected", row.cluster_id)
            self._load_tracks_for_cluster(row.cluster_id)

            # Update stats display
            stats = next((s for s in self._cluster_stats if s.cluster_id == row.cluster_id), None)
            self._update_stats_display(stats)

    def _on_generate_clicked(self, _btn: Gtk.Button) -> None:
        """Handle Generate Playlists button click."""
        if not self._metadata_map:
            logger.warning("No tracks loaded for clustering")
            return

        self._set_loading_state(True)

        # Run analysis and clustering in background thread
        thread = threading.Thread(target=self._generate_worker, daemon=True)
        thread.start()

    def _on_unit_changed(self, dropdown: Gtk.DropDown, _param: object) -> None:
        """Handle unit dropdown change (tracks/minutes)."""
        selected = dropdown.get_selected()
        if selected == 0:
            # Tracks mode: range 5-200, step 1, value 20
            self._size_spin.set_range(5, 200)
            self._size_spin.set_increments(1, 10)
            self._size_spin.set_value(20)
        else:
            # Minutes mode: range 30-240, step 5, value 60
            self._size_spin.set_range(30, 240)
            self._size_spin.set_increments(5, 15)
            self._size_spin.set_value(60)

    def _on_harmonic_switch_toggled(self, switch: Gtk.Switch, _param: object) -> None:
        """Handle harmonic mixing switch toggle - reload tracks with new sequencing."""
        # Reload current cluster tracks if one is selected
        if self._selected_cluster_id is not None:
            self._load_tracks_for_cluster(self._selected_cluster_id)
            logger.info("Harmonic mixing %s", "enabled" if switch.get_active() else "disabled")

    def _generate_worker(self) -> None:
        """Background worker for intensity analysis and clustering."""
        try:
            config = get_config()
            cache_dir = config.get_cache_dir() / "intensity"

            # Read control values
            size_value = int(self._size_spin.get_value())
            unit_selected = self._unit_dropdown.get_selected()
            n_playlists = int(self._playlists_spin.get_value())
            n_playlists = n_playlists if n_playlists > 0 else None

            # Analyze intensities
            analyzer = IntensityAnalyzer(cache_dir=cache_dir)
            self._intensity_map = analyzer.analyze_batch(list(self._metadata_map.keys()))

            # Create clusterer based on unit selection
            if unit_selected == 0:
                # Tracks mode
                clusterer = PlaylistClusterer(target_tracks_per_playlist=size_value)
            else:
                # Minutes mode
                clusterer = PlaylistClusterer(target_duration_per_playlist=size_value)

            # Cluster by features with optional n_playlists override
            self._clusters = clusterer.cluster_by_features(
                self._metadata_map,
                self._intensity_map,
                n_playlists=n_playlists,
            )

            # Convert to stats
            self._cluster_stats = ClusterStats.from_results(self._clusters)

            # Update UI on main thread
            GLib.idle_add(self._on_generate_complete)
        except Exception:
            logger.exception("Error during playlist generation")
            GLib.idle_add(self._on_generate_error)

    def _on_generate_complete(self) -> bool:
        """Handle completion of playlist generation."""
        self._set_loading_state(False)
        self._refresh_cluster_sidebar()

        # Update count label
        count = len(self._cluster_stats)
        noun = "cluster" if count == 1 else "clusters"
        self._count_label.set_text(f"{count} {noun}")

        logger.info("Generated %d playlists", count)
        return False

    def _on_generate_error(self) -> bool:
        """Handle error during playlist generation."""
        self._set_loading_state(False)
        logger.error("Playlist generation failed")
        return False

    def _set_loading_state(self, loading: bool) -> None:
        """Set the loading state (spinner visibility, button sensitivity)."""
        self._spinner.set_visible(loading)
        if loading:
            self._spinner.start()
        else:
            self._spinner.stop()
        self._generate_btn.set_sensitive(not loading)

    def _refresh_cluster_sidebar(self) -> None:
        """Refresh the cluster sidebar with current stats."""
        # Clear existing rows
        while True:
            row = self._cluster_list.get_row_at_index(0)
            if row is None:
                break
            self._cluster_list.remove(row)

        # Add new rows
        for stats in self._cluster_stats:
            row = ClusterRowWidget(stats)
            self._cluster_list.append(row)

        # Select first row if available
        if self._cluster_stats:
            first_row = self._cluster_list.get_row_at_index(0)
            if first_row:
                self._cluster_list.select_row(first_row)

    def _load_tracks_for_cluster(self, cluster_id: int | str) -> None:
        """Load tracks for the selected cluster into the track list."""
        # Find the cluster result
        cluster = next((c for c in self._clusters if str(c.cluster_id) == str(cluster_id)), None)
        if not cluster:
            self._track_list.clear()
            return

        # Apply harmonic sequencing if switch is enabled
        track_order = cluster.tracks
        if self._harmonic_switch.get_active():
            from playchitect.core.sequencer import sequence_harmonic

            try:
                track_order = sequence_harmonic(
                    list(cluster.tracks),
                    self._intensity_map,
                )
            except (ValueError, KeyError) as e:
                logger.warning("Harmonic sequencing failed: %s", e)
                # Fall back to original order

        # Build track models
        tracks: list[TrackModel] = []
        for path in track_order:
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
                cluster=int(cluster_id) if isinstance(cluster_id, int) else -1,
                duration=meta.duration or 0.0,
                audio_format=path.suffix,
                mood=intensity.mood_label if intensity else "",
                camelot_key=intensity.camelot_key if intensity else "",
            )
            tracks.append(model)

        self._track_list.load_tracks(tracks)

    # ── Public API ───────────────────────────────────────────────────────────

    def set_metadata(self, metadata_map: dict[Path, TrackMetadata]) -> None:
        """Set the metadata map for tracks."""
        self._metadata_map = metadata_map
        self._generate_btn.set_sensitive(len(metadata_map) > 0)

    def load_clusters(self, clusters: list[ClusterResult]) -> None:
        """Load clusters and refresh the sidebar."""
        self._clusters = clusters
        self._cluster_stats = ClusterStats.from_results(clusters)
        self._refresh_cluster_sidebar()

        # Update count
        count = len(self._cluster_stats)
        noun = "cluster" if count == 1 else "clusters"
        self._count_label.set_text(f"{count} {noun}")

    def get_selected_cluster_id(self) -> int | str | None:
        """Return the currently selected cluster ID."""
        return self._selected_cluster_id

    def get_track_list(self) -> TrackListWidget:
        """Return the track list widget for external access."""
        return self._track_list

    def clear(self) -> None:
        """Clear all clusters and tracks."""
        self._clusters = []
        self._cluster_stats = []
        self._selected_cluster_id = None
        self._refresh_cluster_sidebar()
        self._track_list.clear()
        self._count_label.set_text("0 clusters")
        self._update_stats_display(None)
