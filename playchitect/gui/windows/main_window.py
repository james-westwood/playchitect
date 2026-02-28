from __future__ import annotations

import logging
import threading
from pathlib import Path

import gi

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")

from gi.repository import Adw, GLib, Gtk  # type: ignore[unresolved-import]  # noqa: E402

from playchitect.core.audio_scanner import AudioScanner  # noqa: E402
from playchitect.core.clustering import ClusterResult, PlaylistClusterer  # noqa: E402
from playchitect.core.intensity_analyzer import IntensityAnalyzer, IntensityFeatures  # noqa: E402
from playchitect.core.metadata_extractor import MetadataExtractor, TrackMetadata  # noqa: E402
from playchitect.core.sequencer import Sequencer  # noqa: E402
from playchitect.core.track_previewer import TrackPreviewer  # noqa: E402
from playchitect.gui.widgets.cluster_stats import ClusterStats  # noqa: E402
from playchitect.gui.widgets.cluster_view import ClusterViewPanel  # noqa: E402
from playchitect.gui.widgets.track_list import TrackListWidget, TrackModel  # noqa: E402
from playchitect.utils.config import get_config  # noqa: E402

logger = logging.getLogger(__name__)

# How long (ms) the "Previewing…" title stays before reverting.
_PREVIEW_TITLE_TIMEOUT_MS: int = 3000


class PlaychitectWindow(Adw.ApplicationWindow):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)

        self.set_title("Playchitect")
        self.set_default_size(1000, 700)

        # ── Preview service ───────────────────────────────────────────────────
        self._previewer = TrackPreviewer()
        self._track_title = "Playchitect"  # restored after preview flash

        # ── Header bar ────────────────────────────────────────────────────────
        header = Adw.HeaderBar()
        self._spinner = Gtk.Spinner()
        header.pack_end(self._spinner)

        # Preview availability chip (right side of header)
        self._preview_chip = Gtk.Label()
        self._preview_chip.add_css_class("caption")
        self._update_preview_chip()
        header.pack_start(self._preview_chip)

        # Cluster button
        self._cluster_btn = Gtk.Button(label="Cluster")
        self._cluster_btn.add_css_class("suggested-action")
        self._cluster_btn.connect("clicked", self._on_cluster_clicked)
        self._cluster_btn.set_sensitive(False)
        header.pack_start(self._cluster_btn)

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

        toolbar_view = Adw.ToolbarView()
        toolbar_view.add_top_bar(header)

        # ── State ─────────────────────────────────────────────────────────────
        self._metadata_map: dict[Path, TrackMetadata] = {}
        self._intensity_map: dict[Path, IntensityFeatures] = {}
        self._clusters: list[ClusterResult] = []

        # ── Split pane: cluster panel (left) + track list (right) ────────────
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.set_position(280)
        paned.set_shrink_start_child(False)
        paned.set_shrink_end_child(False)

        self.cluster_panel = ClusterViewPanel()
        self.cluster_panel.set_size_request(220, -1)
        self.cluster_panel.connect("cluster-selected", self._on_cluster_selected)
        paned.set_start_child(self.cluster_panel)

        self.track_list = TrackListWidget()
        self.track_list.connect("preview-requested", self._on_preview_requested)
        paned.set_end_child(self.track_list)

        toolbar_view.set_content(paned)
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

            # Sequence each cluster (default: ramp)
            sequencer = Sequencer()
            for cluster in self._clusters:
                cluster.tracks = sequencer.sequence(
                    cluster, self._metadata_map, self._intensity_map, mode="ramp"
                )

            GLib.idle_add(self._on_cluster_complete)
        except Exception:
            logger.exception("Error during clustering")
            GLib.idle_add(self._on_cluster_error)

    def _on_cluster_complete(self) -> bool:
        """Update UI with clustering results."""
        self._spinner.stop()
        self._cluster_btn.set_sensitive(True)

        stats = ClusterStats.from_results(self._clusters)
        self.cluster_panel.load_clusters(stats)

        self._track_title = f"Playchitect — {len(self._clusters)} clusters"
        self.set_title(self._track_title)
        return False

    def _on_cluster_error(self) -> bool:
        self._spinner.stop()
        self._cluster_btn.set_sensitive(True)
        self.set_title("Playchitect — clustering failed")
        return False

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
            )
            cluster_tracks.append(model)

        self.track_list.load_tracks(cluster_tracks)
        self.track_list._search_entry.set_text("")
        self.set_title(f"Playchitect — Cluster {cluster_id}")
