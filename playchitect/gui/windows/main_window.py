from __future__ import annotations

import logging
import threading
from pathlib import Path

import gi

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")

from gi.repository import Adw, GLib, Gtk  # type: ignore[unresolved-import]  # noqa: E402

from playchitect.core.audio_scanner import AudioScanner  # noqa: E402
from playchitect.core.metadata_extractor import MetadataExtractor  # noqa: E402
from playchitect.gui.widgets.cluster_view import ClusterViewPanel  # noqa: E402
from playchitect.gui.widgets.track_list import TrackListWidget, TrackModel  # noqa: E402
from playchitect.utils.config import get_config  # noqa: E402

logger = logging.getLogger(__name__)


class PlaychitectWindow(Adw.ApplicationWindow):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)

        self.set_title("Playchitect")
        self.set_default_size(1000, 700)

        header = Adw.HeaderBar()
        self._spinner = Gtk.Spinner()
        header.pack_end(self._spinner)

        toolbar_view = Adw.ToolbarView()
        toolbar_view.add_top_bar(header)

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
        paned.set_end_child(self.track_list)

        toolbar_view.set_content(paned)

        self.set_content(toolbar_view)

        # Load default music directory after the window is shown
        config = get_config()
        music_path = config.get_test_music_path()
        if music_path and music_path.is_dir():
            GLib.idle_add(self._start_scan, music_path)

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
            metadata_map = extractor.extract_batch(audio_files)

            tracks = [
                TrackModel(
                    filepath=str(meta.filepath),
                    title=meta.title or "",
                    artist=meta.artist or "",
                    bpm=meta.bpm or 0.0,
                    duration=meta.duration or 0.0,
                    audio_format=meta.filepath.suffix,
                )
                for meta in metadata_map.values()
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
        self.set_title(f"Playchitect — {len(tracks)} tracks")
        return False

    def _on_scan_error(self) -> bool:
        self._spinner.stop()
        self.set_title("Playchitect — scan failed")
        return False

    def _on_cluster_selected(self, _panel: ClusterViewPanel, cluster_id: object) -> None:
        """Filter the track list to show only tracks in the selected cluster."""
        # cluster_id is emitted as GObject.TYPE_PYOBJECT — cast for the search entry.
        self.track_list._search_entry.set_text("")
        # Re-emit as a track-list cluster filter when clustering is wired in.
        # For now, scroll the track list to show tracks in this cluster by
        # updating the window title so the user knows which cluster is active.
        self.set_title(f"Playchitect — Cluster {cluster_id}")
