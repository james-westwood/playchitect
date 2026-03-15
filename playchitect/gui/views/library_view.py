"""Library view for browsing and managing audio tracks.

Provides a searchable, filterable track list with format chips and folder scanning.
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
    Gio,
    GLib,
    GObject,
    Gtk,
)

from playchitect.core.audio_scanner import AudioScanner  # noqa: E402
from playchitect.core.metadata_extractor import MetadataExtractor  # noqa: E402

logger = logging.getLogger(__name__)


class LibraryTrackModel(GObject.Object):
    """GObject backing model for a library track row.

    Properties:
        title (str): Track title
        artist (str): Track artist
        bpm (float): Beats per minute
        duration_secs (float): Duration in seconds
        filepath (str): Full file path
        file_format (str): Audio format (e.g., "FLAC", "MP3")
    """

    __gtype_name__ = "LibraryTrackModel"

    title = GObject.Property(type=str, default="")
    artist = GObject.Property(type=str, default="")
    bpm = GObject.Property(type=float, default=0.0)
    duration_secs = GObject.Property(type=float, default=0.0)
    filepath = GObject.Property(type=str, default="")
    file_format = GObject.Property(type=str, default="")

    def __init__(
        self,
        title: str = "",
        artist: str = "",
        bpm: float = 0.0,
        duration_secs: float = 0.0,
        filepath: str = "",
        file_format: str = "",
    ) -> None:
        super().__init__()
        self.title = title
        self.artist = artist
        self.bpm = bpm
        self.duration_secs = duration_secs
        self.filepath = filepath
        self.file_format = file_format

    @property
    def display_title(self) -> str:
        """Return title or filename stem as fallback."""
        if self.title:
            return self.title
        name = self.filepath.rsplit("/", 1)[-1]
        return name.rsplit(".", 1)[0] if "." in name else name

    @property
    def duration_formatted(self) -> str:
        """Format duration as M:SS."""
        total = max(0, int(self.duration_secs))
        return f"{total // 60}:{total % 60:02d}"

    @property
    def bpm_formatted(self) -> str:
        """Format BPM as integer string."""
        if self.bpm > 0:
            return f"{self.bpm:.0f}"
        return "—"


def _setup_label(_factory: Gtk.SignalListItemFactory, item: Gtk.ListItem) -> None:
    """Setup callback for label factory."""
    item.set_child(Gtk.Label(xalign=0.0))


def _unbind_label(_factory: Gtk.SignalListItemFactory, item: Gtk.ListItem) -> None:
    """Unbind callback to clear label text."""
    label = item.get_child()
    if isinstance(label, Gtk.Label):
        label.set_text("")


def _compare_tracks(attr: str) -> Any:
    """Return a CustomSorter compare function that sorts by attribute."""

    def _cmp(a: LibraryTrackModel, b: LibraryTrackModel, _data: object) -> Gtk.Ordering:
        va = getattr(a, attr)
        vb = getattr(b, attr)
        if va < vb:
            return Gtk.Ordering.SMALLER
        if va > vb:
            return Gtk.Ordering.LARGER
        return Gtk.Ordering.EQUAL

    return _cmp


class LibraryView(Gtk.Box):
    """Library view widget with track list, search, and format filtering.

    Signals:
        scan-complete(Gio.ListStore): Emitted when scanning finishes
        track-selected(LibraryTrackModel): Emitted when a track is selected
    """

    __gsignals__ = {
        "scan-complete": (GObject.SignalFlags.RUN_FIRST, None, (GObject.Object,)),
        "track-selected": (GObject.SignalFlags.RUN_FIRST, None, (GObject.Object,)),
    }

    # Format chips configuration
    FORMATS = ["All", "FLAC", "MP3", "OGG", "WAV"]

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # State
        self._search_text = ""
        self._selected_format = "All"
        self._scan_thread: threading.Thread | None = None

        # Model chain: ListStore → FilterListModel → SortListModel → Selection
        self._store = Gio.ListStore(item_type=LibraryTrackModel)

        self._filter = Gtk.CustomFilter.new(self._filter_func, None)
        self._filter_model = Gtk.FilterListModel(model=self._store, filter=self._filter)

        self._sort_model = Gtk.SortListModel(model=self._filter_model)
        self._selection = Gtk.SingleSelection(model=self._sort_model)
        self._selection.connect("selection-changed", self._on_selection_changed)

        # Build UI
        self._build_toolbar()
        self._build_format_chips()
        self._build_column_view()
        self._build_footer()

    def _build_toolbar(self) -> None:
        """Build header toolbar with search toggle and open folder button."""
        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        toolbar.set_margin_start(12)
        toolbar.set_margin_end(12)
        toolbar.set_margin_top(12)
        toolbar.set_margin_bottom(6)

        # Open Folder button
        self._open_btn = Gtk.Button()
        self._open_btn.set_icon_name("folder-open-symbolic")
        self._open_btn.set_tooltip_text("Open Folder")
        self._open_btn.connect("clicked", self._on_open_folder_clicked)
        toolbar.append(self._open_btn)

        # Spinner (hidden by default)
        self._spinner = Gtk.Spinner()
        self._spinner.set_visible(False)
        toolbar.append(self._spinner)

        # Search toggle button
        self._search_toggle = Gtk.ToggleButton()
        self._search_toggle.set_icon_name("system-search-symbolic")
        self._search_toggle.set_tooltip_text("Toggle Search")
        self._search_toggle.connect("toggled", self._on_search_toggled)
        toolbar.append(self._search_toggle)

        # SearchBar (hidden by default)
        self._search_bar = Gtk.SearchBar()
        self._search_bar.set_search_mode_enabled(False)
        self._search_bar.set_key_capture_widget(self)

        search_entry = Gtk.SearchEntry()
        search_entry.set_placeholder_text("Search tracks…")
        search_entry.connect("search-changed", self._on_search_changed)
        self._search_entry = search_entry

        self._search_bar.set_child(search_entry)

        # Add toolbar and search bar to main box
        self.append(toolbar)
        self.append(self._search_bar)

    def _build_format_chips(self) -> None:
        """Build format filter chips below the header."""
        chips_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        chips_box.set_margin_start(12)
        chips_box.set_margin_end(12)
        chips_box.set_margin_top(6)
        chips_box.set_margin_bottom(6)

        self._format_buttons: dict[str, Gtk.ToggleButton] = {}

        for i, fmt in enumerate(self.FORMATS):
            btn = Gtk.ToggleButton(label=fmt)
            btn.set_active(i == 0)  # "All" is active by default
            btn.connect("toggled", self._on_format_toggled, fmt)
            if i > 0:
                btn.set_group(list(self._format_buttons.values())[0])
            self._format_buttons[fmt] = btn
            chips_box.append(btn)

        self.append(chips_box)

    def _build_column_view(self) -> None:
        """Build the ColumnView with track columns."""
        scroll = Gtk.ScrolledWindow()
        scroll.set_vexpand(True)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._column_view = Gtk.ColumnView(model=self._selection)
        self._column_view.set_show_row_separators(True)
        self._column_view.set_show_column_separators(False)
        self._column_view.add_css_class("data-table")

        # Define columns: (header, width, sort_attr, bind_callback)
        columns = [
            ("Title", 250, "title", self._bind_title),
            ("Artist", 180, "artist", self._bind_artist),
            ("BPM", 70, "bpm", self._bind_bpm),
            ("Duration", 80, "duration_secs", self._bind_duration),
            ("Format", 70, "file_format", self._bind_format),
        ]

        for header, width, sort_attr, bind_fn in columns:
            factory = Gtk.SignalListItemFactory()
            factory.connect("setup", _setup_label)
            factory.connect("bind", bind_fn)
            factory.connect("unbind", _unbind_label)

            col = Gtk.ColumnViewColumn(title=header, factory=factory)
            col.set_fixed_width(width)
            col.set_resizable(True)
            if header == "Title":
                col.set_expand(True)

            if sort_attr:
                col.set_sorter(Gtk.CustomSorter.new(_compare_tracks(sort_attr), None))

            self._column_view.append_column(col)

        # Wire sorter
        self._sort_model.set_sorter(self._column_view.get_sorter())

        scroll.set_child(self._column_view)
        self.append(scroll)

    def _bind_title(self, _factory: Gtk.SignalListItemFactory, item: Gtk.ListItem) -> None:
        """Bind title column."""
        track: LibraryTrackModel = item.get_item()
        label: Gtk.Label = item.get_child()
        label.set_text(track.display_title)

    def _bind_artist(self, _factory: Gtk.SignalListItemFactory, item: Gtk.ListItem) -> None:
        """Bind artist column."""
        track: LibraryTrackModel = item.get_item()
        label: Gtk.Label = item.get_child()
        label.set_text(track.artist or "—")

    def _bind_bpm(self, _factory: Gtk.SignalListItemFactory, item: Gtk.ListItem) -> None:
        """Bind BPM column with integer formatting."""
        track: LibraryTrackModel = item.get_item()
        label: Gtk.Label = item.get_child()
        label.set_xalign(1.0)
        label.set_text(track.bpm_formatted)

    def _bind_duration(self, _factory: Gtk.SignalListItemFactory, item: Gtk.ListItem) -> None:
        """Bind duration column with M:SS formatting."""
        track: LibraryTrackModel = item.get_item()
        label: Gtk.Label = item.get_child()
        label.set_xalign(1.0)
        label.set_text(track.duration_formatted)

    def _bind_format(self, _factory: Gtk.SignalListItemFactory, item: Gtk.ListItem) -> None:
        """Bind format column."""
        track: LibraryTrackModel = item.get_item()
        label: Gtk.Label = item.get_child()
        label.set_xalign(0.5)
        label.set_text(track.file_format.upper() if track.file_format else "—")

    def _build_footer(self) -> None:
        """Build footer showing track count."""
        footer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        footer.set_margin_start(12)
        footer.set_margin_end(12)
        footer.set_margin_top(6)
        footer.set_margin_bottom(12)

        self._footer_label = Gtk.Label()
        self._footer_label.set_xalign(0.0)
        self._footer_label.add_css_class("caption")
        footer.append(self._footer_label)

        self.append(footer)
        self._update_footer()

    def _update_footer(self) -> None:
        """Update footer label with track count."""
        count = self._filter_model.get_n_items()
        self._footer_label.set_text(f"{count} tracks")

    # ── Event Handlers ─────────────────────────────────────────────────────────

    def _on_search_toggled(self, btn: Gtk.ToggleButton) -> None:
        """Toggle search bar visibility."""
        self._search_bar.set_search_mode_enabled(btn.get_active())

    def _on_search_changed(self, entry: Gtk.SearchEntry) -> None:
        """Handle search text changes."""
        self._search_text = entry.get_text().lower()
        self._filter.changed(Gtk.FilterChange.DIFFERENT)
        self._update_footer()

    def _on_format_toggled(self, btn: Gtk.ToggleButton, fmt: str) -> None:
        """Handle format chip selection."""
        if btn.get_active():
            self._selected_format = fmt
            self._filter.changed(Gtk.FilterChange.DIFFERENT)
            self._update_footer()

    def _on_selection_changed(self, selection: Gtk.SingleSelection, _position: int) -> None:
        """Emit track-selected signal when selection changes."""
        item = selection.get_selected_item()
        if item is not None:
            self.emit("track-selected", item)

    def _on_open_folder_clicked(self, _btn: Gtk.Button) -> None:
        """Open file dialog to select music folder."""
        dialog = Gtk.FileDialog()
        dialog.set_title("Select Music Folder")

        # Set folder mode by not setting any filters
        dialog.select_folder(self.get_root(), None, self._on_folder_selected)

    def _on_folder_selected(self, dialog: Gtk.FileDialog, result: Any) -> None:
        """Handle folder selection from file dialog."""
        try:
            folder = dialog.select_folder_finish(result)
            if folder is not None:
                path = Path(folder.get_path())
                self._start_scan(path)
        except Exception as e:
            logger.error("Failed to select folder: %s", e)

    # ── Scanning ────────────────────────────────────────────────────────────────

    def _start_scan(self, directory: Path) -> None:
        """Start background scanning of directory."""
        self._spinner.set_visible(True)
        self._spinner.start()
        self._open_btn.set_sensitive(False)

        self._scan_thread = threading.Thread(
            target=self._scan_worker, args=(directory,), daemon=True
        )
        self._scan_thread.start()

    def _scan_worker(self, directory: Path) -> None:
        """Worker thread for scanning and extracting metadata."""
        try:
            scanner = AudioScanner()
            audio_files = scanner.scan(directory)
            logger.info("Found %d audio files in %s", len(audio_files), directory)

            extractor = MetadataExtractor()
            metadata_map = extractor.extract_batch(audio_files)

            # Convert to LibraryTrackModel objects
            tracks: list[LibraryTrackModel] = []
            for path, meta in metadata_map.items():
                if meta is None:
                    continue

                ext = path.suffix.lstrip(".").upper()
                track = LibraryTrackModel(
                    title=meta.title or "",
                    artist=meta.artist or "",
                    bpm=meta.bpm or 0.0,
                    duration_secs=meta.duration or 0.0,
                    filepath=str(path),
                    file_format=ext,
                )
                tracks.append(track)

            # Sort by title
            tracks.sort(key=lambda t: t.display_title.lower())

            # Update UI on main thread
            GLib.idle_add(self._on_scan_complete, tracks)
        except Exception:
            logger.exception("Error scanning %s", directory)
            GLib.idle_add(self._on_scan_error)

    def _on_scan_complete(self, tracks: list[LibraryTrackModel]) -> bool:
        """Handle scan completion on main thread."""
        self._store.remove_all()
        for track in tracks:
            self._store.append(track)

        self._spinner.stop()
        self._spinner.set_visible(False)
        self._open_btn.set_sensitive(True)
        self._update_footer()

        # Emit signal with the store
        self.emit("scan-complete", self._store)
        return False  # Don't repeat

    def _on_scan_error(self) -> bool:
        """Handle scan error on main thread."""
        self._spinner.stop()
        self._spinner.set_visible(False)
        self._open_btn.set_sensitive(True)
        self._update_footer()
        return False

    # ── Filter ──────────────────────────────────────────────────────────────────

    def _filter_func(self, item: Any, _user_data: object) -> bool:
        """Filter function combining search and format filters."""
        track: LibraryTrackModel = item

        # Format filter
        if self._selected_format != "All":
            if track.file_format.upper() != self._selected_format.upper():
                return False

        # Search filter (title and artist, case-insensitive)
        if self._search_text:
            search_lower = self._search_text.lower()
            title_match = search_lower in track.display_title.lower()
            artist_match = search_lower in (track.artist or "").lower()
            if not (title_match or artist_match):
                return False

        return True

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_tracks(self, tracks: list[LibraryTrackModel]) -> None:
        """Load tracks into the library view."""
        self._store.remove_all()
        for track in tracks:
            self._store.append(track)
        self._update_footer()

    def clear(self) -> None:
        """Clear all tracks."""
        self._store.remove_all()
        self._update_footer()

    def get_store(self) -> Gio.ListStore:
        """Get the underlying ListStore."""
        return self._store

    def get_selected_track(self) -> LibraryTrackModel | None:
        """Get the currently selected track."""
        return self._selection.get_selected_item()

    def set_search_active(self, active: bool) -> None:
        """Set search bar visibility."""
        self._search_toggle.set_active(active)

    def set_search_text(self, text: str) -> None:
        """Set search text programmatically."""
        self._search_entry.set_text(text)
        self._search_text = text.lower()
        self._filter.changed(Gtk.FilterChange.DIFFERENT)
        self._update_footer()

    def select_format(self, fmt: str) -> None:
        """Select a format filter programmatically."""
        if fmt in self._format_buttons:
            self._format_buttons[fmt].set_active(True)
