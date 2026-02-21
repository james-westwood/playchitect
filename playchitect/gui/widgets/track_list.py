"""Track list widget using Gtk.ColumnView with sorting, filtering, and multi-selection.

Architecture (GTK4 model chain):
    Gio.ListStore → Gtk.FilterListModel → Gtk.SortListModel → Gtk.MultiSelection → Gtk.ColumnView

Gtk.ColumnView handles virtual scrolling natively — only visible rows are rendered,
so 10k+ tracks render in <100ms.
"""

from __future__ import annotations

from typing import Any

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import (  # type: ignore[unresolved-import]  # noqa: E402
    Gdk,
    Gio,
    GObject,
    Gtk,
    Pango,
)


class TrackModel(GObject.Object):
    """GObject backing model for a single track row.

    All fields are GObject.Property so GTK4's model machinery can observe them.
    Pure Python helpers (duration_str, intensity_bars) have no GObject dependency.
    """

    __gtype_name__ = "TrackModel"

    filepath = GObject.Property(type=str, default="")
    title = GObject.Property(type=str, default="")
    artist = GObject.Property(type=str, default="")
    bpm = GObject.Property(type=float, default=0.0)
    intensity = GObject.Property(type=float, default=0.0)
    hardness = GObject.Property(type=float, default=0.0)
    cluster = GObject.Property(type=int, default=-1)
    duration = GObject.Property(type=float, default=0.0)  # seconds
    audio_format = GObject.Property(type=str, default="")  # ".flac", ".mp3", …

    def __init__(
        self,
        filepath: str,
        title: str = "",
        artist: str = "",
        bpm: float = 0.0,
        intensity: float = 0.0,
        hardness: float = 0.0,
        cluster: int = -1,
        duration: float = 0.0,
        audio_format: str = "",
    ) -> None:
        super().__init__()
        self.filepath = filepath
        self.title = title
        self.artist = artist
        self.bpm = bpm
        self.intensity = intensity
        self.hardness = hardness
        self.cluster = cluster
        self.duration = duration
        self.audio_format = audio_format

    @property
    def duration_str(self) -> str:
        """Format duration seconds as M:SS (e.g. 386.4 → '6:26')."""
        total = max(0, int(self.duration))
        return f"{total // 60}:{total % 60:02d}"

    @property
    def intensity_bars(self) -> str:
        """Five-character unicode bar representing hardness in [0, 1]."""
        # Using hardness for the visual bars as it's the more robust metric
        filled = round(max(0.0, min(1.0, self.hardness)) * 5)
        return "█" * filled + "░" * (5 - filled)

    @property
    def display_title(self) -> str:
        """Title, falling back to the filename stem."""
        if self.title:
            return self.title
        # Avoid importing Path at module level just for the fallback
        name = self.filepath.rsplit("/", 1)[-1]
        return name.rsplit(".", 1)[0] if "." in name else name


# ── Column factories ─────────────────────────────────────────────────────────


def _make_label_factory(
    bind_fn: type[Gtk.SignalListItemFactory],
) -> Gtk.SignalListItemFactory:
    """Return a factory whose bind callback is *bind_fn*."""
    factory = Gtk.SignalListItemFactory()
    factory.connect("setup", _setup_label)
    factory.connect("bind", bind_fn)
    factory.connect("unbind", _unbind_label)
    return factory


def _setup_label(_factory: Gtk.SignalListItemFactory, item: Gtk.ListItem) -> None:
    item.set_child(Gtk.Label(xalign=0.0))


def _unbind_label(_factory: Gtk.SignalListItemFactory, item: Gtk.ListItem) -> None:
    label = item.get_child()
    if isinstance(label, Gtk.Label):
        label.set_text("")


def _compare_tracks(attr: str) -> object:
    """Return a CustomSorter compare-function that sorts TrackModel by *attr*."""

    def _cmp(a: TrackModel, b: TrackModel, _data: object) -> Gtk.Ordering:
        va = getattr(a, attr)
        vb = getattr(b, attr)
        if va < vb:
            return Gtk.Ordering.SMALLER
        if va > vb:
            return Gtk.Ordering.LARGER
        return Gtk.Ordering.EQUAL

    return _cmp


# ── Main widget ──────────────────────────────────────────────────────────────


class TrackListWidget(Gtk.Box):
    """Track list widget embedding a sortable, filterable Gtk.ColumnView.

    Signals:
        track-activated(TrackModel): emitted on double-click / Enter
        selection-changed(int):      emitted when selection count changes
    """

    __gsignals__ = {
        "track-activated": (GObject.SignalFlags.RUN_FIRST, None, (GObject.Object,)),
        "selection-changed": (GObject.SignalFlags.RUN_FIRST, None, (int,)),
        "preview-requested": (GObject.SignalFlags.RUN_FIRST, None, ()),
    }

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # ── Model chain ──────────────────────────────────────────────────────
        self._store = Gio.ListStore(item_type=TrackModel)
        self._total_duration: float = 0.0

        self._filter = Gtk.CustomFilter.new(self._filter_func, None)
        self._filter_model = Gtk.FilterListModel(model=self._store, filter=self._filter)

        self._sort_model = Gtk.SortListModel(model=self._filter_model)

        self._selection = Gtk.MultiSelection(model=self._sort_model)
        self._selection.connect("selection-changed", self._on_selection_changed)

        # ── Build UI ─────────────────────────────────────────────────────────
        self._search_text = ""
        self._build_search_bar()
        self.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))
        self._build_column_view()
        self.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))
        self._build_footer()
        self._build_context_menu()

    # ── Search bar ───────────────────────────────────────────────────────────

    def _build_search_bar(self) -> None:
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        box.set_margin_start(6)
        box.set_margin_end(6)
        box.set_margin_top(6)
        box.set_margin_bottom(6)

        self._search_entry = Gtk.SearchEntry()
        self._search_entry.set_placeholder_text("Search tracks…")
        self._search_entry.set_hexpand(True)
        self._search_entry.connect("search-changed", self._on_search_changed)

        box.append(self._search_entry)
        self.append(box)

    # ── Column view ──────────────────────────────────────────────────────────

    def _build_column_view(self) -> None:
        scroll = Gtk.ScrolledWindow()
        scroll.set_vexpand(True)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._column_view = Gtk.ColumnView(model=self._selection)
        self._column_view.set_show_row_separators(True)
        self._column_view.set_show_column_separators(False)
        self._column_view.set_reorderable(True)
        self._column_view.add_css_class("data-table")
        self._column_view.connect("activate", self._on_row_activated)

        # Right-click for context menu
        right_click = Gtk.GestureClick()
        right_click.set_button(3)
        right_click.connect("pressed", self._on_right_click)
        self._column_view.add_controller(right_click)

        # Spacebar → preview-requested signal
        key_ctrl = Gtk.EventControllerKey()
        key_ctrl.connect("key-pressed", self._on_key_pressed)
        self._column_view.add_controller(key_ctrl)

        self._add_columns()

        # Wire the ColumnView's composite sorter into the sort model so that
        # clicking column headers automatically re-sorts the list.
        self._sort_model.set_sorter(self._column_view.get_sorter())

        scroll.set_child(self._column_view)
        self.append(scroll)

    def _add_columns(self) -> None:
        col_specs: list[tuple[str, int, bool, str | None]] = [
            # (header, width, resizable, sort_attr)
            ("♪", 40, False, None),
            ("Title", 220, True, "title"),
            ("Artist", 160, True, "artist"),
            ("BPM", 70, True, "bpm"),
            ("Hardness", 100, False, "hardness"),
            ("Cluster", 80, True, "cluster"),
            ("Time", 70, True, "duration"),
        ]
        bind_fns = [
            self._bind_fmt,
            self._bind_title,
            self._bind_artist,
            self._bind_bpm,
            self._bind_intensity,
            self._bind_cluster,
            self._bind_duration,
        ]

        for (header, width, resizable, sort_attr), bind_fn in zip(col_specs, bind_fns):
            factory = Gtk.SignalListItemFactory()
            factory.connect("setup", _setup_label)
            factory.connect("bind", bind_fn)
            factory.connect("unbind", _unbind_label)

            col = Gtk.ColumnViewColumn(title=header, factory=factory)
            col.set_fixed_width(width)
            col.set_resizable(resizable)
            col.set_expand(header == "Title")

            if sort_attr is not None:
                col.set_sorter(Gtk.CustomSorter.new(_compare_tracks(sort_attr), None))

            self._column_view.append_column(col)

    # ── Factory bind callbacks ────────────────────────────────────────────────

    def _bind_fmt(self, _factory: Gtk.SignalListItemFactory, item: Gtk.ListItem) -> None:
        track: TrackModel = item.get_item()
        label: Gtk.Label = item.get_child()
        label.set_xalign(0.5)
        ext = track.audio_format.lstrip(".").upper()
        label.set_text("🎵" if ext else "♪")
        label.set_tooltip_text(ext or "Unknown format")

    def _bind_title(self, _factory: Gtk.SignalListItemFactory, item: Gtk.ListItem) -> None:
        track: TrackModel = item.get_item()
        label: Gtk.Label = item.get_child()
        label.set_text(track.display_title)
        label.set_ellipsize(Pango.EllipsizeMode.END)

    def _bind_artist(self, _factory: Gtk.SignalListItemFactory, item: Gtk.ListItem) -> None:
        track: TrackModel = item.get_item()
        label: Gtk.Label = item.get_child()
        label.set_text(track.artist or "—")
        label.set_ellipsize(Pango.EllipsizeMode.END)

    def _bind_bpm(self, _factory: Gtk.SignalListItemFactory, item: Gtk.ListItem) -> None:
        track: TrackModel = item.get_item()
        label: Gtk.Label = item.get_child()
        label.set_xalign(1.0)
        label.set_text(str(int(track.bpm)) if track.bpm > 0 else "—")

    def _bind_intensity(self, _factory: Gtk.SignalListItemFactory, item: Gtk.ListItem) -> None:
        track: TrackModel = item.get_item()
        label: Gtk.Label = item.get_child()
        label.set_text(track.intensity_bars)
        label.set_tooltip_text(f"Hardness: {track.hardness:.2f}")

    def _bind_cluster(self, _factory: Gtk.SignalListItemFactory, item: Gtk.ListItem) -> None:
        track: TrackModel = item.get_item()
        label: Gtk.Label = item.get_child()
        label.set_xalign(0.5)
        label.set_text(str(track.cluster) if track.cluster >= 0 else "—")

    def _bind_duration(self, _factory: Gtk.SignalListItemFactory, item: Gtk.ListItem) -> None:
        track: TrackModel = item.get_item()
        label: Gtk.Label = item.get_child()
        label.set_xalign(1.0)
        label.set_text(track.duration_str)

    # ── Keyboard shortcuts ────────────────────────────────────────────────────

    def _on_key_pressed(
        self,
        _ctrl: Gtk.EventControllerKey,
        keyval: int,
        _keycode: int,
        _state: Gdk.ModifierType,
    ) -> bool:
        if keyval == Gdk.KEY_space and self.get_selected_tracks():
            self.emit("preview-requested")
            return True
        return False

    # ── Context menu ─────────────────────────────────────────────────────────

    def _build_context_menu(self) -> None:
        menu = Gio.Menu()
        menu.append("Quick Look", "track.preview")
        menu.append("Move Up", "track.move_up")
        menu.append("Move Down", "track.move_down")
        menu.append("Export to M3U…", "track.export")
        menu.append("Remove from list", "track.remove")

        self._context_menu = Gtk.PopoverMenu(menu_model=menu)
        self._context_menu.set_parent(self)
        self._context_menu.set_has_arrow(False)

        # Actions
        group = Gio.SimpleActionGroup()
        self.insert_action_group("track", group)

        action_up = Gio.SimpleAction.new("move_up", None)
        action_up.connect("activate", self._on_move_up)
        group.add_action(action_up)

        action_down = Gio.SimpleAction.new("move_down", None)
        action_down.connect("activate", self._on_move_down)
        group.add_action(action_down)

    def _on_move_up(self, _action: Gio.SimpleAction, _param: Any) -> None:
        # Clear sorting to allow manual reordering
        self._sort_model.set_sorter(None)

        tracks = self.get_selected_tracks()
        if len(tracks) != 1:
            return
        track = tracks[0]

        # Find index in _store
        n = self._store.get_n_items()
        pos = -1
        for i in range(n):
            if self._store.get_item(i) == track:
                pos = i
                break

        if pos > 0:
            self._store.remove(pos)
            self._store.insert(pos - 1, track)
            self._selection.select_item(pos - 1, True)

    def _on_move_down(self, _action: Gio.SimpleAction, _param: Any) -> None:
        # Clear sorting to allow manual reordering
        self._sort_model.set_sorter(None)

        tracks = self.get_selected_tracks()
        if len(tracks) != 1:
            return
        track = tracks[0]

        # Find index in _store
        n = self._store.get_n_items()
        pos = -1
        for i in range(n):
            if self._store.get_item(i) == track:
                pos = i
                break

        if pos >= 0 and pos < self._store.get_n_items() - 1:
            self._store.remove(pos)
            self._store.insert(pos + 1, track)
            self._selection.select_item(pos + 1, True)

    def _on_right_click(
        self,
        _gesture: Gtk.GestureClick,
        _n_press: int,
        x: float,
        y: float,
    ) -> None:
        rect = Gdk.Rectangle()
        rect.x, rect.y, rect.width, rect.height = int(x), int(y), 1, 1
        self._context_menu.set_pointing_to(rect)
        self._context_menu.popup()

    # ── Search / filter ──────────────────────────────────────────────────────

    def _on_search_changed(self, entry: Gtk.SearchEntry) -> None:
        self._search_text = entry.get_text().lower()
        self._filter.changed(Gtk.FilterChange.DIFFERENT)
        self._update_footer()

    def _filter_func(self, item: Any, _user_data: object) -> bool:
        if not self._search_text:
            return True
        needle = self._search_text
        return (
            needle in item.display_title.lower()
            or needle in (item.artist or "").lower()
            or needle in str(int(item.bpm))
        )

    # ── Selection ─────────────────────────────────────────────────────────────

    def _on_selection_changed(
        self,
        _selection: Gtk.MultiSelection,
        _position: int,
        _n_items: int,
    ) -> None:
        count = len(self.get_selected_tracks())
        self.emit("selection-changed", count)

    def _on_row_activated(self, _view: Gtk.ColumnView, position: int) -> None:
        item = self._sort_model.get_item(position)
        if item is not None:
            self.emit("track-activated", item)

    # ── Footer ────────────────────────────────────────────────────────────────

    def _build_footer(self) -> None:
        footer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        footer.set_margin_start(8)
        footer.set_margin_end(8)
        footer.set_margin_top(4)
        footer.set_margin_bottom(4)

        self._footer_label = Gtk.Label()
        self._footer_label.set_xalign(0.0)
        self._footer_label.add_css_class("caption")
        footer.append(self._footer_label)

        self.append(footer)
        self._update_footer()

    def _update_footer(self) -> None:
        visible = self._filter_model.get_n_items()
        total = self._store.get_n_items()

        secs = int(self._total_duration)
        hours, remainder = divmod(secs, 3600)
        minutes = remainder // 60
        dur_str = f"{hours}h {minutes}m" if hours else f"{minutes}m"

        if visible < total:
            text = f"{visible} of {total} tracks • {dur_str} total"
        else:
            text = f"{total} tracks • {dur_str} total"

        self._footer_label.set_text(text)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_tracks(self, tracks: list[TrackModel]) -> None:
        """Replace the entire track list with *tracks*."""
        self._total_duration = sum(t.duration for t in tracks)
        self._store.splice(0, self._store.get_n_items(), tracks)
        self._update_footer()

    def append_track(self, track: TrackModel) -> None:
        """Append a single track to the list."""
        self._total_duration += track.duration
        self._store.append(track)
        self._update_footer()

    def clear(self) -> None:
        """Remove all tracks."""
        self._total_duration = 0.0
        self._store.remove_all()
        self._update_footer()

    def get_selected_tracks(self) -> list[TrackModel]:
        """Return a list of currently selected TrackModel objects."""
        n = self._sort_model.get_n_items()
        return [self._sort_model.get_item(i) for i in range(n) if self._selection.is_selected(i)]

    @property
    def track_count(self) -> int:
        """Total tracks in the store (unfiltered)."""
        return self._store.get_n_items()

    @property
    def filtered_count(self) -> int:
        """Number of tracks visible after applying the current search filter."""
        return self._filter_model.get_n_items()

    def get_selected_paths(self) -> list[str]:
        """Return filepath strings for all currently selected tracks."""
        return [t.filepath for t in self.get_selected_tracks()]
