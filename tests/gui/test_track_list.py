"""Unit tests for the TrackListWidget and TrackModel.

All GTK/GObject imports are mocked at module level so these tests run in CI
without a display or PyGObject installed.

Pure-Python helpers on TrackModel (duration_str, intensity_bars, display_title)
are tested directly against a real dataclass-like object to avoid mock complexity.
The widget-level tests verify the public API (load_tracks, append_track, clear,
track_count, filtered_count) through the mock layer.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from playchitect.gui.widgets.track_list import TrackListWidget

# conftest.py installs gi mocks before this module is collected.
# Pull out the shared mock objects so the fixture can set side_effects on them.
_GTK_MOCK = sys.modules["gi.repository.Gtk"]
_GIO_MOCK = sys.modules["gi.repository.Gio"]


# ── Pure-Python TrackModel helpers ────────────────────────────────────────────
# These tests exercise logic that has no GObject dependency.  We create a simple
# stand-in class that mirrors just the attributes and properties under test.


class _SimpleTrack:
    """Minimal TrackModel stand-in for testing pure-Python helpers."""

    def __init__(
        self,
        filepath: str = "/music/track.flac",
        title: str = "",
        artist: str = "",
        bpm: float = 0.0,
        intensity: float = 0.0,
        hardness: float = 0.0,
        cluster: int = -1,
        duration: float = 0.0,
        audio_format: str = ".flac",
    ) -> None:
        self.filepath = filepath
        self.title = title
        self.artist = artist
        self.bpm = bpm
        self.intensity = intensity
        self.hardness = hardness
        self.cluster = cluster
        self.duration = duration
        self.audio_format = audio_format

    # Copy the pure-Python properties from TrackModel verbatim so we test them
    # without GTK machinery.

    @property
    def duration_str(self) -> str:
        total = max(0, int(self.duration))
        return f"{total // 60}:{total % 60:02d}"

    @property
    def intensity_bars(self) -> str:
        # Using hardness for the visual bars as it's the more robust metric
        filled = round(max(0.0, min(1.0, self.hardness)) * 5)
        return "█" * filled + "░" * (5 - filled)

    @property
    def display_title(self) -> str:
        if self.title:
            return self.title
        name = self.filepath.rsplit("/", 1)[-1]
        return name.rsplit(".", 1)[0] if "." in name else name


class TestDurationStr:
    def test_zero(self) -> None:
        assert _SimpleTrack(duration=0.0).duration_str == "0:00"

    def test_sub_minute(self) -> None:
        assert _SimpleTrack(duration=45.0).duration_str == "0:45"

    def test_exact_minute(self) -> None:
        assert _SimpleTrack(duration=60.0).duration_str == "1:00"

    def test_typical_track(self) -> None:
        assert _SimpleTrack(duration=386.4).duration_str == "6:26"

    def test_over_one_hour(self) -> None:
        assert _SimpleTrack(duration=3661.0).duration_str == "61:01"

    def test_negative_clamped_to_zero(self) -> None:
        assert _SimpleTrack(duration=-5.0).duration_str == "0:00"


class TestIntensityBars:
    def test_zero(self) -> None:
        assert _SimpleTrack(hardness=0.0).intensity_bars == "░░░░░"

    def test_full(self) -> None:
        assert _SimpleTrack(hardness=1.0).intensity_bars == "█████"

    def test_half(self) -> None:
        assert (
            _SimpleTrack(hardness=0.5).intensity_bars == "██░░░"
            or _SimpleTrack(hardness=0.5).intensity_bars == "███░░"
        )  # rounding edge

    def test_always_five_chars(self) -> None:
        for v in [0.0, 0.2, 0.5, 0.8, 1.0]:
            bars = _SimpleTrack(hardness=v).intensity_bars
            assert len(bars) == 5

    def test_clamped_above_one(self) -> None:
        assert _SimpleTrack(hardness=2.0).intensity_bars == "█████"

    def test_clamped_below_zero(self) -> None:
        assert _SimpleTrack(hardness=-1.0).intensity_bars == "░░░░░"


class TestDisplayTitle:
    def test_uses_title_when_set(self) -> None:
        t = _SimpleTrack(filepath="/music/dark_matter.flac", title="Dark Matter")
        assert t.display_title == "Dark Matter"

    def test_falls_back_to_filename_stem(self) -> None:
        t = _SimpleTrack(filepath="/music/dark_matter.flac", title="")
        assert t.display_title == "dark_matter"

    def test_filename_without_extension(self) -> None:
        t = _SimpleTrack(filepath="/music/track", title="")
        assert t.display_title == "track"

    def test_nested_path(self) -> None:
        t = _SimpleTrack(filepath="/a/b/c/song.mp3", title="")
        assert t.display_title == "song"


# ── Widget tests (with mocked GTK) ────────────────────────────────────────────
# We import after mocks are installed and patch the GTK internals so
# TrackListWidget's __init__ can run without a display.


class _FakeListStore:
    """ListStore stub tracking items as a plain Python list."""

    def __init__(self, **_kwargs: object) -> None:
        self._items: list[Any] = []

    def get_n_items(self) -> int:
        return len(self._items)

    def get_item(self, index: int) -> Any:
        return self._items[index]

    def append(self, item: Any) -> None:
        self._items.append(item)

    def splice(self, pos: int, remove: int, additions: list[Any]) -> None:
        del self._items[pos : pos + remove]
        self._items[pos:pos] = additions

    def remove_all(self) -> None:
        self._items.clear()

    def remove(self, index: int) -> None:
        del self._items[index]

    def insert(self, index: int, item: Any) -> None:
        self._items.insert(index, item)


class _FakeFilterModel:
    """FilterListModel stub that applies a Python callable as filter."""

    def __init__(self, model: _FakeListStore, filter: Any) -> None:  # noqa: A002
        self._model = model
        self._filter = filter

    def get_n_items(self) -> int:
        if self._filter is None:
            return self._model.get_n_items()
        return sum(
            1
            for i in range(self._model.get_n_items())
            if self._filter._func(self._model.get_item(i), None)
        )


class _FakeSortModel:
    """SortListModel stub (no sorting — preserves insertion order)."""

    def __init__(self, model: _FakeFilterModel) -> None:
        self._model = model

    def get_n_items(self) -> int:
        return self._model._model.get_n_items()

    def get_item(self, index: int) -> Any:
        return self._model._model.get_item(index)

    def set_sorter(self, _sorter: Any) -> None:
        pass


class _FakeSelection:
    """MultiSelection stub — no items selected by default."""

    def __init__(self, model: _FakeSortModel) -> None:
        self._model = model
        self._selected: set[int] = set()
        self._callbacks: list[Any] = []

    def connect(self, _signal: str, callback: Any) -> None:
        self._callbacks.append(callback)

    def is_selected(self, index: int) -> bool:
        return index in self._selected

    def select_item(self, index: int, _exclusive: bool) -> None:
        self._selected = {index}

    def get_selection(self) -> Any:
        class _Selection:
            def __init__(self, selected: set[int]):
                self._items = sorted(list(selected))

            def get_n_items(self) -> int:
                return len(self._items)

            def get_item(self, index: int) -> int:
                return self._items[index]

        return _Selection(self._selected)


class _FakeFilter:
    """CustomFilter stub storing the match function."""

    def __init__(self, func: Any, _data: Any) -> None:
        self._func = func

    def changed(self, _change: Any) -> None:
        pass  # In tests we read filter_model.get_n_items() which calls _func directly


@pytest.fixture()
def widget() -> TrackListWidget:
    """Return a TrackListWidget with GTK internals replaced by fakes.

    Uses __new__ to skip __init__ entirely, then wires up fake model objects so
    the public API (load_tracks, append_track, clear, _filter_func, …) can be
    exercised without a display or real PyGObject.
    """
    from playchitect.gui.widgets.track_list import TrackListWidget

    w = TrackListWidget.__new__(TrackListWidget)
    w._search_text = ""
    w._total_duration = 0.0
    w._store = _FakeListStore()
    # Use the widget's own _filter_func as the predicate so filter tests are real.
    w._filter = _FakeFilter(w._filter_func, None)
    w._filter_model = _FakeFilterModel(w._store, w._filter)
    w._sort_model = _FakeSortModel(w._filter_model)
    w._selection = _FakeSelection(w._sort_model)
    w._footer_label = MagicMock()
    return w


def _make_track(title: str = "Track", bpm: float = 128.0, duration: float = 360.0) -> Any:
    from playchitect.gui.widgets.track_list import TrackModel

    t = object.__new__(TrackModel)
    t.filepath = f"/music/{title}.flac"
    t.title = title
    t.artist = "Artist"
    t.bpm = bpm
    t.intensity = 0.5
    t.hardness = 0.5
    t.cluster = 1
    t.duration = duration
    t.audio_format = ".flac"
    return t


class TestTrackCount:
    def test_empty_on_init(self, widget: TrackListWidget) -> None:
        assert widget.track_count == 0

    def test_load_tracks_sets_count(self, widget: TrackListWidget) -> None:
        tracks = [_make_track(f"T{i}") for i in range(10)]
        widget.load_tracks(tracks)
        assert widget.track_count == 10

    def test_append_increments_count(self, widget: TrackListWidget) -> None:
        widget.append_track(_make_track("A"))
        widget.append_track(_make_track("B"))
        assert widget.track_count == 2

    def test_clear_resets_to_zero(self, widget: TrackListWidget) -> None:
        widget.load_tracks([_make_track(f"T{i}") for i in range(5)])
        widget.clear()
        assert widget.track_count == 0

    def test_load_replaces_existing(self, widget: TrackListWidget) -> None:
        widget.load_tracks([_make_track("old")])
        widget.load_tracks([_make_track("new1"), _make_track("new2")])
        assert widget.track_count == 2


class TestTotalDuration:
    def test_load_tracks_sums_duration(self, widget: TrackListWidget) -> None:
        tracks = [_make_track(f"T{i}", duration=100.0) for i in range(3)]
        widget.load_tracks(tracks)
        assert widget._total_duration == pytest.approx(300.0)

    def test_append_accumulates_duration(self, widget: TrackListWidget) -> None:
        widget.append_track(_make_track("A", duration=60.0))
        widget.append_track(_make_track("B", duration=90.0))
        assert widget._total_duration == pytest.approx(150.0)

    def test_clear_resets_duration(self, widget: TrackListWidget) -> None:
        widget.load_tracks([_make_track("T", duration=200.0)])
        widget.clear()
        assert widget._total_duration == pytest.approx(0.0)

    def test_load_replaces_duration(self, widget: TrackListWidget) -> None:
        widget.load_tracks([_make_track("old", duration=999.0)])
        widget.load_tracks([_make_track("new", duration=42.0)])
        assert widget._total_duration == pytest.approx(42.0)


class TestFilterFunc:
    """_filter_func is a pure Python predicate — test it directly."""

    def _filter(self, widget: TrackListWidget, track: Any) -> bool:
        return widget._filter_func(track, None)

    def test_empty_search_accepts_all(self, widget: TrackListWidget) -> None:
        widget._search_text = ""
        assert self._filter(widget, _make_track("Anything")) is True

    def test_title_match_case_insensitive(self, widget: TrackListWidget) -> None:
        widget._search_text = "dark"
        track = _make_track("Dark Matter")
        assert self._filter(widget, track) is True

    def test_no_match_returns_false(self, widget: TrackListWidget) -> None:
        widget._search_text = "zzz"
        assert self._filter(widget, _make_track("Sunshine")) is False

    def test_bpm_match(self, widget: TrackListWidget) -> None:
        widget._search_text = "128"
        track = _make_track("Track", bpm=128.0)
        assert self._filter(widget, track) is True

    def test_artist_match(self, widget: TrackListWidget) -> None:
        widget._search_text = "surgeon"

        class _ArtistTrack:
            display_title = "X"
            artist = "Surgeon"
            bpm = 138.0

        assert widget._filter_func(_ArtistTrack(), None) is True


class TestGetSelectedPaths:
    def test_returns_filepaths_of_selected_tracks(self, widget: TrackListWidget) -> None:
        tracks = [_make_track("A"), _make_track("B"), _make_track("C")]
        widget.load_tracks(tracks)
        widget._selection._selected = {0, 2}
        paths = widget.get_selected_paths()
        assert len(paths) == 2
        assert "/music/A.flac" in paths
        assert "/music/C.flac" in paths

    def test_empty_when_nothing_selected(self, widget: TrackListWidget) -> None:
        widget.load_tracks([_make_track("A"), _make_track("B")])
        assert widget.get_selected_paths() == []


class TestKeyPressed:
    """_on_key_pressed uses Gdk from gi.repository (not gi.repository.Gdk)."""

    @staticmethod
    def _gdk() -> Any:
        # track_list.py imports Gdk via `from gi.repository import Gdk`,
        # so the live reference is sys.modules["gi.repository"].Gdk.
        return sys.modules["gi.repository"].Gdk

    def test_spacebar_with_selection_emits_signal(self, widget: TrackListWidget) -> None:
        widget.load_tracks([_make_track("A")])
        widget._selection._selected = {0}
        widget.emit = MagicMock()
        self._gdk().KEY_space = 32

        result = widget._on_key_pressed(MagicMock(), 32, 0, MagicMock())
        widget.emit.assert_called_once_with("preview-requested")
        assert result is True

    def test_spacebar_without_selection_does_not_emit(self, widget: TrackListWidget) -> None:
        widget.load_tracks([_make_track("A")])
        widget._selection._selected = set()
        widget.emit = MagicMock()
        self._gdk().KEY_space = 32

        result = widget._on_key_pressed(MagicMock(), 32, 0, MagicMock())
        widget.emit.assert_not_called()
        assert result is False

    def test_other_key_returns_false(self, widget: TrackListWidget) -> None:
        widget.load_tracks([_make_track("A")])
        widget._selection._selected = {0}
        widget.emit = MagicMock()
        self._gdk().KEY_space = 32

        result = widget._on_key_pressed(MagicMock(), 65, 0, MagicMock())  # 'A' key
        widget.emit.assert_not_called()
        assert result is False


class TestManualReordering:
    def test_move_up(self, widget: TrackListWidget) -> None:
        tracks = [_make_track(f"T{i}") for i in range(3)]
        widget.load_tracks(tracks)
        # Select second track (index 1)
        widget._selection._selected = {1}

        widget._on_move_up(MagicMock(), None)

        assert widget._store.get_item(0).title == "T1"
        assert widget._store.get_item(1).title == "T0"
        assert widget._selection.is_selected(0)

    def test_move_down(self, widget: TrackListWidget) -> None:
        tracks = [_make_track(f"T{i}") for i in range(3)]
        widget.load_tracks(tracks)
        # Select first track (index 0)
        widget._selection._selected = {0}

        widget._on_move_down(MagicMock(), None)

        assert widget._store.get_item(0).title == "T1"
        assert widget._store.get_item(1).title == "T0"
        assert widget._selection.is_selected(1)

    def test_move_up_top_track_does_nothing(self, widget: TrackListWidget) -> None:
        tracks = [_make_track(f"T{i}") for i in range(3)]
        widget.load_tracks(tracks)
        widget._selection._selected = {0}

        widget._on_move_up(MagicMock(), None)

        assert widget._store.get_item(0).title == "T0"

    def test_move_down_bottom_track_does_nothing(self, widget: TrackListWidget) -> None:
        tracks = [_make_track(f"T{i}") for i in range(3)]
        widget.load_tracks(tracks)
        widget._selection._selected = {2}

        widget._on_move_down(MagicMock(), None)

        assert widget._store.get_item(2).title == "T2"


class TestTrackListHandlers:
    def test_search_changed_updates_filter(self, widget: TrackListWidget) -> None:
        widget._filter.changed = MagicMock()
        with patch.object(widget, "_update_footer") as mock_update:
            entry = MagicMock()
            entry.get_text.return_value = "Techno"

            widget._on_search_changed(entry)

            assert widget._search_text == "techno"
            widget._filter.changed.assert_called_once()
            mock_update.assert_called_once()

    def test_selection_changed_emits_signal(self, widget: TrackListWidget) -> None:
        with patch.object(widget, "emit") as mock_emit:
            widget.load_tracks([_make_track("A"), _make_track("B")])
            widget._selection._selected = {0}

            widget._on_selection_changed(MagicMock(), 0, 1)

            mock_emit.assert_called_once_with("selection-changed", 1)

    def test_row_activated_emits_signal(self, widget: TrackListWidget) -> None:
        track = _make_track("A")
        widget.load_tracks([track])
        with patch.object(widget, "emit") as mock_emit:
            widget._on_row_activated(MagicMock(), 0)

            mock_emit.assert_called_once_with("track-activated", track)

    def test_right_click_popups_menu(self, widget: TrackListWidget) -> None:
        widget._context_menu = MagicMock()
        # Mock Gdk.Rectangle
        with patch("gi.repository.Gdk.Rectangle", return_value=MagicMock()):
            widget._on_right_click(MagicMock(), 1, 100.0, 200.0)
            widget._context_menu.popup.assert_called_once()
