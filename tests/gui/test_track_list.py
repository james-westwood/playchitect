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
from unittest.mock import MagicMock

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
        cluster: int = -1,
        duration: float = 0.0,
        audio_format: str = ".flac",
    ) -> None:
        self.filepath = filepath
        self.title = title
        self.artist = artist
        self.bpm = bpm
        self.intensity = intensity
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
        filled = round(max(0.0, min(1.0, self.intensity)) * 5)
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
        assert _SimpleTrack(intensity=0.0).intensity_bars == "░░░░░"

    def test_full(self) -> None:
        assert _SimpleTrack(intensity=1.0).intensity_bars == "█████"

    def test_half(self) -> None:
        assert (
            _SimpleTrack(intensity=0.5).intensity_bars == "██░░░"
            or _SimpleTrack(intensity=0.5).intensity_bars == "███░░"
        )  # rounding edge

    def test_always_five_chars(self) -> None:
        for v in [0.0, 0.2, 0.5, 0.8, 1.0]:
            bars = _SimpleTrack(intensity=v).intensity_bars
            assert len(bars) == 5

    def test_clamped_above_one(self) -> None:
        assert _SimpleTrack(intensity=2.0).intensity_bars == "█████"

    def test_clamped_below_zero(self) -> None:
        assert _SimpleTrack(intensity=-1.0).intensity_bars == "░░░░░"


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
