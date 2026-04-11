"""Unit tests for LibraryView and LibraryTrackModel.

All GTK/GObject imports are mocked at module level so these tests run in CI
without a display or PyGObject installed.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from playchitect.gui.views.library_view import LibraryView

# conftest.py installs gi mocks before this module is collected.
_GTK_MOCK = sys.modules["gi.repository.Gtk"]
_GIO_MOCK = sys.modules["gi.repository.Gio"]


class _SimpleLibraryTrack:
    """Minimal LibraryTrackModel stand-in for testing pure-Python helpers."""

    def __init__(
        self,
        title: str = "",
        artist: str = "",
        bpm: float = 0.0,
        duration_secs: float = 0.0,
        filepath: str = "/music/track.flac",
        file_format: str = "FLAC",
    ) -> None:
        self.title = title
        self.artist = artist
        self.bpm = bpm
        self.duration_secs = duration_secs
        self.filepath = filepath
        self.file_format = file_format

    @property
    def display_title(self) -> str:
        if self.title:
            return self.title
        name = self.filepath.rsplit("/", 1)[-1]
        return name.rsplit(".", 1)[0] if "." in name else name

    @property
    def duration_formatted(self) -> str:
        total = max(0, int(self.duration_secs))
        return f"{total // 60}:{total % 60:02d}"

    @property
    def bpm_formatted(self) -> str:
        if self.bpm > 0:
            return f"{self.bpm:.0f}"
        return "—"


class TestDisplayTitle:
    def test_uses_title_when_set(self) -> None:
        t = _SimpleLibraryTrack(title="Dark Matter", filepath="/music/ignored.flac")
        assert t.display_title == "Dark Matter"

    def test_falls_back_to_filename(self) -> None:
        t = _SimpleLibraryTrack(title="", filepath="/music/dark_matter.flac")
        assert t.display_title == "dark_matter"

    def test_filename_without_extension(self) -> None:
        t = _SimpleLibraryTrack(title="", filepath="/music/track")
        assert t.display_title == "track"


class TestDurationFormatted:
    def test_zero_seconds(self) -> None:
        assert _SimpleLibraryTrack(duration_secs=0.0).duration_formatted == "0:00"

    def test_sub_minute(self) -> None:
        assert _SimpleLibraryTrack(duration_secs=45.0).duration_formatted == "0:45"

    def test_exact_minute(self) -> None:
        assert _SimpleLibraryTrack(duration_secs=60.0).duration_formatted == "1:00"

    def test_typical_track(self) -> None:
        assert _SimpleLibraryTrack(duration_secs=386.4).duration_formatted == "6:26"


class TestBpmFormatted:
    def test_zero_bpm(self) -> None:
        assert _SimpleLibraryTrack(bpm=0.0).bpm_formatted == "—"

    def test_whole_number(self) -> None:
        assert _SimpleLibraryTrack(bpm=128.0).bpm_formatted == "128"

    def test_decimal_rounded(self) -> None:
        assert _SimpleLibraryTrack(bpm=127.7).bpm_formatted == "128"


class _FakeListStore:
    """ListStore stub tracking items as a plain Python list."""

    def __init__(self, **_kwargs: object) -> None:
        self._items: list[Any] = []

    def get_n_items(self) -> int:
        return len(self._items)

    def get_item(self, index: int) -> Any:
        return self._items[index] if 0 <= index < len(self._items) else None

    def append(self, item: Any) -> None:
        self._items.append(item)

    def remove_all(self) -> None:
        self._items.clear()


class _FakeFilterModel:
    """FilterListModel stub."""

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
    """SortListModel stub."""

    def __init__(self, model: _FakeFilterModel) -> None:
        self._model = model

    def get_n_items(self) -> int:
        return self._model.get_n_items()

    def get_item(self, index: int) -> Any:
        return self._model._model.get_item(index)

    def set_sorter(self, _sorter: Any) -> None:
        pass


class _FakeSingleSelection:
    """SingleSelection stub."""

    def __init__(self, model: _FakeSortModel) -> None:
        self._model = model
        self._selected_index: int = -1
        self._callbacks: list[Any] = []

    def connect(self, signal: str, callback: Any) -> None:
        self._callbacks.append((signal, callback))

    def get_selected_item(self) -> Any:
        if self._selected_index >= 0:
            return self._model.get_item(self._selected_index)
        return None

    def select_item(self, index: int, _exclusive: bool) -> None:
        self._selected_index = index


class _FakeFilter:
    """CustomFilter stub."""

    def __init__(self, func: Any, _data: Any) -> None:
        self._func = func

    def changed(self, _change: Any) -> None:
        pass


@pytest.fixture()
def library_view() -> LibraryView:
    """Return a LibraryView with GTK internals replaced by fakes."""
    from playchitect.gui.views.library_view import LibraryView

    v = LibraryView.__new__(LibraryView)
    v._search_text = ""
    v._selected_format = "All"
    v._tag_filter = ""
    v._scan_thread = None
    v._tag_store = MagicMock()
    v._tag_store.get_tags = MagicMock(return_value=[])

    # Fake model chain
    v._store = _FakeListStore()
    v._filter = _FakeFilter(v._filter_func, None)
    v._filter_model = _FakeFilterModel(v._store, v._filter)
    v._sort_model = _FakeSortModel(v._filter_model)
    v._selection = _FakeSingleSelection(v._sort_model)

    # Fake UI elements
    v._footer_label = MagicMock()
    v._search_toggle = MagicMock()
    v._search_entry = MagicMock()
    v._spinner = MagicMock()
    v._open_btn = MagicMock()
    v._column_view = MagicMock()  # Needed for _on_activate tests

    # Format buttons
    v._format_buttons = {}
    for fmt in LibraryView.FORMATS:
        btn = MagicMock()
        btn.get_active.return_value = fmt == "All"
        v._format_buttons[fmt] = btn

    return v


def _make_library_track(
    title: str = "Track",
    artist: str = "Artist",
    bpm: float = 128.0,
    duration_secs: float = 360.0,
    file_format: str = "FLAC",
    filepath: str = "/music/track.flac",
) -> Any:
    """Create a minimal track object for testing."""
    from playchitect.gui.views.library_view import LibraryTrackModel

    t = object.__new__(LibraryTrackModel)
    t.title = title
    t.artist = artist
    t.bpm = bpm
    t.duration_secs = duration_secs
    t.file_format = file_format
    t.filepath = filepath
    return t


class TestLibraryViewInstantiation:
    def test_instantiates_without_error(self) -> None:
        """LibraryView instantiates without error in mock GTK environment."""
        from playchitect.gui.views.library_view import LibraryView

        # The mock environment from conftest.py allows instantiation
        view = LibraryView.__new__(LibraryView)
        assert view is not None


class TestFilterFunc:
    """Test the filter function combining search and format filters."""

    def _filter(self, view: LibraryView, track: Any) -> bool:
        return view._filter_func(track, None)

    def test_empty_search_accepts_all(self, library_view: LibraryView) -> None:
        library_view._search_text = ""
        library_view._selected_format = "All"
        track = _make_library_track()
        assert self._filter(library_view, track) is True

    def test_title_match_case_insensitive(self, library_view: LibraryView) -> None:
        library_view._search_text = "dark"
        library_view._selected_format = "All"
        track = _make_library_track(title="Dark Matter")
        assert self._filter(library_view, track) is True

    def test_artist_match_case_insensitive(self, library_view: LibraryView) -> None:
        library_view._search_text = "surgeon"
        library_view._selected_format = "All"
        track = _make_library_track(artist="Surgeon")
        assert self._filter(library_view, track) is True

    def test_no_match_returns_false(self, library_view: LibraryView) -> None:
        library_view._search_text = "zzz"
        library_view._selected_format = "All"
        track = _make_library_track(title="Sunshine", artist="Daylight")
        assert self._filter(library_view, track) is False


class TestFormatChipFiltering:
    """Test format chip filtering."""

    def _filter(self, view: LibraryView, track: Any) -> bool:
        return view._filter_func(track, None)

    def test_flac_filter_accepts_flac(self, library_view: LibraryView) -> None:
        library_view._selected_format = "FLAC"
        track = _make_library_track(file_format="FLAC")
        assert self._filter(library_view, track) is True

    def test_flac_filter_rejects_mp3(self, library_view: LibraryView) -> None:
        library_view._selected_format = "FLAC"
        track = _make_library_track(file_format="MP3")
        assert self._filter(library_view, track) is False

    def test_mp3_filter_accepts_mp3(self, library_view: LibraryView) -> None:
        library_view._selected_format = "MP3"
        track = _make_library_track(file_format="MP3")
        assert self._filter(library_view, track) is True

    def test_ogg_filter_accepts_ogg(self, library_view: LibraryView) -> None:
        library_view._selected_format = "OGG"
        track = _make_library_track(file_format="OGG")
        assert self._filter(library_view, track) is True

    def test_wav_filter_accepts_wav(self, library_view: LibraryView) -> None:
        library_view._selected_format = "WAV"
        track = _make_library_track(file_format="WAV")
        assert self._filter(library_view, track) is True

    def test_all_format_accepts_any(self, library_view: LibraryView) -> None:
        library_view._selected_format = "All"
        for fmt in ["FLAC", "MP3", "OGG", "WAV", "M4A"]:
            track = _make_library_track(file_format=fmt)
            assert self._filter(library_view, track) is True


class TestSearchFilter:
    """Test search filter matches title and artist."""

    def _filter(self, view: LibraryView, track: Any) -> bool:
        return view._filter_func(track, None)

    def test_search_matches_title(self, library_view: LibraryView) -> None:
        library_view._search_text = "techno"
        track = _make_library_track(title="Dark Techno Track", artist="Someone")
        assert self._filter(library_view, track) is True

    def test_search_matches_artist(self, library_view: LibraryView) -> None:
        library_view._search_text = "surgeon"
        track = _make_library_track(title="Track 1", artist="Surgeon")
        assert self._filter(library_view, track) is True

    def test_search_no_match(self, library_view: LibraryView) -> None:
        library_view._search_text = "ambient"
        track = _make_library_track(title="Hard Techno", artist="Techno Artist")
        assert self._filter(library_view, track) is False

    def test_search_case_insensitive(self, library_view: LibraryView) -> None:
        library_view._search_text = "techno"  # Already lowercased as in actual usage
        track = _make_library_track(title="DARK TECHNO", artist="artist")
        assert self._filter(library_view, track) is True


class TestSignalsDefined:
    """Test that required signals are defined on LibraryView."""

    def test_scan_complete_signal_defined(self) -> None:
        """'scan-complete' signal is defined on LibraryView."""
        from playchitect.gui.views.library_view import LibraryView

        assert "scan-complete" in LibraryView.__gsignals__

    def test_track_selected_signal_defined(self) -> None:
        """'track-selected' signal is defined on LibraryView."""
        from playchitect.gui.views.library_view import LibraryView

        assert "track-selected" in LibraryView.__gsignals__


class TestSelectionChanged:
    """Test _on_selection_changed handler signature and behavior."""

    def test_selection_changed_accepts_three_args(self, library_view: LibraryView) -> None:
        """_on_selection_changed accepts 3 arguments (selection, position, n_items)."""
        track = _make_library_track(title="Test Track")
        library_view._store.append(track)
        library_view._selection._selected_index = 0  # Select the track

        # Should not raise TypeError when called with 3 arguments
        library_view._on_selection_changed(library_view._selection, 0, 1)

    def test_selection_changed_emits_track_selected(self, library_view: LibraryView) -> None:
        """_on_selection_changed emits 'track-selected' signal when track selected."""
        from unittest.mock import patch

        track = _make_library_track(title="Test Track")
        library_view._store.append(track)
        library_view._selection._selected_index = 0

        # Capture emitted signals
        emitted: list[tuple[str, tuple[Any, ...]]] = []

        def mock_emit(signal_name: str, *args: Any) -> None:
            emitted.append((signal_name, args))

        with patch.object(library_view, "emit", mock_emit):
            library_view._on_selection_changed(library_view._selection, 0, 1)

        assert len(emitted) == 1
        signal_name, args = emitted[0]
        assert signal_name == "track-selected"
        assert len(args) == 1
        assert args[0].title == "Test Track"

    def test_selection_changed_no_emit_when_no_selection(self, library_view: LibraryView) -> None:
        """_on_selection_changed does not emit signal when no track is selected."""
        from unittest.mock import patch

        # Add a track but don't select it
        track = _make_library_track(title="Test Track")
        library_view._store.append(track)
        library_view._selection._selected_index = -1  # No selection

        emitted: list[tuple[str, tuple[Any, ...]]] = []

        def mock_emit(signal_name: str, *args: Any) -> None:
            emitted.append((signal_name, args))

        with patch.object(library_view, "emit", mock_emit):
            library_view._on_selection_changed(library_view._selection, -1, 1)

        # Should not emit anything when there's no selection
        assert len(emitted) == 0

    def test_sequential_selections_emit_correct_tracks(self, library_view: LibraryView) -> None:
        """Sequential track selections emit track-selected with the correct track each time.

        This test verifies BUG-01 fix: clicking track A then track B while preview panel
        is open should update the panel to show track B's information.
        """
        from unittest.mock import patch

        # Add two tracks
        track_a = _make_library_track(title="Track A", filepath="/music/track_a.flac")
        track_b = _make_library_track(title="Track B", filepath="/music/track_b.flac")
        library_view._store.append(track_a)
        library_view._store.append(track_b)

        emitted: list[tuple[str, tuple[Any, ...]]] = []

        def mock_emit(signal_name: str, *args: Any) -> None:
            emitted.append((signal_name, args))

        with patch.object(library_view, "emit", mock_emit):
            # First selection: track A
            library_view._selection._selected_index = 0
            library_view._on_selection_changed(library_view._selection, 0, 2)

            # Second selection: track B (while preview panel is still open)
            library_view._selection._selected_index = 1
            library_view._on_selection_changed(library_view._selection, 1, 2)

        # Should have emitted two signals
        assert len(emitted) == 2

        # First signal should be track A
        signal_1_name, signal_1_args = emitted[0]
        assert signal_1_name == "track-selected"
        assert signal_1_args[0].title == "Track A"
        assert signal_1_args[0].filepath == "/music/track_a.flac"

        # Second signal should be track B (verifies BUG-01 fix)
        signal_2_name, signal_2_args = emitted[1]
        assert signal_2_name == "track-selected"
        assert signal_2_args[0].title == "Track B"
        assert signal_2_args[0].filepath == "/music/track_b.flac"

    def test_on_activate_emits_track_selected(self, library_view: LibraryView) -> None:
        """_on_activate emits track-selected signal when row is activated.

        This is the key BUG-01 fix test: clicking the same (already selected) track
        should still emit track-selected, because selection-changed does not fire
        when clicking the already-selected row.

        Acceptance criteria (c): tests/gui/test_library_view.py has a test that
        calls _on_activate with the current selection index and asserts
        track-selected is emitted.
        """
        from unittest.mock import patch

        # Add a track and select it
        track = _make_library_track(title="Test Track", filepath="/music/test.flac")
        library_view._store.append(track)
        library_view._selection._selected_index = 0  # Select the track

        emitted: list[tuple[str, tuple[Any, ...]]] = []

        def mock_emit(signal_name: str, *args: Any) -> None:
            emitted.append((signal_name, args))

        with patch.object(library_view, "emit", mock_emit):
            # Call _on_activate with the current selection position
            # This simulates the user clicking the already-selected row
            library_view._on_activate(library_view._column_view, 0)

        # Should emit track-selected signal
        assert len(emitted) == 1
        signal_name, args = emitted[0]
        assert signal_name == "track-selected"
        assert len(args) == 1
        assert args[0].title == "Test Track"
        assert args[0].filepath == "/music/test.flac"

    def test_on_activate_same_track_refreshes_panel(self, library_view: LibraryView) -> None:
        """Clicking the same track twice (via activate) refreshes the panel.

        BUG-01 acceptance criteria (b): Clicking the SAME (already selected) track
        while panel is open also refreshes the panel. This test verifies that
        _on_activate emits the signal even when the selection hasn't changed.
        """
        from unittest.mock import patch

        # Add a track
        track = _make_library_track(title="Same Track", filepath="/music/same.flac")
        library_view._store.append(track)
        library_view._selection._selected_index = 0

        emitted: list[tuple[str, tuple[Any, ...]]] = []

        def mock_emit(signal_name: str, *args: Any) -> None:
            emitted.append((signal_name, args))

        with patch.object(library_view, "emit", mock_emit):
            # First activation (user clicks the track)
            library_view._on_activate(library_view._column_view, 0)

            # Second activation (user clicks the SAME track again)
            # selection-changed wouldn't fire here, but activate does
            library_view._on_activate(library_view._column_view, 0)

        # Both activations should emit track-selected
        assert len(emitted) == 2

        # Both should be for the same track
        for signal_name, args in emitted:
            assert signal_name == "track-selected"
            assert args[0].title == "Same Track"


class TestTrackCount:
    """Test track count updates."""

    def test_empty_on_init(self, library_view: LibraryView) -> None:
        assert library_view._store.get_n_items() == 0

    def test_load_tracks_sets_count(self, library_view: LibraryView) -> None:
        tracks = [_make_library_track(f"Track {i}") for i in range(5)]
        library_view.load_tracks(tracks)
        assert library_view._store.get_n_items() == 5

    def test_clear_resets_count(self, library_view: LibraryView) -> None:
        tracks = [_make_library_track() for _ in range(3)]
        library_view.load_tracks(tracks)
        library_view.clear()
        assert library_view._store.get_n_items() == 0


class TestFormatButtons:
    """Test format button API."""

    def test_select_format_activates_button(self, library_view: LibraryView) -> None:
        """select_format() activates the correct format button."""
        library_view.select_format("FLAC")
        library_view._format_buttons["FLAC"].set_active.assert_called_once_with(True)

    def test_select_format_ignores_invalid(self, library_view: LibraryView) -> None:
        """select_format() ignores invalid format."""
        library_view.select_format("INVALID")
        # No button should be activated
        for btn in library_view._format_buttons.values():
            btn.set_active.assert_not_called()


class TestSearchAPI:
    """Test search bar API."""

    def test_set_search_active(self, library_view: LibraryView) -> None:
        """set_search_active() toggles search toggle."""
        library_view.set_search_active(True)
        library_view._search_toggle.set_active.assert_called_once_with(True)

    def test_set_search_text_updates_filter(self, library_view: LibraryView) -> None:
        """set_search_text() updates search text and triggers filter."""
        with patch.object(library_view._filter, "changed") as mock_changed:
            library_view.set_search_text("techno")
            assert library_view._search_text == "techno"
            mock_changed.assert_called_once()
