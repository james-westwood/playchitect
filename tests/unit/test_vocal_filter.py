"""Unit tests for vocal filter logic in PlaylistsView.

Tests the _apply_vocal_filter and _get_vocal_filter_thresholds methods
without requiring GTK runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

# ── Setup GTK mocks (must be before any GUI imports) ─────────────────────────


class _FakeGObject:
    """Minimal GObject.Object stand-in that supports real Python subclassing."""

    def __init__(self, **_kwargs: object) -> None:
        pass


def _property(*, type: object, default: object = None) -> object:  # noqa: A002
    """GObject.Property stub — returns the default value as a class attribute."""
    return default


_gobject_mock = MagicMock()
_gobject_mock.Object = _FakeGObject
_gobject_mock.Property = _property
_gobject_mock.SignalFlags = MagicMock()
_gobject_mock.SignalFlags.RUN_FIRST = 0
_gobject_mock.TYPE_PYOBJECT = object


class _FakeGtkBase:
    """Real base class so GTK widget subclasses remain proper Python classes."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass


_gtk_mock = MagicMock()
_gtk_mock.Box = _FakeGtkBase
_gtk_mock.ActionBar = _FakeGtkBase
_gtk_mock.Button = _FakeGtkBase
_gtk_mock.ToggleButton = _FakeGtkBase
_gtk_mock.SpinButton = _FakeGtkBase
_gtk_mock.DropDown = _FakeGtkBase
_gtk_mock.Scale = _FakeGtkBase
_gtk_mock.Switch = _FakeGtkBase
_gtk_mock.Spinner = _FakeGtkBase
_gtk_mock.Label = _FakeGtkBase
_gtk_mock.Paned = _FakeGtkBase
_gtk_mock.ListBox = _FakeGtkBase
_gtk_mock.ListBoxRow = _FakeGtkBase
_gtk_mock.ScrolledWindow = _FakeGtkBase
_gtk_mock.Separator = _FakeGtkBase
_gtk_mock.ColumnView = _FakeGtkBase
_gtk_mock.StringList = MagicMock()
_gtk_mock.Align = MagicMock()
_gtk_mock.Align.CENTER = 0
_gtk_mock.Orientation = MagicMock()
_gtk_mock.Orientation.HORIZONTAL = 0
_gtk_mock.Orientation.VERTICAL = 1
_gtk_mock.PolicyType = MagicMock()
_gtk_mock.PolicyType.NEVER = 0
_gtk_mock.PolicyType.AUTOMATIC = 1
_gtk_mock.SelectionMode = MagicMock()
_gtk_mock.SelectionMode.SINGLE = 0
_gtk_mock.Justification = MagicMock()
_gtk_mock.Justification.CENTER = 0
_gtk_mock.PositionType = MagicMock()
_gtk_mock.PositionType.RIGHT = 0
_gtk_mock.FilterChange = MagicMock()
_gtk_mock.FilterChange.DIFFERENT = 0
_gtk_mock.Ordering = MagicMock()
_gtk_mock.Ordering.SMALLER = -1
_gtk_mock.Ordering.EQUAL = 0
_gtk_mock.Ordering.LARGER = 1
_gtk_mock.CustomFilter = MagicMock()
_gtk_mock.FilterListModel = _FakeGtkBase
_gtk_mock.SortListModel = _FakeGtkBase
_gtk_mock.MultiSelection = _FakeGtkBase
_gtk_mock.ColumnViewColumn = _FakeGtkBase
_gtk_mock.SignalListItemFactory = _FakeGtkBase
_gtk_mock.CustomSorter = MagicMock()
_gtk_mock.SearchEntry = _FakeGtkBase
_gtk_mock.GestureClick = _FakeGtkBase
_gtk_mock.EventControllerKey = _FakeGtkBase
_gtk_mock.DrawingArea = _FakeGtkBase
_gtk_mock.PopoverMenu = _FakeGtkBase
_gtk_mock.SimpleActionGroup = MagicMock()
_gtk_mock.SimpleAction = MagicMock()
_gtk_mock.GestureLongPress = _FakeGtkBase
_gtk_mock.Entry = _FakeGtkBase

_gdk_mock = MagicMock()
_gdk_mock.KEY_space = 32
_gdk_mock.ModifierType = MagicMock()
_gdk_mock.Rectangle = MagicMock()

_gio_mock = MagicMock()
_gio_mock.ListStore = _FakeGtkBase
_gio_mock.Menu = MagicMock()
_glib_mock = MagicMock()

_pango_mock = MagicMock()
_pango_mock.EllipsizeMode = MagicMock()
_pango_mock.EllipsizeMode.END = 3

_adw_mock = MagicMock()

_gi_mod = ModuleType("gi")
_gi_mod.require_version = MagicMock()  # ty: ignore[unresolved-attribute]

_repo_mod = ModuleType("gi.repository")
_repo_mod.GObject = _gobject_mock  # ty: ignore[unresolved-attribute]
_repo_mod.Gtk = _gtk_mock  # ty: ignore[unresolved-attribute]
_repo_mod.Gdk = _gdk_mock  # ty: ignore[unresolved-attribute]
_repo_mod.Gio = _gio_mock  # ty: ignore[unresolved-attribute]
_repo_mod.GLib = _glib_mock  # ty: ignore[unresolved-attribute]
_repo_mod.Pango = _pango_mock  # ty: ignore[unresolved-attribute]
_repo_mod.Adw = _adw_mock  # ty: ignore[unresolved-attribute]

sys.modules.setdefault("gi", _gi_mod)
sys.modules.setdefault("gi.repository", _repo_mod)
sys.modules.setdefault("gi.repository.GObject", _gobject_mock)
sys.modules.setdefault("gi.repository.Gtk", _gtk_mock)
sys.modules.setdefault("gi.repository.Gdk", _gdk_mock)
sys.modules.setdefault("gi.repository.Gio", _gio_mock)
sys.modules.setdefault("gi.repository.GLib", _glib_mock)
sys.modules.setdefault("gi.repository.Pango", _pango_mock)
sys.modules.setdefault("gi.repository.Adw", _adw_mock)

# Now we can import the PlaylistsView  # noqa: E402
from playchitect.gui.views.playlists_view import PlaylistsView  # noqa: E402

# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_view_with_vocal_state(
    instrumental_active: bool = False,
    vocal_active: bool = False,
) -> PlaylistsView:
    """Return a PlaylistsView with mocked vocal filter buttons."""
    view = PlaylistsView.__new__(PlaylistsView)
    view._vocal_btn_any = MagicMock()
    view._vocal_btn_instrumental = MagicMock()
    view._vocal_btn_vocal = MagicMock()

    # Configure button states
    view._vocal_btn_any.get_active.return_value = not instrumental_active and not vocal_active
    view._vocal_btn_instrumental.get_active.return_value = instrumental_active
    view._vocal_btn_vocal.get_active.return_value = vocal_active

    return view


def _make_intensity_features(
    vocal_presence: float = 0.0,
    intro_length_secs: float = 0.0,
) -> MagicMock:
    """Create a mock IntensityFeatures object."""
    features = MagicMock()
    features.vocal_presence = vocal_presence
    features.intro_length_secs = intro_length_secs
    return features


# ── TestVocalFilterThresholds ───────────────────────────────────────────────


class TestVocalFilterThresholds:
    """Tests for _get_vocal_filter_thresholds method."""

    def test_any_filter_returns_none(self):
        """Test that 'Any' filter returns None (no filtering)."""
        view = _make_view_with_vocal_state(instrumental_active=False, vocal_active=False)

        result = view._get_vocal_filter_thresholds()

        assert result is None

    def test_instrumental_filter_returns_thresholds(self):
        """Test that 'Instrumental' filter returns (0.0, 0.3)."""
        view = _make_view_with_vocal_state(instrumental_active=True, vocal_active=False)

        result = view._get_vocal_filter_thresholds()

        assert result == (0.0, 0.3)

    def test_vocal_filter_returns_thresholds(self):
        """Test that 'Vocal' filter returns (0.6, 1.0)."""
        view = _make_view_with_vocal_state(instrumental_active=False, vocal_active=True)

        result = view._get_vocal_filter_thresholds()

        assert result == (0.6, 1.0)


# ── TestApplyVocalFilter ────────────────────────────────────────────────────


class TestApplyVocalFilter:
    """Tests for _apply_vocal_filter method."""

    def test_any_filter_returns_all_tracks(self):
        """Test that 'Any' filter returns all tracks unchanged."""
        view = _make_view_with_vocal_state(instrumental_active=False, vocal_active=False)

        # Create test data
        path1 = Path("/test/track1.mp3")
        path2 = Path("/test/track2.mp3")
        path3 = Path("/test/track3.mp3")

        metadata_map = {
            path1: MagicMock(),
            path2: MagicMock(),
            path3: MagicMock(),
        }
        intensity_map = {
            path1: _make_intensity_features(vocal_presence=0.1),
            path2: _make_intensity_features(vocal_presence=0.5),
            path3: _make_intensity_features(vocal_presence=0.9),
        }

        filtered_metadata, filtered_intensity = view._apply_vocal_filter(
            metadata_map,  # ty: ignore[invalid-argument-type]
            intensity_map,
        )

        assert len(filtered_metadata) == 3
        assert len(filtered_intensity) == 3
        assert path1 in filtered_metadata
        assert path2 in filtered_metadata
        assert path3 in filtered_metadata

    def test_instrumental_filter_returns_low_vocal_tracks(self):
        """Test that 'Instrumental' filter returns only tracks with vocal < 0.3."""
        view = _make_view_with_vocal_state(instrumental_active=True, vocal_active=False)

        # Create test data with various vocal_presence values
        path1 = Path("/test/track1.mp3")  # vocal = 0.1 (< 0.3, keep)
        path2 = Path("/test/track2.mp3")  # vocal = 0.3 (= 0.3, keep)
        path3 = Path("/test/track3.mp3")  # vocal = 0.5 (> 0.3, discard)
        path4 = Path("/test/track4.mp3")  # vocal = 0.8 (> 0.3, discard)

        metadata_map = {
            path1: MagicMock(),
            path2: MagicMock(),
            path3: MagicMock(),
            path4: MagicMock(),
        }
        intensity_map = {
            path1: _make_intensity_features(vocal_presence=0.1),
            path2: _make_intensity_features(vocal_presence=0.3),
            path3: _make_intensity_features(vocal_presence=0.5),
            path4: _make_intensity_features(vocal_presence=0.8),
        }

        filtered_metadata, filtered_intensity = view._apply_vocal_filter(
            metadata_map,  # ty: ignore[invalid-argument-type]
            intensity_map,
        )

        assert len(filtered_metadata) == 2
        assert path1 in filtered_metadata
        assert path2 in filtered_metadata
        assert path3 not in filtered_metadata
        assert path4 not in filtered_metadata

    def test_vocal_filter_returns_high_vocal_tracks(self):
        """Test that 'Vocal' filter returns only tracks with vocal > 0.6."""
        view = _make_view_with_vocal_state(instrumental_active=False, vocal_active=True)

        # Create test data with various vocal_presence values
        path1 = Path("/test/track1.mp3")  # vocal = 0.1 (< 0.6, discard)
        path2 = Path("/test/track2.mp3")  # vocal = 0.5 (< 0.6, discard)
        path3 = Path("/test/track3.mp3")  # vocal = 0.6 (= 0.6, keep)
        path4 = Path("/test/track4.mp3")  # vocal = 0.9 (> 0.6, keep)

        metadata_map = {
            path1: MagicMock(),
            path2: MagicMock(),
            path3: MagicMock(),
            path4: MagicMock(),
        }
        intensity_map = {
            path1: _make_intensity_features(vocal_presence=0.1),
            path2: _make_intensity_features(vocal_presence=0.5),
            path3: _make_intensity_features(vocal_presence=0.6),
            path4: _make_intensity_features(vocal_presence=0.9),
        }

        filtered_metadata, filtered_intensity = view._apply_vocal_filter(
            metadata_map,  # ty: ignore[invalid-argument-type]
            intensity_map,
        )

        assert len(filtered_metadata) == 2
        assert path1 not in filtered_metadata
        assert path2 not in filtered_metadata
        assert path3 in filtered_metadata
        assert path4 in filtered_metadata

    def test_filter_handles_missing_intensity_data(self):
        """Test that tracks without intensity data are excluded from filtered results."""
        view = _make_view_with_vocal_state(instrumental_active=True, vocal_active=False)

        path1 = Path("/test/track1.mp3")  # Has intensity
        path2 = Path("/test/track2.mp3")  # Missing intensity

        metadata_map = {
            path1: MagicMock(),
            path2: MagicMock(),
        }
        intensity_map = {
            path1: _make_intensity_features(vocal_presence=0.1),
            # path2 missing from intensity_map
        }

        filtered_metadata, filtered_intensity = view._apply_vocal_filter(
            metadata_map,  # ty: ignore[invalid-argument-type]
            intensity_map,
        )

        assert len(filtered_metadata) == 1
        assert path1 in filtered_metadata
        assert path2 not in filtered_metadata

    def test_filter_handles_missing_vocal_presence_attribute(self):
        """Test that tracks with intensity but no vocal_presence attribute default to 0.0."""
        view = _make_view_with_vocal_state(instrumental_active=True, vocal_active=False)

        path1 = Path("/test/track1.mp3")

        metadata_map = {path1: MagicMock()}
        # Create intensity without vocal_presence attribute
        intensity_without_vocal = MagicMock()
        del intensity_without_vocal.vocal_presence  # Ensure attribute doesn't exist
        intensity_map = {path1: intensity_without_vocal}

        filtered_metadata, filtered_intensity = view._apply_vocal_filter(
            metadata_map,  # ty: ignore[invalid-argument-type]
            intensity_map,
        )

        # vocal_presence defaults to 0.0, which is < 0.3, so track is kept
        assert len(filtered_metadata) == 1
        assert path1 in filtered_metadata


# ── TestSortByIntro ─────────────────────────────────────────────────────────


class TestSortByIntro:
    """Tests for _sort_by_intro method."""

    def test_sort_ascending_short_intro_first(self):
        """Test ascending sort returns tracks ordered by increasing intro length."""
        view = PlaylistsView.__new__(PlaylistsView)
        view._intensity_map = {
            Path("/test/short.mp3"): _make_intensity_features(intro_length_secs=5.0),
            Path("/test/medium.mp3"): _make_intensity_features(intro_length_secs=15.0),
            Path("/test/long.mp3"): _make_intensity_features(intro_length_secs=30.0),
            Path("/test/none.mp3"): _make_intensity_features(intro_length_secs=0.0),
        }

        tracks = [
            Path("/test/long.mp3"),
            Path("/test/short.mp3"),
            Path("/test/none.mp3"),
            Path("/test/medium.mp3"),
        ]

        result = view._sort_by_intro(tracks, ascending=True)

        assert result[0] == Path("/test/none.mp3")  # 0.0s
        assert result[1] == Path("/test/short.mp3")  # 5.0s
        assert result[2] == Path("/test/medium.mp3")  # 15.0s
        assert result[3] == Path("/test/long.mp3")  # 30.0s

    def test_sort_descending_long_intro_first(self):
        """Test descending sort returns tracks ordered by decreasing intro length."""
        view = PlaylistsView.__new__(PlaylistsView)
        view._intensity_map = {
            Path("/test/short.mp3"): _make_intensity_features(intro_length_secs=5.0),
            Path("/test/medium.mp3"): _make_intensity_features(intro_length_secs=15.0),
            Path("/test/long.mp3"): _make_intensity_features(intro_length_secs=30.0),
            Path("/test/none.mp3"): _make_intensity_features(intro_length_secs=0.0),
        }

        tracks = [
            Path("/test/short.mp3"),
            Path("/test/long.mp3"),
            Path("/test/medium.mp3"),
            Path("/test/none.mp3"),
        ]

        result = view._sort_by_intro(tracks, ascending=False)

        assert result[0] == Path("/test/long.mp3")  # 30.0s
        assert result[1] == Path("/test/medium.mp3")  # 15.0s
        assert result[2] == Path("/test/short.mp3")  # 5.0s
        assert result[3] == Path("/test/none.mp3")  # 0.0s

    def test_sort_handles_missing_intensity_data(self):
        """Test that tracks without intensity data are treated as 0.0 intro length."""
        view = PlaylistsView.__new__(PlaylistsView)
        view._intensity_map = {
            Path("/test/with_data.mp3"): _make_intensity_features(intro_length_secs=10.0),
            # Missing entry for /test/no_data.mp3
        }

        tracks = [
            Path("/test/with_data.mp3"),
            Path("/test/no_data.mp3"),
        ]

        result = view._sort_by_intro(tracks, ascending=True)

        # Missing data defaults to 0.0, so it comes first in ascending order
        assert result[0] == Path("/test/no_data.mp3")
        assert result[1] == Path("/test/with_data.mp3")


# ── TestTrackModelProperties ────────────────────────────────────────────────


class TestTrackModelProperties:
    """Tests for TrackModel structural properties."""

    def test_track_model_has_vocal_presence_property(self):
        """Test that TrackModel has vocal_presence GObject property."""
        from playchitect.gui.widgets.track_list import TrackModel

        model = TrackModel(
            filepath="/test/track.mp3",
            vocal_presence=0.5,
        )
        assert model.vocal_presence == 0.5

    def test_track_model_has_intro_length_property(self):
        """Test that TrackModel has intro_length_secs GObject property."""
        from playchitect.gui.widgets.track_list import TrackModel

        model = TrackModel(
            filepath="/test/track.mp3",
            intro_length_secs=12.5,
        )
        assert model.intro_length_secs == 12.5

    def test_track_model_intro_formatted(self):
        """Test that intro_formatted property returns 'Xs' format."""
        from playchitect.gui.widgets.track_list import TrackModel

        model = TrackModel(
            filepath="/test/track.mp3",
            intro_length_secs=12.7,  # Should be truncated to int
        )
        assert model.intro_formatted == "12s"

    def test_track_model_intro_formatted_zero(self):
        """Test that intro_formatted returns '0s' for zero intro length."""
        from playchitect.gui.widgets.track_list import TrackModel

        model = TrackModel(
            filepath="/test/track.mp3",
            intro_length_secs=0.0,
        )
        assert model.intro_formatted == "0s"
