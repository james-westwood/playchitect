"""Shared GTK/GObject mock setup for all GUI tests.

Installed before any test module in tests/gui/ is imported, so both
test_gui_app.py and test_track_list.py see a consistent mock regardless of
collection order.  Keeps individual test files free of module-level sys.modules
manipulation.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock


class _FakeGObject:
    """Minimal GObject.Object stand-in that supports real Python subclassing."""

    def __init__(self, **_kwargs: object) -> None:
        pass


def _property(*, type: object, default: object = None) -> object:  # noqa: A002
    """GObject.Property stub — returns the default value as a class attribute."""
    return default


class _FakeValue:
    """Fake value object for drag-and-drop data."""

    def get_int(self) -> int:
        return 0


_gobject_mock = MagicMock()
_gobject_mock.Object = _FakeGObject
_gobject_mock.Property = _property
_gobject_mock.SignalFlags = MagicMock()
_gobject_mock.SignalFlags.RUN_FIRST = 0


class _FakeGtkBase:
    """Real base class so GTK widget subclasses remain proper Python classes.

    Provides stub implementations of common GObject/GTK/Adw methods so that
    monkeypatch.setattr can replace them in tests.
    """

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def run(self, *_args: object) -> int:
        return 0

    def present(self) -> None:
        pass

    def set_icon_name(self, *_args: object) -> None:
        pass

    def set_tooltip_text(self, *_args: object) -> None:
        pass

    def set_menu_model(self, *_args: object) -> None:
        pass

    def set_sidebar_width_fraction(self, *_args: object) -> None:
        pass

    def set_sidebar(self, *_args: object) -> None:
        pass

    def set_content(self, *_args: object) -> None:
        pass

    def add_top_bar(self, *_args: object) -> None:
        pass

    def connect(self, *_args: object, **_kwargs: object) -> int:
        return 0

    def set_title(self, *_args: object) -> None:
        pass

    def set_default_size(self, *_args: object) -> None:
        pass

    def pack_start(self, *_args: object) -> None:
        pass

    def pack_end(self, *_args: object) -> None:
        pass

    def set_size_request(self, *_args: object) -> None:
        pass

    def set_has_frame(self, *_args: object) -> None:
        pass

    def set_margin_start(self, *_args: object) -> None:
        pass

    def set_margin_top(self, *_args: object) -> None:
        pass

    def set_margin_bottom(self, *_args: object) -> None:
        pass

    def set_margin_end(self, *_args: object) -> None:
        pass

    def set_vexpand(self, *_args: object) -> None:
        pass

    def set_hexpand(self, *_args: object) -> None:
        pass

    def set_modal(self, *_args: object) -> None:
        pass

    def set_destroy_with_parent(self, *_args: object) -> None:
        pass

    def set_transient_for(self, *_args: object) -> None:
        pass

    def add_css_class(self, *_args: object) -> None:
        pass

    def remove_css_class(self, *_args: object) -> None:
        pass

    def set_sensitive(self, *_args: object) -> None:
        pass

    def set_pixel_size(self, *_args: object) -> None:
        pass

    def set_xalign(self, *_args: object) -> None:
        pass

    def set_subtitle(self, *_args: object) -> None:
        pass

    def add_suffix(self, *_args: object) -> None:
        pass

    def set_show_row_separators(self, *_args: object) -> None:
        pass

    def set_show_column_separators(self, *_args: object) -> None:
        pass

    def append_column(self, *_args: object) -> None:
        pass

    def set_reorderable(self, *_args: object) -> None:
        pass

    def get_sorter(self, *_args: object) -> Any:
        return None

    def add_controller(self, *_args: object) -> None:
        pass

    def set_key_capture_widget(self, *_args: object) -> None:
        pass

    def set_search_mode_enabled(self, *_args: object) -> None:
        pass

    def set_placeholder_text(self, *_args: object) -> None:
        pass

    def set_child(self, *_args: object) -> None:
        pass

    def add(self, *_args: object) -> None:
        pass

    def set_selection_mode(self, *_args: object) -> None:
        pass

    def select_row(self, *_args: object) -> None:
        pass

    def get_row_at_index(self, *_args: object) -> None:
        return None

    def get_index(self, *_args: object) -> int:
        return 0

    def set_valign(self, *_args: object) -> None:
        pass

    def set_halign(self, *_args: object) -> None:
        pass

    def set_position(self, *_args: object) -> None:
        pass

    def set_shrink_start_child(self, *_args: object) -> None:
        pass

    def set_shrink_end_child(self, *_args: object) -> None:
        pass

    def set_start_child(self, *_args: object) -> None:
        pass

    def set_end_child(self, *_args: object) -> None:
        pass

    def set_visible_child_name(self, *_args: object) -> None:
        pass

    def add_titled(self, *_args: object) -> None:
        pass

    def append(self, *_args: object) -> None:
        pass

    def emit(self, *_args: object, **_kwargs: object) -> None:
        pass

    def set_visible(self, *_args: object) -> None:
        pass

    def get_visible(self, *_args: object) -> bool:
        return True

    def get_item(self, *_args: object) -> Any:
        return None

    def get_first_child(self, *_args: object) -> Any:
        return None

    def get_next_sibling(self, *_args: object) -> Any:
        return None

    def remove(self, *_args: object) -> None:
        pass

    def get_n_items(self, *_args: object) -> int:
        return 0

    def clear(self, *_args: object) -> None:
        pass

    def set_text(self, *_args: object) -> None:
        pass

    def set_markup(self, *_args: object) -> None:
        pass

    def set_ellipsize(self, *_args: object) -> None:
        pass

    def set_opacity(self, *_args: object) -> None:
        pass

    def set_policy(self, *_args: object) -> None:
        pass

    def set_propagate_natural_width(self, *_args: object) -> None:
        pass

    def set_model(self, *_args: object) -> None:
        pass

    def set_fixed_width(self, *_args: object) -> None:
        pass

    def set_expand(self, *_args: object) -> None:
        pass

    def set_expanded(self, *_args: object) -> None:
        pass

    def set_max_width_chars(self, *_args: object) -> None:
        pass

    def set_width_chars(self, *_args: object) -> None:
        pass

    def set_range(self, *_args: object) -> None:
        pass

    def set_draw_value(self, *_args: object) -> None:
        pass

    def set_value(self, *_args: object) -> None:
        pass

    def set_draw_func(self, *_args: object) -> None:
        pass

    def queue_draw(self) -> None:
        pass

    def set_content_fit(self, *_args: object) -> None:
        pass

    def set_fraction(self, *_args: object) -> None:
        pass

    def set_wide_handle(self, *_args: object) -> None:
        pass

    def set_active(self, *_args: object) -> None:
        pass

    def get_active(self) -> bool:
        return False

    def set_group(self, *_args: object) -> None:
        pass

    def set_button(self, *_args: object) -> None:
        pass

    def set_actions(self, *_args: object) -> None:
        pass

    def get_value(self) -> _FakeValue:
        return _FakeValue()

    @classmethod
    def new_from_icon_name(cls, *_args: object) -> _FakeGtkBase:
        return cls()

    @classmethod
    def new(cls, *_args: object, **_kwargs: object) -> _FakeGtkBase:
        return cls()

    @classmethod
    def new_for_value(cls, *_args: object, **_kwargs: object) -> _FakeGtkBase:
        return cls()


_gtk_mock = MagicMock()
_gtk_mock.Box = _FakeGtkBase
_gtk_mock.ColumnView = _FakeGtkBase
_gtk_mock.ColumnViewColumn = _FakeGtkBase
_gtk_mock.Frame = _FakeGtkBase  # ClusterCard base class
_gtk_mock.Frame = _FakeGtkBase  # EnergyBlockCard base class
_gtk_mock.ListBox = _FakeGtkBase
_gtk_mock.ListBoxRow = _FakeGtkBase
_gtk_mock.ListStore = _FakeGtkBase
_gtk_mock.ScrolledWindow = _FakeGtkBase
_gtk_mock.Button = _FakeGtkBase
_gtk_mock.Label = _FakeGtkBase
_gtk_mock.Picture = _FakeGtkBase
_gtk_mock.Image = _FakeGtkBase
_gtk_mock.Scale = _FakeGtkBase
_gtk_mock.Expander = _FakeGtkBase
_gtk_mock.SingleSelection = _FakeGtkBase
_gtk_mock.SignalListItemFactory = _FakeGtkBase
_gtk_mock.ListItem = _FakeGtkBase
_gtk_mock.GestureClick = _FakeGtkBase
_gtk_mock.ProgressBar = _FakeGtkBase
_gtk_mock.DrawingArea = _FakeGtkBase
_gtk_mock.DropTarget = _FakeGtkBase
_gtk_mock.DragSource = _FakeGtkBase
_gtk_mock.Paned = _FakeGtkBase
_gtk_mock.ToggleButton = _FakeGtkBase
_gtk_mock.Separator = _FakeGtkBase
_gtk_mock.SearchBar = _FakeGtkBase
_gtk_mock.SearchEntry = _FakeGtkBase
_gtk_mock.FileDialog = _FakeGtkBase
_gtk_mock.Spinner = _FakeGtkBase
_gtk_mock.PopoverMenu = _FakeGtkBase
_gtk_mock.CustomFilter = _FakeGtkBase
_gtk_mock.FilterListModel = _FakeGtkBase
_gtk_mock.SortListModel = _FakeGtkBase
_gtk_mock.MultiSelection = _FakeGtkBase
_gtk_mock.Ordering = MagicMock()
_gtk_mock.Ordering.SMALLER = -1
_gtk_mock.Ordering.EQUAL = 0
_gtk_mock.Ordering.LARGER = 1
_gtk_mock.EventControllerKey = _FakeGtkBase
_gtk_mock.Align = MagicMock()
_gtk_mock.Align.CENTER = 0
_gtk_mock.PolicyType = MagicMock()
_gtk_mock.PolicyType.AUTOMATIC = 0
_gtk_mock.PolicyType.NEVER = 1
_gtk_mock.Orientation = MagicMock()
_gtk_mock.Orientation.HORIZONTAL = 0
_gtk_mock.Orientation.VERTICAL = 1
_gtk_mock.SelectionMode = MagicMock()
_gtk_mock.SelectionMode.SINGLE = 0
_gtk_mock.ContentFit = MagicMock()
_gtk_mock.ContentFit.COVER = 0

_adw_mock = MagicMock()
_adw_mock.Application = _FakeGtkBase
_adw_mock.ApplicationWindow = _FakeGtkBase
_adw_mock.HeaderBar = _FakeGtkBase
_adw_mock.ToolbarView = _FakeGtkBase
_adw_mock.AlertDialog = _FakeGtkBase
_adw_mock.OverlaySplitView = _FakeGtkBase
_adw_mock.ViewStack = _FakeGtkBase
_adw_mock.ViewStackPage = _FakeGtkBase
_adw_mock.MenuButton = _FakeGtkBase
_adw_mock.PreferencesWindow = _FakeGtkBase
_adw_mock.PreferencesPage = _FakeGtkBase
_adw_mock.PreferencesGroup = _FakeGtkBase
_adw_mock.ActionRow = _FakeGtkBase

_pango_mock = MagicMock()
_pango_mock.EllipsizeMode = MagicMock()
_pango_mock.EllipsizeMode.END = 3

_gi_mod = ModuleType("gi")
_gi_mod.require_version = MagicMock()  # type: ignore[attr-defined]

_repo_mod = ModuleType("gi.repository")
_repo_mod.GObject = _gobject_mock  # type: ignore[attr-defined]
_repo_mod.Gtk = _gtk_mock  # type: ignore[attr-defined]

# Gdk mock with DragAction
_gdk_mock = MagicMock()
_gdk_mock.DragAction = MagicMock()
_gdk_mock.DragAction.MOVE = 1
_gdk_mock.DragAction.COPY = 2
_gdk_mock.KEY_space = 32
_gdk_mock.Rectangle = MagicMock()
_gdk_mock.ModifierType = MagicMock()

_repo_mod.Gdk = _gdk_mock  # type: ignore[attr-defined]
_repo_mod.Gio = MagicMock()  # type: ignore[attr-defined]
_repo_mod.GLib = MagicMock()  # type: ignore[attr-defined]
_repo_mod.Pango = _pango_mock  # type: ignore[attr-defined]
_repo_mod.Adw = _adw_mock  # type: ignore[attr-defined]

sys.modules.setdefault("gi", _gi_mod)
sys.modules.setdefault("gi.repository", _repo_mod)
sys.modules.setdefault("gi.repository.GObject", _gobject_mock)
sys.modules.setdefault("gi.repository.Gtk", _gtk_mock)
sys.modules.setdefault("gi.repository.Gdk", MagicMock())
sys.modules.setdefault("gi.repository.Gio", MagicMock())
sys.modules.setdefault("gi.repository.GLib", MagicMock())
sys.modules.setdefault("gi.repository.Pango", _pango_mock)
sys.modules.setdefault("gi.repository.Adw", _adw_mock)
