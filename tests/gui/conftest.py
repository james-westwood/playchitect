"""Shared GTK/GObject mock setup for all GUI tests.

Installed before any test module in tests/gui/ is imported, so both
test_gui_app.py and test_track_list.py see a consistent mock regardless of
collection order.  Keeps individual test files free of module-level sys.modules
manipulation.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock


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

    def set_child(self, *_args: object) -> None:
        pass

    def set_visible_child_name(self, *_args: object) -> None:
        pass

    def add_titled(self, *_args: object) -> None:
        pass

    def append(self, *_args: object) -> None:
        pass

    def emit(self, *_args: object, **_kwargs: object) -> None:
        pass


_gtk_mock = MagicMock()
_gtk_mock.Box = _FakeGtkBase
_gtk_mock.ColumnView = _FakeGtkBase
_gtk_mock.Frame = _FakeGtkBase  # ClusterCard base class
_gtk_mock.ListBox = _FakeGtkBase
_gtk_mock.ListBoxRow = _FakeGtkBase

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
_repo_mod.Gdk = MagicMock()  # type: ignore[attr-defined]
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
