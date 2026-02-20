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


_gtk_mock = MagicMock()
_gtk_mock.Box = _FakeGtkBase
_gtk_mock.ColumnView = _FakeGtkBase

_adw_mock = MagicMock()
_adw_mock.Application = _FakeGtkBase
_adw_mock.ApplicationWindow = _FakeGtkBase
_adw_mock.HeaderBar = _FakeGtkBase
_adw_mock.ToolbarView = _FakeGtkBase

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
_repo_mod.Pango = _pango_mock  # type: ignore[attr-defined]
_repo_mod.Adw = _adw_mock  # type: ignore[attr-defined]

sys.modules.setdefault("gi", _gi_mod)
sys.modules.setdefault("gi.repository", _repo_mod)
sys.modules.setdefault("gi.repository.GObject", _gobject_mock)
sys.modules.setdefault("gi.repository.Gtk", _gtk_mock)
sys.modules.setdefault("gi.repository.Gdk", MagicMock())
sys.modules.setdefault("gi.repository.Gio", MagicMock())
sys.modules.setdefault("gi.repository.Pango", _pango_mock)
sys.modules.setdefault("gi.repository.Adw", _adw_mock)
