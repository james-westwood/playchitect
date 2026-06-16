"""Tests for the "Make playlist" seed button and signal on LibraryView.

These tests define the contract for TASK-10: adding a _make_playlist_btn
to the toolbar and a "make-playlist-seed" signal to LibraryView.

All tests currently FAIL because neither the button attribute nor the
signal exist yet.  Once TASK-10 is implemented:
  - test_make_playlist_button_exists passes (button created in _build_toolbar)
  - test_button_disabled_with_no_selection passes (set_sensitive(False))
  - test_button_enabled_with_one_selection passes (set_sensitive(True))
  - test_dialog_opens_on_click passes (SeedPlaylistDialog.present called)
  - test_signal_make_playlist_seed_exists passes (signal in __gsignals__)
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

# conftest.py installs gi mocks before this module is collected.
_GTK_MOCK = sys.modules["gi.repository.Gtk"]

# ── Patch _FakeGtkBase with stubs that LibraryView.__init__ needs ────────────
# The conftest harness covers most methods, but LibraryView's constructor calls
# a handful that are missing.  Adding them here avoids ImportError / unrelated
# AttributeError failures, ensuring the *real* failure is the missing
# _make_playlist_btn attribute (which TASK-10 will add).

_FakeGtkBase = _GTK_MOCK.Box  # All widget classes inherit from this class

# Stubs missing from conftest.py that LibraryView.__init__ calls.
# Adding them here avoids unrelated AttributeError failures so that
# the *real* test failure is the missing _make_playlist_btn attribute.
for _meth_name in (
    "set_search_mode",       # SearchBar (distinct from set_search_mode_enabled)
    "set_max_content_width", # ScrolledWindow
    "set_resizable",         # ColumnViewColumn
    "set_sorter",            # SortListModel / ColumnViewColumn
):
    if not hasattr(_FakeGtkBase, _meth_name):
        setattr(_FakeGtkBase, _meth_name, lambda *_a, **_k: None)


class TestSeedPlaylistAction:
    """Contract tests for the "Make playlist" seed-button feature (TASK-10)."""

    # ── 1. Button exists ──────────────────────────────────────────────────────

    def test_make_playlist_button_exists(self) -> None:
        """LibraryView has a _make_playlist_btn attribute after construction.

        The button should be created in _build_toolbar() between the preview
        toggle and the preview chip label.
        """
        from playchitect.gui.views.library_view import LibraryView

        view = LibraryView()
        assert hasattr(view, "_make_playlist_btn"), (
            "LibraryView must define _make_playlist_btn in _build_toolbar()"
        )

    # ── 2. Button disabled with no selection ─────────────────────────────────

    def test_button_disabled_with_no_selection(self) -> None:
        """_on_selection_changed disables _make_playlist_btn when n_items is 0.

        With an empty library (no selection), the "Make playlist" button must
        be insensitive so the user cannot click it without a seed track.
        """
        from playchitect.gui.views.library_view import LibraryView

        view = LibraryView()

        # Spy on the real set_sensitive method (defined on _FakeGtkBase)
        with patch.object(
            view._make_playlist_btn,
            "set_sensitive",
            wraps=view._make_playlist_btn.set_sensitive,
        ) as mock_spy:
            view._on_selection_changed(None, 0, 0)
            mock_spy.assert_called_with(False)

    # ── 3. Button enabled with one selection ──────────────────────────────────

    def test_button_enabled_with_one_selection(self) -> None:
        """_on_selection_changed enables _make_playlist_btn when a track is selected.

        When at least one track is selected (n_items > 0), the button must
        become sensitive so the user can seed a playlist from it.
        """
        from playchitect.gui.views.library_view import LibraryView

        view = LibraryView()

        # Simulate a selection existing on the SingleSelection model
        view._selection = MagicMock()
        view._selection.get_selected.return_value = 0

        # Spy on the real set_sensitive method
        with patch.object(
            view._make_playlist_btn,
            "set_sensitive",
            wraps=view._make_playlist_btn.set_sensitive,
        ) as mock_spy:
            view._on_selection_changed(None, 0, 1)
            mock_spy.assert_called_with(True)

    # ── 4. Dialog opens on click ─────────────────────────────────────────────

    def test_dialog_opens_on_click(self) -> None:
        """Clicking _make_playlist_btn opens SeedPlaylistDialog.present().

        The _on_make_playlist_clicked handler must instantiate a
        SeedPlaylistDialog and call .present() on it to show the dialog
        to the user.
        """
        from playchitect.gui.views.library_view import LibraryView

        view = LibraryView()

        with patch(
            "playchitect.gui.views.library_view.SeedPlaylistDialog"
        ) as mock_dialog_cls:
            mock_dialog = mock_dialog_cls.return_value
            view._on_make_playlist_clicked(None)
            mock_dialog.present.assert_called_once()

    # ── 5. Signal make-playlist-seed exists ───────────────────────────────────

    def test_signal_make_playlist_seed_exists(self) -> None:
        """LibraryView declares a 'make-playlist-seed' signal in __gsignals__.

        This signal carries the selected LibraryTrackModel so that other
        components (e.g. the set-builder view) can react to the user
        choosing a seed track.
        """
        from playchitect.gui.views.library_view import LibraryView

        assert "make-playlist-seed" in LibraryView.__gsignals__, (
            "LibraryView must declare a 'make-playlist-seed' signal"
        )
