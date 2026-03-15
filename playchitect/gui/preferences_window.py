"""Preferences window for Playchitect.

Provides user-configurable settings including library path and other preferences.
"""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import (  # type: ignore[unresolved-import]  # noqa: E402
    Adw,
    Gtk,
)


class PreferencesWindow(Adw.PreferencesWindow):
    """Preferences window with settings pages.

    Currently implements:
    - General page with Library path setting
    """

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)

        self.set_title("Preferences")
        self.set_default_size(600, 400)
        self.set_modal(True)
        self.set_destroy_with_parent(True)

        # Build pages
        self._build_general_page()

    def _build_general_page(self) -> None:
        """Build the General preferences page."""
        page = Adw.PreferencesPage(title="General", icon_name="applications-system-symbolic")

        # Library path group
        library_group = Adw.PreferencesGroup(title="Library")

        # Library path row
        self._library_path_row = Adw.ActionRow(title="Library path")
        self._library_path_row.set_subtitle("Path to your music library folder")

        # Add a button to browse for folder (placeholder)
        browse_button = Gtk.Button(label="Browse…")
        browse_button.set_valign(Gtk.Align.CENTER)
        browse_button.connect("clicked", self._on_browse_library_clicked)
        self._library_path_row.add_suffix(browse_button)

        library_group.add(self._library_path_row)
        page.add(library_group)

        self.add(page)

    def _on_browse_library_clicked(self, _button: Gtk.Button) -> None:
        """Handle click on the Browse button for library path."""
        # Placeholder for folder selection dialog
        # Will be implemented in TASK-19 (config persistence)
        self._library_path_row.set_subtitle("/home/user/Music (placeholder)")
