"""Preferences window for Playchitect.

Provides user-configurable settings including library path and other preferences.
"""

from __future__ import annotations

from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import (  # type: ignore[unresolved-import]  # noqa: E402
    Adw,
    Gio,
    Gtk,
)

from playchitect.utils.config import get_config  # noqa: E402


class PreferencesWindow(Adw.PreferencesWindow):
    """Preferences window with settings pages.

    Currently implements:
    - General page with Library path and Mixxx database path settings
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

        # Mixxx database path group
        mixxx_group = Adw.PreferencesGroup(title="DJ Software Integration")

        # Mixxx database path entry row
        self._mixxx_db_path_entry = Adw.EntryRow(title="Mixxx database path")
        self._mixxx_db_path_entry.set_text(str(Path.home() / ".mixxx" / "mixxxdb.sqlite"))
        self._mixxx_db_path_entry.set_show_apply_button(True)
        self._mixxx_db_path_entry.connect("apply", self._on_mixxx_db_path_apply)

        # Add browse button for Mixxx DB
        mixxx_browse_button = Gtk.Button(label="Browse…")
        mixxx_browse_button.set_valign(Gtk.Align.CENTER)
        mixxx_browse_button.connect("clicked", self._on_browse_mixxx_db_clicked)
        self._mixxx_db_path_entry.add_suffix(mixxx_browse_button)

        mixxx_group.add(self._mixxx_db_path_entry)
        page.add(mixxx_group)

        self.add(page)

    def _on_browse_library_clicked(self, _button: Gtk.Button) -> None:
        """Handle click on the Browse button for library path."""
        # Placeholder for folder selection dialog
        # Will be implemented in TASK-19 (config persistence)
        self._library_path_row.set_subtitle("/home/user/Music (placeholder)")

    def _on_mixxx_db_path_apply(self, entry: Adw.EntryRow) -> None:
        """Handle applying the Mixxx database path."""
        path_str = entry.get_text()
        config = get_config()
        config.set("mixxx_db_path", path_str)
        config.save()

    def _on_browse_mixxx_db_clicked(self, _button: Gtk.Button) -> None:
        """Handle click on the Browse button for Mixxx database path."""
        dialog = Gtk.FileDialog()
        dialog.set_title("Select Mixxx Database File")

        # Set initial folder from current entry text if valid
        current_path = Path(self._mixxx_db_path_entry.get_text())
        if current_path.exists():
            dialog.set_initial_file(Gio.File.new_for_path(str(current_path)))
        else:
            dialog.set_initial_folder(Gio.File.new_for_path(str(Path.home())))

        dialog.open(
            self,
            None,  # cancellable
            self._on_mixxx_db_file_selected,
            None,  # user_data
        )

    def _on_mixxx_db_file_selected(
        self,
        dialog: Gtk.FileDialog,
        result: Gio.AsyncResult,
        _user_data: object,
    ) -> None:
        """Handle file selection from dialog."""
        try:
            file = dialog.open_finish(result)
            if file is not None:
                path = Path(file.get_path())
                self._mixxx_db_path_entry.set_text(str(path))
                # Save to config
                config = get_config()
                config.set("mixxx_db_path", str(path))
                config.save()
        except Exception:
            # Dialog was cancelled
            pass
