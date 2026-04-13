import logging
from pathlib import Path

import gi

# ruff: noqa: E402

gi.require_version("Adw", "1")
gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")

from gi.repository import Adw, Gdk, Gio, Gtk  # type: ignore[unresolved-import]

from playchitect.gui.windows.main_window import PlaychitectWindow

logger = logging.getLogger(__name__)


def _load_brand_css() -> None:
    """Load the Playchitect brand CSS for dark theme."""
    provider = Gtk.CssProvider()
    css_path = Path(__file__).parent / "style.css"
    provider.load_from_path(str(css_path))
    display = Gdk.Display.get_default()
    if display is not None:
        Gtk.StyleContext.add_provider_for_display(
            display,
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )


def _ensure_dark_mode() -> None:
    """Ensure dark colour scheme is enforced."""
    style_manager = Adw.StyleManager.get_default()
    if not style_manager.get_dark():
        style_manager.set_color_scheme(Adw.ColorScheme.FORCE_DARK)


def _configure_titlebar_double_click() -> None:
    """Configure titlebar double-click to maximize window (default is toggle-maximize)."""
    settings = Gtk.Settings.get_default()
    if settings is not None:
        settings.set_property("gtk-titlebar-double-click", "toggle-maximize")


class PlaychitectApplication(Adw.Application):
    def __init__(self, **kwargs):
        super().__init__(application_id="com.github.jameswestwood.Playchitect", **kwargs)
        Gtk.Window.set_default_icon_name("com.github.jameswestwood.Playchitect")
        _configure_titlebar_double_click()
        self.connect("activate", self.on_activate)

        # Register menu actions
        self.register_action("open-folder", self._on_open_folder)
        self.register_action("preferences", self._on_preferences)

    def register_action(self, name: str, callback) -> None:
        """Register a simple action on the app."""
        action = Gio.SimpleAction.new(name, None)
        action.connect("activate", callback)
        self.add_action(action)

    def _on_open_folder(self, _action: Gio.SimpleAction, _param: object) -> None:
        """Handle Open Folder menu action - show folder picker and rescan."""
        if not hasattr(self, "window") or self.window is None:
            logger.warning("No window available for folder selection")
            return

        dialog = Gtk.FileDialog()
        dialog.set_title("Select Music Folder")

        # Set initial folder from config
        from playchitect.utils.config import get_config

        config = get_config()
        music_path = config.get_test_music_path()
        if music_path and music_path.is_dir():
            dialog.set_initial_folder(Gio.File.new_for_path(str(music_path)))
        else:
            dialog.set_initial_folder(Gio.File.new_for_path(str(Path.home())))

        dialog.open(
            self.window,
            None,
            self._on_folder_selected,
            None,
        )

    def _on_folder_selected(
        self,
        dialog: Gtk.FileDialog,
        result: Gio.AsyncResult,
        _user_data: object,
    ) -> None:
        """Handle folder selection from dialog."""
        try:
            file = dialog.open_finish(result)
            if file is not None:
                folder_path = Path(file.get_path())
                logger.info("Selected folder: %s", folder_path)
                # Save to config
                from playchitect.utils.config import get_config

                config = get_config()
                config.set("test_music_path", str(folder_path))
                config.save()
                # Trigger rescan in window
                if hasattr(self, "window") and self.window is not None:
                    self.window.rescan_library(folder_path)
        except Exception:
            # Dialog was cancelled
            pass

    def _on_preferences(self, _action: Gio.SimpleAction, _param: object) -> None:
        """Handle Preferences menu action."""
        if hasattr(self, "window") and self.window is not None:
            self.window.show_preferences()

    def on_activate(self, app):
        self.window = PlaychitectWindow(application=app)
        self.window.present()


def main():
    _ensure_dark_mode()
    _load_brand_css()
    app = PlaychitectApplication()
    return app.run([])


if __name__ == "__main__":
    main()
