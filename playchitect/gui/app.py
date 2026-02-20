import gi

# ruff: noqa: E402

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")

from gi.repository import Adw, Gtk  # type: ignore[unresolved-import]

from playchitect.gui.windows.main_window import PlaychitectWindow


class PlaychitectApplication(Adw.Application):
    def __init__(self, **kwargs):
        super().__init__(application_id="com.github.jameswestwood.Playchitect", **kwargs)
        Gtk.Window.set_default_icon_name("com.github.jameswestwood.Playchitect")
        self.connect("activate", self.on_activate)

    def on_activate(self, app):
        self.window = PlaychitectWindow(application=app)
        self.window.present()


def main():
    app = PlaychitectApplication()
    return app.run([])


if __name__ == "__main__":
    main()
