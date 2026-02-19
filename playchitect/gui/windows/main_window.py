import gi  # type: ignore[import-untyped,unresolved-import]
from gi.repository import Adw, Gtk  # type: ignore[import-untyped,unresolved-import]

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")


class PlaychitectWindow(Adw.ApplicationWindow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.set_title("Playchitect")
        self.set_default_size(800, 600)

        header_bar = Gtk.HeaderBar.new()
        self.set_titlebar(header_bar)

        self.set_content(Adw.Bin.new())
