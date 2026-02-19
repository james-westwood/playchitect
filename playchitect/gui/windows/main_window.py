import gi  # type: ignore[import-untyped,unresolved-import]

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")

from gi.repository import Adw  # type: ignore[import-untyped,unresolved-import] # noqa: E402


class PlaychitectWindow(Adw.ApplicationWindow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.set_title("Playchitect")
        self.set_default_size(800, 600)

        header = Adw.HeaderBar()
        toolbar_view = Adw.ToolbarView()
        toolbar_view.add_top_bar(header)
        self.set_content(toolbar_view)
