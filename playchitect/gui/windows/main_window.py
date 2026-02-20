import gi  # type: ignore[import-untyped,unresolved-import]

gi.require_version("Adw", "1")
gi.require_version("Gtk", "4.0")

from gi.repository import Adw  # type: ignore[import-untyped,unresolved-import] # noqa: E402

from playchitect.gui.widgets.track_list import TrackListWidget  # noqa: E402


class PlaychitectWindow(Adw.ApplicationWindow):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)

        self.set_title("Playchitect")
        self.set_default_size(1000, 700)

        header = Adw.HeaderBar()
        toolbar_view = Adw.ToolbarView()
        toolbar_view.add_top_bar(header)

        self.track_list = TrackListWidget()
        toolbar_view.set_content(self.track_list)

        self.set_content(toolbar_view)
