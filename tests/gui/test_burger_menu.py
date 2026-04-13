"""Smoke tests for the burger menu functionality.

gi mocks are installed by tests/gui/conftest.py before this module is collected.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestBurgerMenuActions:
    """Tests for the hamburger menu actions."""

    def test_app_has_open_folder_method(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that application has _on_open_folder method."""
        mock_run = MagicMock(return_value=0)
        monkeypatch.setattr("playchitect.gui.app.PlaychitectApplication.run", mock_run)

        from playchitect.gui.app import PlaychitectApplication

        app = PlaychitectApplication()
        assert hasattr(app, "_on_open_folder")
        assert callable(app._on_open_folder)

    def test_app_has_preferences_method(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that application has _on_preferences method."""
        mock_run = MagicMock(return_value=0)
        monkeypatch.setattr("playchitect.gui.app.PlaychitectApplication.run", mock_run)

        from playchitect.gui.app import PlaychitectApplication

        app = PlaychitectApplication()
        assert hasattr(app, "_on_preferences")
        assert callable(app._on_preferences)

    def test_preferences_method_calls_window_show_preferences(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that _on_preferences calls window.show_preferences."""
        mock_run = MagicMock(return_value=0)
        monkeypatch.setattr("playchitect.gui.app.PlaychitectApplication.run", mock_run)

        from playchitect.gui.app import PlaychitectApplication

        app = PlaychitectApplication()
        mock_window = MagicMock()
        app.window = mock_window

        app._on_preferences(MagicMock(), None)

        mock_window.show_preferences.assert_called_once()


class TestMainWindowMenuActions:
    """Tests for main window menu structure."""

    def test_menu_contains_open_folder_item(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the menu contains Open Folder item."""
        monkeypatch.setattr("playchitect.gui.windows.main_window.Adw.HeaderBar", MagicMock())
        monkeypatch.setattr("playchitect.gui.windows.main_window.Gtk.MenuButton", MagicMock())
        monkeypatch.setattr("playchitect.gui.windows.main_window.Gio.Menu", MagicMock())

        from playchitect.gui.windows.main_window import PlaychitectWindow

        # Check that the menu building code references "Open Folder"
        import inspect

        source = inspect.getsource(PlaychitectWindow._build_menu)
        assert "Open Folder" in source

    def test_menu_contains_preferences_item(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the menu contains Preferences item."""
        monkeypatch.setattr("playchitect.gui.windows.main_window.Adw.HeaderBar", MagicMock())
        monkeypatch.setattr("playchitect.gui.windows.main_window.Gtk.MenuButton", MagicMock())
        monkeypatch.setattr("playchitect.gui.windows.main_window.Gio.Menu", MagicMock())

        from playchitect.gui.windows.main_window import PlaychitectWindow

        import inspect

        source = inspect.getsource(PlaychitectWindow._build_menu)
        assert "Preferences" in source

    def test_menu_does_not_have_unimplemented_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that keyboard-shortcuts and about items are removed from menu."""
        monkeypatch.setattr("playchitect.gui.windows.main_window.Adw.HeaderBar", MagicMock())
        monkeypatch.setattr("playchitect.gui.windows.main_window.Gtk.MenuButton", MagicMock())
        monkeypatch.setattr("playchitect.gui.windows.main_window.Gio.Menu", MagicMock())

        from playchitect.gui.windows.main_window import PlaychitectWindow

        import inspect

        source = inspect.getsource(PlaychitectWindow._build_menu)
        # These items should be removed (they were greyed out)
        assert "keyboard-shortcuts" not in source
        assert "about" not in source


class TestMainWindowRescanMethod:
    """Tests for rescan_library method on main window."""

    def test_main_window_has_rescan_library_method(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that PlaychitectWindow has rescan_library method."""
        # Just check method exists by checking the source
        from playchitect.gui.windows.main_window import PlaychitectWindow

        assert hasattr(PlaychitectWindow, "rescan_library")
        assert callable(PlaychitectWindow.rescan_library)

    def test_rescan_library_calls_start_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that rescan_library calls _start_scan with the path."""
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        # We need to test that rescan_library properly forwards to _start_scan
        # This is a unit test on the method implementation
        with (
            patch("playchitect.gui.windows.main_window.Adw") as mock_adw,
            patch("playchitect.gui.windows.main_window.Gtk") as mock_gtk,
            patch("playchitect.gui.windows.main_window.Gio") as mock_gio,
            patch("playchitect.gui.windows.main_window.GLib") as mock_glib,
            patch("playchitect.gui.windows.main_window.AudioScanner") as mock_scanner,
            patch("playchitect.gui.windows.main_window.MetadataExtractor") as mock_extractor,
            patch("playchitect.gui.windows.main_window.get_config") as mock_config,
            patch(
                "playchitect.gui.windows.main_window.PlaychitectWindow.__init__",
                lambda self, **kw: None,
            ),
        ):
            from playchitect.gui.windows.main_window import PlaychitectWindow

            # Create instance without calling __init__
            window = object.__new__(PlaychitectWindow)
            window._start_scan = MagicMock()

            # Call rescan_library
            test_path = Path("/test/music")
            window.rescan_library(test_path)

            # Verify _start_scan was called with the path
            window._start_scan.assert_called_once_with(test_path)
