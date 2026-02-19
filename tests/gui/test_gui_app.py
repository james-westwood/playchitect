"""Smoke tests for the Playchitect GTK4 GUI application."""

import sys
from unittest.mock import MagicMock

import pytest

# Mock gi.repository to prevent import errors if PyGObject is not fully installed
sys.modules["gi.repository"] = MagicMock()
sys.modules["gi.repository.Adw"] = MagicMock()
sys.modules["gi.repository.Gtk"] = MagicMock()

# Import app module after mocking
from playchitect.gui.app import PlaychitectApplication  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def mock_gi_libraries():
    """Ensure gi.repository is mocked for all GUI tests."""
    original_gi = sys.modules.get("gi")
    original_adw = sys.modules.get("gi.repository.Adw")
    original_gtk = sys.modules.get("gi.repository.Gtk")

    sys.modules["gi.repository"] = MagicMock()
    sys.modules["gi.repository.Adw"] = MagicMock()
    sys.modules["gi.repository.Gtk"] = MagicMock()

    yield

    # Restore original modules if they existed
    if original_gi:
        sys.modules["gi"] = original_gi
    else:
        del sys.modules["gi"]
    if original_adw:
        sys.modules["gi.repository.Adw"] = original_adw
    else:
        del sys.modules["gi.repository.Adw"]
    if original_gtk:
        sys.modules["gi.repository.Gtk"] = original_gtk
    else:
        del sys.modules["gi.repository.Gtk"]


class TestPlaychitectApplication:
    """Tests for the main GTK4 application."""

    def test_application_initializes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the application can be initialized without crashing."""
        # Mocking the run method to prevent the app from actually starting a GTK loop
        mock_run = MagicMock(return_value=0)
        monkeypatch.setattr(PlaychitectApplication, "run", mock_run)

        app = PlaychitectApplication()
        assert app is not None
        mock_run.assert_called_once_with([])

    def test_application_window_creation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the main window is created on activation."""
        mock_window_present = MagicMock()
        mock_window_init = MagicMock(return_value=MagicMock(present=mock_window_present))
        monkeypatch.setattr(
            "playchitect.gui.windows.main_window.PlaychitectWindow",
            mock_window_init,
        )
        monkeypatch.setattr(PlaychitectApplication, "run", MagicMock(return_value=0))

        app = PlaychitectApplication()
        app.on_activate(app)  # Manually activate
        mock_window_init.assert_called_once()
        mock_window_present.assert_called_once()

    def test_application_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify the application ID is set correctly."""
        monkeypatch.setattr(PlaychitectApplication, "run", MagicMock(return_value=0))
        app = PlaychitectApplication()
        assert app.get_application_id() == "com.github.jameswestwood.Playchitect"

    def test_main_function_runs_app(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the main function correctly runs the application."""
        mock_app_run = MagicMock(return_value=0)
        monkeypatch.setattr(PlaychitectApplication, "run", mock_app_run)
        monkeypatch.setattr(
            "playchitect.gui.app.PlaychitectApplication",
            MagicMock(return_value=PlaychitectApplication()),
        )

        from playchitect.gui.app import main

        main()
        mock_app_run.assert_called_once_with([])
