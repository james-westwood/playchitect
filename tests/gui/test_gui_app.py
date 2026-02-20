"""Smoke tests for the Playchitect GTK4 GUI application.

gi mocks are installed by tests/gui/conftest.py before this module is collected.
"""

from unittest.mock import MagicMock

import pytest

# Import app module — gi is already mocked by conftest.py
from playchitect.gui.app import PlaychitectApplication  # noqa: E402


class TestPlaychitectApplication:
    """Tests for the main GTK4 application."""

    def test_application_initializes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the application can be initialized without crashing."""
        # Mocking the run method to prevent the app from actually starting a GTK loop
        mock_run = MagicMock(return_value=0)
        monkeypatch.setattr(PlaychitectApplication, "run", mock_run)

        app = PlaychitectApplication()
        assert app is not None

    def test_application_window_creation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that the main window is created on activation."""
        mock_window_present = MagicMock()
        mock_window_init = MagicMock(return_value=MagicMock(present=mock_window_present))
        monkeypatch.setattr(
            "playchitect.gui.app.PlaychitectWindow",
            mock_window_init,
        )
        monkeypatch.setattr(PlaychitectApplication, "run", MagicMock(return_value=0))

        app = PlaychitectApplication()
        app.on_activate(app)  # Manually activate
        mock_window_init.assert_called_once()
        mock_window_present.assert_called_once()

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
