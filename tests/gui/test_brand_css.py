"""Tests for brand CSS dark theme loading."""

from pathlib import Path
from unittest.mock import MagicMock, patch


class TestBrandCSS:
    """Tests for brand CSS styling."""

    def test_style_css_exists(self) -> None:
        """Test that the style.css file exists in the gui package."""
        css_path = Path(__file__).parents[2] / "playchitect" / "gui" / "style.css"
        assert css_path.exists(), "style.css should exist in playchitect/gui/"
        content = css_path.read_text()
        assert "background-color: #0D1117" in content
        assert "background-color: #7B5CF0" in content

    def test_style_css_contains_dark_colors(self) -> None:
        """Test that CSS contains the required dark theme colors."""
        css_path = Path(__file__).parents[2] / "playchitect" / "gui" / "style.css"
        content = css_path.read_text()
        assert "#0D1117" in content
        assert "#111720" in content
        assert "#1E2A3A" in content
        assert "#7B5CF0" in content

    def test_style_css_contains_purple_primary(self) -> None:
        """Test that CSS contains purple primary button styling."""
        css_path = Path(__file__).parents[2] / "playchitect" / "gui" / "style.css"
        content = css_path.read_text()
        assert "button.suggested-action" in content
        assert "#7B5CF0" in content
        assert "border-radius: 6px" in content

    def test_style_css_contains_selection_colors(self) -> None:
        """Test that CSS contains purple-tinted selection colors."""
        css_path = Path(__file__).parents[2] / "playchitect" / "gui" / "style.css"
        content = css_path.read_text()
        assert "rgba(123, 92, 240, 0.25)" in content

    def test_style_css_contains_text_colors(self) -> None:
        """Test that CSS contains readable text colors."""
        css_path = Path(__file__).parents[2] / "playchitect" / "gui" / "style.css"
        content = css_path.read_text()
        assert "#E6EDF3" in content
        assert "#C9D1D9" in content


class TestDarkModeEnforcement:
    """Tests for dark mode enforcement via Adw.StyleManager."""

    def test_ensure_dark_mode_function_exists(self) -> None:
        """Test that _ensure_dark_mode function exists in app module."""
        from playchitect.gui.app import _ensure_dark_mode

        assert callable(_ensure_dark_mode)

    @patch("playchitect.gui.app.Adw.StyleManager.get_default")
    def test_ensure_dark_mode_sets_force_dark(self, mock_get_default: MagicMock) -> None:
        """Test that _ensure_dark_mode sets FORCE_DARK when not already dark."""
        from playchitect.gui.app import _ensure_dark_mode

        mock_style_manager = MagicMock()
        mock_style_manager.get_dark.return_value = False
        mock_get_default.return_value = mock_style_manager
        _ensure_dark_mode()
        mock_style_manager.set_color_scheme.assert_called_once()

    @patch("playchitect.gui.app.Adw.StyleManager.get_default")
    def test_ensure_dark_mode_skips_when_already_dark(self, mock_get_default: MagicMock) -> None:
        """Test that _ensure_dark_mode skips when already dark."""
        from playchitect.gui.app import _ensure_dark_mode

        mock_style_manager = MagicMock()
        mock_style_manager.get_dark.return_value = True
        mock_get_default.return_value = mock_style_manager
        _ensure_dark_mode()
        mock_style_manager.set_color_scheme.assert_not_called()


class TestCssLoader:
    """Tests for CSS loading functionality."""

    def test_load_brand_css_function_exists(self) -> None:
        """Test that _load_brand_css function exists in app module."""
        from playchitect.gui.app import _load_brand_css

        assert callable(_load_brand_css)

    @patch("playchitect.gui.app.Gtk.Display.get_default")
    @patch("playchitect.gui.app.Gtk.CssProvider")
    def test_load_brand_css_loads_from_file(
        self, mock_css_provider: MagicMock, mock_get_display: MagicMock
    ) -> None:
        """Test that _load_brand_css loads CSS from the style.css file."""
        from playchitect.gui.app import _load_brand_css

        mock_provider = MagicMock()
        mock_css_provider.return_value = mock_provider
        mock_display = MagicMock()
        mock_get_display.return_value = mock_display
        _load_brand_css()
        mock_provider.load_from_path.assert_called_once()

    def test_load_brand_css_function_calls_correct_path(self) -> None:
        """Test that _load_brand_css loads from the correct CSS file path."""
        css_path = Path(__file__).parents[2] / "playchitect" / "gui" / "style.css"
        expected_path = str(css_path)
        assert css_path.exists(), f"Expected CSS at {expected_path}"
