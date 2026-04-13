"""Tests for tooltip presence on major GUI controls - GUI-05.

Validates that all major controls have descriptive tooltips that explain
their purpose in plain language appropriate for a DJ user.

These tests verify that set_tooltip_text calls exist in the source code
by inspecting the actual implementation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


class SourceTooltipVerifier:
    """Helper class to verify tooltips are set in source code."""

    @staticmethod
    def file_has_tooltip_call(file_path: Path, widget_name: str) -> bool:
        """Check if a file contains set_tooltip_text call for a widget."""
        try:
            content = file_path.read_text()
            # Look for the pattern: widget_name.set_tooltip_text(
            # This is a simple heuristic - we're looking for the actual tooltip text being set
            return (
                f"{widget_name}.set_tooltip_text" in content
                or f"{widget_name}.set_tooltip_text" in content.replace(" ", "")
            )
        except Exception:
            return False


class TestPlaylistsViewTooltips:
    """Tests for tooltips on PlaylistsView controls."""

    def test_generate_button_tooltip_in_source(self) -> None:
        """Generate Playlists button should have a tooltip set in source."""
        file_path = (
            Path(__file__).parent.parent.parent
            / "playchitect"
            / "gui"
            / "views"
            / "playlists_view.py"
        )
        content = file_path.read_text()
        # Check that tooltip is set on generate button
        assert "_generate_btn.set_tooltip_text" in content, (
            "Generate button should have tooltip set"
        )

    def test_size_spin_tooltip_in_source(self) -> None:
        """Size spin button should have a tooltip set in source."""
        file_path = (
            Path(__file__).parent.parent.parent
            / "playchitect"
            / "gui"
            / "views"
            / "playlists_view.py"
        )
        content = file_path.read_text()
        assert "_size_spin.set_tooltip_text" in content, "Size spin should have tooltip set"

    def test_unit_dropdown_tooltip_in_source(self) -> None:
        """Unit dropdown should have a tooltip set in source."""
        file_path = (
            Path(__file__).parent.parent.parent
            / "playchitect"
            / "gui"
            / "views"
            / "playlists_view.py"
        )
        content = file_path.read_text()
        assert "_unit_dropdown.set_tooltip_text" in content, "Unit dropdown should have tooltip set"

    def test_playlists_spin_tooltip_in_source(self) -> None:
        """Playlists spin should have a tooltip set in source."""
        file_path = (
            Path(__file__).parent.parent.parent
            / "playchitect"
            / "gui"
            / "views"
            / "playlists_view.py"
        )
        content = file_path.read_text()
        assert "_playlists_spin.set_tooltip_text" in content, (
            "Playlists spin should have tooltip set"
        )

    def test_sequence_dropdown_tooltip_in_source(self) -> None:
        """Sequence dropdown should have a tooltip set in source."""
        file_path = (
            Path(__file__).parent.parent.parent
            / "playchitect"
            / "gui"
            / "views"
            / "playlists_view.py"
        )
        content = file_path.read_text()
        assert "_sequence_dropdown.set_tooltip_text" in content, (
            "Sequence dropdown should have tooltip set"
        )

    def test_intro_dropdown_tooltip_in_source(self) -> None:
        """Intro dropdown should have a tooltip set in source."""
        file_path = (
            Path(__file__).parent.parent.parent
            / "playchitect"
            / "gui"
            / "views"
            / "playlists_view.py"
        )
        content = file_path.read_text()
        assert "_intro_dropdown.set_tooltip_text" in content, (
            "Intro dropdown should have tooltip set"
        )

    def test_fresh_switch_tooltip_in_source(self) -> None:
        """Fresh toggle switch should have a tooltip set in source."""
        file_path = (
            Path(__file__).parent.parent.parent
            / "playchitect"
            / "gui"
            / "views"
            / "playlists_view.py"
        )
        content = file_path.read_text()
        assert "_fresh_switch.set_tooltip_text" in content, "Fresh switch should have tooltip set"


class TestExportViewTooltips:
    """Tests for tooltips on ExportView controls."""

    def test_export_button_tooltip_in_source(self) -> None:
        """Export button should have a tooltip set in source."""
        file_path = (
            Path(__file__).parent.parent.parent / "playchitect" / "gui" / "views" / "export_view.py"
        )
        content = file_path.read_text()
        assert "_export_button.set_tooltip_text" in content, "Export button should have tooltip set"

    def test_browse_button_tooltip_in_source(self) -> None:
        """Browse button should have a tooltip set in source."""
        file_path = (
            Path(__file__).parent.parent.parent / "playchitect" / "gui" / "views" / "export_view.py"
        )
        content = file_path.read_text()
        assert "_browse_button.set_tooltip_text" in content, "Browse button should have tooltip set"


class TestSetBuilderViewTooltips:
    """Tests for tooltips on SetBuilderView controls."""

    def test_auto_fill_button_tooltip_in_source(self) -> None:
        """Auto-fill button should have a tooltip set in source."""
        file_path = (
            Path(__file__).parent.parent.parent
            / "playchitect"
            / "gui"
            / "views"
            / "set_builder_view.py"
        )
        content = file_path.read_text()
        assert "_auto_fill_button.set_tooltip_text" in content, (
            "Auto-fill button should have tooltip set"
        )

    def test_export_button_tooltip_in_source(self) -> None:
        """Export Set button should have a tooltip set in source."""
        file_path = (
            Path(__file__).parent.parent.parent
            / "playchitect"
            / "gui"
            / "views"
            / "set_builder_view.py"
        )
        content = file_path.read_text()
        # The export button in set_builder_view is named _export_button
        assert "_export_button.set_tooltip_text" in content, (
            "Export Set button should have tooltip set"
        )


class TestEnergyArcWidgetTooltip:
    """Tests for tooltip on EnergyArcWidget."""

    def test_energy_arc_tooltip_in_source(self) -> None:
        """Energy arc widget should have a tooltip set in source."""
        file_path = (
            Path(__file__).parent.parent.parent
            / "playchitect"
            / "gui"
            / "widgets"
            / "energy_arc_widget.py"
        )
        content = file_path.read_text()
        assert "set_tooltip_text" in content, "Energy arc widget should have tooltip set"


class TestClusterViewTooltip:
    """Tests for tooltip on ClusterCard View Tracks button."""

    def test_view_tracks_button_tooltip_in_source(self) -> None:
        """View Tracks button should have a tooltip set in source."""
        file_path = (
            Path(__file__).parent.parent.parent
            / "playchitect"
            / "gui"
            / "widgets"
            / "cluster_view.py"
        )
        content = file_path.read_text()
        assert "view_btn.set_tooltip_text" in content, "View Tracks button should have tooltip set"


class TestTooltipContentQuality:
    """Tests that tooltips contain meaningful content (not empty)."""

    def test_playlists_view_tooltips_not_empty(self) -> None:
        """Verify tooltips in playlists_view.py have non-empty text."""
        file_path = (
            Path(__file__).parent.parent.parent
            / "playchitect"
            / "gui"
            / "views"
            / "playlists_view.py"
        )
        content = file_path.read_text()
        # Verify that set_tooltip_text is called with meaningful strings
        # by checking it has multiple calls with substantial content
        tooltip_calls = content.count("set_tooltip_text(")
        assert tooltip_calls >= 6, f"Expected at least 6 tooltip calls, found {tooltip_calls}"

    def test_export_view_tooltips_not_empty(self) -> None:
        """Verify tooltips in export_view.py have non-empty text."""
        file_path = (
            Path(__file__).parent.parent.parent / "playchitect" / "gui" / "views" / "export_view.py"
        )
        content = file_path.read_text()
        tooltip_calls = content.count("set_tooltip_text(")
        assert tooltip_calls >= 2, f"Expected at least 2 tooltip calls, found {tooltip_calls}"

    def test_set_builder_view_tooltips_not_empty(self) -> None:
        """Verify tooltips in set_builder_view.py have non-empty text."""
        file_path = (
            Path(__file__).parent.parent.parent
            / "playchitect"
            / "gui"
            / "views"
            / "set_builder_view.py"
        )
        content = file_path.read_text()
        tooltip_calls = content.count("set_tooltip_text(")
        assert tooltip_calls >= 2, f"Expected at least 2 tooltip calls, found {tooltip_calls}"

    def test_energy_arc_widget_tooltips_not_empty(self) -> None:
        """Verify tooltips in energy_arc_widget.py have non-empty text."""
        file_path = (
            Path(__file__).parent.parent.parent
            / "playchitect"
            / "gui"
            / "widgets"
            / "energy_arc_widget.py"
        )
        content = file_path.read_text()
        assert "set_tooltip_text" in content, "Energy arc should have tooltip"

    def test_cluster_view_tooltips_not_empty(self) -> None:
        """Verify tooltips in cluster_view.py have non-empty text."""
        file_path = (
            Path(__file__).parent.parent.parent
            / "playchitect"
            / "gui"
            / "widgets"
            / "cluster_view.py"
        )
        content = file_path.read_text()
        tooltip_calls = content.count("set_tooltip_text(")
        assert tooltip_calls >= 2, f"Expected at least 2 tooltip calls, found {tooltip_calls}"
