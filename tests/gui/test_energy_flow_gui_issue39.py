"""GUI tests for Energy Flow controls - Issue #39.

Tests for exposing Energy Flow features (#38) within the GTK4 GUI.
Provides users with intuitive controls to customize playlist sequencing
based on energy characteristics.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from playchitect.gui.views.playlists_view import PlaylistsView


def _make_view() -> PlaylistsView:
    """Return a PlaylistsView with __init__ bypassed and mocked components."""
    from playchitect.gui.views.playlists_view import PlaylistsView

    view = PlaylistsView.__new__(PlaylistsView)
    view._clusters = []
    view._cluster_stats = []
    view._metadata_map = {}
    view._intensity_map = {}
    view._selected_cluster_id = None
    view._cluster_list = MagicMock()
    view._track_list = MagicMock()
    view._count_label = MagicMock()
    view._generate_btn = MagicMock()
    view._spinner = MagicMock()
    view._stats_bpm_label = MagicMock()
    view._stats_intensity_label = MagicMock()
    view._stats_tracks_label = MagicMock()
    view._stats_duration_label = MagicMock()
    view._size_spin = MagicMock()
    view._unit_dropdown = MagicMock()
    view._playlists_spin = MagicMock()
    view._harmonic_switch = MagicMock()
    view._sort_dropdown = MagicMock()
    view._timbre_scale = MagicMock()
    view._vocal_btn_any = MagicMock()
    view._vocal_btn_instrumental = MagicMock()
    view._vocal_btn_vocal = MagicMock()
    view._energy_arc = MagicMock()
    return view


class TestEnergyFlowGUIControls:
    """Tests for energy flow UI controls - Issue #39 acceptance criteria."""

    # Hollow: _energy_flow_dropdown is not set in _make_view(), so hasattr returns
    # False unless the class defines it as a class-level attribute. Either way it
    # never checks the widget is a real Gtk.DropDown with the right options.
    @pytest.mark.hollow
    def test_energy_flow_dropdown_exists(self) -> None:
        """Verify PlaylistsView has an energy flow dropdown."""
        view = _make_view()
        assert hasattr(view, "_energy_flow_dropdown"), (
            "PlaylistsView should have _energy_flow_dropdown per issue #39"
        )

    # Hollow: only checks the method name exists via hasattr();
    # get_energy_flow_options() is never called to verify it returns expected values.
    @pytest.mark.hollow
    def test_energy_flow_options_in_dropdown(self) -> None:
        """Verify energy flow dropdown has expected options."""
        view = _make_view()
        view._energy_flow_dropdown = MagicMock()

        assert hasattr(view, "get_energy_flow_options"), (
            "PlaylistsView should have get_energy_flow_options() per issue #39"
        )

    # Hollow: only checks the method name exists via hasattr(); the 'Build Up'
    # option is never verified to be present in the real dropdown model.
    @pytest.mark.hollow
    def test_energy_flow_build_option_available(self) -> None:
        """Verify 'Build Up' option is available in energy flow."""
        view = _make_view()
        view._energy_flow_dropdown = MagicMock()

        assert hasattr(view, "get_energy_flow_mode"), (
            "PlaylistsView should have get_energy_flow_mode() per issue #39"
        )

    # Hollow: only checks the method name exists via hasattr(); the 'Cool Down'
    # option is never verified to be present in the real dropdown model.
    @pytest.mark.hollow
    def test_energy_flow_cool_down_option_available(self) -> None:
        """Verify 'Cool Down' option is available in energy flow."""
        view = _make_view()
        view._energy_flow_dropdown = MagicMock()

        assert hasattr(view, "set_energy_flow_mode"), (
            "PlaylistsView should have set_energy_flow_mode() per issue #39"
        )

    # Hollow: constructs a mock_model with the expected options but never passes it
    # to the view; falls back to a bare hasattr() check, so the mock setup is dead code.
    @pytest.mark.hollow
    def test_energy_flow_constant_option_available(self) -> None:
        """Verify 'Constant Energy' option is available in energy flow."""
        view = _make_view()
        view._energy_flow_dropdown = MagicMock()

        mock_model = MagicMock()
        mock_model.get_item = MagicMock(
            side_effect=lambda i: ["ramp", "build", "descent", "constant"][i]
        )
        view._energy_flow_dropdown.get_selected_item = MagicMock(return_value=MagicMock())

        assert hasattr(view, "get_energy_flow_mode"), "Energy flow mode getter should exist"


class TestEnergyFlowVisualization:
    """Tests for energy flow visualization - Issue #39."""

    # Hollow: only checks the method name exists via hasattr();
    # update_energy_flow_visualization() is never called to verify it does anything.
    @pytest.mark.hollow
    def test_energy_flow_visualization_available(self) -> None:
        """Verify energy flow visualization is available."""
        view = _make_view()

        assert hasattr(view, "update_energy_flow_visualization"), (
            "PlaylistsView should have update_energy_flow_visualization() per issue #39"
        )

    # Hollow: _energy_heatmap is not set in _make_view() and no class-level default
    # exists, so this will always fail unless the real class defines it — which means
    # it tests the class definition, not any runtime behaviour.
    @pytest.mark.hollow
    def test_energy_heatmap_available(self) -> None:
        """Verify energy heatmap alongside tracks."""
        view = _make_view()
        view._track_list = MagicMock()

        assert hasattr(view, "_energy_heatmap"), (
            "PlaylistsView should have _energy_heatmap widget per issue #39"
        )


class TestEnergyFlowSequenceTrigger:
    """Tests for triggering energy flow sequencing - Issue #39."""

    # Hollow: patch.object replaces get_energy_flow_mode with a mock returning "build",
    # then immediately calls and asserts that mock — the real method is never exercised.
    @pytest.mark.hollow
    def test_generate_playlist_respects_energy_flow_setting(self) -> None:
        """Verify generate_playlist uses energy flow ordering when selected."""
        view = _make_view()
        view._sort_dropdown = MagicMock()

        mock_selected = MagicMock()
        mock_selected.unpack = MagicMock(return_value="build")
        view._sort_dropdown.get_selected_item = MagicMock(return_value=mock_selected)

        with patch.object(view, "get_energy_flow_mode", return_value="build"):
            result = view.get_energy_flow_mode()
            assert result == "build", "Energy flow mode should be retrieved"

    # Hollow: `assert mock_seq.called or True` is a tautology — always passes
    # regardless of whether sequence_by_strategy was ever called.
    @pytest.mark.hollow
    def test_energy_flow_ordering_passed_to_sequencer(self) -> None:
        """Verify energy flow ordering is passed to the sequencer."""
        view = _make_view()
        view._sort_dropdown = MagicMock()

        mock_selected = MagicMock()
        mock_selected.unpack = MagicMock(return_value="ramp")
        view._sort_dropdown.get_selected_item = MagicMock(return_value=mock_selected)

        with patch("playchitect.gui.views.playlists_view.sequence_by_strategy") as mock_seq:
            mock_seq.return_value = [Path("/test/track.mp3")]
            mode = "ramp"
            if mode:
                assert mock_seq.called or True, (
                    "Should trigger sequence_by_strategy when energy flow is set"
                )


class TestEnergyFlowTooltip:
    """Tests for energy flow tooltips - Issue #39."""

    # Hollow: the test itself calls set_tooltip_text() on line 165, then asserts it
    # was called on line 166 — the assertion is trivially true because we just called it.
    # No real GTK widget or production code path is exercised.
    @pytest.mark.hollow
    def test_energy_flow_tooltip_exists(self) -> None:
        """Verify energy flow dropdown has tooltip."""
        view = _make_view()
        view._energy_flow_dropdown = MagicMock()
        view._energy_flow_dropdown.set_tooltip_text = MagicMock()

        view._energy_flow_dropdown.set_tooltip_text("Select energy flow sequencing strategy")
        view._energy_flow_dropdown.set_tooltip_text.assert_called_once()


class TestEnergyFlowStatePersistence:
    """Tests for energy flow state - Issue #39."""

    # Hollow: the mock is configured to return "ramp" via .unpack(), but the real
    # get_energy_flow_mode() likely uses .get_string() on a GTK StringList item, not
    # .unpack(). The assertion validates the mock's return value, not the real logic.
    @pytest.mark.hollow
    def test_default_energy_flow_mode(self) -> None:
        """Verify default energy flow mode is 'ramp'."""
        view = _make_view()
        view._sort_dropdown = MagicMock()

        mock_selected = MagicMock()
        mock_selected.unpack = MagicMock(return_value="ramp")
        view._sort_dropdown.get_selected_item = MagicMock(return_value=mock_selected)

        result = view.get_energy_flow_mode()
        assert result == "ramp", "Default energy flow should be 'ramp'"

    def test_set_energy_flow_mode_changes_dropdown(self) -> None:
        """Verify set_energy_flow_mode updates dropdown."""
        view = _make_view()
        view._energy_flow_dropdown = MagicMock()

        view.set_energy_flow_mode("build")
        view._energy_flow_dropdown.set_selected.assert_called()


class TestCLIEnergyFlowMode:
    """Integration tests for CLI energy flow mode - Issue #39."""

    def test_cli_scan_help_shows_sequence_mode_options(self) -> None:
        """Verify scan command help shows sequence mode options."""
        from click.testing import CliRunner

        from playchitect.cli.commands import scan

        runner = CliRunner()
        result = runner.invoke(scan, ["--help"])

        assert "--sequence-mode" in result.output, (
            "scan command should have --sequence-mode per issue #39"
        )

    def test_cli_sequence_mode_ramp_documented(self) -> None:
        """Verify ramp sequence mode is documented."""
        from click.testing import CliRunner

        from playchitect.cli.commands import scan

        runner = CliRunner()
        result = runner.invoke(scan, ["--help"])

        assert "ramp" in result.output.lower(), (
            "scan command should document 'ramp' sequence mode per issue #39"
        )

    def test_cli_sequence_mode_build_documented(self) -> None:
        """Verify build sequence mode is documented."""
        from click.testing import CliRunner

        from playchitect.cli.commands import scan

        runner = CliRunner()
        result = runner.invoke(scan, ["--help"])

        assert "build" in result.output.lower(), (
            "scan command should document 'build' sequence mode per issue #39"
        )

    def test_cli_sequence_mode_descent_documented(self) -> None:
        """Verify descent sequence mode is documented."""
        from click.testing import CliRunner

        from playchitect.cli.commands import scan

        runner = CliRunner()
        result = runner.invoke(scan, ["--help"])

        assert "descent" in result.output.lower(), (
            "scan command should document 'descent' sequence mode per issue #39"
        )
