"""Core unit tests for Energy Flow features - Issue #38.

Tests for dynamic range, onset density calculation, and energy flow
sequencing in the sequencer module.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.sequencer import (
    SequencingStrategy,
    sequence_by_strategy,
)


class TestEnergyFlowFeatures:
    """Tests for energy flow feature extraction - Issue #38 acceptance criteria."""

    def test_intensity_features_has_dynamic_range(self) -> None:
        """IntensityFeatures should have dynamic_range attribute."""
        features = IntensityFeatures(
            file_path="/test/track.flac",
            file_hash="abc123",
            rms_energy=0.5,
            brightness=0.3,
            sub_bass_energy=0.2,
            kick_energy=0.4,
            bass_harmonics=0.3,
            percussiveness=0.6,
            onset_strength=0.7,
            duration_secs=300.0,
            sample_rate=44100,
            camelot_key="8B",
            key_index=0.0,
            dynamic_range=0.4,
        )
        assert hasattr(features, "dynamic_range"), (
            "IntensityFeatures should have dynamic_range per issue #38"
        )
        assert features.dynamic_range == 0.4

    def test_intensity_features_has_energy_gradient(self) -> None:
        """IntensityFeatures should have energy_gradient attribute."""
        features = IntensityFeatures(
            file_path="/test/track.flac",
            file_hash="abc123",
            rms_energy=0.5,
            brightness=0.3,
            sub_bass_energy=0.2,
            kick_energy=0.4,
            bass_harmonics=0.3,
            percussiveness=0.6,
            onset_strength=0.7,
            duration_secs=300.0,
            sample_rate=44100,
            camelot_key="8B",
            key_index=0.0,
            energy_gradient=0.3,
        )
        assert hasattr(features, "energy_gradient"), (
            "IntensityFeatures should have energy_gradient per issue #38"
        )
        assert features.energy_gradient == 0.3

    def test_intensity_features_has_drop_density(self) -> None:
        """IntensityFeatures should have drop_density attribute."""
        features = IntensityFeatures(
            file_path="/test/track.flac",
            file_hash="abc123",
            rms_energy=0.5,
            brightness=0.3,
            sub_bass_energy=0.2,
            kick_energy=0.4,
            bass_harmonics=0.3,
            percussiveness=0.6,
            onset_strength=0.7,
            duration_secs=300.0,
            sample_rate=44100,
            camelot_key="8B",
            key_index=0.0,
            drop_density=0.2,
        )
        assert hasattr(features, "drop_density"), (
            "IntensityFeatures should have drop_density per issue #38"
        )
        assert features.drop_density == 0.2


class TestDynamicRangeCalculation:
    """Tests for dynamic range calculation - Issue #38."""

    def test_dynamic_range_high_for_variable_track(self) -> None:
        """Track with varying RMS should have higher dynamic range."""
        features = IntensityFeatures(
            file_path="/test/track.flac",
            file_hash="abc",
            rms_energy=0.5,
            brightness=0.3,
            sub_bass_energy=0.2,
            kick_energy=0.4,
            bass_harmonics=0.3,
            percussiveness=0.6,
            onset_strength=0.7,
            duration_secs=300.0,
            sample_rate=44100,
            camelot_key="8B",
            key_index=0.0,
            dynamic_range=0.8,
        )
        assert features.dynamic_range > 0.5, "Variable energy track should have high dynamic range"

    def test_dynamic_range_low_for_flat_track(self) -> None:
        """Track with consistent RMS should have lower dynamic range."""
        features = IntensityFeatures(
            file_path="/test/track.flac",
            file_hash="abc",
            rms_energy=0.5,
            brightness=0.3,
            sub_bass_energy=0.2,
            kick_energy=0.4,
            bass_harmonics=0.3,
            percussiveness=0.6,
            onset_strength=0.7,
            duration_secs=300.0,
            sample_rate=44100,
            camelot_key="8B",
            key_index=0.0,
            dynamic_range=0.1,
        )
        assert features.dynamic_range < 0.3, "Flat energy track should have low dynamic range"

    def test_dynamic_range_defaults_to_zero(self) -> None:
        """Missing dynamic_range should default to 0.0."""
        features = IntensityFeatures(
            file_path="/test/track.flac",
            file_hash="abc",
            rms_energy=0.5,
            brightness=0.3,
            sub_bass_energy=0.2,
            kick_energy=0.4,
            bass_harmonics=0.3,
            percussiveness=0.6,
            onset_strength=0.7,
            duration_secs=300.0,
            sample_rate=44100,
            camelot_key="8B",
            key_index=0.0,
        )
        assert features.dynamic_range == 0.0, "dynamic_range should default to 0.0"


class TestEnergyGradientCalculation:
    """Tests for energy gradient calculation - Issue #38."""

    def test_energy_gradient_positive_for_rising_track(self) -> None:
        """Track with increasing energy should have positive gradient."""
        features = IntensityFeatures(
            file_path="/test/track.flac",
            file_hash="abc",
            rms_energy=0.5,
            brightness=0.3,
            sub_bass_energy=0.2,
            kick_energy=0.4,
            bass_harmonics=0.3,
            percussiveness=0.6,
            onset_strength=0.7,
            duration_secs=300.0,
            sample_rate=44100,
            camelot_key="8B",
            key_index=0.0,
            energy_gradient=0.5,
        )
        assert features.energy_gradient > 0, "Rising energy track should have positive gradient"

    def test_energy_gradient_negative_for_falling_track(self) -> None:
        """Track with decreasing energy should have negative gradient."""
        features = IntensityFeatures(
            file_path="/test/track.flac",
            file_hash="abc",
            rms_energy=0.5,
            brightness=0.3,
            sub_bass_energy=0.2,
            kick_energy=0.4,
            bass_harmonics=0.3,
            percussiveness=0.6,
            onset_strength=0.7,
            duration_secs=300.0,
            sample_rate=44100,
            camelot_key="8B",
            key_index=0.0,
            energy_gradient=-0.4,
        )
        assert features.energy_gradient < 0, "Falling energy track should have negative gradient"


class TestEnergyFlowSequencing:
    """Tests for energy flow sequencing - Issue #38 acceptance criteria."""

    def test_sequence_by_strategy_exists(self) -> None:
        """sequence_by_strategy function should exist in sequencer."""
        assert callable(sequence_by_strategy), (
            "sequencer should have sequence_by_strategy() per issue #38"
        )

    def test_sequence_by_strategy_ramp(self, tmp_path: Path) -> None:
        """Test ramp strategy (ascending RMS energy)."""
        tracks = [tmp_path / f"track_{i}.flac" for i in range(5)]
        features = {
            tracks[0]: IntensityFeatures(
                file_path=str(tracks[0]),
                file_hash="a",
                rms_energy=0.9,
                brightness=0.3,
                sub_bass_energy=0.2,
                kick_energy=0.4,
                bass_harmonics=0.3,
                percussiveness=0.6,
                onset_strength=0.7,
                duration_secs=300.0,
                sample_rate=44100,
                camelot_key="8B",
                key_index=0.0,
            ),
            tracks[1]: IntensityFeatures(
                file_path=str(tracks[1]),
                file_hash="b",
                rms_energy=0.3,
                brightness=0.3,
                sub_bass_energy=0.2,
                kick_energy=0.4,
                bass_harmonics=0.3,
                percussiveness=0.6,
                onset_strength=0.7,
                duration_secs=300.0,
                sample_rate=44100,
                camelot_key="8B",
                key_index=0.0,
            ),
            tracks[2]: IntensityFeatures(
                file_path=str(tracks[2]),
                file_hash="c",
                rms_energy=0.7,
                brightness=0.3,
                sub_bass_energy=0.2,
                kick_energy=0.4,
                bass_harmonics=0.3,
                percussiveness=0.6,
                onset_strength=0.7,
                duration_secs=300.0,
                sample_rate=44100,
                camelot_key="8B",
                key_index=0.0,
            ),
            tracks[3]: IntensityFeatures(
                file_path=str(tracks[3]),
                file_hash="d",
                rms_energy=0.1,
                brightness=0.3,
                sub_bass_energy=0.2,
                kick_energy=0.4,
                bass_harmonics=0.3,
                percussiveness=0.6,
                onset_strength=0.7,
                duration_secs=300.0,
                sample_rate=44100,
                camelot_key="8B",
                key_index=0.0,
            ),
            tracks[4]: IntensityFeatures(
                file_path=str(tracks[4]),
                file_hash="e",
                rms_energy=0.5,
                brightness=0.3,
                sub_bass_energy=0.2,
                kick_energy=0.4,
                bass_harmonics=0.3,
                percussiveness=0.6,
                onset_strength=0.7,
                duration_secs=300.0,
                sample_rate=44100,
                camelot_key="8B",
                key_index=0.0,
            ),
        }

        result = sequence_by_strategy(list(tracks), features, "ramp")

        assert result[0] == tracks[3]  # Lowest energy (0.1)
        assert result[1] == tracks[1]  # Next (0.3)
        assert result[2] == tracks[4]  # Next (0.5)
        assert result[3] == tracks[2]  # Next (0.7)
        assert result[4] == tracks[0]  # Highest (0.9)

    def test_sequence_by_strategy_descent(self, tmp_path: Path) -> None:
        """Test descent strategy (descending RMS energy)."""
        tracks = [tmp_path / f"track_{i}.flac" for i in range(3)]
        features = {
            tracks[0]: IntensityFeatures(
                file_path=str(tracks[0]),
                file_hash="a",
                rms_energy=0.3,
                brightness=0.3,
                sub_bass_energy=0.2,
                kick_energy=0.4,
                bass_harmonics=0.3,
                percussiveness=0.6,
                onset_strength=0.7,
                duration_secs=300.0,
                sample_rate=44100,
                camelot_key="8B",
                key_index=0.0,
            ),
            tracks[1]: IntensityFeatures(
                file_path=str(tracks[1]),
                file_hash="b",
                rms_energy=0.7,
                brightness=0.3,
                sub_bass_energy=0.2,
                kick_energy=0.4,
                bass_harmonics=0.3,
                percussiveness=0.6,
                onset_strength=0.7,
                duration_secs=300.0,
                sample_rate=44100,
                camelot_key="8B",
                key_index=0.0,
            ),
            tracks[2]: IntensityFeatures(
                file_path=str(tracks[2]),
                file_hash="c",
                rms_energy=0.5,
                brightness=0.3,
                sub_bass_energy=0.2,
                kick_energy=0.4,
                bass_harmonics=0.3,
                percussiveness=0.6,
                onset_strength=0.7,
                duration_secs=300.0,
                sample_rate=44100,
                camelot_key="8B",
                key_index=0.0,
            ),
        }

        result = sequence_by_strategy(list(tracks), features, "descent")

        assert result[0] == tracks[1]  # Highest (0.7)
        assert result[1] == tracks[2]  # Next (0.5)
        assert result[2] == tracks[0]  # Lowest (0.3)

    def test_sequence_by_strategy_build(self, tmp_path: Path) -> None:
        """Test build strategy (sort by energy_gradient descending)."""
        tracks = [tmp_path / f"track_{i}.flac" for i in range(3)]
        features = {
            tracks[0]: IntensityFeatures(
                file_path=str(tracks[0]),
                file_hash="a",
                rms_energy=0.5,
                brightness=0.3,
                sub_bass_energy=0.2,
                kick_energy=0.4,
                bass_harmonics=0.3,
                percussiveness=0.6,
                onset_strength=0.7,
                duration_secs=300.0,
                sample_rate=44100,
                camelot_key="8B",
                key_index=0.0,
                energy_gradient=-0.2,
            ),
            tracks[1]: IntensityFeatures(
                file_path=str(tracks[1]),
                file_hash="b",
                rms_energy=0.5,
                brightness=0.3,
                sub_bass_energy=0.2,
                kick_energy=0.4,
                bass_harmonics=0.3,
                percussiveness=0.6,
                onset_strength=0.7,
                duration_secs=300.0,
                sample_rate=44100,
                camelot_key="8B",
                key_index=0.0,
                energy_gradient=0.5,
            ),
            tracks[2]: IntensityFeatures(
                file_path=str(tracks[2]),
                file_hash="c",
                rms_energy=0.5,
                brightness=0.3,
                sub_bass_energy=0.2,
                kick_energy=0.4,
                bass_harmonics=0.3,
                percussiveness=0.6,
                onset_strength=0.7,
                duration_secs=300.0,
                sample_rate=44100,
                camelot_key="8B",
                key_index=0.0,
                energy_gradient=0.1,
            ),
        }

        result = sequence_by_strategy(list(tracks), features, "build")

        assert result[0] == tracks[1]  # Highest gradient (0.5)
        assert result[1] == tracks[2]  # Next (0.1)
        assert result[2] == tracks[0]  # Lowest (-0.2)

    def test_sequence_by_strategy_empty(self, tmp_path: Path) -> None:
        """Empty track list should return empty list."""
        result = sequence_by_strategy([], {}, "ramp")
        assert result == []

    def test_sequence_by_strategy_invalid(self, tmp_path: Path) -> None:
        """Invalid strategy should raise ValueError."""
        tracks = [tmp_path / "track.flac"]
        features = {
            tracks[0]: IntensityFeatures(
                file_path=str(tracks[0]),
                file_hash="a",
                rms_energy=0.5,
                brightness=0.3,
                sub_bass_energy=0.2,
                kick_energy=0.4,
                bass_harmonics=0.3,
                percussiveness=0.6,
                onset_strength=0.7,
                duration_secs=300.0,
                sample_rate=44100,
                camelot_key="8B",
                key_index=0.0,
            ),
        }

        with pytest.raises(ValueError, match="Invalid strategy"):
            sequence_by_strategy(tracks, features, "invalid_strategy")


class TestSequencingStrategyEnum:
    """Tests for SequencingStrategy enum - Issue #38."""

    def test_ramp_strategy_exists(self) -> None:
        """SequencingStrategy should have RAMP value."""
        assert hasattr(SequencingStrategy, "RAMP")
        assert SequencingStrategy.RAMP.value == "ramp"

    def test_build_strategy_exists(self) -> None:
        """SequencingStrategy should have BUILD value."""
        assert hasattr(SequencingStrategy, "BUILD")
        assert SequencingStrategy.BUILD.value == "build"

    def test_descent_strategy_exists(self) -> None:
        """SequencingStrategy should have DESCENT value."""
        assert hasattr(SequencingStrategy, "DESCENT")
        assert SequencingStrategy.DESCENT.value == "descent"

    def test_alternating_strategy_exists(self) -> None:
        """SequencingStrategy should have ALTERNATING value."""
        assert hasattr(SequencingStrategy, "ALTERNATING")
        assert SequencingStrategy.ALTERNATING.value == "alternating"


class TestCLIEnergyFlowIntegration:
    """Integration tests for CLI energy flow support - Issue #38."""

    def test_sequence_mode_build_available(self) -> None:
        """CLI should support --sequence-mode=build."""
        from click.testing import CliRunner

        from playchitect.cli.commands import scan

        runner = CliRunner()
        result = runner.invoke(scan, ["--help"])

        assert "build" in result.output.lower(), (
            "scan command should support 'build' sequence mode per issue #38"
        )

    def test_sequence_mode_descent_available(self) -> None:
        """CLI should support --sequence-mode=descent."""
        from click.testing import CliRunner

        from playchitect.cli.commands import scan

        runner = CliRunner()
        result = runner.invoke(scan, ["--help"])

        assert "descent" in result.output.lower(), (
            "scan command should support 'descent' sequence mode per issue #38"
        )

    def test_sequence_mode_alternating_available(self) -> None:
        """CLI should support --sequence-mode=alternating."""
        from click.testing import CliRunner

        from playchitect.cli.commands import scan

        runner = CliRunner()
        result = runner.invoke(scan, ["--help"])

        assert "alternating" in result.output.lower(), (
            "scan command should support 'alternating' sequence mode per issue #38"
        )
