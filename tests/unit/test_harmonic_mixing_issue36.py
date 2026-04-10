"""Tests for Key/Harmonic Mixing features - Issue #36.

Tests for musical key detection, key compatibility scoring,
and harmonic ordering in the sequencer.
"""

import numpy as np
import pytest

from playchitect.core.intensity_analyzer import (
    IntensityFeatures,
    harmonic_compatibility,
)
from playchitect.core.sequencer import (
    sequence_harmonic,
)


class TestKeyDetection:
    """Tests for musical key detection - Issue #36 acceptance criteria."""

    def test_intensity_features_has_camelot_key(self) -> None:
        """IntensityFeatures should have camelot_key attribute."""
        features = IntensityFeatures(
            file_path="/test/track.flac",
            file_hash="abc123",
            rms_energy=0.5,
            brightness=0.3,
            sub_bass=0.2,
            kick_energy=0.4,
            bass_harmonics=0.3,
            percussiveness=0.6,
            onset_strength=0.7,
            duration_secs=300.0,
            sample_rate=44100,
        )
        assert hasattr(features, "camelot_key"), (
            "IntensityFeatures should have camelot_key attribute per issue #36"
        )

    def test_camelot_key_format(self) -> None:
        """Camelot key should be in correct format (e.g., '8B', '11A')."""
        features = IntensityFeatures(
            file_path="/test/track.flac",
            file_hash="abc123",
            rms_energy=0.5,
            brightness=0.3,
            sub_bass=0.2,
            kick_energy=0.4,
            bass_harmonics=0.3,
            percussiveness=0.6,
            onset_strength=0.7,
            duration_secs=300.0,
            sample_rate=44100,
            camelot_key="8B",
        )
        # Format should be number + letter (e.g., "8B", "11A")
        assert features.camelot_key is not None
        assert len(features.camelot_key) >= 2
        assert features.camelot_key[-1] in "AB"


class TestKeyCompatibility:
    """Tests for key compatibility scoring - Issue #36 acceptance criteria."""

    def test_harmonic_compatibility_same_key(self) -> None:
        """Same Camelot key should be compatible."""
        assert harmonic_compatibility("8B", "8B") is True

    def test_harmonic_compatibility_adjacent_same_letter(self) -> None:
        """Adjacent number, same letter should be compatible."""
        assert harmonic_compatibility("8B", "9B") is True
        assert harmonic_compatibility("8B", "7B") is True

    def test_harmonic_compatibility_relative_minor(self) -> None:
        """Relative minor/major should be compatible."""
        # 8B (C major) is relative to 5A (A minor)
        assert harmonic_compatibility("8B", "5A") is True
        assert harmonic_compatibility("5A", "8B") is True

    def test_harmonic_compatibility_incompatible(self) -> None:
        """Incompatible keys should return False."""
        # 8B and 3B are not compatible
        assert harmonic_compatibility("8B", "3B") is False


class TestHarmonicSequencing:
    """Tests for harmonic sequencing - Issue #36 acceptance criteria."""

    def test_sequence_harmonic_function_exists(self) -> None:
        """sequence_harmonic function should exist in sequencer."""
        assert callable(sequence_harmonic)

    def test_sequence_harmonic_empty_input(self) -> None:
        """Empty track list should return empty list."""
        result = sequence_harmonic([], {})
        assert result == []

    def test_sequence_harmonic_single_track(self, tmp_path: Path) -> None:
        """Single track should be returned as-is."""
        from pathlib import Path

        track = tmp_path / "track.flac"
        features = {
            track: IntensityFeatures(
                file_path=str(track),
                file_hash="abc",
                rms_energy=0.5,
                brightness=0.3,
                sub_bass=0.2,
                kick_energy=0.4,
                bass_harmonics=0.3,
                percussiveness=0.6,
                onset_strength=0.7,
                duration_secs=300.0,
                sample_rate=44100,
                camelot_key="8B",
            )
        }

        result = sequence_harmonic([track], features)
        assert result == [track]

    def test_sequence_harmonic_orders_by_compatibility(self, tmp_path: Path) -> None:
        """Should order tracks by harmonic compatibility."""
        from pathlib import Path

        tracks = [tmp_path / f"track_{i}.flac" for i in range(5)]
        features = {
            tracks[0]: IntensityFeatures(  # Start with highest energy
                file_path=str(tracks[0]),
                file_hash="a",
                rms_energy=0.9,
                brightness=0.3,
                sub_bass=0.2,
                kick_energy=0.4,
                bass_harmonics=0.3,
                percussiveness=0.6,
                onset_strength=0.7,
                duration_secs=300.0,
                sample_rate=44100,
                camelot_key="8B",
            ),
            tracks[1]: IntensityFeatures(  # Compatible key
                file_path=str(tracks[1]),
                file_hash="b",
                rms_energy=0.7,
                brightness=0.3,
                sub_bass=0.2,
                kick_energy=0.4,
                bass_harmonics=0.3,
                percussiveness=0.6,
                onset_strength=0.7,
                duration_secs=300.0,
                sample_rate=44100,
                camelot_key="9B",  # Compatible with 8B
            ),
            tracks[2]: IntensityFeatures(  # Incompatible
                file_path=str(tracks[2]),
                file_hash="c",
                rms_energy=0.5,
                brightness=0.3,
                sub_bass=0.2,
                kick_energy=0.4,
                bass_harmonics=0.3,
                percussiveness=0.6,
                onset_strength=0.7,
                duration_secs=300.0,
                sample_rate=44100,
                camelot_key="3B",  # Incompatible with 8B
            ),
            tracks[3]: IntensityFeatures(  # Relative minor
                file_path=str(tracks[3]),
                file_hash="d",
                rms_energy=0.4,
                brightness=0.3,
                sub_bass=0.2,
                kick_energy=0.4,
                bass_harmonics=0.3,
                percussiveness=0.6,
                onset_strength=0.7,
                duration_secs=300.0,
                sample_rate=44100,
                camelot_key="5A",  # Relative minor of 8B
            ),
            tracks[4]: IntensityFeatures(  # Another compatible
                file_path=str(tracks[4]),
                file_hash="e",
                rms_energy=0.3,
                brightness=0.3,
                sub_bass=0.2,
                kick_energy=0.4,
                bass_harmonics=0.3,
                percussiveness=0.6,
                onset_strength=0.7,
                duration_secs=300.0,
                sample_rate=44100,
                camelot_key="7B",  # Compatible with 8B
            ),
        }

        result = sequence_harmonic(tracks, features)

        # First track should be the highest energy one
        assert result[0] == tracks[0]
        # The rest should be ordered by harmonic compatibility
        # 9B, 7B (compatible), 5A (relative), 3B (incompatible)
        assert result[1] == tracks[1]  # 9B compatible


class TestCLIHarmonicOrdering:
    """Integration tests for CLI harmonic ordering - Issue #36."""

    def test_sequence_mode_harmonic_available(self) -> None:
        """CLI should support --sequence-mode=harmonic."""
        from click.testing import CliRunner

        from playchitect.cli.commands import scan

        runner = CliRunner()
        result = runner.invoke(scan, ["--help"])

        assert "harmonic" in result.output.lower(), (
            "scan command should support 'harmonic' sequence mode per issue #36"
        )

    def test_cluster_mode_harmonic_available(self) -> None:
        """CLI should support --cluster-mode with harmonic consideration."""
        from click.testing import CliRunner

        from playchitect.cli.commands import scan

        runner = CliRunner()
        result = runner.invoke(scan, ["--help"])

        # Should have cluster mode options
        assert "cluster-mode" in result.output
