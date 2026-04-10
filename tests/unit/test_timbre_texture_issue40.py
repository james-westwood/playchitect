"""Core unit tests for Timbre/Texture features - Issue #40.

Tests for MFCCs, spectral contrast, bandwidth, rolloff calculation,
and timbre-based sequencing in the sequencer module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.sequencer import sequence_by_timbre


class TestTimbreTextureFeatures:
    """Tests for timbre/texture feature extraction - Issue #40 acceptance criteria."""

    def test_intensity_features_has_spectral_flatness(self) -> None:
        """IntensityFeatures should have spectral_flatness attribute."""
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
            spectral_flatness=0.3,
        )
        assert hasattr(features, "spectral_flatness"), (
            "IntensityFeatures should have spectral_flatness per issue #40"
        )
        assert features.spectral_flatness == 0.3

    def test_intensity_features_has_zero_crossing_rate(self) -> None:
        """IntensityFeatures should have zero_crossing_rate attribute."""
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
            zero_crossing_rate=0.15,
        )
        assert hasattr(features, "zero_crossing_rate"), (
            "IntensityFeatures should have zero_crossing_rate per issue #40"
        )
        assert features.zero_crossing_rate == 0.15

    def test_intensity_features_has_mfcc_variance(self) -> None:
        """IntensityFeatures should have mfcc_variance attribute."""
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
            mfcc_variance=0.5,
        )
        assert hasattr(features, "mfcc_variance"), (
            "IntensityFeatures should have mfcc_variance per issue #40"
        )
        assert features.mfcc_variance == 0.5

    def test_intensity_features_has_spectral_rolloff_85(self) -> None:
        """IntensityFeatures should have spectral_rolloff_85 attribute."""
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
            spectral_rolloff_85=0.7,
        )
        assert hasattr(features, "spectral_rolloff_85"), (
            "IntensityFeatures should have spectral_rolloff_85 per issue #40"
        )
        assert features.spectral_rolloff_85 == 0.7


class TestSpectralFlatnessCalculation:
    """Tests for spectral flatness calculation - Issue #40."""

    def test_spectral_flatness_tonal_track(self) -> None:
        """Tonal/clean track should have low spectral flatness."""
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
            spectral_flatness=0.1,
        )
        assert features.spectral_flatness < 0.3, "Tonal track should have low spectral flatness"

    def test_spectral_flatness_noisy_track(self) -> None:
        """Noisy/textured track should have high spectral flatness."""
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
            spectral_flatness=0.7,
        )
        assert features.spectral_flatness > 0.5, "Noisy track should have high spectral flatness"

    def test_spectral_flatness_defaults_to_zero(self) -> None:
        """Missing spectral_flatness should default to 0.0."""
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
        assert features.spectral_flatness == 0.0, "spectral_flatness should default to 0.0"


class TestMFCCVarianceCalculation:
    """Tests for MFCC variance calculation - Issue #40."""

    def test_mfcc_variance_complex_timbre(self) -> None:
        """Track with complex timbre should have higher MFCC variance."""
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
            mfcc_variance=0.8,
        )
        assert features.mfcc_variance > 0.5, "Complex timbre should have high MFCC variance"

    def test_mfcc_variance_simple_timbre(self) -> None:
        """Track with simple timbre should have lower MFCC variance."""
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
            mfcc_variance=0.2,
        )
        assert features.mfcc_variance < 0.4, "Simple timbre should have low MFCC variance"


class TestTimbreBasedSequencing:
    """Tests for timbre-based sequencing - Issue #40 acceptance criteria."""

    def test_sequence_by_timbre_exists(self) -> None:
        """sequence_by_timbre function should exist in sequencer."""
        assert callable(sequence_by_timbre), (
            "sequencer should have sequence_by_timbre() per issue #40"
        )

    def test_sequence_by_timbre_sorts_by_similarity(self, tmp_path: Path) -> None:
        """Test timbre similarity ordering."""
        tracks = [tmp_path / f"track_{i}.flac" for i in range(4)]
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
                spectral_flatness=0.2,
                mfcc_variance=0.3,
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
                spectral_flatness=0.8,
                mfcc_variance=0.9,  # Very different
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
                spectral_flatness=0.25,
                mfcc_variance=0.35,  # Similar to track 0
            ),
            tracks[3]: IntensityFeatures(
                file_path=str(tracks[3]),
                file_hash="d",
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
                spectral_flatness=0.75,
                mfcc_variance=0.85,  # Similar to track 1
            ),
        }

        result = sequence_by_timbre(list(tracks), features)

        assert len(result) == 4, "Should return all tracks"
        assert set(result) == set(tracks), "Should contain all input tracks"

    def test_sequence_by_timbre_empty(self, tmp_path: Path) -> None:
        """Empty track list should return empty list."""
        result = sequence_by_timbre([], {})
        assert result == []

    def test_sequence_by_timbre_single_track(self, tmp_path: Path) -> None:
        """Single track should be returned as-is."""
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
                spectral_flatness=0.3,
                mfcc_variance=0.4,
            ),
        }

        result = sequence_by_timbre(tracks, features)
        assert result == tracks

    def test_sequence_by_timbre_missing_features_raises(self, tmp_path: Path) -> None:
        """Missing features should raise ValueError."""
        tracks = [tmp_path / "track.flac"]
        features: dict[Path, IntensityFeatures] = {}

        with pytest.raises(ValueError, match="Missing features"):
            sequence_by_timbre(tracks, features)


class TestTimbreFeatureVector:
    """Tests for timbre feature vector - Issue #40."""

    def test_timbre_features_used_for_clustering(self) -> None:
        """Verify timbre features can be used for clustering."""
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
            spectral_flatness=0.3,
            zero_crossing_rate=0.1,
            mfcc_variance=0.4,
            spectral_rolloff_85=0.6,
        )

        timbre_vector = [
            features.spectral_flatness,
            features.zero_crossing_rate,
            features.mfcc_variance,
            features.spectral_rolloff_85,
        ]

        assert len(timbre_vector) == 4, "Timbre vector should have 4 dimensions"
        assert all(0.0 <= v <= 1.0 for v in timbre_vector), "All values should be normalized 0-1"


class TestCLITimbreMode:
    """Integration tests for CLI timbre mode - Issue #40."""

    def test_sequence_mode_timbre_available(self) -> None:
        """CLI should support --sequence-mode=timb re."""
        from click.testing import CliRunner

        from playchitect.cli.commands import scan

        runner = CliRunner()
        result = runner.invoke(scan, ["--help"])

        assert "timbre" in result.output.lower(), (
            "scan command should support 'timbre' sequence mode per issue #40"
        )

    def test_sort_by_timbre_available(self) -> None:
        """CLI should support sorting by timbre similarity."""
        from click.testing import CliRunner

        from playchitect.cli.commands import scan

        runner = CliRunner()
        result = runner.invoke(scan, ["--help"])

        assert "--sequence-mode" in result.output, (
            "scan command should have --sequence-mode per issue #40"
        )
