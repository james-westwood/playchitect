"""Tests for playchitect.core.features.build_feature_vector."""

from pathlib import Path

import numpy as np

from playchitect.core.features import build_feature_vector
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata


def make_metadata(name: str, bpm: float | None = 128.0, duration: float = 360.0) -> TrackMetadata:
    """Create TrackMetadata for testing."""
    return TrackMetadata(filepath=Path(name), bpm=bpm, duration=duration)


def make_intensity(
    name: str,
    rms: float = 0.5,
    brightness: float = 0.5,
    sub_bass: float = 0.3,
    kick: float = 0.6,
    harmonics: float = 0.4,
    perc: float = 0.5,
    onset: float = 0.5,
) -> IntensityFeatures:
    """Create IntensityFeatures for testing."""
    return IntensityFeatures(
        file_path=Path(name),
        file_hash="deadbeef",  # pragma: allowlist secret
        rms_energy=rms,
        brightness=brightness,
        sub_bass_energy=sub_bass,
        kick_energy=kick,
        bass_harmonics=harmonics,
        percussiveness=perc,
        onset_strength=onset,
        camelot_key="8B",
        key_index=0.0,
    )


class TestBuildFeatureVector:
    """Tests for build_feature_vector(metadata, features) -> np.ndarray | None."""

    def test_returns_array_of_length_8(self) -> None:
        """The returned array must have exactly 8 elements."""
        meta = make_metadata("track1.mp3", bpm=128.0)
        feats = make_intensity("track1.mp3")
        result = build_feature_vector(meta, feats)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == 8

    def test_correct_ordering_matches_feature_names(self) -> None:
        """The 8 values must match FEATURE_NAMES ordering:
        bpm, rms_energy, brightness, sub_bass_energy, kick_energy,
        bass_harmonics, percussiveness, onset_strength.
        """
        meta = make_metadata("track1.mp3", bpm=128.0)
        feats = make_intensity(
            "track1.mp3",
            rms=0.1,
            brightness=0.2,
            sub_bass=0.3,
            kick=0.4,
            harmonics=0.5,
            perc=0.6,
            onset=0.7,
        )
        result = build_feature_vector(meta, feats)
        assert result is not None
        expected = np.array([128.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        np.testing.assert_array_almost_equal(result, expected)

    def test_returns_none_when_bpm_is_none(self) -> None:
        """Returns None when metadata.bpm is None."""
        meta = make_metadata("track1.mp3", bpm=None)
        feats = make_intensity("track1.mp3")
        result = build_feature_vector(meta, feats)
        assert result is None

    def test_returns_none_when_features_is_none(self) -> None:
        """Returns None when features is None."""
        meta = make_metadata("track1.mp3", bpm=128.0)
        result = build_feature_vector(meta, None)  # type: ignore[arg-type]
        assert result is None

    def test_returned_array_is_a_copy(self) -> None:
        """The returned array is a copy — mutating it does not change
        the original feature values.
        """
        meta = make_metadata("track1.mp3", bpm=128.0)
        feats = make_intensity("track1.mp3", rms=0.9)
        result = build_feature_vector(meta, feats)
        assert result is not None
        # Mutate the returned array
        result[0] = 999.0
        result[1] = 888.0
        # Original feature values unchanged
        assert feats.rms_energy == 0.9
        assert meta.bpm == 128.0
        # Second call returns original values again
        result2 = build_feature_vector(meta, feats)
        assert result2 is not None
        assert result2[0] == 128.0
        assert result2[1] == 0.9
