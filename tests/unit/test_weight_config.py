"""Unit tests for weight_config module."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from playchitect.core.clustering import PlaylistClusterer
from playchitect.core.weighting import FEATURE_NAMES
from playchitect.utils.weight_config import (
    WeightOverrides,
    apply_weight_overrides,
    load_weight_overrides,
)


class TestWeightOverrides:
    """Tests for the WeightOverrides dataclass."""

    def test_default_values_are_none(self) -> None:
        """All fields should default to None."""
        overrides = WeightOverrides()
        assert overrides.bpm is None
        assert overrides.rms_energy is None
        assert overrides.brightness is None
        assert overrides.sub_bass is None
        assert overrides.kick_energy is None
        assert overrides.bass_harmonics is None
        assert overrides.percussiveness is None
        assert overrides.onset_strength is None

    def test_partial_initialization(self) -> None:
        """Can initialize with specific values while others remain None."""
        overrides = WeightOverrides(bpm=2.0, brightness=1.5)
        assert overrides.bpm == 2.0
        assert overrides.brightness == 1.5
        assert overrides.rms_energy is None
        assert overrides.sub_bass is None

    def test_all_fields_initialization(self) -> None:
        """Can initialize all fields at once."""
        overrides = WeightOverrides(
            bpm=1.0,
            rms_energy=2.0,
            brightness=3.0,
            sub_bass=4.0,
            kick_energy=5.0,
            bass_harmonics=6.0,
            percussiveness=7.0,
            onset_strength=8.0,
        )
        assert overrides.bpm == 1.0
        assert overrides.rms_energy == 2.0
        assert overrides.brightness == 3.0
        assert overrides.sub_bass == 4.0
        assert overrides.kick_energy == 5.0
        assert overrides.bass_harmonics == 6.0
        assert overrides.percussiveness == 7.0
        assert overrides.onset_strength == 8.0


class TestLoadWeightOverrides:
    """Tests for the load_weight_overrides function."""

    def test_load_single_override(self, tmp_path: Path) -> None:
        """Load a YAML with a single weight override."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("weights:\n  bpm: 2.0\n")

        overrides = load_weight_overrides(config_path)

        assert overrides.bpm == 2.0
        assert overrides.rms_energy is None
        assert overrides.brightness is None
        assert overrides.sub_bass is None
        assert overrides.kick_energy is None
        assert overrides.bass_harmonics is None
        assert overrides.percussiveness is None
        assert overrides.onset_strength is None

    def test_load_multiple_overrides(self, tmp_path: Path) -> None:
        """Load a YAML with multiple weight overrides."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("weights:\n  bpm: 2.0\n  rms_energy: 1.5\n  brightness: 0.5\n")

        overrides = load_weight_overrides(config_path)

        assert overrides.bpm == 2.0
        assert overrides.rms_energy == 1.5
        assert overrides.brightness == 0.5
        assert overrides.sub_bass is None
        assert overrides.kick_energy is None

    def test_load_all_overrides(self, tmp_path: Path) -> None:
        """Load a YAML with all 8 weight overrides."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "weights:\n"
            "  bpm: 1.0\n"
            "  rms_energy: 2.0\n"
            "  brightness: 3.0\n"
            "  sub_bass: 4.0\n"
            "  kick_energy: 5.0\n"
            "  bass_harmonics: 6.0\n"
            "  percussiveness: 7.0\n"
            "  onset_strength: 8.0\n"
        )

        overrides = load_weight_overrides(config_path)

        assert overrides.bpm == 1.0
        assert overrides.rms_energy == 2.0
        assert overrides.brightness == 3.0
        assert overrides.sub_bass == 4.0
        assert overrides.kick_energy == 5.0
        assert overrides.bass_harmonics == 6.0
        assert overrides.percussiveness == 7.0
        assert overrides.onset_strength == 8.0

    def test_load_empty_weights(self, tmp_path: Path) -> None:
        """Load a YAML with empty weights section."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("weights:\n")

        overrides = load_weight_overrides(config_path)

        assert overrides.bpm is None
        assert overrides.rms_energy is None
        assert overrides.brightness is None

    def test_load_no_weights_key(self, tmp_path: Path) -> None:
        """Load a YAML without a weights key."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("other_config:\n  key: value\n")

        overrides = load_weight_overrides(config_path)

        assert overrides.bpm is None
        assert overrides.rms_energy is None

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """Load an empty YAML file."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        overrides = load_weight_overrides(config_path)

        assert overrides.bpm is None
        assert overrides.rms_energy is None

    def test_load_null_values(self, tmp_path: Path) -> None:
        """Load a YAML with explicit null values."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("weights:\n  bpm: null\n  rms_energy: 2.0\n")

        overrides = load_weight_overrides(config_path)

        assert overrides.bpm is None
        assert overrides.rms_energy == 2.0

    def test_load_ignores_invalid_fields(self, tmp_path: Path) -> None:
        """Invalid field names in YAML are ignored."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "weights:\n  bpm: 2.0\n  invalid_field: 999\n  another_invalid: 100\n"
        )

        overrides = load_weight_overrides(config_path)

        assert overrides.bpm == 2.0
        # invalid fields should not cause errors
        assert overrides.rms_energy is None

    def test_load_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Loading a nonexistent file raises FileNotFoundError."""
        config_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            load_weight_overrides(config_path)

    def test_load_invalid_yaml_raises(self, tmp_path: Path) -> None:
        """Loading invalid YAML raises YAMLError."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("invalid: yaml: content: [\n")

        with pytest.raises(yaml.YAMLError):
            load_weight_overrides(config_path)


class TestApplyWeightOverrides:
    """Tests for the apply_weight_overrides function."""

    def test_apply_single_override_1d(self) -> None:
        """Apply a single override to a 1D weight array."""
        weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        overrides = WeightOverrides(bpm=2.0)

        result = apply_weight_overrides(weights, overrides, FEATURE_NAMES)

        assert result[0] == 2.0  # bpm is first feature
        assert result[1] == 0.2  # unchanged
        assert result[2] == 0.3  # unchanged
        # Original weights should not be modified
        assert weights[0] == 0.1

    def test_apply_multiple_overrides_1d(self) -> None:
        """Apply multiple overrides to a 1D weight array."""
        weights = np.ones(8) * 0.125  # uniform weights
        overrides = WeightOverrides(bpm=2.0, rms_energy=1.5, brightness=0.5)

        result = apply_weight_overrides(weights, overrides, FEATURE_NAMES)

        assert result[0] == 2.0  # bpm
        assert result[1] == 1.5  # rms_energy
        assert result[2] == 0.5  # brightness
        assert result[3] == 0.125  # unchanged

    def test_apply_no_overrides_1d(self) -> None:
        """Apply no overrides returns copy of original."""
        weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        overrides = WeightOverrides()

        result = apply_weight_overrides(weights, overrides, FEATURE_NAMES)

        np.testing.assert_array_equal(result, weights)
        # Should be a copy, not the same object
        assert result is not weights

    def test_apply_overrides_2d(self) -> None:
        """Apply overrides to a 2D per-cluster weight array."""
        # 3 clusters, 8 features
        weights = np.ones((3, 8)) * 0.125
        overrides = WeightOverrides(bpm=2.0)

        result = apply_weight_overrides(weights, overrides, FEATURE_NAMES)

        # All clusters should have bpm overridden
        assert result[0, 0] == 2.0
        assert result[1, 0] == 2.0
        assert result[2, 0] == 2.0
        # Other features unchanged
        assert result[0, 1] == 0.125

    def test_apply_all_overrides_1d(self) -> None:
        """Apply overrides to all 8 features."""
        weights = np.ones(8) * 0.125
        overrides = WeightOverrides(
            bpm=1.0,
            rms_energy=2.0,
            brightness=3.0,
            sub_bass=4.0,
            kick_energy=5.0,
            bass_harmonics=6.0,
            percussiveness=7.0,
            onset_strength=8.0,
        )

        result = apply_weight_overrides(weights, overrides, FEATURE_NAMES)

        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        np.testing.assert_array_equal(result, expected)

    def test_apply_with_custom_feature_names(self) -> None:
        """Apply overrides with custom feature name order."""
        # Custom order: reverse of default
        custom_names = tuple(reversed(FEATURE_NAMES))
        weights = np.ones(8) * 0.125
        overrides = WeightOverrides(bpm=2.0)  # bpm is last in reversed order

        result = apply_weight_overrides(weights, overrides, custom_names)

        # bpm should be at index 7 in reversed order
        assert result[7] == 2.0
        assert result[0] == 0.125  # onset_strength (was last, now first)

    def test_apply_mismatched_length_raises(self) -> None:
        """Mismatched weights length and feature_names length raises error."""
        weights = np.ones(5)  # Wrong length
        overrides = WeightOverrides(bpm=2.0)

        # This won't raise during apply, but will simply not find the feature
        # if the names don't match
        result = apply_weight_overrides(weights, overrides, FEATURE_NAMES)
        # No error, just doesn't find 'bpm' in first 5 custom features

    def test_apply_invalid_ndim_raises(self) -> None:
        """Weights with ndim > 2 raises ValueError."""
        weights = np.ones((2, 3, 4))  # 3D array
        overrides = WeightOverrides(bpm=2.0)

        with pytest.raises(ValueError, match="Unsupported weights ndim"):
            apply_weight_overrides(weights, overrides, FEATURE_NAMES)


class TestPlaylistClustererIntegration:
    """Integration tests for PlaylistClusterer with weight overrides."""

    def test_clusterer_accepts_weight_overrides(self) -> None:
        """PlaylistClusterer accepts weight_overrides in constructor."""
        overrides = WeightOverrides(bpm=2.0)

        clusterer = PlaylistClusterer(
            target_tracks_per_playlist=10,
            weight_overrides=overrides,
        )

        assert clusterer.weight_overrides is overrides
        assert clusterer.weight_overrides.bpm == 2.0

    def test_clusterer_none_overrides(self) -> None:
        """PlaylistClusterer works with None weight_overrides."""
        clusterer = PlaylistClusterer(
            target_tracks_per_playlist=10,
            weight_overrides=None,
        )

        assert clusterer.weight_overrides is None

    def test_clusterer_default_overrides_is_none(self) -> None:
        """PlaylistClusterer defaults weight_overrides to None."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=10)

        assert clusterer.weight_overrides is None
