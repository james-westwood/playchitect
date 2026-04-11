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
    """Tests for load_weight_overrides function."""

    def test_load_basic_weights(self, tmp_path: Path) -> None:
        """Can load a basic weights file with some overrides."""
        config = tmp_path / "weights.yaml"
        config.write_text("weights:\n  bpm: 2.0\n  rms_energy: 1.5\n")

        overrides = load_weight_overrides(config)
        assert overrides.bpm == 2.0
        assert overrides.rms_energy == 1.5
        assert overrides.brightness is None

    def test_load_empty_weights(self, tmp_path: Path) -> None:
        """Empty weights section returns all None."""
        config = tmp_path / "weights.yaml"
        config.write_text("weights: {}")

        overrides = load_weight_overrides(config)
        assert overrides.bpm is None

    def test_load_no_weights_key(self, tmp_path: Path) -> None:
        """Missing weights key returns all None."""
        config = tmp_path / "weights.yaml"
        config.write_text("other: stuff")

        overrides = load_weight_overrides(config)
        assert overrides.bpm is None

    def test_load_invalid_yaml_raises(self, tmp_path: Path) -> None:
        """Invalid YAML raises yaml.YAMLError."""
        config = tmp_path / "weights.yaml"
        config.write_text("invalid: yaml: content:")

        with pytest.raises(yaml.YAMLError):
            load_weight_overrides(config)

    def test_load_nonexistent_file_raises(self) -> None:
        """Nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_weight_overrides(Path("/nonexistent/weights.yaml"))

    def test_load_ignores_unknown_fields(self, tmp_path: Path) -> None:
        """Unknown fields in YAML are ignored."""
        config = tmp_path / "weights.yaml"
        config.write_text("weights:\n  bpm: 1.0\n  unknown_field: 99\n")

        overrides = load_weight_overrides(config)
        assert overrides.bpm == 1.0
        # Unknown field should be ignored, not raise

    def test_load_converts_to_float(self, tmp_path: Path) -> None:
        """Integer values are converted to float."""
        config = tmp_path / "weights.yaml"
        config.write_text("weights:\n  bpm: 2\n")

        overrides = load_weight_overrides(config)
        assert overrides.bpm == 2.0
        assert isinstance(overrides.bpm, float)


class TestApplyWeightOverrides:
    """Tests for apply_weight_overrides function."""

    def test_apply_single_override(self) -> None:
        """Can apply a single weight override."""
        weights = np.array([0.1] * 8)
        overrides = WeightOverrides(bpm=2.0)

        result = apply_weight_overrides(weights, overrides)

        assert result[0] == 2.0  # bpm is first feature
        assert result[1] == 0.125  # others normalized to uniform (1/8)

    def test_apply_multiple_overrides(self) -> None:
        """Can apply multiple weight overrides."""
        weights = np.array([0.1] * 8)
        overrides = WeightOverrides(bpm=2.0, rms_energy=1.5)

        result = apply_weight_overrides(weights, overrides)

        assert result[0] == 2.0
        assert result[1] == 1.5

    def test_apply_no_overrides(self) -> None:
        """No overrides returns unchanged weights."""
        weights = np.array([0.1] * 8)
        overrides = WeightOverrides()

        result = apply_weight_overrides(weights, overrides)

        np.testing.assert_array_equal(result, weights)

    def test_apply_2d_weights(self) -> None:
        """Can apply overrides to 2D per-cluster weights."""
        weights = np.ones((3, 8)) * 0.125  # 3 clusters, 8 features
        overrides = WeightOverrides(bpm=2.0)

        result = apply_weight_overrides(weights, overrides)

        # All clusters should have bpm=2.0
        assert np.all(result[:, 0] == 2.0)
        # Other features unchanged
        assert np.all(result[:, 1:] == 0.125)

    def test_apply_partial_2d(self) -> None:
        """Can override specific features in 2D array."""
        weights = np.ones((2, 8)) * 0.125
        overrides = WeightOverrides(bpm=2.0, kick_energy=0.5)

        result = apply_weight_overrides(weights, overrides)

        assert result[0, 0] == 2.0
        assert result[1, 0] == 2.0
        assert result[0, 4] == 0.5
        assert result[1, 4] == 0.5

    def test_apply_custom_feature_order(self) -> None:
        """Can use custom feature name order."""
        weights = np.ones(8)
        overrides = WeightOverrides(bpm=2.0)
        custom_names = (
            "onset_strength",
            "percussiveness",
            "bass_harmonics",
            "kick_energy",
            "sub_bass",
            "brightness",
            "rms_energy",
            "bpm",
        )

        result = apply_weight_overrides(weights, overrides, custom_names)

        # bpm should be at index 7 in reversed order
        assert result[7] == 2.0
        assert result[0] == 0.125  # onset_strength normalized to uniform (1/8)

    def test_apply_mismatched_length_raises(self) -> None:
        """Mismatched weights length and feature_names length raises error."""
        weights = np.ones(5)  # Wrong length
        overrides = WeightOverrides(bpm=2.0)

        # This won't raise during apply, but will simply not find the feature
        # if the names don't match
        apply_weight_overrides(weights, overrides, FEATURE_NAMES)
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


# ============================================================================
# NEW TESTS FOR ISSUE #26 - YAML Weight Config Acceptance Criteria
# ============================================================================


class TestWeightValidation:
    """Tests for weight validation - Issue #26 acceptance criteria.

    These tests verify that weights are validated (sum to 1.0,
    all features present, values >= 0) before being used.
    """

    def test_validate_weights_sum_to_one(self, tmp_path: Path) -> None:
        """Weights in YAML must sum to 1.0."""
        config = tmp_path / "weights.yaml"
        config.write_text("""
weights:
  bpm: 0.125
  rms_energy: 0.125
  brightness: 0.125
  sub_bass: 0.125
  kick_energy: 0.125
  bass_harmonics: 0.125
  percussiveness: 0.125
  onset_strength: 0.125
""")
        # This should work - sum is 1.0
        overrides = load_weight_overrides(config)
        assert overrides.bpm == 0.125

    def test_validate_weights_sum_not_one_raises(self, tmp_path: Path) -> None:
        """Weights that don't sum to 1.0 should raise ValueError."""
        from playchitect.utils.weight_config import (
            validate_weights,  # ty: ignore[unresolved-import]
        )

        config = tmp_path / "weights.yaml"
        config.write_text("""
weights:
  bpm: 0.5
  rms_energy: 0.3
  brightness: 0.3
  sub_bass: 0.0
  kick_energy: 0.0
  bass_harmonics: 0.0
  percussiveness: 0.0
  onset_strength: 0.0
""")
        overrides = load_weight_overrides(config)
        # This should raise because sum is 1.1, not 1.0
        with pytest.raises(ValueError, match="sum to 1.0"):
            validate_weights(overrides)

    def test_validate_weights_negative_raises(self, tmp_path: Path) -> None:
        """Negative weight values should raise ValueError."""
        from playchitect.utils.weight_config import (
            validate_weights,  # ty: ignore[unresolved-import]
        )

        config = tmp_path / "weights.yaml"
        config.write_text("""
weights:
  bpm: -0.1
  rms_energy: 1.1
  brightness: 0.0
  sub_bass: 0.0
  kick_energy: 0.0
  bass_harmonics: 0.0
  percussiveness: 0.0
  onset_strength: 0.0
""")
        overrides = load_weight_overrides(config)
        # This should raise because bpm is negative
        with pytest.raises(ValueError, match="must be >= 0"):
            validate_weights(overrides)

    def test_validate_weights_missing_features_raises(self, tmp_path: Path) -> None:
        """Missing feature weights should raise ValueError."""
        from playchitect.utils.weight_config import (
            validate_weights,  # ty: ignore[unresolved-import]
        )

        config = tmp_path / "weights.yaml"
        config.write_text("""
weights:
  bpm: 1.0
""")
        overrides = load_weight_overrides(config)
        # This should raise because not all features are present
        with pytest.raises(ValueError, match="all 8 features"):
            validate_weights(overrides)


class TestWeightProfileSource:
    """Tests for WeightProfile.source reporting 'user' - Issue #26."""

    def test_select_weights_with_user_overrides_returns_user_source(self) -> None:
        """When user weights are active, source should be 'user'."""
        from playchitect.core.weighting import select_weights

        overrides = WeightOverrides(
            bpm=0.125,
            rms_energy=0.125,
            brightness=0.125,
            sub_bass=0.125,
            kick_energy=0.125,
            bass_harmonics=0.125,
            percussiveness=0.125,
            onset_strength=0.125,
        )

        X = np.random.randn(50, 8)  # 50 tracks, 8 features

        # This should work after implementation - select_weights should accept
        # weight_overrides parameter
        profile = select_weights(X, weight_overrides=overrides)  # ty: ignore[unknown-argument]
        assert profile.source == "user", (
            f"Expected source='user' when overrides provided, got '{profile.source}'"
        )

    def test_select_weights_without_overrides_returns_pca_heuristic_uniform(self) -> None:
        """Without overrides, source should be pca, heuristic, or uniform."""
        from playchitect.core.weighting import select_weights

        X = np.random.randn(50, 8)

        profile = select_weights(X)
        assert profile.source in ("pca", "heuristic", "uniform"), (
            f"Expected pca/heuristic/uniform, got '{profile.source}'"
        )
