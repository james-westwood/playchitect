"""Unit tests for arc_sequencer module."""

from pathlib import Path

import pytest

from playchitect.core.arc_sequencer import (
    BUILTIN_PRESETS,
    EnergyArcPreset,
    apply_arc,
)
from playchitect.core.clustering import ClusterResult


class TestEnergyArcPreset:
    """Test EnergyArcPreset dataclass."""

    def test_valid_preset_creation(self) -> None:
        """Test creating a valid EnergyArcPreset."""
        preset = EnergyArcPreset(
            name="Test Arc",
            description="A test preset",
            arc_curve=[0.0, 0.5, 1.0],
        )
        assert preset.name == "Test Arc"
        assert preset.description == "A test preset"
        assert preset.arc_curve == [0.0, 0.5, 1.0]

    def test_empty_arc_curve_raises(self) -> None:
        """Test that empty arc_curve raises ValueError."""
        with pytest.raises(ValueError, match="arc_curve must not be empty"):
            EnergyArcPreset(
                name="Invalid",
                description="Should fail",
                arc_curve=[],
            )

    def test_value_below_zero_raises(self) -> None:
        """Test that values below 0.0 raise ValueError."""
        with pytest.raises(ValueError, match="must be in range"):
            EnergyArcPreset(
                name="Invalid",
                description="Should fail",
                arc_curve=[-0.1, 0.5, 1.0],
            )

    def test_value_above_one_raises(self) -> None:
        """Test that values above 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="must be in range"):
            EnergyArcPreset(
                name="Invalid",
                description="Should fail",
                arc_curve=[0.0, 0.5, 1.1],
            )

    def test_boundary_values_allowed(self) -> None:
        """Test that boundary values 0.0 and 1.0 are allowed."""
        preset = EnergyArcPreset(
            name="Boundary Test",
            description="Tests boundary values",
            arc_curve=[0.0, 1.0],
        )
        assert preset.arc_curve == [0.0, 1.0]


class TestBuiltinPresets:
    """Test builtin presets are correctly defined."""

    def test_has_five_presets(self) -> None:
        """Test that exactly 5 builtin presets exist."""
        assert len(BUILTIN_PRESETS) == 5

    def test_all_presets_have_valid_curves(self) -> None:
        """Test all builtin presets have 5-element curves in valid range."""
        expected_names = {
            "Warmup Ramp",
            "Peak Hour",
            "Sunrise",
            "Deep Journey",
            "Closing Down",
        }
        actual_names = {p.name for p in BUILTIN_PRESETS}
        assert actual_names == expected_names

        for preset in BUILTIN_PRESETS:
            assert len(preset.arc_curve) == 5
            assert all(0.0 <= v <= 1.0 for v in preset.arc_curve)

    def test_warmup_ramp_curve(self) -> None:
        """Test Warmup Ramp has expected ascending curve."""
        preset = next(p for p in BUILTIN_PRESETS if p.name == "Warmup Ramp")
        assert preset.arc_curve == [0.2, 0.4, 0.6, 0.8, 1.0]

    def test_closing_down_curve(self) -> None:
        """Test Closing Down has expected descending curve."""
        preset = next(p for p in BUILTIN_PRESETS if p.name == "Closing Down")
        assert preset.arc_curve == [1.0, 0.9, 0.7, 0.5, 0.3]


class TestApplyArc:
    """Test apply_arc function."""

    def _make_cluster(self, cluster_id: int, rms_energy: float) -> ClusterResult:
        """Create a ClusterResult with specified RMS energy."""
        return ClusterResult(
            cluster_id=cluster_id,
            tracks=[Path(f"track_{cluster_id}.mp3")],
            bpm_mean=120.0,
            bpm_std=0.0,
            track_count=1,
            total_duration=180.0,
            feature_means={"rms_energy": rms_energy},
        )

    def test_empty_clusters_raises(self) -> None:
        """Test that empty clusters list raises ValueError."""
        preset = EnergyArcPreset(
            name="Test",
            description="Test",
            arc_curve=[0.5, 0.5],
        )
        with pytest.raises(ValueError, match="clusters must not be empty"):
            apply_arc([], preset)

    def test_warmup_ramp_ascending_order(self) -> None:
        """Test Warmup Ramp on 5 clusters of ascending energies returns ascending order."""
        # Create 5 clusters with ascending RMS energy
        clusters = [
            self._make_cluster(0, 0.2),
            self._make_cluster(1, 0.4),
            self._make_cluster(2, 0.6),
            self._make_cluster(3, 0.8),
            self._make_cluster(4, 1.0),
        ]

        preset = next(p for p in BUILTIN_PRESETS if p.name == "Warmup Ramp")
        result = apply_arc(clusters, preset)

        # Result should be in ascending order by energy (matching the curve)
        energies = [r.feature_means["rms_energy"] for r in result if r.feature_means]
        assert energies == sorted(energies)

    def test_closing_down_descending_order(self) -> None:
        """Test Closing Down on 5 clusters returns them in descending energy order."""
        # Create 5 clusters with different energies
        clusters = [
            self._make_cluster(0, 0.2),
            self._make_cluster(1, 0.4),
            self._make_cluster(2, 0.6),
            self._make_cluster(3, 0.8),
            self._make_cluster(4, 1.0),
        ]

        preset = next(p for p in BUILTIN_PRESETS if p.name == "Closing Down")
        result = apply_arc(clusters, preset)

        # Result should be in descending order by energy
        energies = [r.feature_means["rms_energy"] for r in result if r.feature_means]
        assert energies == sorted(energies, reverse=True)

    def test_returns_same_clusters_reordered(self) -> None:
        """Test that apply_arc returns the same clusters, just reordered."""
        clusters = [
            self._make_cluster(1, 0.3),
            self._make_cluster(2, 0.7),
            self._make_cluster(3, 0.5),
        ]

        preset = EnergyArcPreset(
            name="Test",
            description="Test",
            arc_curve=[0.3, 0.5, 0.7],
        )
        result = apply_arc(clusters, preset)

        # Same clusters, same count
        assert len(result) == len(clusters)
        assert set(r.cluster_id for r in result) == set(c.cluster_id for c in clusters)

    def test_cluster_without_feature_means(self) -> None:
        """Test clusters without feature_means use fallback energy value."""
        cluster_no_features = ClusterResult(
            cluster_id=0,
            tracks=[Path("track.mp3")],
            bpm_mean=120.0,
            bpm_std=0.0,
            track_count=1,
            total_duration=180.0,
            feature_means=None,
        )

        preset = EnergyArcPreset(
            name="Test",
            description="Test",
            arc_curve=[0.5],
        )
        result = apply_arc([cluster_no_features], preset)

        assert len(result) == 1
        assert result[0].cluster_id == 0

    def test_single_cluster(self) -> None:
        """Test apply_arc with single cluster works correctly."""
        clusters = [self._make_cluster(0, 0.5)]

        preset = EnergyArcPreset(
            name="Test",
            description="Test",
            arc_curve=[0.5, 0.6, 0.7],
        )
        result = apply_arc(clusters, preset)

        assert len(result) == 1
        assert result[0].cluster_id == 0

    def test_more_clusters_than_positions(self) -> None:
        """Test apply_arc with more clusters than arc positions."""
        # 5 clusters, 3 arc positions
        clusters = [self._make_cluster(i, 0.2 + i * 0.15) for i in range(5)]

        preset = EnergyArcPreset(
            name="Test",
            description="Test",
            arc_curve=[0.2, 0.5, 0.8],
        )
        result = apply_arc(clusters, preset)

        # Should still return all clusters
        assert len(result) == 5

    def test_fewer_clusters_than_positions(self) -> None:
        """Test apply_arc with fewer clusters than arc positions."""
        # 2 clusters, 5 arc positions
        clusters = [
            self._make_cluster(0, 0.3),
            self._make_cluster(1, 0.7),
        ]

        preset = next(p for p in BUILTIN_PRESETS if p.name == "Warmup Ramp")
        result = apply_arc(clusters, preset)

        # Should return both clusters
        assert len(result) == 2

    def test_peak_hour_preset(self) -> None:
        """Test Peak Hour preset orders clusters appropriately."""
        # Create clusters with different energies
        clusters = [
            self._make_cluster(0, 0.4),
            self._make_cluster(1, 0.7),
            self._make_cluster(2, 1.0),
            self._make_cluster(3, 1.0),
            self._make_cluster(4, 0.8),
        ]

        preset = next(p for p in BUILTIN_PRESETS if p.name == "Peak Hour")
        result = apply_arc(clusters, preset)

        # Should have all 5 clusters
        assert len(result) == 5

    def test_sunrise_preset(self) -> None:
        """Test Sunrise preset (dip then rise) orders clusters correctly."""
        clusters = [
            self._make_cluster(0, 0.6),
            self._make_cluster(1, 0.4),
            self._make_cluster(2, 0.2),
            self._make_cluster(3, 0.4),
            self._make_cluster(4, 0.8),
        ]

        preset = next(p for p in BUILTIN_PRESETS if p.name == "Sunrise")
        result = apply_arc(clusters, preset)

        # Should reorder to match the sunrise curve
        assert len(result) == 5
