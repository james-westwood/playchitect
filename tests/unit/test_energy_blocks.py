"""Unit tests for energy_blocks module.

Tests for the EnergyBlock dataclass and suggest_blocks function.
"""

from pathlib import Path

import pytest

from playchitect.core.clustering import ClusterResult
from playchitect.core.energy_blocks import (
    EnergyBlock,
    create_custom_block,
    get_block_energy_range,
    suggest_blocks,
)
from playchitect.core.intensity_analyzer import IntensityFeatures

# ── Helpers ────────────────────────────────────────────────────────────────────


def make_cluster(
    cluster_id: int,
    track_count: int = 5,
    rms_energy: float = 0.5,
) -> ClusterResult:
    """Create a ClusterResult with specified properties."""
    tracks = [Path(f"track_{cluster_id}_{i}.mp3") for i in range(track_count)]
    return ClusterResult(
        cluster_id=cluster_id,
        tracks=tracks,
        bpm_mean=128.0,
        bpm_std=2.0,
        track_count=track_count,
        total_duration=track_count * 360.0,  # 6 min per track
        feature_means={"rms_energy": rms_energy},
    )


def make_intensity_features(
    filepath: Path,
    rms_energy: float = 0.5,
) -> IntensityFeatures:
    """Create IntensityFeatures with specified RMS energy."""
    return IntensityFeatures(
        file_path=filepath,
        file_hash="deadbeef",
        rms_energy=rms_energy,
        brightness=0.5,
        sub_bass_energy=0.3,
        kick_energy=0.6,
        bass_harmonics=0.4,
        percussiveness=0.5,
        onset_strength=0.5,
        camelot_key="8B",
        key_index=0.0,
    )


def make_cluster_with_features(
    cluster_id: int,
    rms_energy: float,
    track_count: int = 5,
) -> tuple[ClusterResult, list[tuple[Path, IntensityFeatures]]]:
    """Create a cluster and corresponding intensity features."""
    cluster = make_cluster(cluster_id, track_count, rms_energy)
    features = [(path, make_intensity_features(path, rms_energy)) for path in cluster.tracks]
    return cluster, features


# ── Tests for EnergyBlock dataclass ─────────────────────────────────────────


class TestEnergyBlock:
    """Tests for EnergyBlock dataclass."""

    def test_energy_block_creation(self) -> None:
        """Test basic EnergyBlock creation."""
        block = EnergyBlock(
            id="test-block",
            name="Test Block",
            target_duration_min=60.0,
            energy_min=0.2,
            energy_max=0.4,
            cluster_ids=[1, 2, 3],
        )
        assert block.id == "test-block"
        assert block.name == "Test Block"
        assert block.target_duration_min == 60.0
        assert block.energy_min == 0.2
        assert block.energy_max == 0.4
        assert block.cluster_ids == [1, 2, 3]


# ── Tests for suggest_blocks ─────────────────────────────────────────────────


class TestSuggestBlocks:
    """Tests for suggest_blocks function."""

    def test_empty_clusters_raises_error(self) -> None:
        """Test that empty cluster list raises ValueError."""
        with pytest.raises(ValueError, match="empty cluster list"):
            suggest_blocks([], {})

    def test_single_cluster_returns_one_block(self) -> None:
        """Test that single cluster creates a Warm-up block."""
        cluster, features_list = make_cluster_with_features(0, 0.5)
        features = {path: feat for path, feat in features_list}

        blocks = suggest_blocks([cluster], features)

        assert len(blocks) == 1
        assert blocks[0].name == "Warm-up"
        assert blocks[0].id == "warm-up"
        assert 0 in blocks[0].cluster_ids

    def test_ten_clusters_returns_4_to_5_blocks(self) -> None:
        """Test that 10 clusters are divided into 4-5 blocks.

        This is the primary acceptance criterion from the task spec.
        """
        clusters = []
        all_features: dict[Path, IntensityFeatures] = {}

        # Create 10 clusters with varying energy levels
        for i in range(10):
            rms = 0.1 + (i * 0.09)  # 0.1, 0.19, 0.28, ..., 0.91
            cluster, features = make_cluster_with_features(i, rms)
            clusters.append(cluster)
            all_features.update(dict(features))

        blocks = suggest_blocks(clusters, all_features)

        # Verify we get 4-5 blocks
        assert 4 <= len(blocks) <= 5

        # Verify all clusters are assigned
        assigned_ids = set()
        for block in blocks:
            assigned_ids.update(block.cluster_ids)
        assert assigned_ids == set(range(10))

    def test_energy_ranges_are_non_overlapping(self) -> None:
        """Test that energy ranges of consecutive blocks don't overlap."""
        clusters = []
        all_features: dict[Path, IntensityFeatures] = {}

        # Create 10 clusters with distinct energy levels
        for i in range(10):
            rms = 0.1 + (i * 0.09)
            cluster, features = make_cluster_with_features(i, rms)
            clusters.append(cluster)
            all_features.update(dict(features))

        blocks = suggest_blocks(clusters, all_features)

        # Check that energy ranges are non-overlapping (sorted by energy)
        for i in range(len(blocks) - 1):
            current_max = blocks[i].energy_max
            next_min = blocks[i + 1].energy_min
            assert current_max <= next_min, (
                f"Blocks {i} and {i + 1} have overlapping ranges: "
                f"Block {i}: [{blocks[i].energy_min}, {blocks[i].energy_max}], "
                f"Block {i + 1}: [{blocks[i + 1].energy_min}, {blocks[i + 1].energy_max}]"
            )

    def test_blocks_sorted_by_energy(self) -> None:
        """Test that blocks are returned sorted by energy level."""
        clusters = []
        all_features: dict[Path, IntensityFeatures] = {}

        # Create clusters with varying energy
        for i in range(8):
            rms = 0.1 + (i * 0.11)
            cluster, features = make_cluster_with_features(i, rms)
            clusters.append(cluster)
            all_features.update(dict(features))

        blocks = suggest_blocks(clusters, all_features)

        # Verify blocks are sorted by energy_min
        for i in range(len(blocks) - 1):
            assert blocks[i].energy_min <= blocks[i + 1].energy_min

    def test_target_duration_proportional_to_track_count(self) -> None:
        """Test that target_duration_min is proportional to track count."""
        clusters = []
        all_features: dict[Path, IntensityFeatures] = {}

        # Create clusters with different track counts
        cluster_sizes = [3, 5, 8, 10, 6]
        for i, size in enumerate(cluster_sizes):
            rms = 0.1 + (i * 0.18)
            cluster, features = make_cluster_with_features(i, rms, track_count=size)
            clusters.append(cluster)
            all_features.update(dict(features))

        blocks = suggest_blocks(clusters, all_features)

        # Larger track counts should generally result in larger durations
        # (within the same block)
        for block in blocks:
            total_tracks = sum(c.track_count for c in clusters if c.cluster_id in block.cluster_ids)
            expected_duration = total_tracks * 6.0  # 6 min per track
            assert block.target_duration_min == expected_duration

    def test_block_names_in_order(self) -> None:
        """Test that standard block names appear in expected order."""
        clusters = []
        all_features: dict[Path, IntensityFeatures] = {}

        for i in range(10):
            rms = 0.1 + (i * 0.09)
            cluster, features = make_cluster_with_features(i, rms)
            clusters.append(cluster)
            all_features.update(dict(features))

        blocks = suggest_blocks(clusters, all_features)

        # First block should always be Warm-up
        assert blocks[0].name == "Warm-up"

        # Block names should follow expected progression
        expected_names = ["Warm-up", "Build", "Peak", "Sustain", "Wind Down"]
        for i, block in enumerate(blocks):
            assert block.name in expected_names


# ── Tests for create_custom_block ────────────────────────────────────────────


class TestCreateCustomBlock:
    """Tests for create_custom_block function."""

    def test_custom_block_creation(self) -> None:
        """Test creating a custom block."""
        block = create_custom_block("custom-1")

        assert block.id == "custom-1"
        assert block.name == "Custom"
        assert block.target_duration_min == 60.0
        assert block.energy_min == 0.0
        assert block.energy_max == 1.0
        assert block.cluster_ids == []

    def test_custom_block_with_custom_params(self) -> None:
        """Test creating a custom block with custom parameters."""
        block = create_custom_block(
            block_id="my-block",
            target_duration_min=45.0,
            energy_min=0.3,
            energy_max=0.7,
        )

        assert block.id == "my-block"
        assert block.name == "Custom"
        assert block.target_duration_min == 45.0
        assert block.energy_min == 0.3
        assert block.energy_max == 0.7


# ── Tests for get_block_energy_range ────────────────────────────────────────


class TestGetBlockEnergyRange:
    """Tests for get_block_energy_range function."""

    def test_empty_blocks(self) -> None:
        """Test that empty list returns default range."""
        min_e, max_e = get_block_energy_range([])
        assert min_e == 0.0
        assert max_e == 1.0

    def test_single_block(self) -> None:
        """Test range from single block."""
        block = EnergyBlock(
            id="test",
            name="Test",
            target_duration_min=60.0,
            energy_min=0.3,
            energy_max=0.5,
            cluster_ids=[1],
        )
        min_e, max_e = get_block_energy_range([block])
        assert min_e == 0.3
        assert max_e == 0.5

    def test_multiple_blocks(self) -> None:
        """Test range across multiple blocks."""
        blocks = [
            EnergyBlock(
                id="low",
                name="Low",
                target_duration_min=60.0,
                energy_min=0.1,
                energy_max=0.3,
                cluster_ids=[1],
            ),
            EnergyBlock(
                id="high",
                name="High",
                target_duration_min=60.0,
                energy_min=0.7,
                energy_max=0.9,
                cluster_ids=[2],
            ),
        ]
        min_e, max_e = get_block_energy_range(blocks)
        assert min_e == 0.1
        assert max_e == 0.9


# ── Integration-style tests ─────────────────────────────────────────────────


class TestSuggestBlocksIntegration:
    """Integration-style tests for suggest_blocks."""

    def test_realistic_cluster_scenario(self) -> None:
        """Test with a realistic scenario of 12 clusters."""
        clusters = []
        all_features: dict[Path, IntensityFeatures] = {}

        # Simulate a realistic distribution: more low-energy, fewer high-energy
        rms_values = [0.15, 0.18, 0.22, 0.28, 0.35, 0.42, 0.55, 0.65, 0.75, 0.82, 0.88, 0.92]
        track_counts = [8, 7, 9, 6, 8, 7, 5, 4, 4, 3, 2, 3]

        for i, (rms, count) in enumerate(zip(rms_values, track_counts)):
            cluster, features = make_cluster_with_features(i, rms, track_count=count)
            clusters.append(cluster)
            all_features.update(dict(features))

        blocks = suggest_blocks(clusters, all_features)

        # Should create 4-5 blocks
        assert 4 <= len(blocks) <= 5

        # All clusters should be covered
        all_cluster_ids = set(range(12))
        covered_ids = set()
        for block in blocks:
            covered_ids.update(block.cluster_ids)
        assert covered_ids == all_cluster_ids

        # Energy should generally increase through blocks
        for i in range(len(blocks) - 1):
            assert blocks[i].energy_min <= blocks[i + 1].energy_min

    def test_handles_clusters_without_features(self) -> None:
        """Test that clusters with missing features get 0.0 energy."""
        cluster = make_cluster(0, 3, 0.5)

        # Only provide features for some tracks
        features = {
            cluster.tracks[0]: make_intensity_features(cluster.tracks[0], 0.5),
            cluster.tracks[1]: make_intensity_features(cluster.tracks[1], 0.5),
            # Missing features for tracks[2]
        }

        blocks = suggest_blocks([cluster], features)

        assert len(blocks) == 1
        assert blocks[0].name == "Warm-up"
        # Should average available features
        assert blocks[0].energy_min == 0.5  # Mean of two 0.5 values
        assert blocks[0].energy_max == 0.5
