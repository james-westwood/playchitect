"""
Unit tests for clustering module.
"""

import pytest
import numpy as np
from pathlib import Path
from playchitect.core.clustering import PlaylistClusterer, ClusterResult
from playchitect.core.metadata_extractor import TrackMetadata


class TestPlaylistClusterer:
    """Test PlaylistClusterer class."""

    def test_initialization_requires_target(self) -> None:
        """Test that initialization requires a target parameter."""
        with pytest.raises(ValueError):
            PlaylistClusterer()

    def test_initialization_with_track_target(self) -> None:
        """Test initialization with track count target."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=25)
        assert clusterer.target_tracks == 25
        assert clusterer.target_duration is None

    def test_initialization_with_duration_target(self) -> None:
        """Test initialization with duration target."""
        clusterer = PlaylistClusterer(target_duration_per_playlist=60)
        assert clusterer.target_tracks is None
        assert clusterer.target_duration == 3600  # 60 minutes in seconds

    def test_cluster_empty_dict(self) -> None:
        """Test clustering with empty metadata dict."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=25)
        result = clusterer.cluster_by_bpm({})
        assert result == []

    def test_cluster_no_bpm_metadata(self) -> None:
        """Test clustering with tracks that have no BPM."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=25)

        metadata_dict = {
            Path("track1.mp3"): TrackMetadata(filepath=Path("track1.mp3"), bpm=None),
            Path("track2.mp3"): TrackMetadata(filepath=Path("track2.mp3"), bpm=None),
        }

        result = clusterer.cluster_by_bpm(metadata_dict)
        assert result == []

    def test_cluster_single_track(self) -> None:
        """Test clustering with single track."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=25)

        metadata_dict = {
            Path("track1.mp3"): TrackMetadata(
                filepath=Path("track1.mp3"), bpm=128.0, duration=180.0
            ),
        }

        results = clusterer.cluster_by_bpm(metadata_dict)

        assert len(results) == 1
        assert results[0].track_count == 1
        assert results[0].bpm_mean == 128.0

    def test_cluster_two_distinct_bpms(self) -> None:
        """Test clustering with two distinct BPM ranges."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=5, min_clusters=2)

        # Create tracks with distinct BPMs: 120-125 and 135-140
        metadata_dict = {}
        for i in range(10):
            bpm = 120 + i % 5 if i < 5 else 135 + i % 5
            metadata_dict[Path(f"track{i}.mp3")] = TrackMetadata(
                filepath=Path(f"track{i}.mp3"), bpm=float(bpm), duration=180.0
            )

        results = clusterer.cluster_by_bpm(metadata_dict)

        assert len(results) >= 2
        assert sum(r.track_count for r in results) == 10

        # Check that BPM means are distinct
        bpm_means = [r.bpm_mean for r in results]
        assert max(bpm_means) - min(bpm_means) > 5

    def test_cluster_respects_target_tracks(self) -> None:
        """Test that clustering respects target tracks per playlist."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=5)

        # Create 20 tracks with similar BPMs (should cluster into ~4 groups)
        metadata_dict = {}
        for i in range(20):
            bpm = 128.0 + np.random.randn() * 2  # BPM around 128 ± 2
            metadata_dict[Path(f"track{i}.mp3")] = TrackMetadata(
                filepath=Path(f"track{i}.mp3"), bpm=float(bpm), duration=180.0
            )

        results = clusterer.cluster_by_bpm(metadata_dict)

        # Should create clusters close to target size
        avg_cluster_size = sum(r.track_count for r in results) / len(results)
        assert 3 <= avg_cluster_size <= 7  # Within reasonable range of target (5)

    def test_cluster_result_contains_all_fields(self) -> None:
        """Test that ClusterResult contains all required fields."""
        # Use 10 tracks with very similar BPMs
        clusterer = PlaylistClusterer(target_tracks_per_playlist=25)

        metadata_dict = {}
        for i in range(10):
            metadata_dict[Path(f"track{i}.mp3")] = TrackMetadata(
                filepath=Path(f"track{i}.mp3"),
                bpm=128.0 + i * 0.1,  # Very similar BPMs (128.0-128.9)
                duration=180.0,
            )

        results = clusterer.cluster_by_bpm(metadata_dict)

        # Verify at least one cluster was created
        assert len(results) > 0

        # Check that all clusters have required fields
        for result in results:
            assert hasattr(result, "cluster_id")
            assert hasattr(result, "tracks")
            assert hasattr(result, "bpm_mean")
            assert hasattr(result, "bpm_std")
            assert hasattr(result, "track_count")
            assert hasattr(result, "total_duration")

            # Verify counts are consistent
            assert result.track_count == len(result.tracks)
            assert result.track_count > 0
            assert result.total_duration > 0

        # Total tracks across all clusters should equal input
        total_tracks = sum(r.track_count for r in results)
        assert total_tracks == 10

    def test_clusters_sorted_by_bpm(self) -> None:
        """Test that clusters are sorted by BPM mean."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=3, min_clusters=3)

        # Create tracks at different BPM ranges
        metadata_dict = {}
        for i in range(9):
            bpm_base = [120, 135, 150][i // 3]
            bpm = bpm_base + np.random.randn()
            metadata_dict[Path(f"track{i}.mp3")] = TrackMetadata(
                filepath=Path(f"track{i}.mp3"), bpm=float(bpm), duration=180.0
            )

        results = clusterer.cluster_by_bpm(metadata_dict)

        if len(results) > 1:
            bpm_means = [r.bpm_mean for r in results]
            assert bpm_means == sorted(bpm_means)

    def test_split_cluster_small_cluster(self) -> None:
        """Test that splitting a small cluster returns it unchanged."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=25)

        cluster = ClusterResult(
            cluster_id=0,
            tracks=[Path(f"track{i}.mp3") for i in range(10)],
            bpm_mean=128.0,
            bpm_std=2.0,
            track_count=10,
            total_duration=1800.0,
        )

        result = clusterer.split_cluster(cluster, target_size=20)

        assert len(result) == 1
        assert result[0] == cluster

    def test_split_cluster_large_cluster(self) -> None:
        """Test splitting a large cluster into sub-clusters."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=25)

        cluster = ClusterResult(
            cluster_id=0,
            tracks=[Path(f"track{i}.mp3") for i in range(50)],
            bpm_mean=128.0,
            bpm_std=2.0,
            track_count=50,
            total_duration=9000.0,
        )

        result = clusterer.split_cluster(cluster, target_size=20)

        # Should split into 3 sub-clusters (50 tracks / 20 target = 2.5 → 3)
        assert len(result) == 3
        assert sum(r.track_count for r in result) == 50

    def test_reproducible_clustering(self) -> None:
        """Test that clustering is reproducible with same random_state."""
        metadata_dict = {}
        for i in range(20):
            bpm = 128.0 + np.random.randn() * 5
            metadata_dict[Path(f"track{i}.mp3")] = TrackMetadata(
                filepath=Path(f"track{i}.mp3"), bpm=float(bpm), duration=180.0
            )

        clusterer1 = PlaylistClusterer(target_tracks_per_playlist=5, random_state=42)
        clusterer2 = PlaylistClusterer(target_tracks_per_playlist=5, random_state=42)

        results1 = clusterer1.cluster_by_bpm(metadata_dict)
        results2 = clusterer2.cluster_by_bpm(metadata_dict)

        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.track_count == r2.track_count
            assert abs(r1.bpm_mean - r2.bpm_mean) < 0.01


class TestClusterResult:
    """Test ClusterResult dataclass."""

    def test_cluster_result_creation(self) -> None:
        """Test creating a ClusterResult."""
        result = ClusterResult(
            cluster_id=0,
            tracks=[Path("track1.mp3"), Path("track2.mp3")],
            bpm_mean=128.5,
            bpm_std=1.5,
            track_count=2,
            total_duration=360.0,
        )

        assert result.cluster_id == 0
        assert len(result.tracks) == 2
        assert result.bpm_mean == 128.5
        assert result.bpm_std == 1.5
        assert result.track_count == 2
        assert result.total_duration == 360.0
