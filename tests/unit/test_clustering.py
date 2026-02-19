"""
Unit tests for clustering module.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict
from playchitect.core.clustering import PlaylistClusterer, ClusterResult, FEATURE_NAMES
from playchitect.core.metadata_extractor import TrackMetadata
from playchitect.core.intensity_analyzer import IntensityFeatures


# ── Helpers ────────────────────────────────────────────────────────────────────


def make_metadata(name: str, bpm: float = 128.0, duration: float = 360.0) -> TrackMetadata:
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
    return IntensityFeatures(
        filepath=Path(name),
        file_hash="deadbeef",
        rms_energy=rms,
        brightness=brightness,
        sub_bass_energy=sub_bass,
        kick_energy=kick,
        bass_harmonics=harmonics,
        percussiveness=perc,
        onset_strength=onset,
    )


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


class TestClusterByFeatures:
    """Test multi-dimensional cluster_by_features method."""

    def _make_hard_techno(
        self, n: int, bpm_base: float = 138.0
    ) -> tuple[Dict[Path, TrackMetadata], Dict[Path, IntensityFeatures]]:
        """Hard techno: high perc, high kick, low brightness."""
        meta: Dict[Path, TrackMetadata] = {}
        intensity: Dict[Path, IntensityFeatures] = {}
        for i in range(n):
            name = f"hard_{i}.mp3"
            p = Path(name)
            meta[p] = make_metadata(name, bpm=bpm_base + i * 0.2)
            intensity[p] = make_intensity(name, rms=0.8, brightness=0.2, perc=0.9, kick=0.85)
        return meta, intensity

    def _make_ambient(
        self, n: int, bpm_base: float = 138.0
    ) -> tuple[Dict[Path, TrackMetadata], Dict[Path, IntensityFeatures]]:
        """Ambient: same BPM range but low energy, high brightness, low perc."""
        meta: Dict[Path, TrackMetadata] = {}
        intensity: Dict[Path, IntensityFeatures] = {}
        for i in range(n):
            name = f"ambient_{i}.mp3"
            p = Path(name)
            meta[p] = make_metadata(name, bpm=bpm_base + i * 0.2)
            intensity[p] = make_intensity(name, rms=0.1, brightness=0.9, perc=0.05, kick=0.1)
        return meta, intensity

    def test_basic_operation(self) -> None:
        """cluster_by_features returns results for valid inputs."""
        meta, intensity = self._make_hard_techno(10)
        clusterer = PlaylistClusterer(target_tracks_per_playlist=5, min_clusters=2)
        results = clusterer.cluster_by_features(meta, intensity)

        assert isinstance(results, list)
        assert len(results) >= 1
        assert sum(r.track_count for r in results) == 10

    def test_separates_groups_with_same_bpm_different_intensity(self) -> None:
        """Tracks with identical BPM but very different intensity should form separate clusters."""
        hard_meta, hard_intensity = self._make_hard_techno(8, bpm_base=138.0)
        amb_meta, amb_intensity = self._make_ambient(8, bpm_base=138.0)

        meta = {**hard_meta, **amb_meta}
        intensity = {**hard_intensity, **amb_intensity}

        clusterer = PlaylistClusterer(target_tracks_per_playlist=8, min_clusters=2)
        results = clusterer.cluster_by_features(meta, intensity)

        assert len(results) >= 2
        assert sum(r.track_count for r in results) == 16

        # The two groups should have different percussiveness means
        perc_means = [r.feature_means["percussiveness"] for r in results if r.feature_means]
        assert max(perc_means) - min(perc_means) > 0.3

    def test_feature_importance_present_and_normalized(self) -> None:
        """feature_importance is present on results and sums to 1.0."""
        hard_meta, hard_intensity = self._make_hard_techno(6)
        amb_meta, amb_intensity = self._make_ambient(6)

        meta = {**hard_meta, **amb_meta}
        intensity = {**hard_intensity, **amb_intensity}

        clusterer = PlaylistClusterer(target_tracks_per_playlist=6, min_clusters=2)
        results = clusterer.cluster_by_features(meta, intensity)

        for r in results:
            assert r.feature_importance is not None
            assert set(r.feature_importance.keys()) == set(FEATURE_NAMES)
            total = sum(r.feature_importance.values())
            assert abs(total - 1.0) < 1e-6

    def test_feature_means_present_and_correct_keys(self) -> None:
        """feature_means contains all 8 feature names."""
        meta, intensity = self._make_hard_techno(8)
        clusterer = PlaylistClusterer(target_tracks_per_playlist=4, min_clusters=2)
        results = clusterer.cluster_by_features(meta, intensity)

        for r in results:
            assert r.feature_means is not None
            assert set(r.feature_means.keys()) == set(FEATURE_NAMES)
            for val in r.feature_means.values():
                assert isinstance(val, float)

    def test_feature_means_in_valid_range(self) -> None:
        """feature_means values should reflect the input ranges (0-1 for intensity, real BPM)."""
        meta, intensity = self._make_hard_techno(8)
        clusterer = PlaylistClusterer(target_tracks_per_playlist=4, min_clusters=2)
        results = clusterer.cluster_by_features(meta, intensity)

        for r in results:
            assert r.feature_means is not None
            # BPM should be in a plausible range
            assert 60 <= r.feature_means["bpm"] <= 250
            # Intensity features are 0-1
            for fname in FEATURE_NAMES[1:]:
                assert 0.0 <= r.feature_means[fname] <= 1.0

    def test_skips_tracks_missing_from_intensity_dict(self) -> None:
        """Tracks in metadata but not intensity_dict are skipped gracefully."""
        meta = {Path(f"t{i}.mp3"): make_metadata(f"t{i}.mp3", bpm=130.0 + i) for i in range(12)}
        # Only provide intensity for half
        intensity = {p: make_intensity(p.name) for p in list(meta.keys())[:6]}

        clusterer = PlaylistClusterer(target_tracks_per_playlist=3, min_clusters=2)
        results = clusterer.cluster_by_features(meta, intensity)

        assert sum(r.track_count for r in results) == 6  # only the 6 with intensity

    def test_empty_metadata_returns_empty(self) -> None:
        """Empty metadata dict returns empty list."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=5)
        assert clusterer.cluster_by_features({}, {}) == []

    def test_no_overlap_between_dicts_returns_empty(self) -> None:
        """No tracks in common between metadata and intensity dicts returns empty."""
        meta = {Path("a.mp3"): make_metadata("a.mp3", bpm=130.0)}
        intensity = {Path("b.mp3"): make_intensity("b.mp3")}

        clusterer = PlaylistClusterer(target_tracks_per_playlist=5)
        assert clusterer.cluster_by_features(meta, intensity) == []

    def test_no_bpm_in_metadata_returns_empty(self) -> None:
        """Tracks with no BPM in metadata are excluded; all missing → empty list."""
        meta = {Path("a.mp3"): TrackMetadata(filepath=Path("a.mp3"), bpm=None)}
        intensity = {Path("a.mp3"): make_intensity("a.mp3")}

        clusterer = PlaylistClusterer(target_tracks_per_playlist=5)
        assert clusterer.cluster_by_features(meta, intensity) == []

    def test_reproducible_with_same_random_state(self) -> None:
        """Same random_state produces identical cluster assignments."""
        hard_meta, hard_intensity = self._make_hard_techno(8)
        amb_meta, amb_intensity = self._make_ambient(8)
        meta = {**hard_meta, **amb_meta}
        intensity = {**hard_intensity, **amb_intensity}

        c1 = PlaylistClusterer(target_tracks_per_playlist=8, min_clusters=2, random_state=42)
        c2 = PlaylistClusterer(target_tracks_per_playlist=8, min_clusters=2, random_state=42)

        r1 = c1.cluster_by_features(meta, intensity)
        r2 = c2.cluster_by_features(meta, intensity)

        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.track_count == b.track_count
            assert abs(a.bpm_mean - b.bpm_mean) < 1e-6

    def test_bpm_only_still_works_after_import(self) -> None:
        """cluster_by_bpm is unaffected by new code — backwards compatibility."""
        meta = {Path(f"t{i}.mp3"): make_metadata(f"t{i}.mp3", bpm=120.0 + i) for i in range(10)}
        clusterer = PlaylistClusterer(target_tracks_per_playlist=5, min_clusters=2)
        results = clusterer.cluster_by_bpm(meta)

        assert len(results) >= 1
        assert sum(r.track_count for r in results) == 10
        # BPM-only results have no intensity fields
        for r in results:
            assert r.feature_means is None
            assert r.feature_importance is None


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

    def test_optional_fields_default_to_none(self) -> None:
        """feature_means and feature_importance default to None."""
        result = ClusterResult(
            cluster_id=0,
            tracks=[],
            bpm_mean=128.0,
            bpm_std=0.0,
            track_count=0,
            total_duration=0.0,
        )
        assert result.feature_means is None
        assert result.feature_importance is None
