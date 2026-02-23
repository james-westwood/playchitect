"""
Unit tests for clustering module.
"""

from pathlib import Path

import numpy as np
import pytest

from playchitect.core.clustering import FEATURE_NAMES, ClusterResult, PlaylistClusterer
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata

# ── Helpers ────────────────────────────────────────────────────────────────────


def make_metadata(name: str, bpm: float | None = 128.0, duration: float = 360.0) -> TrackMetadata:
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
    ) -> tuple[dict[Path, TrackMetadata], dict[Path, IntensityFeatures]]:
        """Hard techno: high perc, high kick, low brightness."""
        meta: dict[Path, TrackMetadata] = {}
        intensity: dict[Path, IntensityFeatures] = {}
        for i in range(n):
            name = f"hard_{i}.mp3"
            p = Path(name)
            meta[p] = make_metadata(name, bpm=bpm_base + i * 0.2)
            intensity[p] = make_intensity(name, rms=0.8, brightness=0.2, perc=0.9, kick=0.85)
        return meta, intensity

    def _make_ambient(
        self, n: int, bpm_base: float = 138.0
    ) -> tuple[dict[Path, TrackMetadata], dict[Path, IntensityFeatures]]:
        """Ambient: same BPM range but low energy, high brightness, low perc."""
        meta: dict[Path, TrackMetadata] = {}
        intensity: dict[Path, IntensityFeatures] = {}
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
            expected_keys = set(FEATURE_NAMES) | {"hardness"}
            assert set(r.feature_means.keys()) == expected_keys
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

    def test_feature_importance_zero_variance(self) -> None:
        """Identical cluster centroids produce equal feature importances."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=5)
        centroids = np.array([[1.0] * len(FEATURE_NAMES)] * 3)  # 3 clusters, all identical
        importance = clusterer._compute_feature_importance(centroids)

        expected = 1.0 / len(FEATURE_NAMES)
        assert set(importance.keys()) == set(FEATURE_NAMES)
        for val in importance.values():
            assert abs(val - expected) < 1e-9

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


class TestGenreAwareClustering:
    """Test per-genre and mixed-genre clustering modes."""

    def _make_mixed_genre_data(
        self,
    ) -> tuple[dict[Path, TrackMetadata], dict[Path, IntensityFeatures], dict[Path, str]]:
        """Techno 125 BPM + DnB 170 BPM with distinct intensity profiles."""
        meta: dict[Path, TrackMetadata] = {}
        intensity: dict[Path, IntensityFeatures] = {}
        genre_dict: dict[Path, str] = {}
        # Techno: 4 tracks
        for i in range(4):
            p = Path(f"techno_{i}.mp3")
            meta[p] = TrackMetadata(filepath=p, bpm=124.0 + i * 2, duration=360.0)
            intensity[p] = make_intensity(str(p), kick=0.8, perc=0.7, brightness=0.4)
            genre_dict[p] = "techno"
        # DnB: 4 tracks
        for i in range(4):
            p = Path(f"dnb_{i}.mp3")
            meta[p] = TrackMetadata(filepath=p, bpm=168.0 + i * 2, duration=360.0)
            intensity[p] = make_intensity(str(p), kick=0.6, perc=0.9, brightness=0.6)
            genre_dict[p] = "dnb"
        return meta, intensity, genre_dict

    def test_per_genre_produces_genre_homogeneous_clusters(self) -> None:
        """Per-genre mode yields separate clusters per genre."""
        meta, intensity, genre_dict = self._make_mixed_genre_data()
        clusterer = PlaylistClusterer(target_tracks_per_playlist=5, min_clusters=1)

        results = clusterer.cluster_by_features(
            meta, intensity, cluster_mode="per-genre", genre_dict=genre_dict
        )

        # Should have clusters for techno and dnb; each cluster has genre set
        genre_labels = {r.genre for r in results}
        assert "techno" in genre_labels
        assert "dnb" in genre_labels
        for r in results:
            assert r.genre is not None
            assert r.cluster_id is not None
            assert str(r.cluster_id).startswith(r.genre + "_")

    def test_mixed_genre_produces_cross_genre_clusters(self) -> None:
        """Mixed-genre mode yields single K-means with BPM scaling."""
        meta, intensity, genre_dict = self._make_mixed_genre_data()
        clusterer = PlaylistClusterer(target_tracks_per_playlist=5, min_clusters=2)

        results = clusterer.cluster_by_features(
            meta, intensity, cluster_mode="mixed-genre", genre_dict=genre_dict
        )

        # All tracks clustered together; may mix genres in one cluster
        total = sum(r.track_count for r in results)
        assert total == 8
        # genre is None in mixed-genre (single K-means, no per-cluster genre)
        assert all(r.genre is None for r in results)

    def test_single_genre_ignores_genre_dict(self) -> None:
        """Single-genre mode works without genre_dict."""
        meta, intensity, _ = self._make_mixed_genre_data()
        clusterer = PlaylistClusterer(target_tracks_per_playlist=5)

        results = clusterer.cluster_by_features(
            meta, intensity, cluster_mode="single-genre", genre_dict=None
        )
        assert len(results) >= 1
        total = sum(r.track_count for r in results)
        assert total == 8

    def test_cluster_result_has_genre_field(self) -> None:
        """ClusterResult supports optional genre field."""
        r = ClusterResult(
            cluster_id="techno_0",
            tracks=[],
            bpm_mean=125.0,
            bpm_std=2.0,
            track_count=0,
            total_duration=0.0,
            genre="techno",
        )
        assert r.genre == "techno"


class TestClusteringEdgeCases:
    """Test edge cases and more advanced logic in PlaylistClusterer."""

    def test_valid_paths_empty_returns_empty(self) -> None:
        """If no tracks have both BPM and intensity, return empty list."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=5)
        meta = {Path("a.mp3"): make_metadata("a.mp3", bpm=None)}
        intensity = {Path("a.mp3"): make_intensity("a.mp3")}
        assert clusterer.cluster_by_features(meta, intensity) == []

    def test_cluster_with_embeddings(self) -> None:
        """Test clustering logic with block-weighted embeddings (PCA)."""

        class MockEmbedding:
            def __init__(self, n: int = 128):
                self.embedding = np.random.randn(n)
                self.primary_mood = "Aggressive"
                self.moods = [("Aggressive", 0.8), ("Cheerful", 0.2)]

        paths = [Path(f"t{i}.mp3") for i in range(15)]
        meta = {p: make_metadata(str(p)) for p in paths}
        intensity = {p: make_intensity(str(p)) for p in paths}
        embeddings = {p: MockEmbedding() for p in paths}

        clusterer = PlaylistClusterer(target_tracks_per_playlist=5, min_clusters=2)
        results = clusterer.cluster_by_features(meta, intensity, embedding_dict=embeddings)

        assert len(results) >= 2
        assert sum(r.track_count for r in results) == 15

    def test_cluster_with_ewkm(self) -> None:
        """Test EWKM per-cluster weight refinement."""
        # Need at least 12 tracks (MIN_TRACKS_EWKM)
        paths = [Path(f"t{i}.mp3") for i in range(16)]
        meta = {p: make_metadata(str(p)) for p in paths}
        intensity = {p: make_intensity(str(p)) for p in paths}

        clusterer = PlaylistClusterer(target_tracks_per_playlist=5, min_clusters=2)
        results = clusterer.cluster_by_features(meta, intensity, use_ewkm=True)

        assert len(results) >= 1
        for r in results:
            if r.feature_importance:
                # EWKM should provide per-cluster importance that sums to 1.0
                total = sum(r.feature_importance.values())
                assert abs(total - 1.0) < 1e-6

    def test_per_genre_no_tracks_found(self) -> None:
        """Test per-genre mode with non-empty genre_dict but all tracks as 'unknown'."""
        meta, intensity = TestClusterByFeatures()._make_hard_techno(5)
        # We need a non-empty genre_dict to trigger the per-genre branch in cluster_by_features
        # but the tracks can be missing from it (falling back to 'unknown')
        genre_dict = {Path("other.mp3"): "techno"}
        clusterer = PlaylistClusterer(target_tracks_per_playlist=5, min_clusters=1)

        results = clusterer.cluster_by_features(
            meta, intensity, cluster_mode="per-genre", genre_dict=genre_dict
        )
        assert len(results) >= 1
        assert results[0].genre == "unknown"
        assert str(results[0].cluster_id).startswith("unknown_")

    def test_mixed_genre_no_valid_paths(self) -> None:
        """Test mixed-genre mode with empty input paths."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=5)
        # Pass empty metadata, but non-empty intensity to bypass earlier checks
        results = clusterer._cluster_mixed_genre({}, {}, None, [], {}, use_ewkm=False)
        assert results == []

    def test_split_cluster_by_duration(self) -> None:
        """Test splitting a cluster by target duration."""
        clusterer = PlaylistClusterer(target_duration_per_playlist=30)  # 30 mins
        cluster = ClusterResult(
            cluster_id=0,
            tracks=[Path(f"t{i}.mp3") for i in range(20)],
            bpm_mean=120,
            bpm_std=0,
            track_count=20,
            total_duration=7200,  # 120 minutes
        )
        # 120 / 30 = 4 sub-clusters
        results = clusterer.split_cluster(cluster, target_size=5)
        assert len(results) == 4
        assert sum(r.track_count for r in results) == 20
