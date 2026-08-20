"""
Failing-first unit tests for TASK-15: split_cluster recluster rewrite.

These tests define the new contract for PlaylistClusterer.split_cluster:

- Features path: re-run KMeans on the stored standardised feature matrix,
  compute per-sub-cluster stats (bpm_mean, bpm_std, total_duration,
  feature_means) from members, and fall back to deterministic ordering when
  KMeans produces empty clusters or degenerate vectors.
- BPM-only path: sort tracks by BPM and take contiguous slices, computing
  per-sub-cluster stats from members.
- No-data fallback: contiguous slices in the given track order, never shuffled.
- No random.shuffle / random.Random.shuffle is used.

The implementation under test does not exist yet; these tests are expected to
fail against the current code.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from playchitect.core.clustering import FEATURE_NAMES, ClusterResult, PlaylistClusterer
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_metadata(path: Path, bpm: float, duration: float = 300.0) -> TrackMetadata:
    return TrackMetadata(filepath=path, bpm=bpm, duration=duration)


def make_intensity(
    path: Path,
    rms: float = 0.5,
    brightness: float = 0.5,
    sub_bass: float = 0.3,
    kick: float = 0.6,
    harmonics: float = 0.4,
    perc: float = 0.5,
    onset: float = 0.5,
) -> IntensityFeatures:
    return IntensityFeatures(
        file_path=path,
        file_hash="deadbeef",
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


def make_uniform_features(
    n: int,
    bpm_base: float = 128.0,
    rms_base: float = 0.5,
) -> tuple[dict[Path, TrackMetadata], dict[Path, IntensityFeatures]]:
    """Return n tracks with identical metadata and intensity features."""
    metadata_dict: dict[Path, TrackMetadata] = {}
    intensity_dict: dict[Path, IntensityFeatures] = {}
    for i in range(n):
        p = Path(f"track{i}.mp3")
        metadata_dict[p] = make_metadata(p, bpm=bpm_base)
        intensity_dict[p] = make_intensity(p, rms=rms_base)
    return metadata_dict, intensity_dict


def member_bpms(tracks: list[Path], metadata_dict: dict[Path, TrackMetadata]) -> list[float]:
    values: list[float] = []
    for t in tracks:
        bpm = metadata_dict[t].bpm
        assert bpm is not None
        values.append(bpm)
    return values


def member_durations(tracks: list[Path], metadata_dict: dict[Path, TrackMetadata]) -> list[float]:
    values: list[float] = []
    for t in tracks:
        duration = metadata_dict[t].duration
        assert duration is not None
        values.append(duration)
    return values


def bpm_sort_key(metadata_dict: dict[Path, TrackMetadata]) -> Callable[[Path], float]:
    def key(t: Path) -> float:
        bpm = metadata_dict[t].bpm
        assert bpm is not None
        return bpm

    return key


# ── Tests ───────────────────────────────────────────────────────────────────────


class TestSplitClusterUndersized:
    """Behaviour for clusters that are already within the target size."""

    def test_undersized_cluster_returned_unchanged(self) -> None:
        """track_count <= target_size returns the original cluster object."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=20, random_state=42)
        cluster = ClusterResult(
            cluster_id=0,
            tracks=[Path(f"track{i}.mp3") for i in range(10)],
            bpm_mean=128.0,
            bpm_std=2.0,
            track_count=10,
            total_duration=3000.0,
        )

        result = clusterer.split_cluster(cluster, target_size=20)

        assert len(result) == 1
        assert result[0] is cluster


class TestSplitClusterFeaturesPath:
    """Features-path split: stored state from cluster_by_features drives reclustering."""

    def test_oversized_cluster_split_count_is_ceil(self) -> None:
        """50 tracks / target_size 20 produces exactly 3 sub-clusters."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=20, random_state=42)
        metadata_dict, intensity_dict = make_uniform_features(50, bpm_base=128.0)
        clusterer.cluster_by_features(metadata_dict, intensity_dict, use_ewkm=False)
        cluster = ClusterResult(
            cluster_id=0,
            tracks=list(metadata_dict.keys()),
            bpm_mean=128.0,
            bpm_std=2.0,
            track_count=50,
            total_duration=15000.0,
        )

        result = clusterer.split_cluster(cluster, target_size=20)

        assert len(result) == 3

    def test_subclusters_disjoint_and_complete(self) -> None:
        """Sub-clusters partition the parent track set with no overlap."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=20, random_state=42)
        metadata_dict, intensity_dict = make_uniform_features(50, bpm_base=128.0)
        clusterer.cluster_by_features(metadata_dict, intensity_dict, use_ewkm=False)
        cluster = ClusterResult(
            cluster_id=0,
            tracks=list(metadata_dict.keys()),
            bpm_mean=128.0,
            bpm_std=2.0,
            track_count=50,
            total_duration=15000.0,
        )

        result = clusterer.split_cluster(cluster, target_size=20)

        union: set[Path] = set()
        total = 0
        for sub in result:
            sub_set = set(sub.tracks)
            assert union.isdisjoint(sub_set)
            union |= sub_set
            total += sub.track_count
            assert sub.track_count == len(sub.tracks)

        assert total == 50
        assert union == set(cluster.tracks)

    def test_subcluster_stats_computed_from_members_not_copied(self) -> None:
        """Each sub-cluster's bpm_mean, bpm_std and total_duration come from members."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=20, random_state=42)
        metadata_dict: dict[Path, TrackMetadata] = {}
        intensity_dict: dict[Path, IntensityFeatures] = {}
        for i in range(50):
            p = Path(f"track{i}.mp3")
            bpm = 124.0 if i < 25 else 132.0
            metadata_dict[p] = make_metadata(p, bpm=bpm, duration=300.0)
            intensity_dict[p] = make_intensity(p)
        clusterer.cluster_by_features(metadata_dict, intensity_dict, use_ewkm=False)

        parent_mean = 128.0
        parent_std = 4.0
        cluster = ClusterResult(
            cluster_id=0,
            tracks=list(metadata_dict.keys()),
            bpm_mean=parent_mean,
            bpm_std=parent_std,
            track_count=50,
            total_duration=15000.0,
        )

        result = clusterer.split_cluster(cluster, target_size=20)

        at_least_one_mean_differs = False
        for sub in result:
            sub_bpms = member_bpms(sub.tracks, metadata_dict)
            sub_durations = member_durations(sub.tracks, metadata_dict)
            assert sub.bpm_mean == pytest.approx(float(np.mean(sub_bpms)))
            assert sub.bpm_std == pytest.approx(float(np.std(sub_bpms)))
            assert sub.total_duration == pytest.approx(float(sum(sub_durations)))
            if abs(sub.bpm_mean - parent_mean) > 0.1:
                at_least_one_mean_differs = True

        assert at_least_one_mean_differs

    def test_subcluster_feature_means_from_members(self) -> None:
        """feature_means is populated per sub-cluster and reflects member averages."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=20, random_state=42)
        metadata_dict: dict[Path, TrackMetadata] = {}
        intensity_dict: dict[Path, IntensityFeatures] = {}
        for i in range(50):
            p = Path(f"track{i}.mp3")
            metadata_dict[p] = make_metadata(p, bpm=128.0, duration=300.0)
            # Vary rms_energy so feature_means are non-trivial to compute.
            intensity_dict[p] = make_intensity(p, rms=0.1 + i * 0.01)
        clusterer.cluster_by_features(metadata_dict, intensity_dict, use_ewkm=False)
        cluster = ClusterResult(
            cluster_id=0,
            tracks=list(metadata_dict.keys()),
            bpm_mean=128.0,
            bpm_std=0.0,
            track_count=50,
            total_duration=15000.0,
        )

        result = clusterer.split_cluster(cluster, target_size=20)

        for sub in result:
            assert sub.feature_means is not None
            for name in FEATURE_NAMES:
                assert name in sub.feature_means
            sub_rms = [intensity_dict[t].rms_energy for t in sub.tracks]
            assert sub.feature_means["rms_energy"] == pytest.approx(float(np.mean(sub_rms)))

    def test_kmeans_split_separates_distinct_blobs(self) -> None:
        """Two well-separated feature blobs produce mostly pure sub-clusters."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=20, random_state=42)
        metadata_dict: dict[Path, TrackMetadata] = {}
        intensity_dict: dict[Path, IntensityFeatures] = {}
        for i in range(50):
            p = Path(f"track{i}.mp3")
            metadata_dict[p] = make_metadata(p, bpm=128.0, duration=300.0)
            # Two tight blobs: high energy (0.89-0.91) and low energy (0.09-0.11).
            base_rms = 0.9 if i < 25 else 0.1
            intensity_dict[p] = make_intensity(p, rms=base_rms + (i % 25) * 0.0008)
        clusterer.cluster_by_features(metadata_dict, intensity_dict, use_ewkm=False)
        cluster = ClusterResult(
            cluster_id=0,
            tracks=list(metadata_dict.keys()),
            bpm_mean=128.0,
            bpm_std=0.0,
            track_count=50,
            total_duration=15000.0,
        )

        result = clusterer.split_cluster(cluster, target_size=20)

        high_blob = {Path(f"track{i}.mp3") for i in range(25)}
        low_blob = {Path(f"track{i}.mp3") for i in range(25, 50)}
        pure_subclusters = 0
        for sub in result:
            high_count = sum(1 for t in sub.tracks if t in high_blob)
            low_count = sum(1 for t in sub.tracks if t in low_blob)
            purity = max(high_count, low_count) / sub.track_count
            if purity > 0.9:
                pure_subclusters += 1

        assert pure_subclusters >= 2

    def test_degenerate_features_fall_back_to_energy_order(self) -> None:
        """KMeans producing an empty cluster triggers rms_energy-ordered fallback."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=20, random_state=42)
        metadata_dict: dict[Path, TrackMetadata] = {}
        intensity_dict: dict[Path, IntensityFeatures] = {}
        # rms_energy strictly increases with index, but cluster tracks are supplied
        # in reverse order so that a rms-energy sort is visibly different from the
        # supplied order.
        for i in range(50):
            p = Path(f"track{i}.mp3")
            metadata_dict[p] = make_metadata(p, bpm=128.0, duration=300.0)
            intensity_dict[p] = make_intensity(p, rms=0.1 + i * 0.01)
        clusterer.cluster_by_features(metadata_dict, intensity_dict, use_ewkm=False)

        cluster = ClusterResult(
            cluster_id=0,
            tracks=[Path(f"track{i}.mp3") for i in range(49, -1, -1)],
            bpm_mean=128.0,
            bpm_std=0.0,
            track_count=50,
            total_duration=15000.0,
        )
        expected_energy_order = sorted(cluster.tracks, key=lambda t: intensity_dict[t].rms_energy)

        # Simulate KMeans returning an empty cluster (label 1 is never used).
        def fake_labels(X: np.ndarray) -> np.ndarray:
            n = X.shape[0]
            labels = np.zeros(n, dtype=int)
            labels[n // 2 :] = 2
            return labels

        mock_kmeans = MagicMock()
        mock_kmeans.fit_predict.side_effect = fake_labels
        mock_kmeans.cluster_centers_ = np.zeros((3, 8))

        with patch("playchitect.core.clustering.KMeans", return_value=mock_kmeans):
            result1 = clusterer.split_cluster(cluster, target_size=20)
            result2 = clusterer.split_cluster(cluster, target_size=20)

        assert len(result1) == 3
        assert all(sub.track_count > 0 for sub in result1)

        # Deterministic across calls.
        for s1, s2 in zip(result1, result2):
            assert s1.tracks == s2.tracks

        # Fallback orders by rms_energy, not by the supplied cluster order.
        concatenated1 = [t for sub in result1 for t in sub.tracks]
        assert concatenated1 == expected_energy_order

        # Disjoint and complete.
        union = set()
        for sub in result1:
            assert union.isdisjoint(set(sub.tracks))
            union |= set(sub.tracks)
        assert union == set(cluster.tracks)

    def test_features_path_reproducible_with_random_state(self) -> None:
        """Two clusterers with the same random_state produce identical partitions."""
        metadata_dict, intensity_dict = make_uniform_features(50, bpm_base=128.0)
        cluster = ClusterResult(
            cluster_id=0,
            tracks=list(metadata_dict.keys()),
            bpm_mean=128.0,
            bpm_std=2.0,
            track_count=50,
            total_duration=15000.0,
        )
        clusterer1 = PlaylistClusterer(target_tracks_per_playlist=20, random_state=42)
        clusterer2 = PlaylistClusterer(target_tracks_per_playlist=20, random_state=42)
        clusterer1.cluster_by_features(metadata_dict, intensity_dict, use_ewkm=False)
        clusterer2.cluster_by_features(metadata_dict, intensity_dict, use_ewkm=False)

        result1 = clusterer1.split_cluster(cluster, target_size=20)
        result2 = clusterer2.split_cluster(cluster, target_size=20)

        assert len(result1) == len(result2)
        for s1, s2 in zip(result1, result2):
            assert s1.tracks == s2.tracks


class TestSplitClusterBpmOnlyPath:
    """BPM-only split: stored state from cluster_by_bpm drives deterministic slicing."""

    def _make_scrambled_bpm_data(self, n: int = 50) -> dict[Path, TrackMetadata]:
        metadata_dict: dict[Path, TrackMetadata] = {}
        for i in range(n):
            p = Path(f"track{i}.mp3")
            # Scatter BPMs deterministically across 115-149.
            bpm = 115.0 + (i * 7) % 35
            metadata_dict[p] = make_metadata(p, bpm=bpm, duration=300.0)
        return metadata_dict

    def test_bpm_only_path_sorted_contiguous_slices(self) -> None:
        """Sub-clusters concatenate to the BPM-ascending sorted track list."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=20, random_state=42)
        metadata_dict = self._make_scrambled_bpm_data(50)
        clusterer.cluster_by_bpm(metadata_dict)
        cluster = ClusterResult(
            cluster_id=0,
            tracks=list(metadata_dict.keys()),
            bpm_mean=130.0,
            bpm_std=10.0,
            track_count=50,
            total_duration=15000.0,
        )

        result = clusterer.split_cluster(cluster, target_size=20)

        expected_order = sorted(cluster.tracks, key=bpm_sort_key(metadata_dict))
        concatenated = [t for sub in result for t in sub.tracks]
        assert concatenated == expected_order

        for sub in result:
            sub_bpms = member_bpms(sub.tracks, metadata_dict)
            sub_durations = member_durations(sub.tracks, metadata_dict)
            assert sub.bpm_mean == pytest.approx(float(np.mean(sub_bpms)))
            assert sub.bpm_std == pytest.approx(float(np.std(sub_bpms)))
            assert sub.total_duration == pytest.approx(float(sum(sub_durations)))

    def test_bpm_only_path_deterministic_across_calls(self) -> None:
        """Two identical BPM-only splits produce identical ordered partitions."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=20, random_state=42)
        metadata_dict = self._make_scrambled_bpm_data(50)
        clusterer.cluster_by_bpm(metadata_dict)
        cluster = ClusterResult(
            cluster_id=0,
            tracks=list(metadata_dict.keys()),
            bpm_mean=130.0,
            bpm_std=10.0,
            track_count=50,
            total_duration=15000.0,
        )

        result1 = clusterer.split_cluster(cluster, target_size=20)
        result2 = clusterer.split_cluster(cluster, target_size=20)

        assert len(result1) == len(result2)
        for s1, s2 in zip(result1, result2):
            assert s1.tracks == s2.tracks


class TestSplitClusterNoShuffle:
    """Shuffle must never be invoked inside split_cluster."""

    def test_no_random_shuffle_used(self, monkeypatch) -> None:
        """random.Random.shuffle raising during split_cluster would break both paths."""

        def fail_shuffle(*args: object, **kwargs: object) -> None:
            raise AssertionError("shuffle used")

        monkeypatch.setattr(random.Random, "shuffle", fail_shuffle)

        # Features path.
        clusterer_features = PlaylistClusterer(target_tracks_per_playlist=20, random_state=42)
        metadata_dict, intensity_dict = make_uniform_features(50, bpm_base=128.0)
        clusterer_features.cluster_by_features(metadata_dict, intensity_dict, use_ewkm=False)
        cluster_features = ClusterResult(
            cluster_id=0,
            tracks=list(metadata_dict.keys()),
            bpm_mean=128.0,
            bpm_std=2.0,
            track_count=50,
            total_duration=15000.0,
        )
        clusterer_features.split_cluster(cluster_features, target_size=20)

        # BPM-only path.
        clusterer_bpm = PlaylistClusterer(target_tracks_per_playlist=20, random_state=42)
        metadata_dict_bpm: dict[Path, TrackMetadata] = {}
        for i in range(50):
            p = Path(f"track{i}.mp3")
            metadata_dict_bpm[p] = make_metadata(p, bpm=115.0 + (i * 7) % 35)
        clusterer_bpm.cluster_by_bpm(metadata_dict_bpm)
        cluster_bpm = ClusterResult(
            cluster_id=0,
            tracks=list(metadata_dict_bpm.keys()),
            bpm_mean=130.0,
            bpm_std=10.0,
            track_count=50,
            total_duration=15000.0,
        )
        clusterer_bpm.split_cluster(cluster_bpm, target_size=20)


class TestSplitClusterNamingAndFallback:
    """Sub-cluster naming and the no-data fallback path."""

    def test_subcluster_id_naming(self) -> None:
        """Sub-cluster ids are f"{parent_id}_{i}" for i in 0..K-1."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=20, random_state=42)
        metadata_dict, intensity_dict = make_uniform_features(30, bpm_base=128.0)
        clusterer.cluster_by_features(metadata_dict, intensity_dict, use_ewkm=False)
        cluster = ClusterResult(
            cluster_id="parent",
            tracks=list(metadata_dict.keys()),
            bpm_mean=128.0,
            bpm_std=2.0,
            track_count=30,
            total_duration=9000.0,
        )

        result = clusterer.split_cluster(cluster, target_size=20)

        assert len(result) == 2
        for i, sub in enumerate(result):
            assert sub.cluster_id == f"parent_{i}"

    def test_no_data_fallback_contiguous_slices(self) -> None:
        """Fresh clusterer never shuffles; slices preserve the supplied track order."""
        clusterer = PlaylistClusterer(target_tracks_per_playlist=20, random_state=42)
        cluster = ClusterResult(
            cluster_id=0,
            tracks=[Path(f"track{i}.mp3") for i in range(50)],
            bpm_mean=128.0,
            bpm_std=2.0,
            track_count=50,
            total_duration=15000.0,
        )

        result = clusterer.split_cluster(cluster, target_size=20)

        assert len(result) == 3
        concatenated = [t for sub in result for t in sub.tracks]
        assert concatenated == cluster.tracks

        union: set[Path] = set()
        for sub in result:
            assert union.isdisjoint(set(sub.tracks))
            union |= set(sub.tracks)
        assert union == set(cluster.tracks)
