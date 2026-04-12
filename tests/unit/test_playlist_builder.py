"""
Unit tests for playlist builder module.
"""

from pathlib import Path

import numpy as np
import pytest

from playchitect.core.clustering import ClusterResult
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata
from playchitect.core.playlist_builder import build_duration_constrained_playlists


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


def make_cluster(
    cluster_id: int | str,
    tracks: list[Path],
    centroid: np.ndarray | None = None,
) -> ClusterResult:
    """Create ClusterResult for testing."""
    return ClusterResult(
        cluster_id=cluster_id,
        tracks=tracks,
        bpm_mean=128.0,
        bpm_std=5.0,
        track_count=len(tracks),
        total_duration=len(tracks) * 360.0,
        centroid=centroid,
    )


class TestBuildDurationConstrainedPlaylists:
    """Test build_duration_constrained_playlists function."""

    def test_empty_clusters_raises(self) -> None:
        """Test that empty clusters list raises ValueError."""
        with pytest.raises(ValueError):
            build_duration_constrained_playlists([], 90.0)

    def test_negative_target_raises(self) -> None:
        """Test that negative target raises ValueError."""
        clusters = [make_cluster(0, [Path("t1.mp3")])]
        with pytest.raises(ValueError):
            build_duration_constrained_playlists(clusters, -10.0)

    def test_zero_target_raises(self) -> None:
        """Test that zero target raises ValueError."""
        clusters = [make_cluster(0, [Path("t1.mp3")])]
        with pytest.raises(ValueError):
            build_duration_constrained_playlists(clusters, 0.0)

    def test_90min_target_between_80_and_100(self) -> None:
        """Test 90-min target produces playlists between 80-100 mins."""
        tracks = [Path(f"track{i}.mp3") for i in range(90)]
        metadata_dict = {t: make_metadata(str(t), 128.0, 180.0) for t in tracks}

        features_dict = {}
        for i, track in enumerate(tracks):
            features = make_intensity(str(track), rms=float(i) / 90.0)
            features_dict[track] = features

        centroid0 = np.array([0.2, 0.5, 0.3, 0.6, 0.4, 0.5, 0.5])
        centroid1 = np.array([0.5, 0.5, 0.3, 0.6, 0.4, 0.5, 0.5])
        centroid2 = np.array([0.8, 0.5, 0.3, 0.6, 0.4, 0.5, 0.5])

        clusters = [
            make_cluster(0, tracks[:30], centroid0),
            make_cluster(1, tracks[30:60], centroid1),
            make_cluster(2, tracks[60:], centroid2),
        ]

        result = build_duration_constrained_playlists(
            clusters,
            target_duration_mins=90.0,
            tolerance=0.1,
            metadata_dict=metadata_dict,
            features_dict=features_dict,
        )

        non_empty = [c for c in result if c.track_count > 0]
        assert len(non_empty) >= 1

        all_durations = [c.total_duration / 60.0 for c in non_empty]

        all_between_80_and_100 = all(80 <= d <= 100 for d in all_durations)

        assert all_between_80_and_100, f"Not all playlists between 80-100 mins: {all_durations}"

    def test_fewer_tracks_than_needed(self) -> None:
        """Test returns all available when fewer tracks than needed."""
        tracks = [Path(f"track{i}.mp3") for i in range(3)]
        metadata_dict = {t: make_metadata(str(t), 128.0, 200.0) for t in tracks}

        centroid = np.array([0.5, 0.5, 0.3, 0.6, 0.4, 0.5, 0.5])
        clusters = [make_cluster(0, tracks, centroid)]

        result = build_duration_constrained_playlists(
            clusters,
            target_duration_mins=90.0,
            tolerance=0.1,
            metadata_dict=metadata_dict,
            features_dict={},
        )

        assert len(result) == 1
        assert result[0].track_count == 3

    def test_zero_duration_tracks_skipped(self) -> None:
        """Test that zero-duration tracks are skipped."""
        tracks = [
            Path("valid1.mp3"),
            Path("zero.mp3"),
            Path("valid2.mp3"),
        ]
        metadata_dict = {
            Path("valid1.mp3"): make_metadata("valid1.mp3", 128.0, 300.0),
            Path("zero.mp3"): make_metadata("zero.mp3", 128.0, 0.0),
            Path("valid2.mp3"): make_metadata("valid2.mp3", 128.0, 300.0),
        }

        centroid = np.array([0.5, 0.5, 0.3, 0.6, 0.4, 0.5, 0.5])
        clusters = [make_cluster(0, tracks, centroid)]

        result = build_duration_constrained_playlists(
            clusters,
            target_duration_mins=10.0,
            tolerance=0.1,
            metadata_dict=metadata_dict,
            features_dict={},
        )

        assert len(result) == 1
        assert result[0].track_count == 2

    def test_single_track_short_duration(self) -> None:
        """Test single cluster with total duration under target."""
        tracks = [Path("track1.mp3")]
        metadata_dict = {Path("track1.mp3"): make_metadata("track1.mp3", 128.0, 180.0)}

        centroid = np.array([0.5, 0.5, 0.3, 0.6, 0.4, 0.5, 0.5])
        clusters = [make_cluster(0, tracks, centroid)]

        result = build_duration_constrained_playlists(
            clusters,
            target_duration_mins=90.0,
            tolerance=0.1,
            metadata_dict=metadata_dict,
            features_dict={},
        )

        assert len(result) == 1
        assert result[0].track_count == 1

    def test_tolerance_boundary(self) -> None:
        """Test tracks respect tolerance boundary exactly."""
        tracks = [Path(f"track{i}.mp3") for i in range(10)]
        metadata_dict = {t: make_metadata(str(t), 128.0, 600.0) for t in tracks}

        centroid = np.array([0.5, 0.5, 0.3, 0.6, 0.4, 0.5, 0.5])
        clusters = [make_cluster(0, tracks, centroid)]

        result = build_duration_constrained_playlists(
            clusters,
            target_duration_mins=90.0,
            tolerance=0.1,
            metadata_dict=metadata_dict,
            features_dict={},
        )

        assert len(result) == 1
        duration_mins = result[0].total_duration / 60.0
        assert duration_mins <= 99.0
