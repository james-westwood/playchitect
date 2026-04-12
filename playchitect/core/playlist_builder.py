"""
Duration-constrained playlist assembly.

Builds playlists that respect target duration constraints by ranking tracks
by distance to cluster centroid and selecting tracks until the target duration
is reached (within tolerance). Tracks that don't fit their primary cluster
are tried against next-closest clusters.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

from playchitect.core.clustering import ClusterResult
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

__all__: list[str] = ["build_duration_constrained_playlists"]


def build_duration_constrained_playlists(
    clusters: list[ClusterResult],
    target_duration_mins: float,
    tolerance: float = 0.1,
    metadata_dict: Union[dict[Path, TrackMetadata], None] = None,
    features_dict: Union[dict[Path, IntensityFeatures], None] = None,
) -> list[ClusterResult]:
    """Build playlists that respect target duration constraints.

    For each cluster, ranks tracks by distance to the cluster centroid (closest first)
    and adds tracks in that order until the cumulative duration reaches
    target_duration_mins * (1 + tolerance). Any track that doesn't fit
    its primary cluster is tried against the next-closest cluster's
    centroid; if it fits there within tolerance, it is added.

    Args:
        clusters: List of ClusterResult objects from clustering
        target_duration_mins: Target duration in minutes per playlist
        tolerance: Tolerance as fraction (default 0.1 = 10%)
        metadata_dict: Optional mapping of track paths to metadata
            (required for duration-aware assembly)
        features_dict: Optional mapping of track paths to intensity features
            (required for distance calculation)

    Returns:
        New list of ClusterResult objects with duration-constrained track lists

    Raises:
        ValueError: If clusters is empty or target_duration_mins <= 0
    """
    if not clusters:
        raise ValueError("clusters list cannot be empty")
    if target_duration_mins <= 0:
        raise ValueError("target_duration_mins must be positive")

    target_duration_secs = target_duration_mins * 60
    max_duration_secs = target_duration_secs * (1 + tolerance)

    cluster_centroids: dict[Union[int, str], np.ndarray] = {}
    cluster_primary_tracks: dict[Union[int, str], list[Path]] = {}

    for cluster in clusters:
        if cluster.centroid is not None:
            cluster_centroids[cluster.cluster_id] = cluster.centroid
            cluster_primary_tracks[cluster.cluster_id] = list(cluster.tracks)

    all_tracks: list[Path] = []
    for cluster in clusters:
        all_tracks.extend(cluster.tracks)
    all_tracks = list(set(all_tracks))

    track_features: dict[Path, IntensityFeatures] = {}
    if features_dict:
        track_features = features_dict

    track_distances: dict[Path, list[Any]] = defaultdict(list)
    for track_path in all_tracks:
        distances: list[tuple[Union[int, str], float]] = []
        for cid, centroid in cluster_centroids.items():
            dist = _calculate_distance(track_path, centroid, track_features)
            distances.append((cid, dist))
        distances.sort(key=lambda x: x[1])
        track_distances[track_path] = distances

    cluster_tracks: dict[Union[int, str], list[Path]] = {}
    for cluster in clusters:
        cluster_tracks[cluster.cluster_id] = []

    for cluster in clusters:
        primary = cluster_primary_tracks[cluster.cluster_id]
        if cluster.centroid is not None:
            ranked = _rank_tracks_by_distance(primary, cluster.centroid, track_features)
        else:
            ranked = [(t, 0.0) for t in primary]

        cumulative_duration = 0.0

        for track_path, _ in ranked:
            track_duration = _get_track_duration(track_path, metadata_dict)
            if track_duration <= 0:
                logger.debug("Skipping zero-duration track: %s", track_path)
                continue

            if cumulative_duration + track_duration > max_duration_secs:
                continue

            cluster_tracks[cluster.cluster_id].append(track_path)
            cumulative_duration += track_duration

            if track_path in track_distances:
                del track_distances[track_path]

    for track_path in list(track_distances.keys()):
        track_duration = _get_track_duration(track_path, metadata_dict)
        if track_duration <= 0:
            continue

        distances = track_distances[track_path]

        assigned = False
        for cluster_id, _ in distances:
            current_tracks = cluster_tracks[cluster_id]
            current_duration = _sum_track_durations(current_tracks, metadata_dict)

            if current_duration + track_duration <= max_duration_secs:
                current_tracks.append(track_path)
                assigned = True
                break

        if assigned and track_path in track_distances:
            del track_distances[track_path]

    result_clusters: list[ClusterResult] = []
    for cluster in clusters:
        selected = cluster_tracks[cluster.cluster_id]
        result_clusters.append(_create_trimmed_cluster(cluster, selected, metadata_dict))

    result_clusters = _filter_empty_clusters(result_clusters)

    return result_clusters


def _calculate_distance(
    track_path: Path,
    centroid: np.ndarray,
    features_dict: dict[Path, IntensityFeatures],
) -> float:
    """Calculate distance from track to centroid."""
    if track_path in features_dict:
        features = features_dict[track_path]
        vector = _features_to_vector(features)
        if vector is not None and len(vector) == len(centroid):
            return float(np.linalg.norm(vector - centroid))

    return float("inf")


def _rank_tracks_by_distance(
    tracks: list[Path],
    centroid: np.ndarray,
    features_dict: dict[Path, IntensityFeatures],
) -> list[tuple[Path, float]]:
    """Rank tracks by distance to centroid (closest first)."""
    distances: list[tuple[Path, float]] = []

    for track_path in tracks:
        if track_path in features_dict:
            features = features_dict[track_path]
            vector = _features_to_vector(features)
            if vector is not None and len(vector) == len(centroid):
                dist = float(np.linalg.norm(vector - centroid))
            else:
                dist = float("inf")
        else:
            dist = float("inf")

        distances.append((track_path, dist))

    distances.sort(key=lambda x: x[1])
    return distances


def _features_to_vector(features: IntensityFeatures) -> Union[np.ndarray, None]:
    """Convert IntensityFeatures to feature vector."""
    if features is None:
        return None

    vector = np.array(
        [
            features.rms_energy,
            features.brightness,
            features.sub_bass_energy,
            features.kick_energy,
            features.bass_harmonics,
            features.percussiveness,
            features.onset_strength,
        ]
    )
    return vector


def _get_track_duration(
    track_path: Path,
    metadata_dict: Union[dict[Path, TrackMetadata], None],
) -> float:
    """Get track duration in seconds."""
    if metadata_dict and track_path in metadata_dict:
        duration = metadata_dict[track_path].duration
        if duration is not None:
            return duration
    return 0.0


def _sum_track_durations(
    tracks: list[Path],
    metadata_dict: Union[dict[Path, TrackMetadata], None],
) -> float:
    """Sum durations of all tracks."""
    total = 0.0
    for track_path in tracks:
        total += _get_track_duration(track_path, metadata_dict)
    return total


def _create_trimmed_cluster(
    original: ClusterResult,
    selected_tracks: list[Path],
    metadata_dict: Union[dict[Path, TrackMetadata], None],
) -> ClusterResult:
    """Create a new ClusterResult with trimmed track list."""
    total_duration = _sum_track_durations(selected_tracks, metadata_dict)

    return ClusterResult(
        cluster_id=original.cluster_id,
        tracks=selected_tracks,
        bpm_mean=original.bpm_mean,
        bpm_std=original.bpm_std,
        track_count=len(selected_tracks),
        total_duration=total_duration,
        feature_means=original.feature_means,
        feature_importance=original.feature_importance,
        weight_source=original.weight_source,
        embedding_variance_explained=original.embedding_variance_explained,
        genre=original.genre,
        mood=original.mood,
        opener=original.opener,
        closer=original.closer,
        centroid=original.centroid,
    )


def _filter_empty_clusters(
    clusters: list[ClusterResult],
) -> list[ClusterResult]:
    """Filter out clusters with no tracks."""
    return [c for c in clusters if len(c.tracks) > 0]
