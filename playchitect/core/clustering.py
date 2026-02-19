"""
K-means clustering for playlist generation.

MVP: BPM-only clustering with auto-K determination using elbow method.
Future: Add intensity features, genre-aware clustering, PCA-based weighting.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from playchitect.core.metadata_extractor import TrackMetadata

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Result of clustering operation."""

    cluster_id: Union[int, str]  # int normally, str for split subclusters
    tracks: List[Path]
    bpm_mean: float
    bpm_std: float
    track_count: int
    total_duration: float  # in seconds


class PlaylistClusterer:
    """Clusters tracks into playlists using K-means."""

    def __init__(
        self,
        target_tracks_per_playlist: Optional[int] = None,
        target_duration_per_playlist: Optional[float] = None,
        min_clusters: int = 2,
        max_clusters: int = 10,
        random_state: int = 42,
    ):
        """
        Initialize playlist clusterer.

        Args:
            target_tracks_per_playlist: Target number of tracks per playlist
            target_duration_per_playlist: Target duration in minutes per playlist
            min_clusters: Minimum number of clusters to consider
            max_clusters: Maximum number of clusters to consider
            random_state: Random seed for reproducibility
        """
        if target_tracks_per_playlist is None and target_duration_per_playlist is None:
            raise ValueError(
                "Must specify either target_tracks_per_playlist or " "target_duration_per_playlist"
            )

        self.target_tracks = target_tracks_per_playlist
        self.target_duration = (
            target_duration_per_playlist * 60 if target_duration_per_playlist else None
        )  # Convert to seconds
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()

    def cluster_by_bpm(self, metadata_dict: Dict[Path, TrackMetadata]) -> List[ClusterResult]:
        """
        Cluster tracks by BPM only (MVP implementation).

        Args:
            metadata_dict: Dictionary mapping file paths to metadata

        Returns:
            List of ClusterResult objects
        """
        # Filter tracks with valid BPM
        valid_tracks = {path: meta for path, meta in metadata_dict.items() if meta.bpm is not None}

        if not valid_tracks:
            logger.error("No tracks with BPM metadata found")
            return []

        if len(valid_tracks) < self.min_clusters:
            logger.warning(f"Only {len(valid_tracks)} tracks, creating single cluster")
            return self._create_single_cluster(valid_tracks)

        logger.info(f"Clustering {len(valid_tracks)} tracks by BPM")

        # Extract BPM values and track paths
        tracks = list(valid_tracks.keys())
        bpms = np.array([valid_tracks[t].bpm for t in tracks]).reshape(-1, 1)

        # Normalize BPM values
        bpms_normalized = self.scaler.fit_transform(bpms)

        # Determine optimal K
        optimal_k = self._determine_optimal_k(bpms_normalized, valid_tracks, len(tracks))

        logger.info(f"Using K={optimal_k} clusters")

        # Perform K-means clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(bpms_normalized)

        # Create ClusterResult objects
        results = []
        for cluster_id in range(optimal_k):
            cluster_mask = cluster_labels == cluster_id
            cluster_tracks = [tracks[i] for i in np.where(cluster_mask)[0]]

            cluster_bpms = [valid_tracks[t].bpm for t in cluster_tracks]
            cluster_durations = [valid_tracks[t].duration or 0 for t in cluster_tracks]

            results.append(
                ClusterResult(
                    cluster_id=cluster_id,
                    tracks=cluster_tracks,
                    bpm_mean=float(np.mean(cluster_bpms)),
                    bpm_std=float(np.std(cluster_bpms)),
                    track_count=len(cluster_tracks),
                    total_duration=sum(cluster_durations),
                )
            )

        # Sort by BPM mean for consistency
        results.sort(key=lambda r: r.bpm_mean)

        # Log cluster statistics
        for result in results:
            logger.info(
                f"Cluster {result.cluster_id}: {result.track_count} tracks, "
                f"BPM: {result.bpm_mean:.1f} ± {result.bpm_std:.1f}, "
                f"Duration: {result.total_duration / 60:.1f} min"
            )

        return results

    def _determine_optimal_k(
        self,
        features: np.ndarray,
        metadata_dict: Dict[Path, TrackMetadata],
        total_tracks: int,
    ) -> int:
        """
        Determine optimal number of clusters using elbow method and constraints.

        Args:
            features: Normalized feature array
            metadata_dict: Track metadata for duration calculation
            total_tracks: Total number of tracks to cluster

        Returns:
            Optimal K value
        """
        # Calculate K based on target constraints
        if self.target_tracks:
            k_from_tracks = max(
                self.min_clusters, min(total_tracks // self.target_tracks, self.max_clusters)
            )
        else:
            k_from_tracks = None

        if self.target_duration:
            total_duration = sum(meta.duration or 0 for meta in metadata_dict.values())
            k_from_duration = max(
                self.min_clusters,
                min(int(total_duration / self.target_duration), self.max_clusters),
            )
        else:
            k_from_duration = None

        # Use constraint-based K if available
        if k_from_tracks and k_from_duration:
            constraint_k = int((k_from_tracks + k_from_duration) / 2)
        elif k_from_tracks:
            constraint_k = k_from_tracks
        elif k_from_duration:
            constraint_k = k_from_duration
        else:
            constraint_k = None

        # Run elbow method
        k_range = range(self.min_clusters, min(self.max_clusters + 1, total_tracks))
        inertias = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)

        # Find elbow point (largest drop in inertia)
        if len(inertias) <= 1:
            elbow_k = self.min_clusters
        else:
            inertia_diffs = np.diff(inertias)
            elbow_k = self.min_clusters + int(np.argmax(np.abs(inertia_diffs)))

        logger.debug(f"Elbow method suggests K={elbow_k}")
        logger.debug(f"Constraint-based K={constraint_k}")

        # Prefer constraint-based K if within reasonable range of elbow
        if constraint_k and abs(constraint_k - elbow_k) <= 2:
            return constraint_k
        else:
            return elbow_k

    def _create_single_cluster(
        self, metadata_dict: Dict[Path, TrackMetadata]
    ) -> List[ClusterResult]:
        """
        Create a single cluster when insufficient tracks.

        Args:
            metadata_dict: Track metadata

        Returns:
            Single-item list with one ClusterResult
        """
        tracks = list(metadata_dict.keys())
        bpms = [metadata_dict[t].bpm for t in tracks]
        durations = [metadata_dict[t].duration or 0 for t in tracks]

        return [
            ClusterResult(
                cluster_id=0,
                tracks=tracks,
                bpm_mean=float(np.mean(bpms)),
                bpm_std=float(np.std(bpms)),
                track_count=len(tracks),
                total_duration=sum(durations),
            )
        ]

    def split_cluster(self, cluster: ClusterResult, target_size: int) -> List[ClusterResult]:
        """
        Split a cluster that exceeds target size.

        Args:
            cluster: ClusterResult to split
            target_size: Target number of tracks per sub-cluster

        Returns:
            List of sub-clusters
        """
        if cluster.track_count <= target_size:
            return [cluster]

        num_splits = (cluster.track_count + target_size - 1) // target_size

        # Simple random splitting for MVP
        # TODO: In future, split by intensity features
        np.random.seed(self.random_state)
        shuffled_tracks = cluster.tracks.copy()
        np.random.shuffle(shuffled_tracks)

        subclusters = []
        for i in range(num_splits):
            start_idx = i * target_size
            end_idx = min((i + 1) * target_size, len(shuffled_tracks))
            subcluster_tracks = shuffled_tracks[start_idx:end_idx]

            # Note: Would need metadata_dict here to calculate actual stats
            # For now, create minimal ClusterResult
            subclusters.append(
                ClusterResult(
                    cluster_id=f"{cluster.cluster_id}_{i}",
                    tracks=subcluster_tracks,
                    bpm_mean=cluster.bpm_mean,
                    bpm_std=cluster.bpm_std,
                    track_count=len(subcluster_tracks),
                    total_duration=0,  # Would need to recalculate
                )
            )

        return subclusters
