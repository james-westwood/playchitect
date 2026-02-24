"""
K-means clustering for playlist generation.

BPM-only clustering is retained for lightweight/MVP usage.
Multi-dimensional clustering (cluster_by_features) uses BPM + 7 intensity
features for character-aware playlist grouping, with adaptive feature weighting
via PCA communality weights and optional EWKM per-cluster refinement.
"""

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata
from playchitect.core.weighting import (
    FEATURE_NAMES,  # re-exported for backwards compatibility
    WeightProfile,
    ewkm_refine,
    select_weights,
)

logger = logging.getLogger(__name__)

# Minimum tracks to activate EWKM per-cluster weight refinement.
_MIN_TRACKS_EWKM = 80

# Block PCA constants for optional MusiCNN embedding integration.
_EMBEDDING_PCA_COMPONENTS: int = 12
_INTENSITY_BLOCK_WEIGHT: float = 0.70
_SEMANTIC_BLOCK_WEIGHT: float = 0.30

# Re-export FEATURE_NAMES so existing callers of
# `from playchitect.core.clustering import FEATURE_NAMES` continue to work.
__all__ = ["CLUSTER_MODES", "FEATURE_NAMES", "ClusterResult", "PlaylistClusterer"]


@dataclass
class ClusterResult:
    """Result of a clustering operation."""

    cluster_id: int | str  # int normally, str for split subclusters
    tracks: list[Path]
    bpm_mean: float
    bpm_std: float
    track_count: int
    total_duration: float  # seconds

    # Populated by cluster_by_features(); None when using cluster_by_bpm().
    feature_means: dict[str, float] | None = field(default=None)
    feature_importance: dict[str, float] | None = field(default=None)
    weight_source: str | None = field(default=None)  # "pca", "heuristic", or "uniform"

    # Populated when embedding_dict is supplied; None otherwise.
    embedding_variance_explained: float | None = field(default=None)

    # Populated in per-genre mode; None otherwise.
    genre: str | None = field(default=None)

    # Populated when moods are available; None otherwise.
    mood: str | None = field(default=None)

    # Populated by sequencer / track selector
    opener: Path | None = field(default=None)
    closer: Path | None = field(default=None)


# Genre-aware clustering modes
_CLUSTER_MODE_SINGLE = "single-genre"
_CLUSTER_MODE_PER_GENRE = "per-genre"
_CLUSTER_MODE_MIXED = "mixed-genre"
CLUSTER_MODES: tuple[str, ...] = (
    _CLUSTER_MODE_SINGLE,
    _CLUSTER_MODE_PER_GENRE,
    _CLUSTER_MODE_MIXED,
)

# BPM scaling for mixed-genre mode. Scale is applied AFTER StandardScaler
# so genre-specific adjustments are preserved in the feature space.
# Industry-typical BPM ranges (techno ~125, house ~120, ambient ~100, dnb ~170).
_REF_BPM: float = 120.0
_GENRE_TYPICAL_BPM: dict[str, float] = {
    "techno": 125.0,
    "house": 120.0,
    "ambient": 100.0,
    "dnb": 170.0,
    "unknown": 120.0,  # No scaling for unknown; log warning when encountered
}


class PlaylistClusterer:
    """Clusters tracks into playlists using K-means."""

    def __init__(
        self,
        target_tracks_per_playlist: int | None = None,
        target_duration_per_playlist: float | None = None,
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
                "Must specify either target_tracks_per_playlist or target_duration_per_playlist"
            )

        self.target_tracks = target_tracks_per_playlist
        self.target_duration = (
            target_duration_per_playlist * 60 if target_duration_per_playlist else None
        )
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()

    # ── Public API ─────────────────────────────────────────────────────────────

    def cluster_by_bpm(self, metadata_dict: dict[Path, TrackMetadata]) -> list[ClusterResult]:
        """
        Cluster tracks by BPM only (lightweight / MVP mode).

        Args:
            metadata_dict: Mapping of file path → TrackMetadata

        Returns:
            List of ClusterResult objects sorted by BPM mean
        """
        valid_tracks = {p: m for p, m in metadata_dict.items() if m.bpm is not None}

        if not valid_tracks:
            logger.error("No tracks with BPM metadata found")
            return []

        if len(valid_tracks) < self.min_clusters:
            logger.warning(f"Only {len(valid_tracks)} tracks, creating single cluster")
            return self._create_single_cluster(valid_tracks)

        logger.info(f"Clustering {len(valid_tracks)} tracks by BPM")

        tracks = list(valid_tracks.keys())
        bpms = np.array([valid_tracks[t].bpm for t in tracks]).reshape(-1, 1)
        bpms_normalized = self.scaler.fit_transform(bpms)

        optimal_k = self._determine_optimal_k(bpms_normalized, valid_tracks, len(tracks))
        logger.info(f"Using K={optimal_k} clusters")

        kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(bpms_normalized)

        results = self._build_cluster_results(tracks, labels, optimal_k, valid_tracks)
        results.sort(key=lambda r: r.bpm_mean)

        for r in results:
            logger.info(
                f"Cluster {r.cluster_id}: {r.track_count} tracks, "
                f"BPM: {r.bpm_mean:.1f} ± {r.bpm_std:.1f}, "
                f"Duration: {r.total_duration / 60:.1f} min"
            )

        return results

    def cluster_by_features(
        self,
        metadata_dict: dict[Path, TrackMetadata],
        intensity_dict: dict[Path, IntensityFeatures],
        embedding_dict: dict[Path, Any] | None = None,
        genre: str | None = None,
        use_ewkm: bool = True,
        cluster_mode: str = _CLUSTER_MODE_SINGLE,
        genre_dict: dict[Path, str] | None = None,
        bpm_scaling: dict[Path, float] | None = None,
    ) -> list[ClusterResult]:
        """
        Cluster tracks using BPM + 7 intensity features (8-dimensional).

        When embedding_dict is supplied the space expands to 20 dimensions via
        Block PCA: the 8D intensity block (×0.70) is stacked with 12 PCA
        components extracted from the 128-dim MusiCNN embeddings (×0.30).

        Feature weights for the intensity block are selected adaptively:
          - PCA communality weights when ≥40 tracks (bootstrap-validated)
          - Heuristic genre weights when genre is specified
          - Uniform weights (1/8 each) as fallback

        EWKM per-cluster weight refinement is applied when ≥80 tracks,
        use_ewkm=True, AND no embedding_dict is supplied (EWKM operates in
        the 8D intensity space only).

        Only tracks present in both dicts with valid BPM are clustered.
        Tracks missing from intensity_dict or (when embedding_dict is given)
        from embedding_dict are skipped and logged.

        Cluster modes (when genre_dict is supplied):
          - single-genre: Full feature space, single K-means (default)
          - per-genre: Separate K-means per genre → genre-homogeneous playlists
          - mixed-genre: Single K-means with BPM scaled per genre → cross-genre playlists

        Args:
            metadata_dict:  Mapping of file path → TrackMetadata
            intensity_dict: Mapping of file path → IntensityFeatures
            embedding_dict: Optional mapping of file path → EmbeddingFeatures.
                            When supplied, Block PCA (70/30) is used.
            genre:          Optional genre hint ('techno', 'house', 'ambient', 'dnb')
            use_ewkm:       Apply EWKM per-cluster weight refinement (8D mode only)
            cluster_mode:   'single-genre' | 'per-genre' | 'mixed-genre'
            genre_dict:     Per-track genre (from genre resolver); required for
                            per-genre and mixed-genre modes.
            bpm_scaling:    Optional per-path BPM scaling (e.g. for mixed-genre);
                            raw BPM * scale used in features, original for reporting.

        Returns:
            List of ClusterResult objects sorted by BPM mean, each with
            feature_means, feature_importance, and weight_source populated.
            embedding_variance_explained is set when embedding_dict is used.
            genre is set in per-genre mode.
        """
        # Intersect: only tracks with both metadata and intensity features
        common = set(metadata_dict.keys()) & set(intensity_dict.keys())
        valid_paths = sorted(p for p in common if metadata_dict[p].bpm is not None)

        if cluster_mode == _CLUSTER_MODE_PER_GENRE and genre_dict:
            return self._cluster_per_genre(
                metadata_dict,
                intensity_dict,
                embedding_dict,
                valid_paths,
                genre_dict,
                use_ewkm,
            )
        if cluster_mode == _CLUSTER_MODE_MIXED and genre_dict:
            return self._cluster_mixed_genre(
                metadata_dict,
                intensity_dict,
                embedding_dict,
                valid_paths,
                genre_dict,
                use_ewkm,
            )

        # single-genre: continue with original logic

        skipped = len(metadata_dict) - len(valid_paths)
        if skipped > 0:
            logger.warning(f"Skipped {skipped} tracks missing intensity features or BPM")

        if not valid_paths:
            logger.error("No tracks with both BPM and intensity features found")
            return []

        if len(valid_paths) < self.min_clusters:
            logger.warning(f"Only {len(valid_paths)} tracks, creating single cluster")
            valid_meta = {p: metadata_dict[p] for p in valid_paths}
            return self._create_single_cluster(valid_meta)

        logger.info(f"Clustering {len(valid_paths)} tracks on {len(FEATURE_NAMES)} features")

        # Build (N, 8) feature matrix: BPM column + 7 intensity columns (raw BPM)
        bpm_col = np.array([[metadata_dict[p].bpm or 120.0] for p in valid_paths])
        intensity_matrix = np.array([intensity_dict[p].to_feature_vector() for p in valid_paths])
        features = np.hstack([bpm_col, intensity_matrix])  # (N, 8)

        features_normalized = self.scaler.fit_transform(features)

        # Apply BPM scaling AFTER StandardScaler so genre adjustments are preserved
        # (StandardScaler would otherwise re-normalize and undo pre-scaled BPM)
        if bpm_scaling:
            bpm_idx = 0
            for i, p in enumerate(valid_paths):
                features_normalized[i, bpm_idx] *= bpm_scaling.get(p, 1.0)

        embedding_pca_variance: float | None = None

        if embedding_dict is not None:
            # Further filter to tracks that also have embeddings
            valid_paths = [p for p in valid_paths if p in embedding_dict]
            if not valid_paths:
                logger.error("No tracks with embeddings found; cannot use embedding_dict")
                return []

            # Rebuild feature matrix for the embedding-filtered subset of paths
            bpm_col_f = np.array([[metadata_dict[p].bpm or 120.0] for p in valid_paths])
            intensity_matrix_f = np.array(
                [intensity_dict[p].to_feature_vector() for p in valid_paths]
            )
            features = np.hstack([bpm_col_f, intensity_matrix_f])  # (N', 8)
            features_normalized = self.scaler.fit_transform(features)

            if bpm_scaling:
                for i, p in enumerate(valid_paths):
                    features_normalized[i, 0] *= bpm_scaling.get(p, 1.0)

            # PCA-compress 128-dim embeddings → up to 12 semantic components
            n_comp = min(len(valid_paths), _EMBEDDING_PCA_COMPONENTS)
            emb_matrix = np.array([embedding_dict[p].embedding for p in valid_paths])  # (N', 128)
            pca = PCA(n_components=n_comp, random_state=self.random_state)
            emb_pca = pca.fit_transform(emb_matrix)  # (N', n_comp)
            emb_scaler = StandardScaler()
            emb_scaled = emb_scaler.fit_transform(emb_pca)  # (N', n_comp)

            # Mood features (5 clusters from MIREX)
            # EmbeddingFeatures.moods is a list of (label, prob) tuples sorted descending.
            # We need a stable vector order.
            mood_labels = sorted(list({m[0] for p in valid_paths for m in embedding_dict[p].moods}))
            if mood_labels:
                mood_matrix = []
                for p in valid_paths:
                    mood_dict = dict(embedding_dict[p].moods)
                    mood_matrix.append([mood_dict.get(label, 0.0) for label in mood_labels])
                mood_matrix_np = np.array(mood_matrix)  # (N', 5)
                mood_scaler = StandardScaler()
                mood_scaled = mood_scaler.fit_transform(mood_matrix_np)
                semantic_block = np.hstack([emb_scaled, mood_scaled])
            else:
                semantic_block = emb_scaled

            # Block-weighted concatenation: intensity 70% + semantic 30%
            # We treat mood as part of the semantic block.
            X_intensity = features_normalized * _INTENSITY_BLOCK_WEIGHT  # (N', 8)
            X_semantic = semantic_block * _SEMANTIC_BLOCK_WEIGHT  # (N', 17 or 12)
            features_for_kmeans = np.hstack([X_intensity, X_semantic])  # (N', 25 or 20)

            embedding_pca_variance = float(pca.explained_variance_ratio_.sum())
            logger.info(
                "Embedding PCA: %d components, %.1f%% variance explained",
                _EMBEDDING_PCA_COMPONENTS,
                embedding_pca_variance * 100,
            )
            weight_source = "block_pca"
        else:
            # Standard 8D path: per-feature adaptive weighting
            profile: WeightProfile = select_weights(
                features_normalized, genre=genre, random_state=self.random_state
            )
            w_sqrt = np.sqrt(profile.weights)
            features_for_kmeans = features_normalized * w_sqrt[np.newaxis, :]
            weight_source = profile.source

        valid_meta = {p: metadata_dict[p] for p in valid_paths}
        optimal_k = self._determine_optimal_k(features_for_kmeans, valid_meta, len(valid_paths))
        logger.info(f"Using K={optimal_k} clusters (weight source: {weight_source})")

        kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(features_for_kmeans)

        # Determine per-cluster feature importance
        per_cluster_importance: list[dict[str, float]] | None = None

        # EWKM applies only in the 8D intensity mode (not when embeddings are active)
        if embedding_dict is None and use_ewkm and len(valid_paths) >= _MIN_TRACKS_EWKM:
            # EWKM operates in normalized (unweighted) space; de-weight centroids first
            w_sqrt = np.sqrt(profile.weights)
            centroids_norm = kmeans.cluster_centers_ / w_sqrt[np.newaxis, :]
            labels, ewkm_weights = ewkm_refine(features_normalized, labels, centroids_norm)
            per_cluster_importance = [
                {name: float(ewkm_weights[k, i]) for i, name in enumerate(FEATURE_NAMES)}
                for k in range(optimal_k)
            ]
            logger.info("EWKM per-cluster weights applied")
        else:
            # Fall back to global centroid-variance importance (8D features only)
            centres_8d = kmeans.cluster_centers_[:, : len(FEATURE_NAMES)]
            global_importance = self._compute_feature_importance(centres_8d)
            per_cluster_importance = [global_importance] * optimal_k

        results = self._build_cluster_results(
            valid_paths,
            labels,
            optimal_k,
            valid_meta,
            intensity_dict=intensity_dict,
            embedding_dict=embedding_dict,
            raw_features=features,
            per_cluster_importance=per_cluster_importance,
            weight_source=weight_source,
        )
        results.sort(key=lambda r: r.bpm_mean)

        # Propagate PCA variance to all cluster results when embeddings were used
        if embedding_pca_variance is not None:
            for r in results:
                r.embedding_variance_explained = embedding_pca_variance

        for r in results:
            if r.feature_importance:
                top = max(r.feature_importance, key=lambda k: r.feature_importance[k])  # type: ignore[index]
                logger.info(
                    f"Cluster {r.cluster_id}: {r.track_count} tracks, "
                    f"BPM: {r.bpm_mean:.1f} ± {r.bpm_std:.1f}, "
                    f"top feature: {top} ({r.feature_importance[top]:.2f})"
                )

        return results

    def _cluster_per_genre(
        self,
        metadata_dict: dict[Path, TrackMetadata],
        intensity_dict: dict[Path, IntensityFeatures],
        embedding_dict: dict[Path, Any] | None,
        valid_paths: list[Path],
        genre_dict: dict[Path, str],
        use_ewkm: bool,
    ) -> list[ClusterResult]:
        """Run separate K-means per genre; merge results with genre-prefixed IDs."""
        if embedding_dict is not None:
            valid_paths = [p for p in valid_paths if p in embedding_dict]

        # Group paths by genre
        by_genre: dict[str, list[Path]] = {}
        for p in valid_paths:
            g = genre_dict.get(p, "unknown")
            by_genre.setdefault(g, []).append(p)

        if not by_genre:
            logger.error("No tracks with genre found for per-genre clustering")
            return []

        all_results: list[ClusterResult] = []
        for g, paths in sorted(by_genre.items()):
            if len(paths) < self.min_clusters:
                logger.warning(
                    "Genre '%s' has only %d tracks; creating single cluster",
                    g,
                    len(paths),
                )
                meta_sub = {p: metadata_dict[p] for p in paths}
                sub_results = self._create_single_cluster(meta_sub)
            else:
                meta_sub = {p: metadata_dict[p] for p in paths}
                int_sub = {p: intensity_dict[p] for p in paths}
                emb_sub = {p: embedding_dict[p] for p in paths} if embedding_dict else None
                sub_results = self.cluster_by_features(
                    meta_sub,
                    int_sub,
                    embedding_dict=emb_sub,
                    genre=g if g != "unknown" else None,
                    use_ewkm=use_ewkm,
                    cluster_mode=_CLUSTER_MODE_SINGLE,
                    genre_dict=None,
                )
            for r in sub_results:
                r.genre = g
                r.cluster_id = f"{g}_{r.cluster_id}"
            all_results.extend(sub_results)

        all_results.sort(key=lambda r: r.bpm_mean)
        return all_results

    def _cluster_mixed_genre(
        self,
        metadata_dict: dict[Path, TrackMetadata],
        intensity_dict: dict[Path, IntensityFeatures],
        embedding_dict: dict[Path, Any] | None,
        valid_paths: list[Path],
        genre_dict: dict[Path, str],
        use_ewkm: bool,
    ) -> list[ClusterResult]:
        """Run single K-means with BPM scaled per genre for cross-genre coherence."""
        if embedding_dict is not None:
            valid_paths = [p for p in valid_paths if p in embedding_dict]

        if not valid_paths:
            logger.error("No valid paths for mixed-genre clustering")
            return []

        # Scale BPM per genre so e.g. 170 DnB and 125 techno align
        bpm_scaling: dict[Path, float] = {}
        unknown_count = 0
        for p in valid_paths:
            g = genre_dict.get(p, "unknown")
            if g not in _GENRE_TYPICAL_BPM:
                unknown_count += 1
                g = "unknown"
            typical = _GENRE_TYPICAL_BPM[g]
            bpm_scaling[p] = _REF_BPM / typical
        if unknown_count > 0:
            logger.warning(
                "Mixed-genre: %d tracks with unknown/unmapped genre (scale=1.0)",
                unknown_count,
            )

        meta_sub = {p: metadata_dict[p] for p in valid_paths}
        int_sub = {p: intensity_dict[p] for p in valid_paths}

        logger.info("Mixed-genre mode: BPM scaled by genre-typical values")
        return self.cluster_by_features(
            meta_sub,
            int_sub,
            embedding_dict=embedding_dict,
            genre=None,  # Mixed: no single genre for weighting
            use_ewkm=use_ewkm,
            cluster_mode=_CLUSTER_MODE_SINGLE,
            genre_dict=None,
            bpm_scaling=bpm_scaling,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_cluster_results(
        self,
        tracks: list[Path],
        labels: np.ndarray,
        n_clusters: int,
        metadata_dict: dict[Path, TrackMetadata],
        intensity_dict: dict[Path, IntensityFeatures] | None = None,
        embedding_dict: dict[Path, Any] | None = None,
        raw_features: np.ndarray | None = None,
        per_cluster_importance: list[dict[str, float]] | None = None,
        feature_importance: dict[str, float] | None = None,
        weight_source: str | None = None,
    ) -> list[ClusterResult]:
        """Build ClusterResult list from K-means labels."""
        results = []

        for cid in range(n_clusters):
            mask = labels == cid
            cluster_tracks = [tracks[i] for i in np.where(mask)[0]]

            # Skip empty clusters — can occur when K-means K exceeds the number
            # of distinct data points (e.g. few unique BPM values).
            if not cluster_tracks:
                logger.debug("Skipping empty cluster %d", cid)
                continue

            cluster_bpms: list[float] = [
                b for t in cluster_tracks if (b := metadata_dict[t].bpm) is not None
            ]
            cluster_durations = [metadata_dict[t].duration or 0 for t in cluster_tracks]

            # Per-feature means (only available in multi-dimensional mode)
            f_means: dict[str, float] | None = None
            if raw_features is not None:
                cluster_raw = raw_features[mask]
                f_means = {
                    name: float(cluster_raw[:, i].mean()) for i, name in enumerate(FEATURE_NAMES)
                }
                # Also include mean hardness if possible
                if intensity_dict:
                    h_vals = [
                        intensity_dict[t].hardness for t in cluster_tracks if t in intensity_dict
                    ]
                    if h_vals:
                        f_means["hardness"] = float(np.mean(h_vals))

            # Per-cluster importance: from EWKM or global centroid-variance
            f_importance = (
                per_cluster_importance[cid]
                if per_cluster_importance is not None
                else feature_importance
            )

            # Determine dominant mood for cluster
            dominant_mood: str | None = None
            if embedding_dict:
                mood_counts: dict[str, float] = {}
                for t in cluster_tracks:
                    if t in embedding_dict and (m := embedding_dict[t].primary_mood):
                        mood_counts[m] = mood_counts.get(m, 0.0) + 1.0
                if mood_counts:
                    dominant_mood = max(mood_counts, key=mood_counts.get)  # type: ignore[arg-type]

            results.append(
                ClusterResult(
                    cluster_id=cid,
                    tracks=cluster_tracks,
                    bpm_mean=float(np.mean(cluster_bpms)),
                    bpm_std=float(np.std(cluster_bpms)),
                    track_count=len(cluster_tracks),
                    total_duration=float(sum(cluster_durations)),
                    feature_means=f_means,
                    feature_importance=f_importance,
                    weight_source=weight_source,
                    mood=dominant_mood,
                )
            )

        return results

    def _compute_feature_importance(self, centroids: np.ndarray) -> dict[str, float]:
        """
        Compute feature importance as variance of cluster centroids.

        Centroids are in StandardScaler-normalized space, so variance
        reflects how much each feature actually separates the clusters.
        Scores are normalized to sum to 1.0.

        Args:
            centroids: Array of shape (n_clusters, n_features)

        Returns:
            Dict mapping feature name → importance score (0-1, sum=1)
        """
        variances = np.var(centroids, axis=0)
        total = float(variances.sum())

        if total > 0:
            scores: list[float] = [float(v / total) for v in variances]
        else:
            scores = [1.0 / len(FEATURE_NAMES)] * len(FEATURE_NAMES)

        return {name: scores[i] for i, name in enumerate(FEATURE_NAMES)}

    def _determine_optimal_k(
        self,
        features: np.ndarray,
        metadata_dict: dict[Path, TrackMetadata],
        total_tracks: int,
    ) -> int:
        """
        Determine optimal number of clusters using elbow method and constraints.

        Args:
            features: Normalized feature array (N, D)
            metadata_dict: Track metadata for duration-based K estimation
            total_tracks: Total number of tracks to cluster

        Returns:
            Optimal K value
        """
        k_from_tracks: int | None = None
        k_from_duration: int | None = None

        if self.target_tracks:
            k_from_tracks = max(
                self.min_clusters,
                min(total_tracks // self.target_tracks, self.max_clusters),
            )

        if self.target_duration:
            total_duration = sum(m.duration or 0 for m in metadata_dict.values())
            k_from_duration = max(
                self.min_clusters,
                min(int(total_duration / self.target_duration), self.max_clusters),
            )

        if k_from_tracks and k_from_duration:
            constraint_k: int | None = int((k_from_tracks + k_from_duration) / 2)
        elif k_from_tracks:
            constraint_k = k_from_tracks
        elif k_from_duration:
            constraint_k = k_from_duration
        else:
            constraint_k = None

        # Elbow method
        k_range = range(self.min_clusters, min(self.max_clusters + 1, total_tracks))
        inertias = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            km.fit(features)
            inertias.append(km.inertia_)

        if len(inertias) <= 1:
            elbow_k = self.min_clusters
        else:
            inertia_diffs = np.diff(inertias)
            elbow_k = self.min_clusters + int(np.argmax(np.abs(inertia_diffs)))

        logger.debug(f"Elbow method suggests K={elbow_k}, constraint K={constraint_k}")

        if constraint_k and abs(constraint_k - elbow_k) <= 2:
            return constraint_k
        return elbow_k

    def _create_single_cluster(
        self, metadata_dict: dict[Path, TrackMetadata]
    ) -> list[ClusterResult]:
        """Create a single cluster when there are insufficient tracks."""
        tracks = list(metadata_dict.keys())
        bpms: list[float] = [b for t in tracks if (b := metadata_dict[t].bpm) is not None]
        durations = [metadata_dict[t].duration or 0 for t in tracks]

        return [
            ClusterResult(
                cluster_id=0,
                tracks=tracks,
                bpm_mean=float(np.mean(bpms)),
                bpm_std=float(np.std(bpms)),
                track_count=len(tracks),
                total_duration=float(sum(durations)),
            )
        ]

    def split_cluster(self, cluster: ClusterResult, target_size: int) -> list[ClusterResult]:
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

        rng = random.Random(self.random_state)
        shuffled = cluster.tracks.copy()
        rng.shuffle(shuffled)

        avg_duration = cluster.total_duration / cluster.track_count if cluster.track_count else 0.0
        subclusters = []
        for i in range(num_splits):
            start = i * target_size
            end = min((i + 1) * target_size, len(shuffled))
            sub_count = end - start
            subclusters.append(
                ClusterResult(
                    cluster_id=f"{cluster.cluster_id}_{i}",
                    tracks=shuffled[start:end],
                    bpm_mean=cluster.bpm_mean,
                    bpm_std=cluster.bpm_std,
                    track_count=sub_count,
                    total_duration=avg_duration * sub_count,
                )
            )

        return subclusters
