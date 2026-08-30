"""
K-means clustering for playlist generation.

BPM-only clustering is retained for lightweight/MVP usage.
Multi-dimensional clustering (cluster_by_features) uses BPM + 7 intensity
features for character-aware playlist grouping, with adaptive feature weighting
via PCA communality weights and optional EWKM per-cluster refinement.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata
from playchitect.core.weighting import (
    FEATURE_NAMES,  # re-exported for backwards compatibility
    WeightProfile,
    ewkm_refine,
    select_weights,
)
from playchitect.utils.weight_config import WeightOverrides, apply_weight_overrides

logger = logging.getLogger(__name__)

# Minimum tracks to activate EWKM per-cluster weight refinement.
_MIN_TRACKS_EWKM = 80

# Silhouette score thresholds for auto-K selection.
_SIL_STRONG: float = 0.5  # silhouette max above this → silhouette is primary signal
_SIL_WEAK: float = 0.3  # silhouette max below this → fall back to elbow method

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

    # Centroid in normalized feature space (for deduplication)
    centroid: np.ndarray | None = field(default=None, repr=False)


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
        weight_overrides: WeightOverrides | None = None,
    ):
        """
        Initialize playlist clusterer.

        Args:
            target_tracks_per_playlist: Target number of tracks per playlist
            target_duration_per_playlist: Target duration in minutes per playlist
            min_clusters: Minimum number of clusters to consider
            max_clusters: Maximum number of clusters to consider
            random_state: Random seed for reproducibility
            weight_overrides: Optional user-specified feature weight overrides
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
        self.weight_overrides = weight_overrides

        # State stashed by cluster_by_* so split_cluster can recompute per-member
        # stats instead of copying parent statistics.
        self._last_metadata_dict: dict[Path, TrackMetadata] | None = None
        self._last_intensity_dict: dict[Path, IntensityFeatures] | None = None
        self._last_features_paths: list[Path] | None = None
        self._last_features_normalized_8d: np.ndarray | None = None

        # Surface a warning when auto-K clustering fails to separate the library.
        self.last_degenerate_warning: str | None = None

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
        self.last_degenerate_warning = None

        tracks = list(valid_tracks.keys())
        bpms = np.array([valid_tracks[t].bpm for t in tracks]).reshape(-1, 1)
        bpms_normalized = self.scaler.fit_transform(bpms)

        optimal_k = self._determine_optimal_k(bpms_normalized, valid_tracks, len(tracks))
        logger.info(f"Using K={optimal_k} clusters")

        kmeans = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(bpms_normalized)

        results = self._build_cluster_results(
            tracks, labels, optimal_k, valid_tracks, cluster_centers=kmeans.cluster_centers_
        )
        results.sort(key=lambda r: r.bpm_mean)

        for r in results:
            logger.info(
                f"Cluster {r.cluster_id}: {r.track_count} tracks, "
                f"BPM: {r.bpm_mean:.1f} ± {r.bpm_std:.1f}, "
                f"Duration: {r.total_duration / 60:.1f} min"
            )

        self._check_degenerate_separation(results, optimal_k, k_was_auto=True)
        self._store_bpm_state(valid_tracks)
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
        n_playlists: int | None = None,
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
            n_playlists:    Optional override for number of clusters. When > 0,
                            bypasses auto-K selection (elbow/silhouette) and uses
                            this exact K value.

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
        self.last_degenerate_warning = None

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
                # Defensive backstop for callers that do not pre-check the
                # embedding stage themselves (GUI, ETL scripts); the CLI
                # fails earlier with a diagnostic naming the cause.
                logger.error(
                    "None of the supplied tracks have an embedding, so embedding-aware "
                    "clustering cannot run. Check that embedding extraction succeeded "
                    "before calling this, or cluster without embeddings."
                )
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
            # Apply user-specified weight overrides if provided
            if self.weight_overrides is not None:
                profile = WeightProfile(
                    weights=apply_weight_overrides(
                        profile.weights, self.weight_overrides, FEATURE_NAMES
                    ),
                    source=f"{profile.source}+override",
                    genre=profile.genre,
                    n_tracks=profile.n_tracks,
                    ci_width=profile.ci_width,
                )
            w_sqrt = np.sqrt(profile.weights)
            features_for_kmeans = features_normalized * w_sqrt[np.newaxis, :]
            weight_source = profile.source

        valid_meta = {p: metadata_dict[p] for p in valid_paths}

        # Stash the 8D standardized feature matrix for split_cluster to reuse.
        self._store_features_state(metadata_dict, intensity_dict, valid_paths, features_normalized)

        # Determine optimal K, respecting n_playlists override if provided
        if n_playlists is not None and n_playlists > 0:
            optimal_k = min(n_playlists, len(valid_paths))
            optimal_k = max(self.min_clusters, optimal_k)
            logger.info(f"Using K={optimal_k} clusters (n_playlists override)")
        else:
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
            cluster_centers=kmeans.cluster_centers_,
        )
        results = self._deduplicate_clusters(results, features_for_kmeans, valid_paths)
        results.sort(key=lambda r: r.bpm_mean)

        # Propagate PCA variance to all cluster results when embeddings were used
        if embedding_pca_variance is not None:
            for r in results:
                r.embedding_variance_explained = embedding_pca_variance

        for r in results:
            if r.feature_importance:
                top = max(r.feature_importance, key=lambda k: r.feature_importance[k])  # type: ignore
                logger.info(
                    f"Cluster {r.cluster_id}: {r.track_count} tracks, "
                    f"BPM: {r.bpm_mean:.1f} ± {r.bpm_std:.1f}, "
                    f"top feature: {top} ({r.feature_importance[top]:.2f})"
                )

        k_was_auto = n_playlists is None or n_playlists <= 0
        self._check_degenerate_separation(results, optimal_k, k_was_auto=k_was_auto)
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

    def _check_degenerate_separation(
        self,
        results: list[ClusterResult],
        optimal_k: int,
        k_was_auto: bool,
    ) -> None:
        """Check if auto-selected clustering produced a degenerate dominant cluster.

        A library is flagged when the largest cluster holds at least 70% of the
        tracks, the final K is at most 2, K was auto-selected, and the library
        has at least 50 tracks. When triggered, a WARNING is logged and the
        message is stored in ``last_degenerate_warning`` so callers can surface it.

        Args:
            results: Cluster results produced by the current run.
            optimal_k: Final K value that was selected.
            k_was_auto: Whether K was auto-selected rather than forced.
        """
        if not k_was_auto:
            return
        if optimal_k > 2:
            return

        total_tracks = sum(r.track_count for r in results)
        if total_tracks < 50:
            return
        if not results:
            return

        largest = max(r.track_count for r in results)
        pct = largest / total_tracks * 100
        if pct < 70:
            return

        msg = (
            f"Clusters did not separate: largest cluster holds {pct:.1f}% "
            f"of {total_tracks} tracks. Consider --use-embeddings or "
            f"--genre hints for better separation."
        )
        logger.warning(msg)
        self.last_degenerate_warning = msg

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
        cluster_centers: np.ndarray | None = None,
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
                    dominant_mood = max(mood_counts, key=mood_counts.get)  # type: ignore

            # Determine centroid for deduplication (use first D dimensions from kmeans center)
            centroid: np.ndarray | None = None
            if cluster_centers is not None and cid < len(cluster_centers):
                centroid = cluster_centers[cid, :]

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
                    centroid=centroid,
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

        # Elbow method + Silhouette Score
        k_range = range(self.min_clusters, min(self.max_clusters + 1, total_tracks))
        inertias: list[float] = []
        sil_scores: list[float] = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(features)
            inertias.append(float(km.inertia_))  # type: ignore
            if k > 1 and len(np.unique(labels)) > 1:
                sil_scores.append(float(silhouette_score(features, labels)))
            else:
                sil_scores.append(-1.0)  # silhouette undefined for k=1 or single-cluster results

        if len(inertias) <= 1:
            elbow_k = self.min_clusters
        else:
            inertia_diffs = np.diff(inertias)
            elbow_k = self.min_clusters + int(np.argmax(np.abs(inertia_diffs)))

        max_sil = max(sil_scores) if sil_scores else 0.0
        best_sil_k = list(k_range)[int(np.argmax(sil_scores))]

        logger.debug(
            f"Silhouette scores: max={max_sil:.3f} at K={best_sil_k}, "
            f"elbow K={elbow_k}, constraint K={constraint_k}"
        )

        if max_sil >= _SIL_STRONG:
            data_k = best_sil_k
        elif max_sil < _SIL_WEAK:
            data_k = elbow_k
        else:
            # Blend: average of silhouette suggestion and elbow, rounded
            data_k = round((best_sil_k + elbow_k) / 2)

        if constraint_k and abs(constraint_k - data_k) <= 2:
            return constraint_k
        return data_k

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

    def _recompute_cluster_stats(
        self,
        tracks: list[Path],
        metadata_dict: dict[Path, TrackMetadata],
        intensity_dict: dict[Path, IntensityFeatures],
    ) -> tuple[float, float, float, dict[str, float] | None]:
        """Recompute BPM/duration/feature-mean stats from a set of surviving tracks.

        Mirrors the formulas used by ``_build_cluster_results`` so that fresh
        stats produced here agree with what a from-scratch build would report
        for the same track set.

        Args:
            tracks: Surviving tracks for a cluster after reassignment.
            metadata_dict: Mapping of file path -> track metadata.
            intensity_dict: Mapping of file path -> intensity features.

        Returns:
            Tuple of (bpm_mean, bpm_std, total_duration, feature_means).
            ``feature_means`` is ``None`` when no intensity data is available.
        """
        bpm_values: list[float] = []
        durations: list[float] = []
        for t in tracks:
            meta = metadata_dict.get(t)
            if meta is None:
                continue
            if meta.bpm is not None:
                bpm_values.append(meta.bpm)
            durations.append(meta.duration or 0.0)

        bpm_mean = float(np.mean(bpm_values)) if bpm_values else 0.0
        bpm_std = float(np.std(bpm_values)) if bpm_values else 0.0
        total_duration = float(sum(durations))

        feature_means: dict[str, float] | None = None
        rows = [t for t in tracks if t in intensity_dict and t in metadata_dict]
        if rows:
            bpm_col = np.array([[metadata_dict[t].bpm or 120.0] for t in rows])
            intensity_matrix = np.array([intensity_dict[t].to_feature_vector() for t in rows])
            raw = np.hstack([bpm_col, intensity_matrix])
            feature_means = {name: float(raw[:, i].mean()) for i, name in enumerate(FEATURE_NAMES)}
            feature_means["hardness"] = float(np.mean([intensity_dict[t].hardness for t in rows]))

        return bpm_mean, bpm_std, total_duration, feature_means

    def _deduplicate_clusters(
        self,
        results: list[ClusterResult],
        features: np.ndarray,
        track_order: list[Path],
    ) -> list[ClusterResult]:
        """
        Conservatively resolve cluster membership by nearest-centroid reassignment.

        Historical context (BUG-02 / #192): downstream consumers require that
        no track appears in more than one cluster's ``tracks`` list. Normal
        K-means labels are already disjoint, but when EWKM per-cluster weight
        refinement runs (>= ``_MIN_TRACKS_EWKM`` tracks, no embeddings; see
        ``cluster_by_features``), the membership recorded on each
        ``ClusterResult`` reflects the EWKM-*refined* labels while the
        ``centroid`` stored on that same result is still the raw (pre-EWKM)
        K-means centroid (see ``_build_cluster_results``). Recomputing each
        track's nearest centroid against those raw centroids can therefore
        legitimately disagree with a track's EWKM-assigned cluster for
        boundary tracks.

        TASK-28: the previous implementation treated that disagreement as "this
        track doesn't belong here" and removed the track from its current
        cluster without ever adding it anywhere else, silently losing tracks
        (verified: 405 tracks in, 398 out on a real library). This version
        performs a full reassignment — every track is placed in the cluster
        whose centroid it is nearest to — rather than a remove-only filter, so
        the set of tracks returned is always exactly the set of tracks passed
        in: never lost, never duplicated. Whenever a cluster's membership
        actually changes as a result, its BPM/duration/feature-mean stats are
        recomputed from the surviving members (see
        ``_recompute_cluster_stats``) instead of being copied stale from the
        pre-dedup result.

        Args:
            results: List of ClusterResult objects (pre-dedup).
            features: Feature matrix (N, D) in the same space as each result's
                ``centroid`` (i.e. whatever matrix produced the K-means
                labels), indexed by ``track_order``.
            track_order: Order of tracks that corresponds to rows in ``features``.

        Returns:
            Deduped ClusterResult list with disjoint, lossless track membership.
        """
        if not results:
            return results

        # Build track -> feature index map
        track_to_idx: dict[Path, int] = {t: i for i, t in enumerate(track_order)}

        # Build cluster_id -> centroid map
        cluster_centroids: dict[int | str, np.ndarray] = {
            r.cluster_id: r.centroid for r in results if r.centroid is not None
        }

        if not cluster_centroids:
            return results

        # Current (pre-dedup) cluster each track sits in, preserving the
        # concatenated cross-cluster track order for stable output ordering.
        current_cluster: dict[Path, int | str] = {}
        ordered_tracks: list[Path] = []
        for r in results:
            for t in r.tracks:
                current_cluster[t] = r.cluster_id
                ordered_tracks.append(t)

        # Reassign every track to its nearest centroid. This is a full
        # reassignment (every track always lands somewhere), not a
        # remove-only filter — see docstring above for why that distinction
        # matters.
        track_to_cluster: dict[Path, int | str] = {}
        missing_feature_count = 0
        for t in ordered_tracks:
            idx = track_to_idx.get(t)
            if idx is None or idx >= len(features):
                # No feature vector available for this track (should not
                # happen via the sole production call site in
                # cluster_by_features); keep it where it already is rather
                # than losing it.
                track_to_cluster[t] = current_cluster[t]
                missing_feature_count += 1
                continue
            track_vec = features[idx]
            best_cluster = current_cluster[t]
            min_dist = float("inf")
            for cluster_id, centroid in cluster_centroids.items():
                dist = float(np.linalg.norm(track_vec - centroid))
                if dist < min_dist:
                    min_dist = dist
                    best_cluster = cluster_id
            track_to_cluster[t] = best_cluster

        if missing_feature_count:
            logger.warning(
                "_deduplicate_clusters: %d track(s) had no feature vector for "
                "nearest-centroid reassignment; kept in their current cluster "
                "to avoid dropping them",
                missing_feature_count,
            )

        # Group tracks by resolved cluster, preserving relative order.
        tracks_by_cluster: dict[int | str, list[Path]] = {}
        for t in ordered_tracks:
            tracks_by_cluster.setdefault(track_to_cluster[t], []).append(t)

        # Stashed by cluster_by_features (the sole caller) before dedup runs;
        # used to recompute stats for clusters whose membership changed. Can
        # be None only when _deduplicate_clusters is invoked directly (e.g.
        # in isolated unit tests) rather than via cluster_by_features.
        metadata_dict = self._last_metadata_dict
        intensity_dict = self._last_intensity_dict

        deduped: list[ClusterResult] = []
        total_kept = 0
        for r in results:
            kept_tracks = tracks_by_cluster.get(r.cluster_id, [])

            if not kept_tracks:
                # Cluster lost all members to nearer centroids elsewhere —
                # drop it, mirroring how _build_cluster_results skips empty
                # clusters rather than emitting a hollow ClusterResult.
                logger.debug("Dedup: cluster %s ended up empty; dropping it", r.cluster_id)
                continue

            total_kept += len(kept_tracks)
            membership_changed = set(kept_tracks) != set(r.tracks)

            if not membership_changed:
                bpm_mean, bpm_std = r.bpm_mean, r.bpm_std
                total_duration = r.total_duration
                feature_means = r.feature_means
            elif metadata_dict is not None and intensity_dict is not None:
                bpm_mean, bpm_std, total_duration, recomputed_means = self._recompute_cluster_stats(
                    kept_tracks, metadata_dict, intensity_dict
                )
                feature_means = (
                    recomputed_means if recomputed_means is not None else r.feature_means
                )
            else:
                # No cached state to recompute from (direct-call path). Stats
                # remain stale in this case, but membership is still correct
                # and lossless — only the isolated-unit-test call path hits
                # this branch, never cluster_by_features.
                logger.debug(
                    "Dedup: cluster %s membership changed but no cached "
                    "metadata/intensity state is available to recompute "
                    "stats; keeping pre-dedup stats",
                    r.cluster_id,
                )
                bpm_mean, bpm_std = r.bpm_mean, r.bpm_std
                total_duration = r.total_duration
                feature_means = r.feature_means

            deduped.append(
                ClusterResult(
                    cluster_id=r.cluster_id,
                    tracks=kept_tracks,
                    bpm_mean=bpm_mean,
                    bpm_std=bpm_std,
                    track_count=len(kept_tracks),
                    total_duration=total_duration,
                    feature_means=feature_means,
                    feature_importance=r.feature_importance,
                    weight_source=r.weight_source,
                    embedding_variance_explained=r.embedding_variance_explained,
                    genre=r.genre,
                    mood=r.mood,
                    opener=r.opener,
                    closer=r.closer,
                    centroid=r.centroid,
                )
            )

        if total_kept != len(ordered_tracks):
            # Should be unreachable given the full-reassignment strategy above
            # (every track in ordered_tracks is placed into exactly one
            # cluster's tracks_by_cluster bucket). Logged loudly rather than
            # silently returning a short result, per TASK-28.
            logger.warning(
                "_deduplicate_clusters: track count mismatch after dedup — "
                "started with %d tracks, returning %d; %d may have been lost",
                len(ordered_tracks),
                total_kept,
                len(ordered_tracks) - total_kept,
            )

        return deduped

    def _store_bpm_state(self, metadata_dict: dict[Path, TrackMetadata]) -> None:
        """Stash BPM-only metadata for deterministic split_cluster slicing.

        Args:
            metadata_dict: Mapping of file paths to track metadata.

        Returns:
            None
        """
        self._last_metadata_dict = dict(metadata_dict)
        self._last_intensity_dict = None
        self._last_features_paths = None
        self._last_features_normalized_8d = None

    def _store_features_state(
        self,
        metadata_dict: dict[Path, TrackMetadata],
        intensity_dict: dict[Path, IntensityFeatures],
        valid_paths: list[Path],
        features_normalized: np.ndarray,
    ) -> None:
        """Stash multi-dimensional state so split_cluster can recluster members.

        Args:
            metadata_dict: Mapping of file paths to track metadata.
            intensity_dict: Mapping of file paths to intensity features.
            valid_paths: Ordered list of valid track file paths corresponding
                to rows in feature matrix.
            features_normalized: Standardized 8D feature matrix as a numpy array.

        Returns:
            None
        """
        self._last_metadata_dict = dict(metadata_dict)
        self._last_intensity_dict = dict(intensity_dict)
        self._last_features_paths = list(valid_paths)
        self._last_features_normalized_8d = features_normalized.copy()

    def _split_by_features(
        self,
        cluster: ClusterResult,
        target_size: int,
        num_splits: int,
    ) -> list[ClusterResult]:
        """Re-run KMeans on the stored 8D standardized feature matrix.

        Args:
            cluster: Over-sized ClusterResult to split.
            target_size: Target number of tracks per sub-cluster.
            num_splits: Number of sub-clusters to create.

        Returns:
            List of new ClusterResult sub-clusters.
        """
        assert self._last_features_paths is not None
        assert self._last_features_normalized_8d is not None

        path_to_idx = {p: i for i, p in enumerate(self._last_features_paths)}
        indices = np.array([path_to_idx[t] for t in cluster.tracks])
        X = self._last_features_normalized_8d[indices]

        # Degenerate input: all points identical -> no separable signal.
        if len(np.unique(X, axis=0)) <= 1:
            logger.warning("Feature vectors are degenerate; falling back to energy order")
            return self._split_by_energy(cluster, target_size, num_splits)

        kmeans = KMeans(n_clusters=num_splits, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X)

        if len(set(labels)) < num_splits:
            logger.warning("KMeans produced empty sub-clusters; falling back to energy order")
            return self._split_by_energy(cluster, target_size, num_splits)

        return self._build_subclusters_from_labels(cluster, labels, num_splits)

    def _split_by_bpm(
        self,
        cluster: ClusterResult,
        target_size: int,
        num_splits: int,
    ) -> list[ClusterResult]:
        """Sort tracks by BPM and take contiguous slices.

        Args:
            cluster: Over-sized ClusterResult to split.
            target_size: Target number of tracks per sub-cluster.
            num_splits: Number of sub-clusters to create.

        Returns:
            List of new ClusterResult sub-clusters.
        """
        assert self._last_metadata_dict is not None
        metadata = self._last_metadata_dict

        def _bpm_key(t: Path) -> float:
            bpm = metadata[t].bpm
            return float("inf") if bpm is None else bpm

        sorted_tracks = sorted(cluster.tracks, key=_bpm_key)
        slices = self._make_contiguous_slices(sorted_tracks, target_size, num_splits)
        return self._build_subclusters_from_slices(cluster, slices, use_features=False)

    def _split_by_energy(
        self,
        cluster: ClusterResult,
        target_size: int,
        num_splits: int,
    ) -> list[ClusterResult]:
        """Fallback: order tracks by rms_energy ascending and slice contiguously.

        Args:
            cluster: Over-sized ClusterResult to split.
            target_size: Target number of tracks per sub-cluster.
            num_splits: Number of sub-clusters to create.

        Returns:
            List of new ClusterResult sub-clusters.
        """
        assert self._last_intensity_dict is not None
        intensity = self._last_intensity_dict

        sorted_tracks = sorted(
            cluster.tracks,
            key=lambda t: intensity[t].rms_energy if t in intensity else 0.0,
        )
        slices = self._make_contiguous_slices(sorted_tracks, target_size, num_splits)
        return self._build_subclusters_from_slices(cluster, slices, use_features=True)

    def _split_fallback(
        self,
        cluster: ClusterResult,
        target_size: int,
        num_splits: int,
    ) -> list[ClusterResult]:
        """Last-resort deterministic split: preserve the supplied track order.

        Args:
            cluster: Over-sized ClusterResult to split.
            target_size: Target number of tracks per sub-cluster.
            num_splits: Number of sub-clusters to create.

        Returns:
            List of new ClusterResult sub-clusters.
        """
        logger.warning("No stored clustering state; splitting by supplied track order")
        slices = self._make_contiguous_slices(cluster.tracks, target_size, num_splits)
        return self._build_subclusters_from_slices(
            cluster, slices, use_features=False, copy_parent_stats=True
        )

    @staticmethod
    def _make_contiguous_slices(
        tracks: list[Path],
        target_size: int,
        num_splits: int,
    ) -> list[list[Path]]:
        """Partition ``tracks`` into ``num_splits`` contiguous slices of up to ``target_size``.

        Args:
            tracks: List of track file paths.
            target_size: Maximum number of tracks per slice.
            num_splits: Number of slices to generate.

        Returns:
            List of track path lists representing contiguous slices.
        """
        slices: list[list[Path]] = []
        for i in range(num_splits):
            start = i * target_size
            end = min((i + 1) * target_size, len(tracks))
            slices.append(tracks[start:end])
        return slices

    def _build_subclusters_from_labels(
        self,
        parent: ClusterResult,
        labels: np.ndarray,
        num_splits: int,
    ) -> list[ClusterResult]:
        """Build ClusterResults from KMeans labels.

        Args:
            parent: Parent ClusterResult being split.
            labels: NumPy array of integer cluster assignments for each track in parent.
            num_splits: Number of sub-clusters.

        Returns:
            List of new ClusterResult sub-clusters.
        """
        assert self._last_metadata_dict is not None
        assert self._last_intensity_dict is not None

        groups: dict[int, list[Path]] = {i: [] for i in range(num_splits)}
        for track, label in zip(parent.tracks, labels):
            groups[int(label)].append(track)

        results: list[ClusterResult] = []
        for i in range(num_splits):
            tracks = groups[i]
            if not tracks:
                continue
            stats = self._member_stats(tracks, fallback_parent=parent)
            feature_means = self._compute_feature_means(tracks)
            results.append(
                ClusterResult(
                    cluster_id=f"{parent.cluster_id}_{i}",
                    tracks=tracks,
                    bpm_mean=stats["bpm_mean"],
                    bpm_std=stats["bpm_std"],
                    track_count=len(tracks),
                    total_duration=stats["total_duration"],
                    feature_means=feature_means,
                    feature_importance=parent.feature_importance,
                    weight_source=parent.weight_source,
                )
            )
        return results

    def _build_subclusters_from_slices(
        self,
        parent: ClusterResult,
        slices: list[list[Path]],
        use_features: bool,
        copy_parent_stats: bool = False,
    ) -> list[ClusterResult]:
        """Build ClusterResults from pre-computed contiguous track slices.

        Args:
            parent: Parent ClusterResult being split.
            slices: List of track path lists for each sub-cluster.
            use_features: Whether to compute feature means from stored state.
            copy_parent_stats: Whether to copy parent stats directly instead of recomputing.

        Returns:
            List of new ClusterResult sub-clusters.
        """
        results: list[ClusterResult] = []
        for i, tracks in enumerate(slices):
            if not tracks:
                continue
            stats = self._member_stats(
                tracks, fallback_parent=parent, copy_parent=copy_parent_stats
            )
            feature_means = self._compute_feature_means(tracks) if use_features else None
            results.append(
                ClusterResult(
                    cluster_id=f"{parent.cluster_id}_{i}",
                    tracks=tracks,
                    bpm_mean=stats["bpm_mean"],
                    bpm_std=stats["bpm_std"],
                    track_count=len(tracks),
                    total_duration=stats["total_duration"],
                    feature_means=feature_means,
                    feature_importance=parent.feature_importance,
                    weight_source=parent.weight_source,
                )
            )
        return results

    def _member_stats(
        self,
        tracks: list[Path],
        fallback_parent: ClusterResult,
        copy_parent: bool = False,
    ) -> dict[str, float]:
        """Compute member-based stats, copying parent values only when no data exists.

        Args:
            tracks: List of track file paths belonging to the sub-cluster.
            fallback_parent: Parent ClusterResult to fall back on if metadata is missing.
            copy_parent: Force copying parent statistics.

        Returns:
            Dictionary containing bpm_mean, bpm_std, and total_duration.
        """
        metadata = self._last_metadata_dict
        if metadata is None or copy_parent:
            avg_duration = (
                fallback_parent.total_duration / fallback_parent.track_count
                if fallback_parent.track_count
                else 0.0
            )
            return {
                "bpm_mean": fallback_parent.bpm_mean,
                "bpm_std": fallback_parent.bpm_std,
                "total_duration": avg_duration * len(tracks),
            }

        cluster_bpms: list[float] = []
        cluster_durations: list[float] = []
        for t in tracks:
            meta = metadata[t]
            if meta.bpm is not None:
                cluster_bpms.append(meta.bpm)
            cluster_durations.append(meta.duration or 0.0)
        bpm_mean = float(np.mean(cluster_bpms)) if cluster_bpms else fallback_parent.bpm_mean
        bpm_std = float(np.std(cluster_bpms)) if cluster_bpms else fallback_parent.bpm_std
        return {
            "bpm_mean": bpm_mean,
            "bpm_std": bpm_std,
            "total_duration": float(sum(cluster_durations)),
        }

    def _compute_feature_means(self, tracks: list[Path]) -> dict[str, float] | None:
        """Compute per-feature means for the given tracks from stored state.

        Args:
            tracks: List of track file paths.

        Returns:
            Dictionary mapping feature names to mean values, or None if state is missing.
        """
        if self._last_metadata_dict is None or self._last_intensity_dict is None:
            return None

        metadata = self._last_metadata_dict
        intensity = self._last_intensity_dict

        vectors: list[np.ndarray] = []
        for t in tracks:
            if t not in intensity:
                continue
            meta = metadata[t]
            bpm = 120.0 if meta.bpm is None else meta.bpm
            feature_vec = intensity[t].to_feature_vector()
            if isinstance(feature_vec, dict):
                continue
            vectors.append(np.concatenate(([float(bpm)], feature_vec)))

        if not vectors:
            return None

        matrix = np.vstack(vectors)
        f_means: dict[str, float] = {
            name: float(matrix[:, i].mean()) for i, name in enumerate(FEATURE_NAMES)
        }

        hardness_values = [intensity[t].hardness for t in tracks if t in intensity]
        if hardness_values:
            f_means["hardness"] = float(np.mean(hardness_values))

        return f_means

    def split_cluster(self, cluster: ClusterResult, target_size: int) -> list[ClusterResult]:
        """Split an over-sized cluster into smaller, coherent sub-clusters.

        The method tries, in order:
          1. Re-cluster on stored standardized feature vectors (features path).
          2. Sort by BPM and slice contiguously (BPM-only path).
          3. Preserve the supplied track order and slice (no-data fallback).

        No random shuffling is used; ``random_state`` only seeds the inner KMeans.

        Args:
            cluster: ClusterResult to split.
            target_size: Target number of tracks per sub-cluster.

        Returns:
            List of sub-clusters partitioning the parent tracks.
        """
        if cluster.track_count <= target_size:
            return [cluster]

        num_splits = int(np.ceil(cluster.track_count / target_size))

        # Features path: stored intensity + standardized feature matrix available.
        if self._last_intensity_dict is not None:
            missing = [t for t in cluster.tracks if t not in self._last_intensity_dict]
            if not missing:
                return self._split_by_features(cluster, target_size, num_splits)
            logger.warning(
                "%d tracks missing from stored intensity features; cannot use features path",
                len(missing),
            )

        # BPM-only path: stored metadata available (no intensity features).
        if self._last_metadata_dict is not None:
            missing = [t for t in cluster.tracks if t not in self._last_metadata_dict]
            if not missing:
                return self._split_by_bpm(cluster, target_size, num_splits)
            logger.warning(
                "%d tracks missing from stored metadata; cannot use BPM path", len(missing)
            )

        # No stored state: deterministic fallback using the supplied order.
        return self._split_fallback(cluster, target_size, num_splits)
