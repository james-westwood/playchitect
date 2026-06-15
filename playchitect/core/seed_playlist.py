"Seed-based playlist generation: build length-targeted playlists from a single seed track."

import logging
from statistics import fmean, pstdev
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from playchitect.core.clustering import ClusterResult
from playchitect.core.features import build_feature_vector
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata
from playchitect.core.sequencer import sequence_by_strategy
from playchitect.core.weighting import select_weights
from playchitect.utils.weight_config import WeightOverrides

logger = logging.getLogger(__name__)

__all__ = ["generate_playlist_from_seed"]

_VALID_SEQUENCE_MODES: frozenset[str] = frozenset(
    {"ramp", "build", "descent", "alternating", "bpm_asc", "bpm_desc"}
)


def rank_by_similarity(
    seed_vec: np.ndarray,
    candidates: dict[Path, np.ndarray],
    *,
    genre: str | None = None,
    weight_overrides: WeightOverrides | None = None,
) -> list[tuple[Path, float]]:
    """Rank candidates by weighted Euclidean distance from the seed vector.

    Uses StandardScaler for normalisation and select_weights() for
    genre-aware feature weighting.

    Args:
        seed_vec: 8-D seed feature vector.
        candidates: Mapping of path → 8-D feature vector.
        genre: Optional genre hint for feature weighting.
        weight_overrides: Optional user-specified weight overrides.

    Returns:
        List of (path, distance) sorted by distance, nearest first.
    """
    if not candidates:
        return []

    paths = list(candidates.keys())
    matrix = np.array([candidates[p] for p in paths])

    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)
    seed_scaled: np.ndarray = scaler.transform(seed_vec.reshape(1, -1))[0]

    weight_profile = select_weights(
        X_scaled=matrix_scaled,
        genre=genre,
        weight_overrides=weight_overrides,
    )
    weights = weight_profile.weights

    matrix_weighted = matrix_scaled * weights
    seed_weighted = seed_scaled * weights

    distances: list[tuple[Path, float]] = []
    for i, path in enumerate(paths):
        dist = float(np.linalg.norm(matrix_weighted[i] - seed_weighted))
        distances.append((path, dist))

    distances.sort(key=lambda x: x[1])
    return distances


def fill_to_duration(
    ranked: list[tuple[Path, float]],
    metadata_dict: dict[Path, TrackMetadata],
    target_secs: float,
    tolerance: float,
    seed_path: Path,
) -> list[Path]:
    """Greedily fill a playlist from ranked candidates up to a target duration.

    The seed track is always included first regardless of ranking.

    Args:
        ranked: Candidates sorted by distance, nearest first.
        metadata_dict: Path → TrackMetadata for duration lookups.
        target_secs: Target total duration in seconds.
        tolerance: Fractional tolerance for exceeding target
                   (e.g. 0.1 allows 10% overshoot).
        seed_path: Path of the seed track — always included first.

    Returns:
        List of Path in the order they should appear in the playlist.
    """
    selected: list[Path] = [seed_path]
    seed_meta_default = TrackMetadata(filepath=seed_path)
    cumulative: float = (
        metadata_dict.get(seed_path, seed_meta_default).duration or 0.0
    )
    max_allowed = target_secs * (1.0 + tolerance)

    for path, _dist in ranked:
        meta = metadata_dict.get(path)
        if meta is None or meta.duration is None or meta.duration <= 0:
            continue

        if cumulative + meta.duration <= max_allowed:
            selected.append(path)
            cumulative += meta.duration

    return selected


def generate_playlist_from_seed(
    seed_path: Path,
    candidate_features: dict[Path, IntensityFeatures],
    metadata_dict: dict[Path, TrackMetadata],
    target_duration_mins: float,
    *,
    genre: str | None = None,
    weight_overrides: WeightOverrides | None = None,
    tolerance: float = 0.1,
    sequence_mode: str = "ramp",
) -> ClusterResult:
    """Generate a playlist of tracks similar to a seed track, filled to a target duration.

    The engine:
    1. Builds 8-D feature vectors for all candidates using build_feature_vector()
    2. Scales and applies genre-aware feature weights via rank_by_similarity()
    3. Greedily fills to the target duration via fill_to_duration()
    4. Sequences selected tracks via sequence_by_strategy()
    5. Returns a ClusterResult with stats and the generated playlist

    Args:
        seed_path: Path to the seed track.
        candidate_features: Mapping of path → IntensityFeatures for all
            candidate tracks (must include the seed).
        metadata_dict: Mapping of path → TrackMetadata for duration/BPM lookups.
        target_duration_mins: Desired playlist length in minutes (must be > 0).
        genre: Optional genre hint for feature weighting
               ('techno', 'house', 'ambient', 'dnb').
        weight_overrides: Optional user-specified weight overrides.
        tolerance: Fractional tolerance for exceeding the target duration
                   (e.g. 0.1 = ±10%).
        sequence_mode: Energy sequencing strategy. One of:
                       'ramp', 'build', 'descent', 'alternating',
                       'bpm_asc', 'bpm_desc'. Defaults to 'ramp'.

    Returns:
        A ClusterResult describing the generated playlist.

    Raises:
        ValueError: If target_duration_mins <= 0, candidate_features is empty,
                    seed_path is missing from candidate_features, or the seed
                    has no valid feature vector.
    """
    # 1. Validate inputs
    if target_duration_mins <= 0:
        raise ValueError("target_duration_mins must be positive")
    if not candidate_features:
        raise ValueError("candidate_features must not be empty")
    if seed_path not in candidate_features:
        raise ValueError("seed_path not found in candidate_features")

    # 2. Build 8-D vectors for all candidates
    candidates: dict[Path, np.ndarray] = {}
    for path, feats in candidate_features.items():
        meta = metadata_dict.get(path)
        if meta is None:
            logger.warning("No metadata for %s — skipping", path)
            continue
        vec = build_feature_vector(meta, feats)
        if vec is None:
            logger.warning("Could not build feature vector for %s — skipping", path)
            continue
        candidates[path] = vec

    # 3. Get seed vector
    if seed_path not in candidates:
        raise ValueError(
            f"Seed track {seed_path} has no valid feature vector "
            "(missing BPM or features)"
        )

    seed_vec = candidates[seed_path]

    # 4. Rank by similarity
    ranked = rank_by_similarity(
        seed_vec=seed_vec,
        candidates={
            p: v for p, v in candidates.items() if p != seed_path
        },
        genre=genre,
        weight_overrides=weight_overrides,
    )

    # 5. Fill to duration
    target_secs = target_duration_mins * 60.0
    selected = fill_to_duration(
        ranked=ranked,
        metadata_dict=metadata_dict,
        target_secs=target_secs,
        tolerance=tolerance,
        seed_path=seed_path,
    )

    # 6. Sequence
    if sequence_mode not in _VALID_SEQUENCE_MODES:
        logger.warning(
            "Unknown sequence_mode '%s' — falling back to 'ramp'", sequence_mode
        )
        sequence_mode = "ramp"

    sequenced_tracks = sequence_by_strategy(
        tracks=selected,
        features=candidate_features,
        strategy=sequence_mode,
        metadata=metadata_dict,
    )

    # 7. Compute stats
    bpm_values = [
        float(track_meta.bpm)
        for p in sequenced_tracks
        if (track_meta := metadata_dict.get(p)) is not None and track_meta.bpm is not None
    ]
    bpm_mean = fmean(bpm_values) if bpm_values else 0.0
    bpm_std = pstdev(bpm_values) if len(bpm_values) > 1 else 0.0
    total_duration = sum(
        metadata_dict[p].duration or 0.0 for p in sequenced_tracks
    )

    # Seed title for auto-naming
    seed_meta = metadata_dict.get(seed_path)
    seed_title = (
        seed_meta.title if seed_meta and seed_meta.title else seed_path.stem
    )

    # Centroid: scaled + weighted seed vector
    # We need to recompute scaling over all candidates for the centroid
    all_vecs = np.array(list(candidates.values()))
    scaler = StandardScaler()
    all_scaled = scaler.fit_transform(all_vecs)
    seed_scaled_centroid: np.ndarray = scaler.transform(seed_vec.reshape(1, -1))[0]
    weight_profile = select_weights(
        X_scaled=all_scaled,
        genre=genre,
        weight_overrides=weight_overrides,
    )
    centroid = seed_scaled_centroid * weight_profile.weights

    # 8. Return ClusterResult
    return ClusterResult(
        cluster_id="seed",
        tracks=sequenced_tracks,
        bpm_mean=bpm_mean,
        bpm_std=bpm_std,
        track_count=len(sequenced_tracks),
        total_duration=total_duration,
        genre=f"Like: {seed_title}",
        centroid=centroid,
    )
