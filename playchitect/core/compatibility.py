"""Track compatibility scoring and next-track suggestions.

Provides algorithms for determining how well two tracks mix together based on
BPM, key compatibility, energy similarity, and timbre similarity.
"""

from __future__ import annotations

from pathlib import Path

from playchitect.core.intensity_analyzer import IntensityFeatures, harmonic_compatibility

# Weight constants for compatibility score components
_BPM_WEIGHT: float = 0.3
_KEY_WEIGHT: float = 0.3
_ENERGY_WEIGHT: float = 0.25
_TIMBRE_WEIGHT: float = 0.15

# BPM difference normalization factor (BPM diff of 20 = score of 0)
_BPM_NORMALIZATION: float = 20.0


def compatibility_score(
    feat_a: IntensityFeatures,
    feat_b: IntensityFeatures,
    bpm_a: float,
    bpm_b: float,
) -> float:
    """Calculate compatibility score between two tracks.

    The score is a weighted combination of four components:
    - BPM score: how close the tempos are (normalized to 20 BPM difference)
    - Key score: whether keys are harmonically compatible (binary 0 or 1)
    - Energy score: similarity in RMS energy levels
    - Timbre score: similarity in spectral brightness

    Args:
        feat_a: Intensity features for track A
        feat_b: Intensity features for track B
        bpm_a: BPM for track A (from TrackMetadata)
        bpm_b: BPM for track B (from TrackMetadata)

    Returns:
        Compatibility score between 0.0 and 1.0, where 1.0 means
        perfectly compatible for mixing.
    """
    # BPM similarity score (0 to 1)
    bpm_diff = abs(bpm_a - bpm_b)
    bpm_score = max(0.0, 1.0 - (bpm_diff / _BPM_NORMALIZATION))

    # Key compatibility score (0 or 1)
    key_score = 1.0 if harmonic_compatibility(feat_a.camelot_key, feat_b.camelot_key) else 0.0

    # Energy similarity score (0 to 1)
    energy_score = 1.0 - abs(feat_a.rms_energy - feat_b.rms_energy)

    # Timbre/brightness similarity score (0 to 1)
    timbre_score = 1.0 - abs(feat_a.brightness - feat_b.brightness)

    # Weighted combination
    final_score = (
        _BPM_WEIGHT * bpm_score
        + _KEY_WEIGHT * key_score
        + _ENERGY_WEIGHT * energy_score
        + _TIMBRE_WEIGHT * timbre_score
    )

    return float(final_score)


def next_track_suggestions(
    current_path: Path,
    current_features: IntensityFeatures,
    current_bpm: float,
    candidates: list[tuple[Path, IntensityFeatures, float]],
    n: int = 5,
) -> list[tuple[Path, float]]:
    """Get top-N next track suggestions based on compatibility scoring.

    Ranks candidate tracks by compatibility score with the current track,
    returning the highest-scoring candidates sorted descending.

    Args:
        current_path: Path to the current track (excluded from results)
        current_features: Intensity features for the current track
        current_bpm: BPM of the current track
        candidates: List of (path, features, bpm) tuples for candidate tracks
        n: Number of suggestions to return (default: 5)

    Returns:
        List of (path, score) tuples for the top N compatible tracks,
        sorted by score in descending order.
    """
    scored: list[tuple[Path, float]] = []

    for candidate_path, candidate_features, candidate_bpm in candidates:
        # Skip if this is the current track
        if candidate_path == current_path:
            continue

        score = compatibility_score(
            current_features,
            candidate_features,
            current_bpm,
            candidate_bpm,
        )
        scored.append((candidate_path, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Return top N
    return scored[:n]
