"""Vibe profiling and salience scoring for playlist naming.

Provides data layer for intelligent playlist naming by computing cluster-level
vibe profiles and identifying salient (statistically distinctive) characteristics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from playchitect.core.clustering import ClusterResult
    from playchitect.core.intensity_analyzer import IntensityFeatures
    from playchitect.core.metadata_extractor import TrackMetadata

# BPM bucketing thresholds
_BPM_SLOW_THRESHOLD: float = 100.0
_BPM_MID_TEMPO_THRESHOLD: float = 120.0
_BPM_PEAK_HOUR_THRESHOLD: float = 135.0

# Energy bucketing thresholds
_ENERGY_SUBTLE_THRESHOLD: float = 0.3
_ENERGY_GROOVY_THRESHOLD: float = 0.5
_ENERGY_ENERGETIC_THRESHOLD: float = 0.7

# Salience scoring threshold
_SALIENCE_ZSCORE_THRESHOLD: float = 1.5


@dataclass
class VibeProfile:
    """Vibe profile for a cluster, used as data layer for playlist naming.

    Attributes:
        cluster_id: Cluster identifier (int or str for subclusters)
        mean_bpm: Average BPM across tracks in cluster
        mean_rms: Average RMS energy (loudness) across tracks
        mean_brightness: Average spectral brightness across tracks
        mean_percussiveness: Average percussiveness across tracks
        mean_vocal_presence: Average vocal presence across tracks
        dominant_mood: Most frequent mood label in cluster
        mood_distribution: Mapping of mood → fraction of tracks (0.0-1.0)
    """

    cluster_id: int | str
    mean_bpm: float
    mean_rms: float
    mean_brightness: float
    mean_percussiveness: float
    mean_vocal_presence: float
    dominant_mood: str
    mood_distribution: dict[str, float]


def compute_vibe_profile(
    cluster: "ClusterResult",
    features: dict[Path, "IntensityFeatures"],
    metadata: dict[Path, "TrackMetadata"] | None = None,
) -> VibeProfile:
    """Compute vibe profile for a cluster.

    Averages each feature field across all tracks in the cluster and computes
    the dominant mood as the most frequent mood_label across tracks.

    Args:
        cluster: ClusterResult containing tracks and basic stats
        features: Mapping of file path → IntensityFeatures
        metadata: Optional mapping of file path → TrackMetadata (for API compliance)

    Returns:
        VibeProfile with averaged features and mood distribution

    Raises:
        ValueError: If cluster has no tracks or features are missing
    """
    if not cluster.tracks:
        raise ValueError("Cannot compute vibe profile for empty cluster")

    # Collect feature values for tracks in this cluster
    cluster_features: list[IntensityFeatures] = []
    for track_path in cluster.tracks:
        if track_path in features:
            cluster_features.append(features[track_path])

    if not cluster_features:
        raise ValueError(f"No intensity features found for cluster {cluster.cluster_id}")

    # Calculate means for each feature
    mean_bpm = cluster.bpm_mean  # Use pre-computed from cluster
    mean_rms = float(np.mean([f.rms_energy for f in cluster_features]))
    mean_brightness = float(np.mean([f.brightness for f in cluster_features]))
    mean_percussiveness = float(np.mean([f.percussiveness for f in cluster_features]))
    mean_vocal_presence = float(np.mean([f.vocal_presence for f in cluster_features]))

    # Compute mood distribution
    mood_counts: dict[str, int] = {}
    for f in cluster_features:
        mood = f.mood_label
        mood_counts[mood] = mood_counts.get(mood, 0) + 1

    total_tracks = len(cluster_features)
    mood_distribution = {mood: count / total_tracks for mood, count in mood_counts.items()}

    # Dominant mood is the most frequent
    dominant_mood: str = "Ethereal"
    if mood_counts:
        # Use list() to ensure we have an iterable for max()
        moods = list(mood_counts.keys())
        dominant_mood = max(moods, key=lambda m: mood_counts[m])

    return VibeProfile(
        cluster_id=cluster.cluster_id,
        mean_bpm=mean_bpm,
        mean_rms=mean_rms,
        mean_brightness=mean_brightness,
        mean_percussiveness=mean_percussiveness,
        mean_vocal_presence=mean_vocal_presence,
        dominant_mood=dominant_mood,
        mood_distribution=mood_distribution,
    )


def score_salience(
    profile: VibeProfile,
    library_profiles: list[VibeProfile],
) -> dict[str, float]:
    """Score how salient (distinctive) a cluster's features are vs. library.

    Computes z-scores for mean_bpm, mean_rms, mean_brightness,
    mean_percussiveness, and mean_vocal_presence against the library mean/std.
    Only returns entries with abs(z) > 1.5.

    Args:
        profile: The cluster's vibe profile to score
        library_profiles: All vibe profiles in the library for comparison

    Returns:
        Dict mapping feature_name → z-score for statistically distinctive features

    Raises:
        ValueError: If library_profiles is empty
    """
    if not library_profiles:
        raise ValueError("Cannot compute salience against empty library")

    # Feature fields to analyze
    feature_fields = [
        "mean_bpm",
        "mean_rms",
        "mean_brightness",
        "mean_percussiveness",
        "mean_vocal_presence",
    ]

    salient_features: dict[str, float] = {}

    for field in feature_fields:
        # Extract values for this field from all library profiles
        values = [getattr(p, field) for p in library_profiles]

        # Compute library mean and std
        lib_mean = float(np.mean(values))
        lib_std = float(np.std(values))

        # Avoid division by zero
        if lib_std == 0:
            continue

        # Compute z-score for this profile
        profile_value = getattr(profile, field)
        z_score = (profile_value - lib_mean) / lib_std

        # Only include if abs(z) > threshold
        if abs(z_score) > _SALIENCE_ZSCORE_THRESHOLD:
            salient_features[field] = round(z_score, 6)

    return salient_features


def bucket_bpm(bpm: float) -> str:
    """Categorize BPM into named buckets.

    Buckets:
        - < 100: 'Slow'
        - 100-119: 'Mid-Tempo'
        - 120-135: 'Peak Hour'
        - > 135: 'High Energy'

    Args:
        bpm: Beats per minute value

    Returns:
        Bucket name as string
    """
    if bpm < _BPM_SLOW_THRESHOLD:
        return "Slow"
    if bpm < _BPM_MID_TEMPO_THRESHOLD:
        return "Mid-Tempo"
    if bpm <= _BPM_PEAK_HOUR_THRESHOLD:
        return "Peak Hour"
    return "High Energy"


def bucket_energy(rms: float) -> str:
    """Categorize RMS energy into named buckets.

    Buckets:
        - < 0.3: 'Subtle'
        - 0.3-0.49: 'Groovy'
        - 0.5-0.7: 'Energetic'
        - > 0.7: 'Intense'

    Args:
        rms: RMS energy value (0.0-1.0 normalized)

    Returns:
        Bucket name as string
    """
    if rms < _ENERGY_SUBTLE_THRESHOLD:
        return "Subtle"
    if rms < _ENERGY_GROOVY_THRESHOLD:
        return "Groovy"
    if rms <= _ENERGY_ENERGETIC_THRESHOLD:
        return "Energetic"
    return "Intense"
