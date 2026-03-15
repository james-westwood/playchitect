"""Intelligent playlist naming system.

Wires vibe profiler and grammar engine into a complete naming system for clusters.
Generates natural-sounding playlist names based on cluster characteristics.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from playchitect.core.naming.grammar_engine import generate_name
from playchitect.core.naming.vibe_profiler import (
    VibeProfile,
    bucket_bpm,
    bucket_energy,
    compute_vibe_profile,
    score_salience,
)

if TYPE_CHECKING:
    from playchitect.core.clustering import ClusterResult
    from playchitect.core.intensity_analyzer import IntensityFeatures
    from playchitect.core.metadata_extractor import TrackMetadata

# Mapping of mood labels and feature buckets to descriptor words
TAG_TO_DESCRIPTORS: dict[str, list[str]] = {
    # Mood-based descriptors
    "Dark": ["Dark", "Shadowed", "Nocturnal", "Obscure", "Gloomy"],
    "Ethereal": ["Ethereal", "Celestial", "Airy", "Delicate", "Dreamy"],
    "Aggressive": ["Aggressive", "Fierce", "Relentless", "Intense", "Forceful"],
    "Melancholic": ["Melancholic", "Somber", "Mournful", "Wistful", "Reflective"],
    "Euphoric": ["Euphoric", "Exultant", "Ecstatic", "Triumphant", "Joyous"],
    "Tense": ["Tense", "Suspenseful", "Nervous", "Edgy", "Uneasy"],
    "Mysterious": ["Mysterious", "Enigmatic", "Cryptic", "Obscure", "Veiled"],
    # BPM bucket descriptors
    "Slow": ["Languid", "Unhurried", "Leisurely", "Relaxed", "Mellow"],
    "Mid-Tempo": ["Steady", "Measured", "Moderate", "Balanced", "Even"],
    "Peak Hour": ["Driving", "Propulsive", "Pulsing", "Thrusting", "Pushing"],
    "High Energy": ["Rapid", "Swift", "Brisk", "Accelerated", "Turbo"],
    # Energy bucket descriptors
    "Subtle": ["Subtle", "Delicate", "Nuanced", "Understated", "Refined"],
    "Groovy": ["Groovy", "Rhythmic", "Hypnotic", "Catchy", "Infectious"],
    "Energetic": ["Energetic", "Vigorous", "Dynamic", "Animated", "Lively"],
    "Intense": ["Intense", "Extreme", "Severe", "Powerful", "Strong"],
    # Feature-based descriptors (mapped from salient features)
    "mean_brightness": ["Bright", "Luminous", "Radiant", "Brilliant", "Vivid"],
    "mean_rms": ["Loud", "Powerful", "Thunderous", "Resonant", "Potent"],
    "mean_percussiveness": ["Percussive", "Punchy", "Driving", "Staccato", "Rhythmic"],
    "mean_vocal_presence": ["Vocal", "Lyrical", "Sung", "Harmonic", "Melodic"],
    "mean_bpm": ["Fast", "Quick", "Rapid", "Swift", "Accelerated"],
}

# Mapping of moods to appropriate nouns
MOOD_TO_NOUN: dict[str, str] = {
    "Dark": "Journey",
    "Ethereal": "Wave",
    "Aggressive": "Set",
    "Melancholic": "Journey",
    "Euphoric": "Wave",
    "Tense": "Set",
    "Mysterious": "Journey",
}

# Default nouns when mood is not in mapping
DEFAULT_NOUNS: list[str] = ["Journey", "Groove", "Set", "Wave"]

# Feature-to-descriptor mapping for salient features
FEATURE_TO_DESCRIPTOR_KEY: dict[str, str] = {
    "mean_brightness": "mean_brightness",
    "mean_rms": "mean_rms",
    "mean_percussiveness": "mean_percussiveness",
    "mean_vocal_presence": "mean_vocal_presence",
    "mean_bpm": "mean_bpm",
}

# Thresholds for descriptor selection
_TOP_SALIENCE_COUNT: int = 2


def _get_noun_for_mood(mood: str, used_nouns: set[str]) -> str:
    """Get appropriate noun for a mood, avoiding used ones.

    Args:
        mood: The dominant mood of the cluster.
        used_nouns: Set of nouns already used in other names.

    Returns:
        An appropriate noun string.
    """
    # Try mood-specific noun first
    preferred = MOOD_TO_NOUN.get(mood)
    if preferred and preferred not in used_nouns:
        return preferred

    # Fall back to defaults in order
    for noun in DEFAULT_NOUNS:
        if noun not in used_nouns:
            return noun

    # All defaults used, just return first default
    return DEFAULT_NOUNS[0]


def _descriptors_from_salience(
    salience: dict[str, float],
    profile: VibeProfile,
) -> list[str]:
    """Generate descriptor words from salient features.

    Args:
        salience: Mapping of feature names to z-scores.
        profile: The cluster's vibe profile.

    Returns:
        List of descriptor words.
    """
    descriptors: list[str] = []

    # Sort by absolute z-score (most distinctive first)
    sorted_features = sorted(
        salience.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    # Take top 2 most salient features
    for feature_name, z_score in sorted_features[:_TOP_SALIENCE_COUNT]:
        descriptor_key = FEATURE_TO_DESCRIPTOR_KEY.get(feature_name)
        if descriptor_key and descriptor_key in TAG_TO_DESCRIPTORS:
            desc_list = TAG_TO_DESCRIPTORS[descriptor_key]
            # Pick descriptor based on direction of z-score
            if z_score > 0:
                # High value - use first (strongest) descriptor
                descriptors.append(desc_list[0])
            else:
                # Low value - look for an opposite descriptor or use a modifier
                # For now, use the second option as a "lighter" variant
                if len(desc_list) > 1:
                    descriptors.append(desc_list[1])
                else:
                    descriptors.append(desc_list[0])

    return descriptors


def _descriptors_from_fallback(
    profile: VibeProfile,
) -> list[str]:
    """Generate descriptor words from dominant mood and BPM bucket.

    Used when no salient features are detected.

    Args:
        profile: The cluster's vibe profile.

    Returns:
        List of descriptor words.
    """
    descriptors: list[str] = []

    # Add mood descriptor
    if profile.dominant_mood in TAG_TO_DESCRIPTORS:
        mood_descs = TAG_TO_DESCRIPTORS[profile.dominant_mood]
        if mood_descs:
            descriptors.append(mood_descs[0])

    # Add BPM bucket descriptor
    bpm_bucket = bucket_bpm(profile.mean_bpm)
    if bpm_bucket in TAG_TO_DESCRIPTORS:
        bpm_descs = TAG_TO_DESCRIPTORS[bpm_bucket]
        if bpm_descs:
            descriptors.append(bpm_descs[0])

    return descriptors


class PlaylistNamer:
    """Generates intelligent names for playlist clusters.

    Uses vibe profiling, salience scoring, and grammar engine to create
    natural-sounding playlist names that reflect cluster characteristics.
    """

    def __init__(self) -> None:
        """Initialize the playlist namer."""
        self._used_names: set[str] = set()
        self._used_nouns: set[str] = set()
        self._name_counter: dict[str, int] = {}

    def name_cluster(
        self,
        profile: VibeProfile,
        salience: dict[str, float],
        library_profiles: list[VibeProfile],
    ) -> str:
        """Generate a name for a single cluster.

        Picks the top 2 salient feature descriptors (or falls back to
        dominant_mood + bpm_bucket) and generates a natural-sounding name.

        Args:
            profile: The cluster's vibe profile.
            salience: Mapping of salient feature names to z-scores.
            library_profiles: All vibe profiles in the library (for context).

        Returns:
            A generated playlist name string.
        """
        # Determine descriptors
        if salience:
            descriptors = _descriptors_from_salience(salience, profile)
        else:
            descriptors = _descriptors_from_fallback(profile)

        # Ensure we have at least one descriptor
        if not descriptors:
            descriptors = ["Mixed"]

        # Select appropriate noun based on mood
        noun = _get_noun_for_mood(profile.dominant_mood, self._used_nouns)
        self._used_nouns.add(noun)

        # Generate name using grammar engine
        name = generate_name(descriptors, noun, self._used_names)

        # Handle duplicates by appending Roman numerals
        final_name = self._ensure_unique(name)

        return final_name

    def _ensure_unique(self, name: str) -> str:
        """Ensure name is unique by appending Roman numerals if needed.

        Args:
            name: The proposed name.

        Returns:
            A unique name (possibly with II, III, etc. appended).
        """
        if name not in self._used_names:
            self._used_names.add(name)
            return name

        # Name collision - append Roman numeral
        base_name = name
        counter = self._name_counter.get(base_name, 1)

        roman_numerals = ["II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]

        while name in self._used_names:
            if counter - 1 < len(roman_numerals):
                name = f"{base_name} {roman_numerals[counter - 1]}"
            else:
                # Fall back to Arabic numerals if we run out of Roman numerals
                name = f"{base_name} {counter + 1}"
            counter += 1

        self._name_counter[base_name] = counter
        self._used_names.add(name)
        return name

    def name_all_clusters(
        self,
        clusters: list["ClusterResult"],
        features: dict[Path, "IntensityFeatures"],
        metadata: dict[Path, "TrackMetadata"],
    ) -> dict[int | str, str]:
        """Generate names for all clusters.

        Computes vibe profiles and salience scores for each cluster,
        generates names, and ensures no duplicates.

        Args:
            clusters: List of cluster results.
            features: Mapping of file path to IntensityFeatures.
            metadata: Mapping of file path to TrackMetadata.

        Returns:
            Mapping of cluster_id to generated name.
        """
        # Reset state for fresh naming session
        self._used_names.clear()
        self._used_nouns.clear()
        self._name_counter.clear()

        # Compute vibe profiles for all clusters
        profiles: dict[int | str, VibeProfile] = {}
        for cluster in clusters:
            try:
                profile = compute_vibe_profile(cluster, features)
                profiles[cluster.cluster_id] = profile
            except ValueError:
                # Skip clusters with no features
                continue

        # Collect all profiles for salience scoring
        library_profiles = list(profiles.values())

        # Generate names for each cluster
        result: dict[int | str, str] = {}
        for cluster in clusters:
            if cluster.cluster_id not in profiles:
                # Skip clusters without profiles
                result[cluster.cluster_id] = f"Cluster {cluster.cluster_id}"
                continue

            profile = profiles[cluster.cluster_id]

            # Compute salience
            if len(library_profiles) > 1:
                salience = score_salience(profile, library_profiles)
            else:
                salience = {}

            # Generate name
            name = self.name_cluster(profile, salience, library_profiles)
            result[cluster.cluster_id] = name

        return result
