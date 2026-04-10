"""Sequencing engine for ordering tracks within clusters.

Creates cohesive DJ set narratives using intensity ramps and smart
opener/closer placement.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from playchitect.core.clustering import ClusterResult
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata
from playchitect.core.track_selector import TrackSelector

if TYPE_CHECKING:
    from playchitect.core.play_history import PlayHistory

logger = logging.getLogger(__name__)


class SequencingStrategy(StrEnum):
    """Available sequencing strategies for energy flow control."""

    RAMP = "ramp"  # Energy ramp (default): sort by RMS energy ascending
    BUILD = "build"  # Build to peak: sort by energy_gradient descending
    DESCENT = "descent"  # Gradual descent: sort by RMS energy descending
    ALTERNATING = "alternating"  # Alternating: interleave high/low energy


class FiveRhythmsPhase(StrEnum):
    """Five Rhythms phases for dance/movement-based sequencing."""

    FLOWING = "flowing"
    STACCATO = "staccato"
    CHAOS = "chaos"
    LYRICAL = "lyrical"
    STILLNESS = "stillness"


# Phase ordering for Five Rhythms sequence
_FIVE_RHYTHMS_ORDER: list[FiveRhythmsPhase] = [
    FiveRhythmsPhase.FLOWING,
    FiveRhythmsPhase.STACCATO,
    FiveRhythmsPhase.CHAOS,
    FiveRhythmsPhase.LYRICAL,
    FiveRhythmsPhase.STILLNESS,
]

# BPM and energy thresholds for Five Rhythms classification
_FIVE_RHYTHMS_THRESHOLDS: dict[str, dict[str, float]] = {
    "flowing": {"bpm_min": 85.0, "bpm_max": 115.0, "rms_max": 0.5},
    "staccato": {"bpm_min": 115.0, "bpm_max": 135.0, "rms_min": 0.4, "rms_max": 0.7},
    "chaos": {"bpm_threshold": 135.0, "rms_threshold": 0.75},
    "stillness": {"bpm_threshold": 85.0, "rms_threshold": 0.25},
}


def classify_five_rhythms_phase(bpm: float, rms_energy: float) -> FiveRhythmsPhase:
    """
    Classify a track into a Five Rhythms phase based on BPM and RMS energy.

    Classification rules:
    - Flowing: BPM 85-115 and RMS < 0.5 (smooth, continuous movement)
    - Staccato: BPM 115-135 and RMS 0.4-0.7 (sharp, percussive)
    - Chaos: BPM > 135 or RMS > 0.75 (high energy, release)
    - Stillness: BPM < 85 or RMS < 0.25 (quiet, meditative)
    - Lyrical: Everything else (expressive, melodic)

    Args:
        bpm: Track tempo in beats per minute (must be > 0)
        rms_energy: RMS energy level (0.0-1.0)

    Returns:
        FiveRhythmsPhase classification for the track

    Raises:
        ValueError: If BPM is not positive
    """
    if bpm <= 0:
        raise ValueError(f"BPM must be positive, got {bpm}")

    thresholds = _FIVE_RHYTHMS_THRESHOLDS

    # Chaos: high BPM or very high energy
    chaos_thresholds = thresholds["chaos"]
    if bpm > chaos_thresholds["bpm_threshold"] or rms_energy > chaos_thresholds["rms_threshold"]:
        return FiveRhythmsPhase.CHAOS

    # Stillness: low BPM or very low energy
    stillness_thresholds = thresholds["stillness"]
    if (
        bpm < stillness_thresholds["bpm_threshold"]
        or rms_energy < stillness_thresholds["rms_threshold"]
    ):
        return FiveRhythmsPhase.STILLNESS

    # Flowing: mid BPM with low energy
    flowing_thresholds = thresholds["flowing"]
    if (
        flowing_thresholds["bpm_min"] <= bpm < flowing_thresholds["bpm_max"]
        and rms_energy < flowing_thresholds["rms_max"]
    ):
        return FiveRhythmsPhase.FLOWING

    # Staccato: higher BPM with mid energy
    staccato_thresholds = thresholds["staccato"]
    if (
        staccato_thresholds["bpm_min"] <= bpm < staccato_thresholds["bpm_max"]
        and staccato_thresholds["rms_min"] <= rms_energy < staccato_thresholds["rms_max"]
    ):
        return FiveRhythmsPhase.STACCATO

    # Default: Lyrical
    return FiveRhythmsPhase.LYRICAL


def sequence_five_rhythms(
    tracks: list[Path],
    metadata: dict[Path, TrackMetadata],
    features: dict[Path, IntensityFeatures],
) -> list[Path]:
    """
    Sequence tracks according to the Five Rhythms flow.

    Groups tracks by Five Rhythms phase, then orders them:
    Flowing → Staccato → Chaos → Lyrical → Stillness
    Within each phase, tracks are sorted by energy ascending.
    Phases with no tracks are skipped.

    Args:
        tracks: List of track paths to sequence
        metadata: Mapping of path to TrackMetadata (contains BPM)
        features: Mapping of path to IntensityFeatures (contains RMS energy)

    Returns:
        List of track paths in Five Rhythms order

    Raises:
        ValueError: If a track is missing from the metadata or features dict
    """
    # Group tracks by phase
    phase_groups: dict[FiveRhythmsPhase, list[Path]] = {phase: [] for phase in FiveRhythmsPhase}

    for track in tracks:
        if track not in metadata:
            raise ValueError(f"Missing metadata for track: {track}")
        if track not in features:
            raise ValueError(f"Missing features for track: {track}")

        track_bpm = metadata[track].bpm
        if track_bpm is None or track_bpm <= 0:
            # Skip tracks without valid BPM - they default to lyrical
            phase_groups[FiveRhythmsPhase.LYRICAL].append(track)
            continue

        rms_energy = features[track].rms_energy
        phase = classify_five_rhythms_phase(track_bpm, rms_energy)
        phase_groups[phase].append(track)

    # Build final sequence in phase order
    result: list[Path] = []
    for phase in _FIVE_RHYTHMS_ORDER:
        group = phase_groups[phase]
        if not group:
            continue

        # Sort within phase by RMS energy ascending
        group.sort(key=lambda t: features[t].rms_energy)
        result.extend(group)

    logger.info(
        "Sequenced %d tracks using Five Rhythms: %s",
        len(result),
        {p.value: len(phase_groups[p]) for p in FiveRhythmsPhase if phase_groups[p]},
    )

    return result


class Sequencer:
    """Orders tracks within a cluster to form a narrative sequence."""

    def __init__(self) -> None:
        """Initialise sequencer."""
        self.selector = TrackSelector()

    def sequence(
        self,
        cluster: ClusterResult,
        metadata_dict: dict[Path, TrackMetadata],
        intensity_dict: dict[Path, IntensityFeatures],
        mode: str = "ramp",
    ) -> list[Path]:
        """
        Order tracks in a cluster based on the specified mode.

        Args:
            cluster: Cluster to sequence.
            metadata_dict: Path -> Metadata mapping.
            intensity_dict: Path -> Intensity mapping.
            mode: 'ramp' (energy build) | 'fixed' (no change)

        Returns:
            List of track paths in sequenced order.
        """
        if mode == "ramp":
            return self._sequence_ramp(cluster, metadata_dict, intensity_dict)

        logger.debug("Unknown or 'fixed' sequence mode '%s', returning original order", mode)
        return cluster.tracks

    def _sequence_ramp(
        self,
        cluster: ClusterResult,
        metadata_dict: dict[Path, TrackMetadata],
        intensity_dict: dict[Path, IntensityFeatures],
    ) -> list[Path]:
        """
        Order tracks from low to high intensity (hardness).
        Ensures best opener is first and best closer is last.
        """
        if len(cluster.tracks) <= 2:
            return cluster.tracks

        # Get opener/closer recommendations
        selection = self.selector.select(cluster, metadata_dict, intensity_dict)
        first = selection.selected_first
        last = selection.selected_last

        # Set them on the cluster object for UI use
        cluster.opener = first
        cluster.closer = last

        # In case the top opener and closer are the same track (rare but possible in small clusters)
        # fallback to second best closer.
        if first == last and len(cluster.tracks) > 1:
            if len(selection.last_tracks) > 1:
                last = selection.last_tracks[1].path
            elif len(selection.first_tracks) > 1:
                first = selection.first_tracks[1].path

        remaining = [t for t in cluster.tracks if t != first and t != last]

        # Sort remaining tracks by hardness
        remaining.sort(key=lambda t: intensity_dict[t].hardness if t in intensity_dict else 0.5)

        logger.info(
            "Sequenced cluster %s as energy ramp (Opener: %s, Closer: %s)",
            cluster.cluster_id,
            first.name,
            last.name,
        )

        return [first] + remaining + [last]


def sequence_fresh(
    tracks: list[Path],
    features: dict[Path, IntensityFeatures],
    history: PlayHistory,
) -> list[Path]:
    """Sequence tracks prioritizing freshness and energy.

    Sorts tracks by freshness_score * rms_energy descending,
    excluding tracks with freshness score < 0.1.

    Args:
        tracks: List of track paths to sequence.
        features: Mapping of path to IntensityFeatures (contains RMS energy).
        history: PlayHistory instance for freshness scores.

    Returns:
        List of track paths sorted by freshness-weighted energy.

    Raises:
        ValueError: If a track is missing from the features dict.
    """

    # Calculate scores for all tracks
    scored_tracks: list[tuple[Path, float]] = []

    for track in tracks:
        if track not in features:
            raise ValueError(f"Missing features for track: {track}")

        freshness = history.get_freshness_score(track)

        # Skip tracks with low freshness score
        if freshness < 0.1:
            logger.debug(
                "Excluding track %s due to low freshness score: %.3f", track.name, freshness
            )
            continue

        rms_energy = features[track].rms_energy
        combined_score = freshness * rms_energy
        scored_tracks.append((track, combined_score))

    # Sort by combined score descending
    scored_tracks.sort(key=lambda x: x[1], reverse=True)

    result = [track for track, _ in scored_tracks]

    logger.info(
        "Sequenced %d tracks using freshness priority (excluded %d)",
        len(result),
        len(tracks) - len(result),
    )

    return result


def sequence_by_strategy(
    tracks: list[Path],
    features: dict[Path, IntensityFeatures],
    strategy: str,
) -> list[Path]:
    """Sequence tracks according to the specified energy flow strategy.

    Strategies:
        - 'ramp': Energy ramp (default) - sort by RMS energy ascending
        - 'build': Build to peak - sort by energy_gradient descending (rising tracks first)
        - 'descent': Gradual descent - sort by RMS energy descending
        - 'alternating': Alternating - interleave high/low energy tracks

    Args:
        tracks: List of track paths to sequence.
        features: Mapping of path to IntensityFeatures (contains RMS energy,
            energy_gradient, etc.).
        strategy: One of 'ramp', 'build', 'descent', 'alternating'.

    Returns:
        List of track paths in the sequenced order.

    Raises:
        ValueError: If a track is missing from the features dict or if
            strategy is invalid.
    """
    if not tracks:
        return []

    # Validate all tracks have features
    for track in tracks:
        if track not in features:
            raise ValueError(f"Missing features for track: {track}")

    # Validate strategy
    valid_strategies = {s.value for s in SequencingStrategy}
    if strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy '{strategy}'. Must be one of: {valid_strategies}")

    result: list[Path] = []

    if strategy == SequencingStrategy.RAMP:
        # Sort by RMS energy ascending (low to high)
        result = sorted(tracks, key=lambda t: features[t].rms_energy)
        logger.info("Sequenced %d tracks using 'ramp' strategy (RMS ascending)", len(result))

    elif strategy == SequencingStrategy.BUILD:
        # Sort by energy_gradient descending (rising tracks first)
        result = sorted(tracks, key=lambda t: features[t].energy_gradient, reverse=True)
        logger.info("Sequenced %d tracks using 'build' strategy (gradient descending)", len(result))

    elif strategy == SequencingStrategy.DESCENT:
        # Sort by RMS energy descending (high to low)
        result = sorted(tracks, key=lambda t: features[t].rms_energy, reverse=True)
        logger.info("Sequenced %d tracks using 'descent' strategy (RMS descending)", len(result))

    elif strategy == SequencingStrategy.ALTERNATING:
        # Interleave high/low energy tracks
        sorted_by_energy = sorted(tracks, key=lambda t: features[t].rms_energy)
        low_half = sorted_by_energy[: len(sorted_by_energy) // 2]
        high_half = sorted_by_energy[len(sorted_by_energy) // 2 :]

        result: list[Path] = []
        # Interleave: take from high (reversed for descending), then low
        high_reversed = list(reversed(high_half))
        low_iter = iter(low_half)
        high_iter = iter(high_reversed)

        # Alternate between high and low
        while True:
            has_low = False
            has_high = False
            try:
                result.append(next(high_iter))
                has_high = True
            except StopIteration:
                pass
            try:
                result.append(next(low_iter))
                has_low = True
            except StopIteration:
                pass
            if not has_low and not has_high:
                break

        logger.info("Sequenced %d tracks using 'alternating' strategy", len(result))

    return result


def _harmonic_score(key_a: str, key_b: str) -> int:
    """Calculate harmonic compatibility score between two Camelot keys.

    Returns:
        Score from 0-2 where:
        - 2: Compatible (same number or adjacent with same letter)
        - 1: Same number, different letter (mode switch - acceptable)
        - 0: Incompatible
    """
    if key_a == key_b:
        return 2  # Same key

    try:
        num_a, letter_a = int(key_a[:-1]), key_a[-1]
        num_b, letter_b = int(key_b[:-1]), key_b[-1]
    except (ValueError, IndexError):
        return 0

    # Same number, different letter = mode switch (score 1)
    if num_a == num_b and letter_a != letter_b:
        return 1

    # Same letter with adjacent numbers (including wrap-around 12->1)
    if letter_a == letter_b:
        diff = abs(num_a - num_b)
        if diff == 1 or diff == 11:
            return 2

    return 0


def sequence_harmonic(
    tracks: list[Path],
    features: dict[Path, IntensityFeatures],
) -> list[Path]:
    """Sequence tracks using greedy nearest-Camelot-neighbour approach.

    Starts from the track with highest energy, then repeatedly picks the next
    track with the highest harmonic compatibility score. Ties are broken by
    energy proximity (prefer tracks with similar RMS energy).

    Args:
        tracks: List of track paths to sequence.
        features: Mapping of path to IntensityFeatures (contains camelot_key).

    Returns:
        List of track paths in harmonic sequence order.

    Raises:
        ValueError: If a track is missing from the features dict or has no key.
    """
    if not tracks:
        return []

    # Validate all tracks have features
    for track in tracks:
        if track not in features:
            raise ValueError(f"Missing features for track: {track}")

    if len(tracks) == 1:
        return tracks

    # Start with highest energy track
    remaining = set(tracks)
    current = max(remaining, key=lambda t: features[t].rms_energy)
    result = [current]
    remaining.remove(current)

    while remaining:
        current_key = features[current].camelot_key
        current_energy = features[current].rms_energy

        # Find best next track by harmonic score, then energy proximity
        best_track: Path | None = None
        best_score = -1
        best_energy_diff = float("inf")

        for candidate in remaining:
            candidate_key = features[candidate].camelot_key
            candidate_energy = features[candidate].rms_energy

            score = _harmonic_score(current_key, candidate_key)
            energy_diff = abs(current_energy - candidate_energy)

            # Prefer higher score, then lower energy difference
            if score > best_score or (score == best_score and energy_diff < best_energy_diff):
                best_track = candidate
                best_score = score
                best_energy_diff = energy_diff

        if best_track is None:
            # Fallback: shouldn't happen but handle gracefully
            best_track = remaining.pop()
        else:
            remaining.remove(best_track)

        result.append(best_track)
        current = best_track

    logger.info(
        "Sequenced %d tracks using harmonic mixing (starting with highest energy)",
        len(result),
    )

    return result


def sequence_by_timbre(
    tracks: list[Path],
    features: dict[Path, IntensityFeatures],
) -> list[Path]:
    """Sequence tracks by timbral similarity using greedy nearest-neighbor approach.

    Starts from the track with highest spectral flatness, then repeatedly picks
    the next track with the closest timbral characteristics. Uses a weighted
    combination of spectral_flatness, mfcc_variance, zero_crossing_rate, and
    spectral_rolloff_85 for distance calculation.

    Args:
        tracks: List of track paths to sequence.
        features: Mapping of path to IntensityFeatures (contains timbre features).

    Returns:
        List of track paths in timbral sequence order.

    Raises:
        ValueError: If a track is missing from the features dict.
    """
    if not tracks:
        return []

    # Validate all tracks have features
    for track in tracks:
        if track not in features:
            raise ValueError(f"Missing features for track: {track}")

    if len(tracks) == 1:
        return tracks

    def timbre_vector(f: IntensityFeatures) -> np.ndarray:
        """Extract 4D timbre feature vector."""
        return np.array(
            [
                f.spectral_flatness,
                f.mfcc_variance,
                f.zero_crossing_rate,
                f.spectral_rolloff_85,
            ]
        )

    # Start with the track that has the most unique timbre (highest variance)
    remaining = set(tracks)
    current = max(remaining, key=lambda t: features[t].mfcc_variance)
    result = [current]
    remaining.remove(current)

    while remaining:
        current_vec = timbre_vector(features[current])

        # Find best next track by timbral similarity (minimum Euclidean distance)
        best_track: Path | None = None
        best_distance = float("inf")

        for candidate in remaining:
            candidate_vec = timbre_vector(features[candidate])
            distance = np.linalg.norm(current_vec - candidate_vec)

            if distance < best_distance:
                best_track = candidate
                best_distance = distance

        if best_track is None:
            # Fallback: shouldn't happen but handle gracefully
            best_track = remaining.pop()
        else:
            remaining.remove(best_track)

        result.append(best_track)
        current = best_track

    logger.info(
        "Sequenced %d tracks using timbral similarity",
        len(result),
    )

    return result
