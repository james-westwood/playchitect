"""
Sequencing engine for ordering tracks within clusters.

Creates cohesive DJ set narratives using intensity ramps and smart
opener/closer placement.
"""

import logging
from pathlib import Path

from playchitect.core.clustering import ClusterResult
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata
from playchitect.core.track_selector import TrackSelector

logger = logging.getLogger(__name__)


class Sequencer:
    """Orders tracks within a cluster to form a narrative sequence."""

    def __init__(self):
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
