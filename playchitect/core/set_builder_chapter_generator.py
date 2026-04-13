"""Chapter generation for Set Builder.

A chapter is a named segment of a DJ set with a defined energy role
and target duration. The generator creates chapters based on energy
blocks with proportionally distributed target durations.

This is independent from the Playlists view clustering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from playchitect.core.clustering import ClusterResult, PlaylistClusterer
from playchitect.core.energy_blocks import EnergyBlock, suggest_blocks
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata

logger = logging.getLogger(__name__)

# Default target set length in minutes
_DEFAULT_TARGET_DURATION_MIN = 90.0

# Scale factor: actual set duration as ratio of target
_MIN_DURATION_RATIO = 0.89  # 80/90 = ~89%
_MAX_DURATION_RATIO = 1.11  # 100/90 = ~111%

# Chapter names mapped to energy block IDs
_CHAPTER_NAMES: dict[str, str] = {
    "warm-up": "Intro",
    "build": "Build",
    "peak": "Peak",
    "sustain": "Sustain",
    "wind-down": "Outro",
}

# Default chapter names when block not in standard map
_DEFAULT_CHAPTER_NAMES: dict[str, str] = {
    "warm-up": "Intro",
    "build": "Build",
    "peak": "Peak",
    "sustain": "Sustain",
    "wind-down": "Outro",
}

# Average track duration in seconds for duration estimation
_AVERAGE_TRACK_DURATION_SEC = 360.0  # 6 minutes


@dataclass
class ChapterTrack:
    """A track assigned to a chapter."""

    path: Path
    metadata: TrackMetadata
    features: IntensityFeatures
    distance_to_centroid: float = 0.0

    @property
    def duration(self) -> float:
        """Get track duration in seconds."""
        return self.metadata.duration or 0.0


@dataclass
class Chapter:
    """A chapter in the Set Builder.

    Attributes:
        id: Unique identifier
        name: Display name (editable)
        energy_block_id: Reference to the energy block this chapter uses
        target_duration_min: Target duration in minutes
        actual_duration_min: Actual duration in minutes
        tracks: Tracks assigned to this chapter
    """

    id: str
    name: str
    energy_block_id: str
    target_duration_min: float
    actual_duration_min: float = 0.0
    tracks: list[ChapterTrack] = field(default_factory=list)

    @property
    def track_count(self) -> int:
        """Get number of tracks in the chapter."""
        return len(self.tracks)

    @property
    def is_filled(self) -> bool:
        """Check if chapter has reached target duration."""
        return self.actual_duration_min >= self.target_duration_min

    def total_duration_min(self) -> float:
        """Calculate total duration in minutes."""
        return sum(t.duration for t in self.tracks) / 60.0


def _get_chapter_name(block: EnergyBlock) -> str:
    """Get the display name for a chapter based on its energy block.

    Args:
        block: EnergyBlock to get name from

    Returns:
        Display name for the chapter
    """
    block_id = block.id.lower()

    if block_id in _DEFAULT_CHAPTER_NAMES:
        return _DEFAULT_CHAPTER_NAMES[block_id]

    return _DEFAULT_CHAPTER_NAMES.get(block_id, block.name)


class ChapterGenerator:
    """Generates chapters for Set Builder from track library.

    This generator:
    1. Runs clustering on the library using energy as primary axis
    2. Creates energy blocks via suggest_blocks()
    3. Creates chapters with auto-generated names and target durations
    4. Fills chapters with tracks ranked by centroid proximity
    """

    def __init__(
        self,
        min_clusters: int = 2,
        max_clusters: int = 10,
        random_state: int = 42,
    ):
        """Initialize chapter generator.

        Args:
            min_clusters: Minimum number of clusters to consider
            max_clusters: Maximum number of clusters to consider
            random_state: Random seed for reproducibility
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.random_state = random_state
        self._clusterer = PlaylistClusterer(
            target_duration_per_playlist=30.0,  # Default 30 min per "playlist"
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            random_state=random_state,
        )

    def generate_chapters(
        self,
        metadata_dict: dict[Path, TrackMetadata],
        intensity_dict: dict[Path, IntensityFeatures],
        target_duration_min: float = _DEFAULT_TARGET_DURATION_MIN,
    ) -> list[Chapter]:
        """Generate chapters for Set Builder from track library.

        This is separate from Playlists view clustering - it's an independent
        workflow using energy as the primary clustering axis.

        Args:
            metadata_dict: Mapping of file path → TrackMetadata
            intensity_dict: Mapping of file path → IntensityFeatures
            target_duration_min: Target total set duration in minutes

        Returns:
            List of Chapter objects with assigned tracks

        Raises:
            ValueError: If metadata_dict or intensity_dict is empty
        """
        if not metadata_dict:
            raise ValueError("Cannot generate chapters from empty metadata_dict")
        if not intensity_dict:
            raise ValueError("Cannot generate chapters from empty intensity_dict")

        # Filter to tracks with both metadata and intensity
        valid_paths = self._get_valid_paths(metadata_dict, intensity_dict)
        if len(valid_paths) < self.min_clusters:
            logger.warning(
                "Only %d valid tracks, creating single chapter",
                len(valid_paths),
            )
            return self._create_single_chapter(
                valid_paths, metadata_dict, intensity_dict, target_duration_min
            )

        logger.info(
            "Generating chapters for %d tracks with target duration %.0f min",
            len(valid_paths),
            target_duration_min,
        )

        # Cluster using energy as primary axis (multi-dimensional)
        valid_meta = {p: metadata_dict[p] for p in valid_paths}
        valid_intensity = {p: intensity_dict[p] for p in valid_paths}

        clusters = self._clusterer.cluster_by_features(
            valid_meta,
            valid_intensity,
        )

        if not clusters:
            logger.warning("No clusters generated, creating single chapter")
            return self._create_single_chapter(
                valid_paths, metadata_dict, intensity_dict, target_duration_min
            )

        # Create energy blocks from clusters
        energy_blocks = suggest_blocks(clusters, valid_intensity)

        if not energy_blocks:
            logger.warning("No energy blocks generated, creating single chapter")
            return self._create_single_chapter(
                valid_paths, metadata_dict, intensity_dict, target_duration_min
            )

        # Distribute target duration proportionally across blocks
        chapter_duration_targets = self._distribute_durations(energy_blocks, target_duration_min)

        # Create chapters and fill with tracks
        chapters = self._create_chapters(
            energy_blocks,
            chapter_duration_targets,
            clusters,
            metadata_dict,
            intensity_dict,
        )

        logger.info(
            "Generated %d chapters with total duration %.0f min",
            len(chapters),
            sum(c.total_duration_min() for c in chapters),
        )

        return chapters

    def _get_valid_paths(
        self,
        metadata_dict: dict[Path, TrackMetadata],
        intensity_dict: dict[Path, IntensityFeatures],
    ) -> list[Path]:
        """Get paths that have both metadata and intensity features."""
        common = set(metadata_dict.keys()) & set(intensity_dict.keys())
        return sorted(p for p in common if metadata_dict[p].bpm is not None)

    def _create_single_chapter(
        self,
        paths: list[Path],
        metadata_dict: dict[Path, TrackMetadata],
        intensity_dict: dict[Path, IntensityFeatures],
        target_duration_min: float,
    ) -> list[Chapter]:
        """Create a single chapter when there are insufficient tracks."""
        chapter_tracks = []
        for path in paths:
            meta = metadata_dict.get(path)
            feat = intensity_dict.get(path)
            if meta and feat:
                chapter_tracks.append(
                    ChapterTrack(
                        path=path,
                        metadata=meta,
                        features=feat,
                    )
                )

        total_duration = sum(t.duration for t in chapter_tracks) / 60.0
        min_duration = target_duration_min * _MIN_DURATION_RATIO

        return [
            Chapter(
                id="chapter-0",
                name="Intro",
                energy_block_id="warm-up",
                target_duration_min=min(min_duration, total_duration),
                actual_duration_min=total_duration,
                tracks=chapter_tracks,
            )
        ]

    def _distribute_durations(
        self,
        blocks: list[EnergyBlock],
        target_duration_min: float,
    ) -> dict[str, float]:
        """Distribute target duration across energy blocks proportionally.

        Args:
            blocks: List of energy blocks
            target_duration_min: Total target duration

        Returns:
            Dict mapping block ID to target duration in minutes
        """
        if not blocks:
            return {}

        # Calculate total track count across all blocks
        total_tracks = sum(len(b.cluster_ids) for b in blocks)

        if total_tracks == 0:
            # Fallback: equal distribution
            per_block = target_duration_min / len(blocks)
            return {b.id: per_block for b in blocks}

        # Distribute proportionally by cluster count
        targets: dict[str, float] = {}
        for block in blocks:
            block_weight = len(block.cluster_ids) / total_tracks
            targets[block.id] = target_duration_min * block_weight

        return targets

    def _create_chapters(
        self,
        blocks: list[EnergyBlock],
        duration_targets: dict[str, float],
        clusters: list[ClusterResult],
        metadata_dict: dict[Path, TrackMetadata],
        intensity_dict: dict[Path, IntensityFeatures],
    ) -> list[Chapter]:
        """Create chapters and fill them with tracks.

        Args:
            blocks: Energy blocks
            duration_targets: Target duration per block
            clusters: Cluster results
            metadata_dict: Track metadata
            intensity_dict: Track intensity features

        Returns:
            List of Chapter objects
        """
        # Build cluster ID to cluster mapping
        cluster_map: dict[int | str, ClusterResult] = {c.cluster_id: c for c in clusters}

        # Track which paths are already assigned
        assigned_paths: set[Path] = set()

        chapters: list[Chapter] = []

        for block in blocks:
            # Get target duration for this block
            target_duration = duration_targets.get(block.id, 30.0)

            # Get tracks from this block's clusters
            block_tracks = self._get_block_tracks(
                block, cluster_map, metadata_dict, intensity_dict, assigned_paths
            )

            # Rank by centroid proximity and fill
            filled_tracks = self._rank_and_fill(
                block_tracks,
                target_duration,
                metadata_dict,
                intensity_dict,
            )

            # Update assigned paths
            for ct in filled_tracks:
                assigned_paths.add(ct.path)

            # Calculate actual duration
            actual_duration = sum(t.duration for t in filled_tracks) / 60.0

            # Create chapter
            chapter = Chapter(
                id=f"chapter-{block.id}",
                name=_get_chapter_name(block),
                energy_block_id=block.id,
                target_duration_min=target_duration,
                actual_duration_min=actual_duration,
                tracks=filled_tracks,
            )
            chapters.append(chapter)

        return chapters

    def _get_block_tracks(
        self,
        block: EnergyBlock,
        cluster_map: dict[int | str, ClusterResult],
        metadata_dict: dict[Path, TrackMetadata],
        intensity_dict: dict[Path, IntensityFeatures],
        already_assigned: set[Path],
    ) -> list[ChapterTrack]:
        """Get tracks belonging to an energy block.

        Args:
            block: Energy block
            cluster_map: Mapping of cluster IDs to results
            metadata_dict: Track metadata
            intensity_dict: Track intensity features
            already_assigned: Paths already assigned to another chapter

        Returns:
            List of ChapterTrack objects
        """
        tracks: list[ChapterTrack] = []

        for cluster_id in block.cluster_ids:
            cluster = cluster_map.get(cluster_id)
            if cluster is None:
                continue

            for path in cluster.tracks:
                if path in already_assigned:
                    continue

                meta = metadata_dict.get(path)
                feat = intensity_dict.get(path)
                if meta and feat:
                    tracks.append(
                        ChapterTrack(
                            path=path,
                            metadata=meta,
                            features=feat,
                        )
                    )

        return tracks

    def _rank_and_fill(
        self,
        tracks: list[ChapterTrack],
        target_duration_min: float,
        metadata_dict: dict[Path, TrackMetadata],
        intensity_dict: dict[Path, IntensityFeatures],
    ) -> list[ChapterTrack]:
        """Rank tracks by centroid proximity and fill to target duration.

        Args:
            tracks: Available tracks
            target_duration_min: Target duration in minutes
            metadata_dict: Track metadata (for reference)
            intensity_dict: Track intensity features (for centroid calculation)

        Returns:
            List of ChapterTrack objects ranked by proximity to centroid
        """
        if not tracks:
            return []

        # Calculate centroid of all tracks (in 8D feature space)
        feature_vectors = np.array([t.features.to_feature_vector() for t in tracks])
        centroid = feature_vectors.mean(axis=0)

        # Calculate distance to centroid for each track
        for track in tracks:
            vec = track.features.to_feature_vector()
            distance = float(np.linalg.norm(vec - centroid))
            track.distance_to_centroid = distance

        # Sort by distance (closest first)
        ranked = sorted(tracks, key=lambda t: t.distance_to_centroid)

        # Fill until target duration reached
        filled: list[ChapterTrack] = []
        current_duration = 0.0

        target_sec = target_duration_min * 60.0

        for track in ranked:
            filled.append(track)
            current_duration += track.duration

            if current_duration >= target_sec:
                break

        return filled


def generate_chapters(
    metadata_dict: dict[Path, TrackMetadata],
    intensity_dict: dict[Path, IntensityFeatures],
    target_duration_min: float = _DEFAULT_TARGET_DURATION_MIN,
) -> list[Chapter]:
    """Generate chapters for Set Builder.

    Convenience function that creates a ChapterGenerator and generates
    chapters from the provided data.

    Args:
        metadata_dict: Mapping of file path → TrackMetadata
        intensity_dict: Mapping of file path → IntensityFeatures
        target_duration_min: Target total set duration in minutes

    Returns:
        List of Chapter objects with assigned tracks
    """
    generator = ChapterGenerator()
    return generator.generate_chapters(
        metadata_dict,
        intensity_dict,
        target_duration_min,
    )
