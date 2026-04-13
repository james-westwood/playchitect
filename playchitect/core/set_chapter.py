"""Set chapter data model for Set Builder.

Chapters represent named sections of a DJ set with their own track lists,
enabling the DJ to organize the narrative arc of their set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playchitect.core.intensity_analyzer import IntensityFeatures
    from playchitect.core.metadata_extractor import TrackMetadata


@dataclass
class SetChapter:
    """Represents a chapter in a DJ set.

    Attributes:
        id: Unique identifier for the chapter
        name: Human-readable chapter name
        tracks: Ordered list of track paths in this chapter
    """

    id: str
    name: str
    tracks: list[Path] = field(default_factory=list)

    def add_track(self, path: Path) -> None:
        """Add a track to the chapter."""
        if path not in self.tracks:
            self.tracks.append(path)

    def remove_track(self, path: Path) -> None:
        """Remove a track from the chapter."""
        if path in self.tracks:
            self.tracks.remove(path)

    def reorder_track(self, from_index: int, to_index: int) -> bool:
        """Reorder a track within the chapter.

        Args:
            from_index: Current index of the track
            to_index: Target index for the track

        Returns:
            True if reorder was successful
        """
        if 0 <= from_index < len(self.tracks) and 0 <= to_index < len(self.tracks):
            track = self.tracks.pop(from_index)
            self.tracks.insert(to_index, track)
            return True
        return False

    def get_track_count(self) -> int:
        """Get the number of tracks in this chapter."""
        return len(self.tracks)

    def calculate_total_duration(
        self,
        metadata_map: dict[Path, TrackMetadata],
    ) -> float:
        """Calculate total duration of all tracks in seconds.

        Args:
            metadata_map: Mapping of paths to track metadata

        Returns:
            Total duration in seconds
        """
        total = 0.0
        for path in self.tracks:
            if path in metadata_map:
                meta = metadata_map[path]
                if meta.duration is not None:
                    total += meta.duration
        return total

    def calculate_energy_range(
        self,
        features_map: dict[Path, IntensityFeatures],
    ) -> tuple[float, float]:
        """Calculate min/max energy range across all tracks.

        Args:
            features_map: Mapping of paths to intensity features

        Returns:
            Tuple of (min_energy, max_energy)
        """
        energies = []
        for path in self.tracks:
            if path in features_map:
                energies.append(features_map[path].rms_energy)

        if not energies:
            return (0.0, 1.0)

        return (min(energies), max(energies))

    def get_summary(
        self,
        metadata_map: dict[Path, TrackMetadata],
        features_map: dict[Path, IntensityFeatures],
    ) -> dict[str, float | str | int]:
        """Get chapter summary information.

        Args:
            metadata_map: Mapping of paths to track metadata
            features_map: Mapping of paths to intensity features

        Returns:
            Dictionary with name, track_count, total_duration, energy_range
        """
        duration = self.calculate_total_duration(metadata_map)
        energy_min, energy_max = self.calculate_energy_range(features_map)

        hours = int(duration) // 3600
        minutes = (int(duration) % 3600) // 60
        secs = int(duration) % 60
        if hours > 0:
            duration_str = f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            duration_str = f"{minutes}:{secs:02d}"

        energy_str = f"{energy_min:.0%}-{energy_max:.0%}"

        return {
            "name": self.name,
            "track_count": len(self.tracks),
            "total_duration": duration,
            "duration_str": duration_str,
            "energy_min": energy_min,
            "energy_max": energy_max,
            "energy_range": energy_str,
        }


class ChapterManager:
    """Manages a collection of chapters for a DJ set."""

    def __init__(self) -> None:
        self._chapters: list[SetChapter] = []
        self._chapter_counter = 0

    def create_chapter(self, name: str | None = None) -> SetChapter:
        """Create a new chapter.

        Args:
            name: Optional chapter name (auto-generated if not provided)

        Returns:
            New SetChapter instance
        """
        self._chapter_counter += 1
        if name is None:
            name = f"Chapter {self._chapter_counter}"

        chapter = SetChapter(id=f"chapter-{self._chapter_counter}", name=name)
        self._chapters.append(chapter)
        return chapter

    def remove_chapter(self, chapter_id: str) -> bool:
        """Remove a chapter by ID.

        Args:
            chapter_id: ID of the chapter to remove

        Returns:
            True if chapter was found and removed
        """
        for i, chapter in enumerate(self._chapters):
            if chapter.id == chapter_id:
                self._chapters.pop(i)
                return True
        return False

    def get_chapter(self, chapter_id: str) -> SetChapter | None:
        """Get a chapter by ID.

        Args:
            chapter_id: ID of the chapter to retrieve

        Returns:
            SetChapter if found, None otherwise
        """
        for chapter in self._chapters:
            if chapter.id == chapter_id:
                return chapter
        return None

    def reorder_chapter(self, from_index: int, to_index: int) -> bool:
        """Reorder chapters in the list.

        Args:
            from_index: Current index of the chapter
            to_index: Target index for the chapter

        Returns:
            True if reorder was successful
        """
        if (
            0 <= from_index < len(self._chapters)
            and 0 <= to_index < len(self._chapters)
            and from_index != to_index
        ):
            chapter = self._chapters.pop(from_index)
            self._chapters.insert(to_index, chapter)
            return True
        return False

    def get_chapters(self) -> list[SetChapter]:
        """Get all chapters in order.

        Returns:
            List of SetChapter objects
        """
        return self._chapters.copy()

    def get_chapter_count(self) -> int:
        """Get the number of chapters."""
        return len(self._chapters)

    def clear(self) -> None:
        """Clear all chapters."""
        self._chapters.clear()
        self._chapter_counter = 0

    def update_chapter_name(self, chapter_id: str, name: str) -> bool:
        """Update the name of a chapter.

        Args:
            chapter_id: ID of the chapter
            name: New name

        Returns:
            True if chapter was found and updated
        """
        chapter = self.get_chapter(chapter_id)
        if chapter:
            chapter.name = name
            return True
        return False


def create_default_chapters() -> list[SetChapter]:
    """Create a default set of chapters for a new DJ set.

    Returns:
        List of 4 default chapters
    """
    return [
        SetChapter(id="intro", name="Intro"),
        SetChapter(id="build", name="Build"),
        SetChapter(id="peak", name="Peak"),
        SetChapter(id="outro", name="Outro"),
    ]
