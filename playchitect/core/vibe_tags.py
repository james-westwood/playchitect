"""Vibe tags for manual track annotation.

Provides a persistent store for user-defined vibe tags that can be attached
to individual tracks for organization and filtering.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_tag(tag: str) -> str:
    """Normalize a tag to lowercase stripped string.

    Args:
        tag: Raw tag string

    Returns:
        Normalized lowercase stripped tag
    """
    return tag.lower().strip()


class VibeTagStore:
    """Persistent store for user-defined vibe tags.

    Maps track file paths to lists of tags. Tags are normalized to lowercase.
    Data is persisted as JSON to the store path.

    Example:
        store = VibeTagStore()
        store.add_tag(Path("/music/track.mp3"), "Techno")
        store.add_tag(Path("/music/track.mp3"), "Peak Time")
        tags = store.get_tags(Path("/music/track.mp3"))
        # Returns: ["techno", "peak time"]
    """

    def __init__(
        self,
        store_path: Path = Path.home() / ".local" / "share" / "playchitect" / "vibe_tags.json",
    ) -> None:
        """Initialize the tag store.

        Args:
            store_path: Path to the JSON persistence file
        """
        self._store_path = Path(store_path)
        self._data: dict[str, list[str]] = {}
        self.load()

    def load(self) -> None:
        """Load tag data from disk.

        Creates the store directory if needed. Silently handles
        missing or corrupt files by starting with empty data.
        """
        self._data = {}

        if not self._store_path.exists():
            logger.debug("Vibe tag store not found, starting fresh")
            return

        try:
            with open(self._store_path, encoding="utf-8") as f:
                raw_data = json.load(f)

            # Validate and normalize loaded data
            if isinstance(raw_data, dict):
                for path_str, tags in raw_data.items():
                    if isinstance(tags, list):
                        normalized = [_normalize_tag(t) for t in tags if isinstance(t, str)]
                        if normalized:
                            self._data[path_str] = sorted(normalized)

            logger.debug("Loaded %d tracks from vibe tag store", len(self._data))

        except json.JSONDecodeError as e:
            logger.warning("Corrupt vibe tag store, starting fresh: %s", e)
        except Exception as e:
            logger.error("Failed to load vibe tag store: %s", e)

    def save(self) -> None:
        """Save tag data to disk.

        Creates parent directories if needed. Removes empty tag lists
        before saving to keep the file clean.
        """
        try:
            # Remove empty entries
            clean_data = {k: v for k, v in self._data.items() if v}

            # Ensure parent directory exists
            self._store_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self._store_path, "w", encoding="utf-8") as f:
                json.dump(clean_data, f, indent=2, sort_keys=True)

            logger.debug("Saved %d tracks to vibe tag store", len(clean_data))

        except Exception as e:
            logger.error("Failed to save vibe tag store: %s", e)

    def add_tag(self, track_path: Path, tag: str) -> None:
        """Add a tag to a track.

        Tags are normalized to lowercase. Duplicate tags are ignored.
        Automatically persists to disk.

        Args:
            track_path: Path to the audio file
            tag: Tag to add (will be normalized)
        """
        path_str = str(track_path)
        normalized = _normalize_tag(tag)

        if not normalized:
            return

        if path_str not in self._data:
            self._data[path_str] = []

        if normalized not in self._data[path_str]:
            self._data[path_str].append(normalized)
            self._data[path_str].sort()
            self.save()
            logger.debug("Added tag '%s' to %s", normalized, path_str)

    def remove_tag(self, track_path: Path, tag: str) -> None:
        """Remove a tag from a track.

        Tag matching is case-insensitive. Automatically persists to disk.

        Args:
            track_path: Path to the audio file
            tag: Tag to remove (case-insensitive match)
        """
        path_str = str(track_path)
        normalized = _normalize_tag(tag)

        if path_str not in self._data:
            return

        if normalized in self._data[path_str]:
            self._data[path_str].remove(normalized)
            if not self._data[path_str]:
                del self._data[path_str]
            self.save()
            logger.debug("Removed tag '%s' from %s", normalized, path_str)

    def get_tags(self, track_path: Path) -> list[str]:
        """Get all tags for a track.

        Args:
            track_path: Path to the audio file

        Returns:
            List of normalized tags (sorted alphabetically)
        """
        path_str = str(track_path)
        tags = self._data.get(path_str, [])
        return tags.copy()

    def search_by_tag(self, tag: str) -> list[Path]:
        """Find all tracks with a specific tag.

        Search is case-insensitive.

        Args:
            tag: Tag to search for

        Returns:
            List of paths for tracks with matching tag
        """
        normalized = _normalize_tag(tag)
        matching: list[Path] = []

        for path_str, tags in self._data.items():
            if normalized in tags:
                matching.append(Path(path_str))

        return sorted(matching)

    def all_tags(self) -> list[str]:
        """Get all unique tags across all tracks.

        Returns:
            Sorted list of unique tag strings
        """
        unique: set[str] = set()
        for tags in self._data.values():
            unique.update(tags)
        return sorted(unique)

    def clear_tags(self, track_path: Path) -> None:
        """Remove all tags from a track.

        Args:
            track_path: Path to the audio file
        """
        path_str = str(track_path)
        if path_str in self._data:
            del self._data[path_str]
            self.save()
            logger.debug("Cleared all tags from %s", path_str)

    def remove_tag_from_all(self, tag: str) -> None:
        """Remove a tag from all tracks.

        Useful for tag management (e.g., deleting a tag entirely).

        Args:
            tag: Tag to remove from all tracks
        """
        normalized = _normalize_tag(tag)
        modified = False

        for path_str in list(self._data.keys()):
            if normalized in self._data[path_str]:
                self._data[path_str].remove(normalized)
                if not self._data[path_str]:
                    del self._data[path_str]
                modified = True

        if modified:
            self.save()
            logger.debug("Removed tag '%s' from all tracks", normalized)

    def to_dict(self) -> dict[str, list[str]]:
        """Export all data as a dictionary.

        Returns:
            Copy of internal data dict (path_str -> tags)
        """
        return {k: v.copy() for k, v in self._data.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any], store_path: Path | None = None) -> VibeTagStore:
        """Create a store from a dictionary.

        Args:
            data: Dictionary mapping path strings to tag lists
            store_path: Optional custom store path

        Returns:
            New VibeTagStore instance with loaded data
        """
        default_path = Path.home() / ".local" / "share" / "playchitect" / "vibe_tags.json"
        store = cls.__new__(cls)
        store._store_path = store_path if store_path else default_path
        store._data = {}

        for path_str, tags in data.items():
            if isinstance(tags, list):
                normalized = [_normalize_tag(t) for t in tags if isinstance(t, str)]
                if normalized:
                    store._data[path_str] = sorted(normalized)

        return store
