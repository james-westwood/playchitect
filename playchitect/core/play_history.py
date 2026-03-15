"""Play history tracking for freshness-aware sequencing.

Tracks which tracks have been played and when, to enable
"Fresh Tracks" sequencing that prioritizes less-recently-played songs.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Days until full freshness restoration (score approaches 1.0)
_FRESHNESS_DAYS_THRESHOLD: int = 30
# Maximum times_used value for score calculation (log scale normalization)
_FRESHNESS_MAX_USAGE_LOG: int = 31


@dataclass
class TrackHistory:
    """History entry for a single track."""

    times_used: int = 0
    last_used: str = ""  # ISO8601 date string (YYYY-MM-DD)


class PlayHistory:
    """Manages play history for freshness-aware track sequencing.

    Tracks how many times each track has been used and when it was
    last played. Provides a freshness score for each track that can
    be used to prioritize less-recently-played songs.
    """

    def __init__(
        self,
        history_path: Path = Path.home() / ".cache" / "playchitect" / "play_history.json",
    ) -> None:
        """Initialize play history manager.

        Args:
            history_path: Path to the JSON history file.
        """
        self.history_path = history_path
        self._history: dict[str, TrackHistory] = {}
        self.load()

    def load(self) -> None:
        """Load history from JSON file.

        Creates empty history if file doesn't exist or is invalid.
        """
        if not self.history_path.exists():
            logger.debug("History file not found, starting fresh: %s", self.history_path)
            self._history = {}
            return

        try:
            with open(self.history_path) as f:
                data: dict[str, Any] = json.load(f)

            self._history = {}
            for path_str, entry in data.items():
                if isinstance(entry, dict):
                    self._history[path_str] = TrackHistory(
                        times_used=entry.get("times_used", 0),
                        last_used=entry.get("last_used", ""),
                    )
                else:
                    # Handle old format or invalid entries
                    self._history[path_str] = TrackHistory()

            logger.debug("Loaded history for %d tracks", len(self._history))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load history, starting fresh: %s", e)
            self._history = {}

    def save(self) -> None:
        """Save history to JSON file."""
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to serializable dict
            data = {
                path_str: {"times_used": entry.times_used, "last_used": entry.last_used}
                for path_str, entry in self._history.items()
            }

            with open(self.history_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug("Saved history for %d tracks", len(self._history))
        except OSError as e:
            logger.warning("Failed to save history: %s", e)

    def record(self, track_path: Path) -> None:
        """Record that a track has been played.

        Increments times_used counter and updates last_used to today.

        Args:
            track_path: Path to the track that was played.
        """
        path_str = str(track_path)
        today = datetime.now(UTC).strftime("%Y-%m-%d")

        if path_str not in self._history:
            self._history[path_str] = TrackHistory()

        entry = self._history[path_str]
        entry.times_used += 1
        entry.last_used = today

        logger.debug("Recorded play for %s (times_used=%d)", track_path.name, entry.times_used)

    def get_freshness_score(self, track_path: Path) -> float:
        """Calculate freshness score for a track.

        Freshness score ranges from 0.0 to 1.0:
        - 1.0 if track has never been used
        - Decreases as times_used increases (logarithmic scale)
        - Approaches 1.0 again as days since last use increases

        Formula:
            If never used: 1.0
            Otherwise: max(0.0, 1.0 - log(times_used + 1) / log(31)) * days_decay
            where days_decay = min(1.0, days_since_use / 30.0)

        Args:
            track_path: Path to the track.

        Returns:
            Freshness score between 0.0 and 1.0.
        """
        path_str = str(track_path)

        # Track has never been played - maximum freshness
        if path_str not in self._history:
            return 1.0

        entry = self._history[path_str]

        # No last_used date - treat as never played
        if not entry.last_used:
            return 1.0

        # Calculate usage penalty (logarithmic)
        # log(times_used + 1) / log(31) gives a value between 0 and 1
        # when times_used is between 0 and 30
        times_factor = math.log(entry.times_used + 1) / math.log(_FRESHNESS_MAX_USAGE_LOG)
        usage_score = max(0.0, 1.0 - times_factor)

        # Calculate days decay factor
        # More days since use = higher freshness
        try:
            last_used_date = datetime.fromisoformat(entry.last_used)
            if last_used_date.tzinfo is None:
                last_used_date = last_used_date.replace(tzinfo=UTC)
            days_since_use = (datetime.now(UTC) - last_used_date).days
        except ValueError:
            # Invalid date format, treat as old
            days_since_use = _FRESHNESS_DAYS_THRESHOLD

        days_decay = min(1.0, days_since_use / _FRESHNESS_DAYS_THRESHOLD)

        # Combined score
        score = usage_score * days_decay

        logger.debug(
            "Freshness score for %s: %.3f (times=%d, days=%d)",
            track_path.name,
            score,
            entry.times_used,
            days_since_use,
        )

        return score

    def get_history(self, track_path: Path) -> TrackHistory | None:
        """Get history entry for a specific track.

        Args:
            track_path: Path to the track.

        Returns:
            TrackHistory entry or None if track not in history.
        """
        path_str = str(track_path)
        return self._history.get(path_str)

    def clear(self) -> None:
        """Clear all history."""
        self._history = {}
        logger.info("Cleared play history")
