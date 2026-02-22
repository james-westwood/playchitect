"""
Metadata extraction from audio files using mutagen.

Extracts BPM, artist, title, album, duration, and other metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from playchitect.utils.warnings import suppress_librosa_warnings

try:
    from mutagen import File as MutagenFile

    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class CuePoint:
    """A saved cue point or loop."""

    position: float  # In seconds
    label: str = ""
    hotcue: int | None = None  # 0-based index if it's a hotcue


@dataclass
class TrackMetadata:
    """Container for track metadata."""

    filepath: Path
    bpm: float | None = None
    artist: str | None = None
    title: str | None = None
    album: str | None = None
    duration: float | None = None  # Duration in seconds
    year: int | None = None
    genre: str | None = None
    rating: int | None = None  # 0-5 stars
    play_count: int | None = None
    last_played: str | None = None  # ISO format string
    cues: list[CuePoint] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "filepath": str(self.filepath),
            "bpm": self.bpm,
            "artist": self.artist,
            "title": self.title,
            "album": self.album,
            "duration": self.duration,
            "year": self.year,
            "genre": self.genre,
            "rating": self.rating,
            "play_count": self.play_count,
            "last_played": self.last_played,
            "cues": (
                [{"position": c.position, "label": c.label, "hotcue": c.hotcue} for c in self.cues]
                if self.cues
                else None
            ),
        }


class MetadataExtractor:
    """Extracts metadata from audio files."""

    # Common BPM tag names across different formats
    BPM_TAGS = [
        "BPM",
        "bpm",
        "TBPM",  # ID3 BPM tag
        "tempo",
        "TEMPO",
        "tmpo",  # iTunes BPM tag
        "fBPM",  # Some MP3 tag formats
    ]

    def __init__(self, cache_enabled: bool = True):
        """
        Initialize metadata extractor.

        Args:
            cache_enabled: Whether to enable metadata caching (future feature)
        """
        if not MUTAGEN_AVAILABLE:
            logger.warning("Mutagen library not available. Metadata extraction will be limited.")

        self.cache_enabled = cache_enabled
        self._cache: dict[Path, TrackMetadata] = {}

    def extract(self, filepath: Path) -> TrackMetadata:
        """
        Extract metadata from audio file.

        Args:
            filepath: Path to audio file

        Returns:
            TrackMetadata object with extracted information
        """
        # Check cache first
        if self.cache_enabled and filepath in self._cache:
            logger.debug(f"Using cached metadata for: {filepath}")
            return self._cache[filepath]

        metadata = TrackMetadata(filepath=filepath)

        if not MUTAGEN_AVAILABLE:
            logger.warning(f"Cannot extract metadata (mutagen not available): {filepath}")
            # Still try to calculate BPM if mutagen is missing
            metadata.bpm = self.calculate_bpm(filepath)
            return metadata

        try:
            audio = MutagenFile(filepath)

            if audio is None:
                logger.warning("Failed to read audio file: %s", filepath)
                # Fallback to calculation
                metadata.bpm = self.calculate_bpm(filepath)
            else:
                # Extract BPM
                metadata.bpm = self._extract_bpm(audio)

                # Extract basic metadata
                metadata.artist = self._extract_text_tag(audio, ["artist", "TPE1", "\xa9ART"])
                metadata.title = self._extract_text_tag(audio, ["title", "TIT2", "\xa9nam"])
                metadata.album = self._extract_text_tag(audio, ["album", "TALB", "\xa9alb"])
                metadata.genre = self._extract_text_tag(audio, ["genre", "TCON", "\xa9gen"])

                # Check if extracted BPM is suspicious or missing
                if metadata.bpm is None or self.is_bpm_suspicious(metadata.bpm, metadata.genre):
                    logger.info(
                        "BPM missing or suspicious (%s) for %s, calculating...",
                        metadata.bpm,
                        filepath.name,
                    )
                    calculated_bpm = self.calculate_bpm(filepath)
                    if calculated_bpm:
                        metadata.bpm = calculated_bpm

                # Extract year
                year_str = self._extract_text_tag(audio, ["date", "year", "TDRC", "\xa9day"])
                if year_str:
                    metadata.year = self._parse_year(year_str)

                # Extract duration
                if hasattr(audio, "info") and hasattr(audio.info, "length"):
                    metadata.duration = float(audio.info.length)

        except Exception as e:
            logger.error("Error extracting metadata from %s: %s", filepath, e)
            # Last ditch effort
            if metadata.bpm is None:
                try:
                    metadata.bpm = self.calculate_bpm(filepath)
                except Exception:
                    pass

        # Cache the result
        if self.cache_enabled:
            self._cache[filepath] = metadata

        return metadata

    def is_bpm_suspicious(self, bpm: float | None, genre: str | None) -> bool:
        """
        Determine if a BPM value is suspicious and needs recalculation.

        Args:
            bpm: The BPM value to check
            genre: Optional genre hint

        Returns:
            True if BPM is suspicious
        """
        if bpm is None:
            return False

        # 1. Non-whole numbers are suspicious for electronic music
        if not np.isclose(bpm, round(bpm), atol=1e-3):
            return True

        # 2. Genre-based mismatch
        if genre:
            genre_lower = genre.lower()
            if "techno" in genre_lower and bpm < 110:
                return True
            if "house" in genre_lower and bpm < 100:
                return True

            dnb_genres = ("dnb", "drum and bass", "drum & bass", "d&b")
            if any(g in genre_lower for g in dnb_genres) and bpm < 150:
                return True

        return False

    def calculate_bpm(self, filepath: Path) -> float | None:
        """
        Calculate BPM using librosa.

        Args:
            filepath: Path to audio file

        Returns:
            Calculated BPM as float, or None if calculation fails
        """
        if not filepath.exists():
            return None

        with suppress_librosa_warnings():
            try:
                import librosa  # noqa: PLC0415

                # Only load first 60s for speed, enough for tempo estimation
                # Use duration=60 but also check file size/duration if possible
                y, sr = librosa.load(filepath, sr=22050, duration=60)

                # Defensive check for empty or near-empty audio
                if y is None or len(y) < 100:
                    return None

                # Onset strength requires some signal
                if np.max(np.abs(y)) < 1e-4:
                    return None

                tempo_result = librosa.beat.beat_track(y=y, sr=sr)

                # tempo_result is usually (tempo, beats)
                tempo = np.atleast_1d(tempo_result[0])[0]
                bpm = float(tempo)

                # Round to nearest whole number as per user preference
                bpm = float(round(bpm))
                return bpm if bpm > 0 else None
            except Exception as e:
                logger.error("Error calculating BPM for %s: %s", filepath, e)
                return None

    def _extract_bpm(self, audio: Any) -> float | None:
        """
        Extract BPM from audio file tags.

        Args:
            audio: Mutagen audio file object

        Returns:
            BPM as float, or None if not found
        """
        for tag in self.BPM_TAGS:
            if tag in audio:
                value = audio[tag]

                # Handle list values (common in ID3 tags)
                if isinstance(value, list) and len(value) > 0:
                    value = value[0]

                # Try to convert to float
                try:
                    bpm = float(str(value))
                    if 0 < bpm < 500:  # Sanity check
                        return bpm
                except (ValueError, TypeError):
                    continue

        return None

    def _extract_text_tag(self, audio: Any, tag_names: list[str]) -> str | None:
        """
        Extract text metadata from various tag formats.

        Args:
            audio: Mutagen audio file object
            tag_names: List of possible tag names to check

        Returns:
            Tag value as string, or None if not found
        """
        for tag in tag_names:
            if tag in audio:
                value = audio[tag]

                # Handle list values
                if isinstance(value, list) and len(value) > 0:
                    value = value[0]

                # Convert to string
                try:
                    text = str(value).strip()
                    if text:
                        return text
                except Exception:
                    continue

        return None

    def _parse_year(self, year_str: str) -> int | None:
        """
        Parse year from various date string formats.

        Args:
            year_str: Year string (e.g., "2023", "2023-05-10")

        Returns:
            Year as integer, or None if parsing fails
        """
        try:
            # Try to extract first 4 digits
            year_part = year_str[:4]
            year = int(year_part)
            if 1900 <= year <= 2100:  # Sanity check
                return year
        except (ValueError, IndexError):
            pass

        return None

    def extract_batch(self, filepaths: list[Path]) -> dict[Path, TrackMetadata]:
        """
        Extract metadata from multiple files.

        Args:
            filepaths: List of file paths

        Returns:
            Dictionary mapping file paths to metadata
        """
        results: dict[Path, TrackMetadata] = {}

        for i, filepath in enumerate(filepaths, 1):
            if i % 50 == 0:
                logger.info(f"Processed {i}/{len(filepaths)} files...")

            results[filepath] = self.extract(filepath)

        return results

    def clear_cache(self) -> None:
        """Clear the metadata cache."""
        self._cache.clear()
        logger.info("Metadata cache cleared")

    def recalculate(self, filepath: Path) -> TrackMetadata:
        """
        Force a recalculation of metadata for a file, bypassing the cache.

        Args:
            filepath: Path to audio file

        Returns:
            TrackMetadata object with fresh information
        """
        # Remove from cache if present to ensure fresh extraction
        if filepath in self._cache:
            del self._cache[filepath]

        # Extract will now run from scratch and cache the result
        return self.extract(filepath)
