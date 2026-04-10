"""Rekordbox XML import functionality.

Imports tracks from Pioneer Rekordbox XML library exports.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Mapping from Rekordbox key notation to Camelot notation
# Rekordbox uses: 1-12 A/B (major/minor) and also traditional notation like Cm, G#m, etc.
_REKORDBOX_TO_CAMELOT: dict[str, str] = {
    # Numeric notation (Rekordbox standard)
    "1A": "1A",  # A-flat minor
    "2A": "2A",  # E-flat minor
    "3A": "3A",  # B-flat minor
    "4A": "4A",  # F minor
    "5A": "5A",  # C minor
    "6A": "6A",  # G minor
    "7A": "7A",  # D minor
    "8A": "8A",  # A minor
    "9A": "9A",  # E minor
    "10A": "10A",  # B minor
    "11A": "11A",  # F-sharp minor
    "12A": "12A",  # D-flat minor
    "1B": "1B",  # B major
    "2B": "2B",  # F-sharp major
    "3B": "3B",  # D-flat major
    "4B": "4B",  # A-flat major
    "5B": "5B",  # E-flat major
    "6B": "6B",  # B-flat major
    "7B": "7B",  # F major
    "8B": "8B",  # C major
    "9B": "9B",  # G major
    "10B": "10B",  # D major
    "11B": "11B",  # A major
    "12B": "12B",  # E major
    # Traditional notation (minor keys)
    "Abm": "1A",
    "G#m": "1A",
    "Ebm": "2A",
    "D#m": "2A",
    "Bbm": "3A",
    "A#m": "3A",
    "Fm": "4A",
    "Cm": "5A",
    "Gm": "6A",
    "Dm": "7A",
    "Am": "8A",
    "Em": "9A",
    "Bm": "10A",
    "F#m": "11A",
    "Gbm": "11A",
    "Dbm": "12A",
    "C#m": "12A",
    # Traditional notation (major keys)
    "B": "1B",
    "F#": "2B",
    "Gb": "2B",
    "Db": "3B",
    "C#": "3B",
    "Ab": "4B",
    "G#": "4B",
    "Eb": "5B",
    "D#": "5B",
    "Bb": "6B",
    "A#": "6B",
    "F": "7B",
    "C": "8B",
    "G": "9B",
    "D": "10B",
    "A": "11B",
    "E": "12B",
}


def rekordbox_key_to_camelot(key_str: str) -> str:
    """Convert Rekordbox key notation to Camelot notation.

    Args:
        key_str: Rekordbox key string (e.g., '6A', '4B', 'Cm', 'G#m')

    Returns:
        Camelot key string (e.g., '6A', '4B'), or empty string if unknown
    """
    if not key_str:
        return ""

    # Strip whitespace first
    key_str = key_str.strip()

    # Check if ends with lowercase 'm' (explicit minor)
    if key_str.endswith("m") and len(key_str) > 1:
        # Explicit minor key like "Cm", "G#m"
        normalized = key_str[:-1].upper() + "m"
    elif key_str.endswith("M") and len(key_str) > 1:
        # Explicit major key like "CM" (C major)
        # Just uppercase the letter part
        normalized = key_str[:-1].upper()
    else:
        # No explicit suffix - could be numeric or traditional major
        normalized = key_str.upper()

    return _REKORDBOX_TO_CAMELOT.get(normalized, "")


def parse_rekordbox_xml(xml_path: Path) -> list[dict[str, Any]]:
    """Parse Rekordbox XML library export and extract track data.

    Args:
        xml_path: Path to Rekordbox XML file

    Returns:
        List of track dictionaries with keys:
        - location: str (absolute file path)
        - bpm: float | None
        - key_rekordbox: str (original key from Tonality attr)
        - cue_points: list[dict] with name, position_ms, type

    Raises:
        FileNotFoundError: If xml_path does not exist
        ET.ParseError: If XML is malformed
    """
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    logger.info(f"Parsing Rekordbox XML: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    tracks: list[dict[str, Any]] = []

    # Find COLLECTION element
    collection = root.find("COLLECTION")
    if collection is None:
        logger.warning("No COLLECTION element found in XML")
        return tracks

    # Iterate over TRACK elements
    for track_elem in collection.findall("TRACK"):
        track_data = _parse_track_element(track_elem)
        if track_data:
            tracks.append(track_data)

    logger.info(f"Parsed {len(tracks)} tracks from Rekordbox XML")

    return tracks


def _parse_track_element(track_elem: ET.Element) -> dict[str, Any] | None:
    """Parse a single TRACK element from Rekordbox XML.

    Args:
        track_elem: XML element representing a track

    Returns:
        Track dictionary or None if track has no location
    """
    # Get location (required)
    location = track_elem.get("Location", "")
    if not location:
        logger.debug("Skipping track with no location")
        return None

    # Strip file://localhost prefix
    if location.startswith("file://localhost"):
        location = location[16:]  # len("file://localhost") == 16
    elif location.startswith("file://"):
        location = location[7:]  # len("file://") == 7

    # Parse BPM
    bpm_str = track_elem.get("AverageBpm", "")
    bpm: float | None = None
    if bpm_str:
        try:
            bpm = float(bpm_str)
        except ValueError:
            logger.debug(f"Invalid BPM value: {bpm_str}")

    # Get key (Tonality attribute)
    key_rekordbox = track_elem.get("Tonality", "")

    # Parse cue points from POSITION_MARK children
    cue_points: list[dict[str, Any]] = []
    for pos_mark in track_elem.findall("POSITION_MARK"):
        cue_data = _parse_position_mark(pos_mark)
        if cue_data:
            cue_points.append(cue_data)

    # Sort cue points by position
    cue_points.sort(key=lambda x: x["position_ms"])

    return {
        "location": location,
        "bpm": bpm,
        "key_rekordbox": key_rekordbox,
        "cue_points": cue_points,
    }


def _parse_position_mark(pos_mark: ET.Element) -> dict[str, Any] | None:
    """Parse a POSITION_MARK element (cue point / hot cue).

    Args:
        pos_mark: XML element representing a position mark

    Returns:
        Cue point dictionary or None if invalid
    """
    # Get position in milliseconds
    pos_str = pos_mark.get("Start", "")
    if not pos_str:
        return None

    try:
        # Rekordbox stores position in milliseconds as float
        position_ms = float(pos_str)
    except ValueError:
        logger.debug(f"Invalid position value: {pos_str}")
        return None

    # Get name (optional)
    name = pos_mark.get("Name", "")

    # Determine type from Num attribute
    # Num 0-7 = hot cues, 8 = memory cue
    num_str = pos_mark.get("Num", "")
    if num_str:
        try:
            num = int(num_str)
            if 0 <= num <= 7:
                cue_type = "hot_cue"
            elif num == 8:
                cue_type = "memory_cue"
            else:
                cue_type = "unknown"
        except ValueError:
            cue_type = "unknown"
    else:
        cue_type = "unknown"

    return {
        "name": name,
        "position_ms": position_ms,
        "type": cue_type,
    }
