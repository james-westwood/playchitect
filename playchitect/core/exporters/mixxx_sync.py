"""Mixxx DJ software bidirectional synchronization.

Provides functions to read from and write to Mixxx's SQLite database,
enabling bidirectional sync of playlists (as Mixxx crates) and library data.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playchitect.core.clustering import ClusterResult

logger = logging.getLogger(__name__)


def read_mixxx_library(db_path: Path) -> list[dict]:
    """
    Read track library from Mixxx database.

    Queries the library and track_locations tables to retrieve track
    information including file path, BPM, rating, and play count.

    Args:
        db_path: Path to the Mixxx mixxxdb.sqlite database file.

    Returns:
        List of dictionaries with keys: location, bpm, rating, timesplayed.
        Each dict represents a track in the Mixxx library that hasn't been
        marked as deleted.

    Raises:
        sqlite3.Error: If the database cannot be opened or queried.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.cursor()
        query = """
            SELECT library.id,
                   track_locations.location,
                   library.bpm,
                   library.rating,
                   library.timesplayed
            FROM library
            JOIN track_locations ON library.location = track_locations.id
            WHERE library.mixxx_deleted = 0
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        result = []
        for row in rows:
            result.append(
                {
                    "location": row["location"],
                    "bpm": row["bpm"],
                    "rating": row["rating"],
                    "timesplayed": row["timesplayed"],
                }
            )

        logger.info(f"Read {len(result)} tracks from Mixxx library")
        return result
    finally:
        conn.close()


def write_mixxx_crate(db_path: Path, crate_name: str, track_paths: list[Path]) -> int:
    """
    Create or update a Mixxx crate with the specified tracks.

    Inserts or replaces the crate in the crates table, then populates
    the crate_tracks table with the matching track IDs from track_locations.

    Args:
        db_path: Path to the Mixxx mixxxdb.sqlite database file.
        crate_name: Name of the crate to create or update.
        track_paths: List of track file paths to include in the crate.

    Returns:
        Count of tracks successfully written to the crate.

    Raises:
        sqlite3.Error: If the database cannot be opened or modified.
    """
    conn = sqlite3.connect(db_path)

    try:
        cursor = conn.cursor()

        # Insert or replace crate
        cursor.execute(
            "INSERT OR REPLACE INTO crates (name) VALUES (?)",
            (crate_name,),
        )
        crate_id = cursor.lastrowid

        # Delete existing crate tracks
        cursor.execute(
            "DELETE FROM crate_tracks WHERE crate_id = ?",
            (crate_id,),
        )

        # Get track IDs for the given paths and insert into crate_tracks
        tracks_written = 0
        for track_path in track_paths:
            # Normalize path to absolute string for matching
            path_str = str(track_path.resolve())

            # Find track ID in track_locations
            cursor.execute(
                "SELECT id FROM track_locations WHERE location = ?",
                (path_str,),
            )
            row = cursor.fetchone()

            if row:
                track_id = row[0]
                cursor.execute(
                    "INSERT INTO crate_tracks (crate_id, track_id) VALUES (?, ?)",
                    (crate_id, track_id),
                )
                tracks_written += 1
            else:
                logger.warning(f"Track not found in Mixxx DB: {path_str}")

        conn.commit()
        logger.info(f"Wrote {tracks_written} tracks to crate '{crate_name}'")
        return tracks_written
    finally:
        conn.close()


def sync_all_playlists(
    db_path: Path,
    clusters: list[ClusterResult],
) -> dict[str, int]:
    """
    Synchronize all cluster results as Mixxx crates.

    Creates a Mixxx crate for each cluster, using the cluster name as the
    crate name. Returns a summary of how many tracks were written per crate.

    Args:
        db_path: Path to the Mixxx mixxxdb.sqlite database file.
        clusters: List of ClusterResult objects to sync as crates.

    Returns:
        Dictionary mapping cluster name to track count written.

    Raises:
        sqlite3.Error: If the database cannot be opened or modified.
    """
    results: dict[str, int] = {}

    for cluster in clusters:
        # Generate crate name from cluster info
        if hasattr(cluster, "cluster_id"):
            crate_name = f"Playchitect {cluster.cluster_id}"
        else:
            crate_name = "Playchitect Playlist"

        # Write the crate
        track_count = write_mixxx_crate(db_path, crate_name, cluster.tracks)
        results[crate_name] = track_count

    logger.info(f"Synced {len(clusters)} playlists to Mixxx")
    return results
