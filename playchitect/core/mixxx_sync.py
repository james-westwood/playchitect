"""
Integration with Mixxx DJ software database.

Provides read-only synchronization of ratings, play counts, history, and cues
from the Mixxx SQLite database.
"""

import logging
import sqlite3
import sys
from pathlib import Path

from playchitect.core.metadata_extractor import CuePoint, TrackMetadata

logger = logging.getLogger(__name__)


class MixxxSync:
    """Read-only interface to Mixxx database."""

    def __init__(self, db_path: Path | None = None):
        """
        Initialize Mixxx synchronizer.

        Args:
            db_path: Path to mixxxdb.sqlite. If None, auto-discovery is attempted.
        """
        self.db_path = db_path or self._find_database()
        if not self.db_path or not self.db_path.exists():
            logger.warning("Mixxx database not found or provided path is invalid")
            self.available = False
        else:
            self.available = True
            logger.info(f"Using Mixxx database: {self.db_path}")

    def _find_database(self) -> Path | None:
        """Attempt to locate mixxxdb.sqlite in standard locations."""
        home = Path.home()
        candidates = []

        if sys.platform == "linux":
            candidates.append(home / ".mixxx" / "mixxxdb.sqlite")
            # Flatpak location
            candidates.append(
                home / ".var" / "app" / "org.mixxx.Mixxx" / ".mixxx" / "mixxxdb.sqlite"
            )
        elif sys.platform == "darwin":
            candidates.append(home / "Library" / "Application Support" / "Mixxx" / "mixxxdb.sqlite")
        elif sys.platform == "win32":
            import os

            local_app_data = os.environ.get("LOCALAPPDATA")
            if local_app_data:
                candidates.append(Path(local_app_data) / "Mixxx" / "mixxxdb.sqlite")

        for path in candidates:
            if path.exists():
                return path

        return None

    def _connect(self) -> sqlite3.Connection | None:
        """Open a read-only connection to the database."""
        if not self.available or not self.db_path:
            return None

        try:
            # URI mode=ro ensures we never accidentally write to the DB
            # absolute path is required for URI
            uri = f"file:{self.db_path.resolve()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to Mixxx DB: {e}")
            return None

    def enrich_track(self, metadata: TrackMetadata) -> TrackMetadata:
        """
        Query Mixxx DB for the given track and update metadata.

        Matches based on absolute file path.
        """
        if not self.available:
            return metadata

        conn = self._connect()
        if not conn:
            return metadata

        try:
            # 1. Find track ID from location
            # Mixxx stores paths in the 'track_locations' table
            # We must handle potential differences in path format (e.g. forward/back slashes)
            # but for now we assume exact match on absolute path string.
            track_path = str(metadata.filepath.resolve())

            cursor = conn.cursor()

            # Use a join to get library info in one go
            query = """
                SELECT
                    l.id,
                    l.rating,
                    l.timesplayed,
                    l.last_played_at,
                    l.samplerate
                FROM library l
                JOIN track_locations tl ON l.location = tl.id
                WHERE tl.location = ? AND tl.fs_deleted = 0
            """

            cursor.execute(query, (track_path,))
            row = cursor.fetchone()

            if row:
                metadata.rating = row["rating"]
                metadata.play_count = row["timesplayed"]
                # last_played_at might be None or a string
                metadata.last_played = row["last_played_at"]

                track_id = row["id"]
                samplerate = row["samplerate"] or 44100.0  # Fallback if missing

                # 2. Get Cues
                cues_query = """
                    SELECT position, label, type, hotcue
                    FROM cues
                    WHERE track_id = ?
                    ORDER BY position ASC
                """
                cursor.execute(cues_query, (track_id,))

                cues = []
                for cue_row in cursor.fetchall():
                    # Position is in samples
                    seconds = cue_row["position"] / samplerate
                    label = cue_row["label"] or ""

                    # Mixxx types: 0=Cue, 1=Hotcue, 2=Loop, etc.
                    # We store hotcue index if type==1
                    hotcue_idx = cue_row["hotcue"] if cue_row["type"] == 1 else None

                    cues.append(CuePoint(position=seconds, label=label, hotcue=hotcue_idx))

                if cues:
                    metadata.cues = cues

                logger.debug(f"Enriched {metadata.filepath.name} from Mixxx DB")

        except sqlite3.Error as e:
            logger.error(f"Database error during sync: {e}")
        finally:
            conn.close()

        return metadata
