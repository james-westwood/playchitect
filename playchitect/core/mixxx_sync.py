"""
Integration with Mixxx DJ software database.

Provides read-only synchronization of ratings, play counts, history, and cues
from the Mixxx SQLite database. Also supports writing cue points for structural
analysis integration.
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

    def write_cue_points(
        self, db_path: Path, track_path: Path, cue_points: dict[str, float]
    ) -> int:
        """
        Write cue points to Mixxx database as hot cues.

        Inserts or replaces rows in the cues table with type=1 (hot cues).
        Positions are converted from milliseconds to samples at 44100 Hz.

        Args:
            db_path: Path to mixxxdb.sqlite (must be writable)
            track_path: Path to the audio file (must exist in track_locations)
            cue_points: Dict mapping cue names to positions in milliseconds
                       (e.g., {'cue_1_ms': 12345.0, 'cue_2_ms': 67890.0})

        Returns:
            Number of cue points written

        Raises:
            FileNotFoundError: If db_path doesn't exist
            ValueError: If track not found in Mixxx library
            sqlite3.Error: If database operations fail
        """
        if not db_path.exists():
            raise FileNotFoundError(f"Mixxx database not found: {db_path}")

        conn: sqlite3.Connection | None = None
        try:
            # Open writable connection (not read-only like _connect())
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Find track_id from track_locations
            cursor.execute(
                "SELECT id FROM track_locations WHERE location = ? AND fs_deleted = 0",
                (str(track_path.resolve()),),
            )
            row = cursor.fetchone()

            if row is None:
                raise ValueError(f"Track not found in Mixxx library: {track_path}")

            track_id = row["id"]

            # Sample rate for position conversion (Mixxx standard is 44100)
            sample_rate = 44100

            # Write each cue point as a hot cue (type=1)
            written = 0
            for idx, (cue_name, position_ms) in enumerate(cue_points.items()):
                # Convert milliseconds to samples
                position_samples = int((position_ms / 1000.0) * sample_rate)

                # Use hotcue index starting from 4 to avoid clobbering user-set cues 0-3
                hotcue_idx = idx + 4

                # Create a label from the cue name
                label = cue_name.replace("_", " ").title()

                # Insert or replace cue
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO cues
                    (track_id, type, position, length, hotcue, label)
                    VALUES (?, 1, ?, 0, ?, ?)
                    """,
                    (track_id, position_samples, hotcue_idx, label),
                )
                written += 1

            conn.commit()
            logger.info(f"Wrote {written} cue points to Mixxx DB for {track_path.name}")
            return written

        except sqlite3.Error as e:
            logger.error(f"Database error writing cues: {e}")
            raise
        finally:
            if conn:
                conn.close()
