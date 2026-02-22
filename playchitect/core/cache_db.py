"""
SQLite-backed cache for IntensityFeatures.

Replaces the per-file JSON cache with a single WAL-mode SQLite database,
allowing the entire cache to be loaded in one query instead of N individual
file reads.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playchitect.core.intensity_analyzer import IntensityFeatures

logger = logging.getLogger(__name__)

# ── Table / column name constants ─────────────────────────────────────────────
_TABLE = "intensity_features"
_COL_FILE_HASH = "file_hash"
_COL_RMS_ENERGY = "rms_energy"
_COL_BRIGHTNESS = "brightness"
_COL_SUB_BASS_ENERGY = "sub_bass_energy"
_COL_KICK_ENERGY = "kick_energy"
_COL_BASS_HARMONICS = "bass_harmonics"
_COL_PERCUSSIVENESS = "percussiveness"
_COL_ONSET_STRENGTH = "onset_strength"

# Ordered tuple used for INSERT / SELECT column lists
_DATA_COLS = (
    _COL_FILE_HASH,
    _COL_RMS_ENERGY,
    _COL_BRIGHTNESS,
    _COL_SUB_BASS_ENERGY,
    _COL_KICK_ENERGY,
    _COL_BASS_HARMONICS,
    _COL_PERCUSSIVENESS,
    _COL_ONSET_STRENGTH,
)

_DDL = f"""
CREATE TABLE IF NOT EXISTS {_TABLE} (
    {_COL_FILE_HASH}       TEXT PRIMARY KEY,
    {_COL_RMS_ENERGY}      REAL NOT NULL,
    {_COL_BRIGHTNESS}      REAL NOT NULL,
    {_COL_SUB_BASS_ENERGY} REAL NOT NULL,
    {_COL_KICK_ENERGY}     REAL NOT NULL,
    {_COL_BASS_HARMONICS}  REAL NOT NULL,
    {_COL_PERCUSSIVENESS}  REAL NOT NULL,
    {_COL_ONSET_STRENGTH}  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_intensity_brightness
    ON {_TABLE} ({_COL_BRIGHTNESS});
CREATE INDEX IF NOT EXISTS idx_intensity_percussiveness
    ON {_TABLE} ({_COL_PERCUSSIVENESS});
"""

_INSERT_SQL = (
    f"INSERT OR REPLACE INTO {_TABLE} "
    f"({', '.join(_DATA_COLS)}) "
    f"VALUES ({', '.join('?' * len(_DATA_COLS))})"
)

_SELECT_ONE_SQL = (
    f"SELECT {', '.join(_DATA_COLS[1:])} "  # all except file_hash (passed as arg)
    f"FROM {_TABLE} WHERE {_COL_FILE_HASH} = ?"
)

_SELECT_ALL_SQL = f"SELECT {', '.join(_DATA_COLS)} FROM {_TABLE}"


def _row_to_features(file_hash: str, row: tuple) -> IntensityFeatures:
    """Construct IntensityFeatures from a DB row (without filepath)."""
    from playchitect.core.intensity_analyzer import IntensityFeatures  # lazy import

    rms, brightness, sub_bass, kick, harmonics, perc, onset = row
    return IntensityFeatures(
        filepath=Path(),  # caller is responsible for setting the real filepath
        file_hash=file_hash,
        rms_energy=rms,
        brightness=brightness,
        sub_bass_energy=sub_bass,
        kick_energy=kick,
        bass_harmonics=harmonics,
        percussiveness=perc,
        onset_strength=onset,
    )


class CacheDB:
    """
    SQLite-backed cache for IntensityFeatures.

    Stores all intensity analysis results in a single WAL-mode SQLite database.
    The key performance improvement over the JSON cache is ``load_all_intensity()``,
    which loads the entire cache in a single query instead of N individual file reads.

    Usage::

        db = CacheDB(Path("~/.cache/playchitect/playchitect.db").expanduser())
        db.put_intensity(file_hash, features)
        cached = db.get_intensity(file_hash)
        all_cached = db.load_all_intensity()  # keyed by file_hash
    """

    def __init__(self, db_path: Path) -> None:
        """
        Open (or create) the database and ensure the schema exists.

        Args:
            db_path: Path to the SQLite file. Parent directories are created
                     automatically.
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_DDL)
        self._conn.commit()

    def get_intensity(self, file_hash: str) -> IntensityFeatures | None:
        """
        Return cached IntensityFeatures for file_hash, or None if not found.

        The returned object's ``filepath`` attribute is set to ``Path()`` —
        callers must assign the real path after retrieval.

        Args:
            file_hash: MD5 hash used as the cache key.

        Returns:
            IntensityFeatures, or None if the hash is not in the cache.
        """
        row = self._conn.execute(_SELECT_ONE_SQL, (file_hash,)).fetchone()
        if row is None:
            return None
        return _row_to_features(file_hash, row)

    def put_intensity(self, file_hash: str, features: IntensityFeatures) -> None:
        """
        Insert or replace IntensityFeatures for file_hash.

        Args:
            file_hash: MD5 hash used as the cache key.
            features:  Features to store (filepath is not persisted).
        """
        self._conn.execute(
            _INSERT_SQL,
            (
                file_hash,
                features.rms_energy,
                features.brightness,
                features.sub_bass_energy,
                features.kick_energy,
                features.bass_harmonics,
                features.percussiveness,
                features.onset_strength,
            ),
        )
        self._conn.commit()

    def load_all_intensity(self) -> dict[str, IntensityFeatures]:
        """
        Load the entire intensity cache in a single query.

        This is the primary performance improvement over the JSON cache:
        one round-trip replaces N individual file reads.

        Returns:
            Mapping of file_hash -> IntensityFeatures.  Each object's
            ``filepath`` attribute is set to ``Path()``; callers set the
            real path when matching against their file list.
        """
        rows = self._conn.execute(_SELECT_ALL_SQL).fetchall()
        result: dict[str, IntensityFeatures] = {}
        for file_hash, *rest in rows:
            result[file_hash] = _row_to_features(file_hash, tuple(rest))
        return result

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> CacheDB:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


def migrate_json_cache(json_cache_dir: Path, db: CacheDB) -> int:
    """
    One-time migration: read all ``*.json`` files from the old cache directory
    into the SQLite DB.

    Skips files whose hash is already present in the DB (idempotent).
    Skips files that cannot be parsed, logging a warning for each.

    Args:
        json_cache_dir: Directory containing ``<file_hash>.json`` files.
        db:             Target CacheDB to migrate into.

    Returns:
        Number of records successfully migrated.
    """
    from playchitect.core.intensity_analyzer import IntensityFeatures  # lazy import

    if not json_cache_dir.exists():
        return 0

    existing = db.load_all_intensity()
    migrated = 0

    for json_path in json_cache_dir.glob("*.json"):
        file_hash = json_path.stem  # filename IS the MD5 hash
        if file_hash in existing:
            continue
        try:
            with open(json_path) as fh:
                data = json.load(fh)
            features = IntensityFeatures.from_dict(data)
            db.put_intensity(file_hash, features)
            migrated += 1
        except Exception as exc:
            logger.warning("Skipping %s during JSON migration: %s", json_path.name, exc)

    if migrated:
        logger.info("Migrated %d JSON cache files to SQLite", migrated)
    return migrated
