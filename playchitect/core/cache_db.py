"""
SQLite-backed cache for IntensityFeatures, Metadata, and Clusters.

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
    from playchitect.core.clustering import ClusterResult
    from playchitect.core.intensity_analyzer import IntensityFeatures
    from playchitect.core.metadata_extractor import TrackMetadata

logger = logging.getLogger(__name__)

# ── Table / column name constants ─────────────────────────────────────────────
_TABLE_INTENSITY = "intensity_features"
_TABLE_METADATA = "track_metadata"
_TABLE_CLUSTERS = "cluster_cache"

_COL_FILE_HASH = "file_hash"
_COL_RMS_ENERGY = "rms_energy"
_COL_BRIGHTNESS = "brightness"
_COL_SUB_BASS_ENERGY = "sub_bass_energy"
_COL_KICK_ENERGY = "kick_energy"
_COL_BASS_HARMONICS = "bass_harmonics"
_COL_PERCUSSIVENESS = "percussiveness"
_COL_ONSET_STRENGTH = "onset_strength"

# Ordered tuple used for INSERT / SELECT column lists
_INTENSITY_COLS = (
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
CREATE TABLE IF NOT EXISTS {_TABLE_INTENSITY} (
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
    ON {_TABLE_INTENSITY} ({_COL_BRIGHTNESS});
CREATE INDEX IF NOT EXISTS idx_intensity_percussiveness
    ON {_TABLE_INTENSITY} ({_COL_PERCUSSIVENESS});

CREATE TABLE IF NOT EXISTS {_TABLE_METADATA} (
    filepath    TEXT PRIMARY KEY,
    bpm         REAL,
    artist      TEXT,
    title       TEXT,
    album       TEXT,
    duration    REAL,
    year        INTEGER,
    genre       TEXT,
    rating      INTEGER,
    play_count  INTEGER,
    last_played TEXT,
    cues_json   TEXT
);

CREATE TABLE IF NOT EXISTS {_TABLE_CLUSTERS} (
    cluster_id         TEXT PRIMARY KEY,
    tracks_json        TEXT NOT NULL,
    bpm_mean           REAL NOT NULL,
    bpm_std            REAL NOT NULL,
    track_count        INTEGER NOT NULL,
    total_duration     REAL NOT NULL,
    genre              TEXT,
    feature_means_json TEXT,
    importance_json    TEXT,
    weight_source      TEXT,
    embedding_variance REAL
);
"""

_INTENSITY_INSERT_SQL = (
    f"INSERT OR REPLACE INTO {_TABLE_INTENSITY} "
    f"({', '.join(_INTENSITY_COLS)}) "
    f"VALUES ({', '.join('?' * len(_INTENSITY_COLS))})"
)

_INTENSITY_SELECT_ONE_SQL = (
    f"SELECT {', '.join(_INTENSITY_COLS[1:])} "  # all except file_hash (passed as arg)
    f"FROM {_TABLE_INTENSITY} WHERE {_COL_FILE_HASH} = ?"
)

_INTENSITY_SELECT_ALL_SQL = f"SELECT {', '.join(_INTENSITY_COLS)} FROM {_TABLE_INTENSITY}"


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
    SQLite-backed cache for IntensityFeatures, Metadata, and Clusters.

    Stores all analysis results in a single WAL-mode SQLite database.
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

    # ── Intensity Features ───────────────────────────────────────────────────

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
        row = self._conn.execute(_INTENSITY_SELECT_ONE_SQL, (file_hash,)).fetchone()
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
            _INTENSITY_INSERT_SQL,
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
        rows = self._conn.execute(_INTENSITY_SELECT_ALL_SQL).fetchall()
        result: dict[str, IntensityFeatures] = {}
        for file_hash, *rest in rows:
            result[file_hash] = _row_to_features(file_hash, tuple(rest))
        return result

    # ── Track Metadata ───────────────────────────────────────────────────────

    def get_metadata(self, filepath: Path) -> TrackMetadata | None:
        """Return cached TrackMetadata for filepath, or None if not found."""
        from playchitect.core.metadata_extractor import CuePoint, TrackMetadata  # lazy import

        sql = f"SELECT * FROM {_TABLE_METADATA} WHERE filepath = ?"
        row = self._conn.execute(sql, (str(filepath),)).fetchone()
        if not row:
            return None

        (
            _,
            bpm,
            artist,
            title,
            album,
            duration,
            year,
            genre,
            rating,
            play_count,
            last_played,
            cues_json,
        ) = row

        cues = None
        if cues_json:
            cues_data = json.loads(cues_json)
            cues = [
                CuePoint(position=c["position"], label=c["label"], hotcue=c.get("hotcue"))
                for c in cues_data
            ]

        return TrackMetadata(
            filepath=filepath,
            bpm=bpm,
            artist=artist,
            title=title,
            album=album,
            duration=duration,
            year=year,
            genre=genre,
            rating=rating,
            play_count=play_count,
            last_played=last_played,
            cues=cues,
        )

    def put_metadata(self, metadata: TrackMetadata) -> None:
        """Insert or replace TrackMetadata."""
        cues_json = None
        if metadata.cues:
            cues_json = json.dumps(
                [
                    {"position": c.position, "label": c.label, "hotcue": c.hotcue}
                    for c in metadata.cues
                ]
            )

        sql = f"""
            INSERT OR REPLACE INTO {_TABLE_METADATA}
            (filepath, bpm, artist, title, album, duration, year, genre,
             rating, play_count, last_played, cues_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self._conn.execute(
            sql,
            (
                str(metadata.filepath),
                metadata.bpm,
                metadata.artist,
                metadata.title,
                metadata.album,
                metadata.duration,
                metadata.year,
                metadata.genre,
                metadata.rating,
                metadata.play_count,
                metadata.last_played,
                cues_json,
            ),
        )
        self._conn.commit()

    def load_all_metadata(self) -> dict[Path, TrackMetadata]:
        """Load all metadata from DB."""
        sql = f"SELECT filepath FROM {_TABLE_METADATA}"
        paths = [Path(r[0]) for r in self._conn.execute(sql).fetchall()]

        result: dict[Path, TrackMetadata] = {}
        for p in paths:
            meta = self.get_metadata(p)
            if meta:
                result[p] = meta
        return result

    # ── Cluster Cache ────────────────────────────────────────────────────────

    def put_clusters(self, clusters: list[ClusterResult]) -> None:
        """Save a set of ClusterResults, replacing any existing ones."""
        self._conn.execute(f"DELETE FROM {_TABLE_CLUSTERS}")

        sql = f"""
            INSERT INTO {_TABLE_CLUSTERS}
            (cluster_id, tracks_json, bpm_mean, bpm_std, track_count, total_duration,
             genre, feature_means_json, importance_json, weight_source, embedding_variance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        for c in clusters:
            tracks_json = json.dumps([str(p) for p in c.tracks])
            means_json = json.dumps(c.feature_means) if c.feature_means else None
            imp_json = json.dumps(c.feature_importance) if c.feature_importance else None

            self._conn.execute(
                sql,
                (
                    str(c.cluster_id),
                    tracks_json,
                    c.bpm_mean,
                    c.bpm_std,
                    c.track_count,
                    c.total_duration,
                    c.genre,
                    means_json,
                    imp_json,
                    c.weight_source,
                    c.embedding_variance_explained,
                ),
            )
        self._conn.commit()

    def load_latest_clusters(self) -> list[ClusterResult]:
        """Load the most recently saved clusters."""
        from playchitect.core.clustering import ClusterResult  # lazy import

        rows = self._conn.execute(f"SELECT * FROM {_TABLE_CLUSTERS}").fetchall()
        results = []
        for row in rows:
            (
                cid,
                t_json,
                bm,
                bs,
                cnt,
                dur,
                genre,
                m_json,
                i_json,
                ws,
                ev,
            ) = row

            tracks = [Path(p) for p in json.loads(t_json)]
            means = json.loads(m_json) if m_json else None
            importance = json.loads(i_json) if i_json else None

            # Handle numeric vs string cluster_id
            try:
                final_cid: int | str = int(cid)
            except ValueError:
                final_cid = cid

            results.append(
                ClusterResult(
                    cluster_id=final_cid,
                    tracks=tracks,
                    bpm_mean=bm,
                    bpm_std=bs,
                    track_count=cnt,
                    total_duration=dur,
                    genre=genre,
                    feature_means=means,
                    feature_importance=importance,
                    weight_source=ws,
                    embedding_variance_explained=ev,
                )
            )
        return results

    # ── Lifecycle ────────────────────────────────────────────────────────────

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
