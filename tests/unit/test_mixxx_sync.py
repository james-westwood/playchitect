"""
Unit tests for Mixxx database synchronization.
"""

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from playchitect.core.clustering import ClusterResult
from playchitect.core.exporters.mixxx_sync import (
    read_mixxx_library,
    sync_all_playlists,
    write_mixxx_crate,
)
from playchitect.core.metadata_extractor import TrackMetadata
from playchitect.core.mixxx_sync import MixxxSync


@pytest.fixture
def mock_mixxx_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite database with Mixxx schema subset."""
    db_path = tmp_path / "mixxxdb.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Create Tables
    cursor.execute("""
        CREATE TABLE track_locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location varchar(512) UNIQUE,
            fs_deleted INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE library (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location INTEGER REFERENCES track_locations(id),
            rating INTEGER DEFAULT 0,
            timesplayed INTEGER DEFAULT 0,
            last_played_at DATETIME,
            samplerate INTEGER DEFAULT 44100,
            mixxx_deleted INTEGER DEFAULT 0,
            bpm REAL DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE cues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER REFERENCES library(id),
            type INTEGER,
            position INTEGER,
            length INTEGER DEFAULT 0,
            hotcue INTEGER,
            label varchar(32)
        )
    """)

    # Add crates tables for bidirectional sync
    cursor.execute("""
        CREATE TABLE crates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    """)

    cursor.execute("""
        CREATE TABLE crate_tracks (
            crate_id INTEGER REFERENCES crates(id),
            track_id INTEGER REFERENCES library(id),
            PRIMARY KEY (crate_id, track_id)
        )
    """)

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def mock_mixxx_db_with_data(tmp_path: Path) -> tuple[Path, list[Path]]:
    """Create a mock Mixxx DB pre-populated with tracks."""
    db_path = tmp_path / "mixxxdb.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE track_locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location varchar(512) UNIQUE,
            fs_deleted INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE library (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location INTEGER REFERENCES track_locations(id),
            rating INTEGER DEFAULT 0,
            timesplayed INTEGER DEFAULT 0,
            last_played_at DATETIME,
            samplerate INTEGER DEFAULT 44100,
            mixxx_deleted INTEGER DEFAULT 0,
            bpm REAL DEFAULT 120.0
        )
    """)

    cursor.execute("""
        CREATE TABLE cues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER REFERENCES library(id),
            type INTEGER,
            position INTEGER,
            length INTEGER DEFAULT 0,
            hotcue INTEGER,
            label varchar(32)
        )
    """)

    cursor.execute("""
        CREATE TABLE crates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    """)

    cursor.execute("""
        CREATE TABLE crate_tracks (
            crate_id INTEGER REFERENCES crates(id),
            track_id INTEGER REFERENCES library(id),
            PRIMARY KEY (crate_id, track_id)
        )
    """)

    # Insert test tracks
    track_paths: list[Path] = []
    for i in range(5):
        track_file = tmp_path / f"track_{i}.mp3"
        track_file.touch()
        track_path = track_file.resolve()
        track_paths.append(track_path)

        cursor.execute(
            "INSERT INTO track_locations (location, fs_deleted) VALUES (?, 0)",
            (str(track_path),),
        )
        loc_id = cursor.lastrowid

        cursor.execute(
            """
            INSERT INTO library (location, rating, timesplayed, bpm, mixxx_deleted)
            VALUES (?, ?, ?, ?, 0)
            """,
            (loc_id, i + 1, i * 10, 120.0 + i),
        )

    conn.commit()
    conn.close()
    return db_path, track_paths


def seed_track(
    db_path: Path, filepath: Path, rating: int = 0, plays: int = 0, cues: list | None = None
):
    """Helper to insert a track into the mock DB."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Insert location
    cursor.execute(
        "INSERT INTO track_locations (location, fs_deleted) VALUES (?, 0)",
        (str(filepath.resolve()),),
    )
    loc_id = cursor.lastrowid

    # Insert library entry
    cursor.execute(
        """
        INSERT INTO library (location, rating, timesplayed, samplerate)
        VALUES (?, ?, ?, 44100)
        """,
        (loc_id, rating, plays),
    )
    track_id = cursor.lastrowid

    # Insert cues
    if cues:
        for c in cues:
            # c = (position_samples, label, type, hotcue_idx)
            cursor.execute(
                """
                INSERT INTO cues (track_id, position, label, type, hotcue)
                VALUES (?, ?, ?, ?, ?)
                """,
                (track_id, c[0], c[1], c[2], c[3]),
            )

    conn.commit()
    conn.close()


class TestMixxxSync:
    def test_init_auto_discovery_fails_gracefully(self):
        """Test that initialization works even if DB not found."""
        # Force a path that doesn't exist
        sync = MixxxSync(db_path=Path("/non/existent/path.sqlite"))
        assert not sync.available

    def test_enrich_track_not_found(self, mock_mixxx_db):
        """Test syncing a track that isn't in Mixxx DB."""
        sync = MixxxSync(db_path=mock_mixxx_db)
        track = TrackMetadata(filepath=Path("/path/to/unknown.mp3"))

        updated = sync.enrich_track(track)

        # Should be unchanged
        assert updated.rating is None
        assert updated.play_count is None

    def test_enrich_track_success(self, mock_mixxx_db, tmp_path):
        """Test successful retrieval of rating and playcount."""
        # Create a dummy file so resolve() works if needed, though we mock the path string in DB
        dummy_file = tmp_path / "techno.mp3"
        dummy_file.touch()

        seed_track(mock_mixxx_db, dummy_file, rating=5, plays=42)

        sync = MixxxSync(db_path=mock_mixxx_db)
        track = TrackMetadata(filepath=dummy_file)

        updated = sync.enrich_track(track)

        assert updated.rating == 5
        assert updated.play_count == 42

    def test_enrich_cues(self, mock_mixxx_db, tmp_path):
        """Test retrieval and conversion of cue points."""
        dummy_file = tmp_path / "cues.mp3"
        dummy_file.touch()

        # 1 sec = 44100 samples
        cues_data = [
            (44100, "Drop", 1, 0),  # Hotcue 1 at 1s
            (88200, "Break", 0, None),  # Memory cue at 2s
        ]
        seed_track(mock_mixxx_db, dummy_file, cues=cues_data)

        sync = MixxxSync(db_path=mock_mixxx_db)
        track = TrackMetadata(filepath=dummy_file)

        updated = sync.enrich_track(track)

        assert updated.cues is not None
        assert len(updated.cues) == 2

        # Check first cue (Hotcue)
        c1 = updated.cues[0]
        assert c1.position == 1.0
        assert c1.label == "Drop"
        assert c1.hotcue == 0

        # Check second cue (Memory)
        c2 = updated.cues[1]
        assert c2.position == 2.0
        assert c2.label == "Break"
        assert c2.hotcue is None

    def test_read_only_safety(self, mock_mixxx_db, tmp_path):
        """Verify that the connection is indeed read-only."""
        dummy_file = tmp_path / "safety.mp3"
        dummy_file.touch()
        seed_track(mock_mixxx_db, dummy_file)

        sync = MixxxSync(db_path=mock_mixxx_db)

        # Access the private connection method to attempt a write
        conn = sync._connect()
        assert conn is not None

        try:
            with pytest.raises(
                sqlite3.OperationalError, match="attempt to write a readonly database"
            ):
                conn.execute("DELETE FROM library")
        finally:
            conn.close()

    def test_discovery_linux(self):
        """Test discovery logic for Linux paths."""
        from unittest.mock import patch

        with patch("sys.platform", "linux"), patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = Path("/mock/home")

            # Use a side_effect that returns True only for the expected path
            # Since pathlib.Path.exists is patched, it won't receive 'self' automatically
            # in the way a bound method does if we just replace the class attribute.
            # However, we can just ensure it returns True for the specific query.

            with patch("pathlib.Path.exists") as mock_exists:
                # We can't easily filter by 'self' path value here because the mock
                # replaces the function entirely.
                # Instead, we'll make it return True, and then verify the *logic*
                # constructed the correct path.
                mock_exists.return_value = True

                sync = MixxxSync()

                # Check that the first candidate path was constructed correctly
                # The first candidate on Linux is ~/.mixxx/mixxxdb.sqlite
                expected = Path("/mock/home/.mixxx/mixxxdb.sqlite")
                assert sync.db_path == expected

    def test_discovery_windows(self):
        """Test discovery logic for Windows paths."""
        from unittest.mock import patch

        with patch("sys.platform", "win32"), patch("os.environ", {"LOCALAPPDATA": "/AppData"}):
            with patch("pathlib.Path.exists") as mock_exists:
                mock_exists.side_effect = lambda: True
                sync = MixxxSync()
                assert sync.db_path == Path("/AppData/Mixxx/mixxxdb.sqlite")

    def test_discovery_macos(self):
        """Test discovery logic for macOS paths."""
        from unittest.mock import patch

        with patch("sys.platform", "darwin"), patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = Path("/Users/dj")
            with patch("pathlib.Path.exists") as mock_exists:
                mock_exists.side_effect = lambda: True
                sync = MixxxSync()
                expected = Path("/Users/dj/Library/Application Support/Mixxx/mixxxdb.sqlite")
                assert sync.db_path == expected


class TestBidirectionalSync:
    """Tests for bidirectional sync functions (TASK-25)."""

    def test_write_mixxx_crate_creates_crate_and_tracks(self, mock_mixxx_db_with_data):
        """Test that write_mixxx_crate creates a crate row and crate_tracks rows."""
        db_path, track_paths = mock_mixxx_db_with_data

        # Write a crate with 3 tracks
        crate_name = "Test Playlist"
        tracks_to_add = track_paths[:3]
        count = write_mixxx_crate(db_path, crate_name, tracks_to_add)

        assert count == 3

        # Verify crate was created
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM crates WHERE name = ?", (crate_name,))
        crate_row = cursor.fetchone()
        assert crate_row is not None
        crate_id = crate_row[0]

        # Verify crate_tracks were created
        cursor.execute("SELECT COUNT(*) FROM crate_tracks WHERE crate_id = ?", (crate_id,))
        track_count = cursor.fetchone()[0]
        assert track_count == 3

        conn.close()

    def test_write_mixxx_crate_updates_existing_crate(self, mock_mixxx_db_with_data):
        """Test that write_mixxx_crate replaces existing crate contents."""
        db_path, track_paths = mock_mixxx_db_with_data

        crate_name = "Test Playlist"

        # First write - 2 tracks
        count1 = write_mixxx_crate(db_path, crate_name, track_paths[:2])
        assert count1 == 2

        # Second write - 4 tracks (should replace)
        count2 = write_mixxx_crate(db_path, crate_name, track_paths[:4])
        assert count2 == 4

        # Verify only 4 tracks in crate
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM crates WHERE name = ?", (crate_name,))
        crate_id = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM crate_tracks WHERE crate_id = ?", (crate_id,))
        track_count = cursor.fetchone()[0]
        assert track_count == 4

        conn.close()

    def test_write_mixxx_crate_skips_unknown_tracks(self, mock_mixxx_db_with_data, tmp_path):
        """Test that write_mixxx_crate skips tracks not in the library."""
        db_path, track_paths = mock_mixxx_db_with_data

        crate_name = "Test Playlist"

        # Create a path that doesn't exist in the DB
        unknown_track = tmp_path / "unknown.mp3"
        unknown_track.touch()

        # Write with mix of known and unknown tracks
        tracks_to_add = [track_paths[0], unknown_track]
        count = write_mixxx_crate(db_path, crate_name, tracks_to_add)

        # Only 1 track should be written (the known one)
        assert count == 1

    def test_sync_all_playlists_returns_dict(self, mock_mixxx_db_with_data):
        """Test that sync_all_playlists returns a dict mapping cluster names to track counts."""
        db_path, track_paths = mock_mixxx_db_with_data

        # Create cluster results
        clusters: list[ClusterResult] = [
            ClusterResult(
                cluster_id=0,
                tracks=track_paths[:2],
                bpm_mean=120.0,
                bpm_std=2.0,
                track_count=2,
                total_duration=600.0,
            ),
            ClusterResult(
                cluster_id=1,
                tracks=track_paths[2:4],
                bpm_mean=125.0,
                bpm_std=3.0,
                track_count=2,
                total_duration=720.0,
            ),
        ]

        results = sync_all_playlists(db_path, clusters)

        # Should return a dict
        assert isinstance(results, dict)
        assert len(results) == 2
        assert "Playchitect 0" in results
        assert "Playchitect 1" in results
        assert results["Playchitect 0"] == 2
        assert results["Playchitect 1"] == 2

    def test_read_mixxx_library_returns_expected_dicts(self, mock_mixxx_db_with_data):
        """Test that read_mixxx_library returns expected dicts on fixture DB."""
        db_path, track_paths = mock_mixxx_db_with_data

        results = read_mixxx_library(db_path)

        # Should return list of dicts
        assert isinstance(results, list)
        assert len(results) == 5  # We created 5 tracks

        # Check structure of first result
        first = results[0]
        assert isinstance(first, dict)
        assert "location" in first
        assert "bpm" in first
        assert "rating" in first
        assert "timesplayed" in first

        # Check values
        assert first["bpm"] == 120.0
        assert first["rating"] == 1
        assert first["timesplayed"] == 0

    def test_read_mixxx_library_excludes_deleted_tracks(self, mock_mixxx_db_with_data):
        """Test that read_mixxx_library excludes tracks with mixxx_deleted=1."""
        db_path, track_paths = mock_mixxx_db_with_data

        # Mark one track as deleted
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE library SET mixxx_deleted = 1 WHERE id = 1")
        conn.commit()
        conn.close()

        results = read_mixxx_library(db_path)

        # Should only have 4 tracks now
        assert len(results) == 4

    def test_read_mixxx_library_empty_db(self, tmp_path):
        """Test read_mixxx_library on empty database."""
        db_path = tmp_path / "empty.sqlite"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create tables but no data
        cursor.execute("""
            CREATE TABLE track_locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location varchar(512) UNIQUE,
                fs_deleted INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE library (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location INTEGER REFERENCES track_locations(id),
                rating INTEGER DEFAULT 0,
                timesplayed INTEGER DEFAULT 0,
                last_played_at DATETIME,
                samplerate INTEGER DEFAULT 44100,
                mixxx_deleted INTEGER DEFAULT 0,
                bpm REAL DEFAULT 0
            )
        """)

        conn.commit()
        conn.close()

        results = read_mixxx_library(db_path)
        assert results == []

    def test_sync_all_playlists_empty_clusters(self, mock_mixxx_db_with_data):
        """Test sync_all_playlists with empty cluster list."""
        db_path, _ = mock_mixxx_db_with_data

        results = sync_all_playlists(db_path, [])

        assert isinstance(results, dict)
        assert len(results) == 0

    def test_sync_all_playlists_with_str_cluster_id(self, mock_mixxx_db_with_data):
        """Test sync_all_playlists handles string cluster IDs."""
        db_path, track_paths = mock_mixxx_db_with_data

        # Create cluster with string ID
        clusters: list[Any] = [
            ClusterResult(
                cluster_id="techno_0",
                tracks=track_paths[:2],
                bpm_mean=130.0,
                bpm_std=2.0,
                track_count=2,
                total_duration=600.0,
            ),
        ]

        results = sync_all_playlists(db_path, clusters)

        assert "Playchitect techno_0" in results
        assert results["Playchitect techno_0"] == 2
