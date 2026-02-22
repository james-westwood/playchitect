"""
Unit tests for Mixxx database synchronization.
"""

import sqlite3
from pathlib import Path

import pytest

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
            samplerate INTEGER DEFAULT 44100
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

    conn.commit()
    conn.close()
    return db_path


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
