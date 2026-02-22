"""
Unit tests for cache_db module.
"""

import json
from pathlib import Path

import pytest

from playchitect.core.cache_db import CacheDB, migrate_json_cache
from playchitect.core.intensity_analyzer import IntensityFeatures


def _make_features(
    file_hash: str = "abc123",
    rms: float = 0.5,
    brightness: float = 0.6,
    sub_bass: float = 0.3,
    kick: float = 0.7,
    harmonics: float = 0.4,
    perc: float = 0.8,
    onset: float = 0.65,
) -> IntensityFeatures:
    return IntensityFeatures(
        filepath=Path("test.mp3"),
        file_hash=file_hash,
        rms_energy=rms,
        brightness=brightness,
        sub_bass_energy=sub_bass,
        kick_energy=kick,
        bass_harmonics=harmonics,
        percussiveness=perc,
        onset_strength=onset,
    )


class TestCacheDBSchema:
    def test_creates_db_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "cache.db"
        CacheDB(db_path)
        assert db_path.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        db_path = tmp_path / "deep" / "nested" / "cache.db"
        CacheDB(db_path)
        assert db_path.exists()

    def test_wal_mode_is_set(self, tmp_path: Path) -> None:
        db = CacheDB(tmp_path / "cache.db")
        row = db._conn.execute("PRAGMA journal_mode").fetchone()
        assert row is not None
        assert row[0] == "wal"

    def test_intensity_table_exists(self, tmp_path: Path) -> None:
        db = CacheDB(tmp_path / "cache.db")
        row = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='intensity_features'"
        ).fetchone()
        assert row is not None

    def test_indexes_exist(self, tmp_path: Path) -> None:
        db = CacheDB(tmp_path / "cache.db")
        indexes = {
            r[0]
            for r in db._conn.execute(
                "SELECT name FROM sqlite_master"
                " WHERE type='index' AND tbl_name='intensity_features'"
            ).fetchall()
        }
        assert "idx_intensity_brightness" in indexes
        assert "idx_intensity_percussiveness" in indexes

    def test_schema_creation_is_idempotent(self, tmp_path: Path) -> None:
        db_path = tmp_path / "cache.db"
        CacheDB(db_path)
        CacheDB(db_path)  # second open must not raise


class TestCacheDBRoundtrip:
    def test_put_then_get(self, tmp_path: Path) -> None:
        db = CacheDB(tmp_path / "cache.db")
        f = _make_features()
        db.put_intensity(f.file_hash, f)

        result = db.get_intensity(f.file_hash)
        assert result is not None
        assert result.file_hash == f.file_hash
        assert result.rms_energy == pytest.approx(f.rms_energy)
        assert result.brightness == pytest.approx(f.brightness)
        assert result.sub_bass_energy == pytest.approx(f.sub_bass_energy)
        assert result.kick_energy == pytest.approx(f.kick_energy)
        assert result.bass_harmonics == pytest.approx(f.bass_harmonics)
        assert result.percussiveness == pytest.approx(f.percussiveness)
        assert result.onset_strength == pytest.approx(f.onset_strength)

    def test_get_nonexistent_returns_none(self, tmp_path: Path) -> None:
        db = CacheDB(tmp_path / "cache.db")
        assert db.get_intensity("no_such_hash") is None

    def test_put_is_idempotent(self, tmp_path: Path) -> None:
        db = CacheDB(tmp_path / "cache.db")
        f = _make_features()
        db.put_intensity(f.file_hash, f)
        db.put_intensity(f.file_hash, f)  # must not raise or duplicate
        rows = db._conn.execute("SELECT COUNT(*) FROM intensity_features").fetchone()
        assert rows is not None
        assert rows[0] == 1

    def test_put_replaces_on_duplicate_hash(self, tmp_path: Path) -> None:
        db = CacheDB(tmp_path / "cache.db")
        f1 = _make_features(rms=0.1)
        f2 = _make_features(rms=0.9)  # same hash, different values
        db.put_intensity(f1.file_hash, f1)
        db.put_intensity(f2.file_hash, f2)
        result = db.get_intensity(f2.file_hash)
        assert result is not None
        assert result.rms_energy == pytest.approx(0.9)

    def test_filepath_is_placeholder_on_get(self, tmp_path: Path) -> None:
        """filepath is not stored in the DB; callers set it after retrieval."""
        db = CacheDB(tmp_path / "cache.db")
        f = _make_features()
        db.put_intensity(f.file_hash, f)
        result = db.get_intensity(f.file_hash)
        assert result is not None
        assert result.filepath == Path()


class TestLoadAllIntensity:
    def test_empty_db_returns_empty_dict(self, tmp_path: Path) -> None:
        db = CacheDB(tmp_path / "cache.db")
        assert db.load_all_intensity() == {}

    def test_returns_all_rows(self, tmp_path: Path) -> None:
        db = CacheDB(tmp_path / "cache.db")
        hashes = ["hash1", "hash2", "hash3"]
        for h in hashes:
            db.put_intensity(h, _make_features(file_hash=h))

        result = db.load_all_intensity()
        assert set(result.keys()) == set(hashes)

    def test_values_match_inserted(self, tmp_path: Path) -> None:
        db = CacheDB(tmp_path / "cache.db")
        f = _make_features(file_hash="h1", rms=0.42, brightness=0.77)
        db.put_intensity(f.file_hash, f)
        result = db.load_all_intensity()
        assert result["h1"].rms_energy == pytest.approx(0.42)
        assert result["h1"].brightness == pytest.approx(0.77)


class TestMigrateJsonCache:
    def _write_json(self, directory: Path, features: IntensityFeatures) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{features.file_hash}.json"
        data = features.to_dict()
        with open(path, "w") as fh:
            json.dump(data, fh)

    def test_migrates_json_files(self, tmp_path: Path) -> None:
        json_dir = tmp_path / "intensity"
        f = _make_features(file_hash="deadbeef")
        self._write_json(json_dir, f)

        db = CacheDB(tmp_path / "cache.db")
        count = migrate_json_cache(json_dir, db)
        assert count == 1
        assert db.get_intensity("deadbeef") is not None

    def test_is_idempotent(self, tmp_path: Path) -> None:
        json_dir = tmp_path / "intensity"
        f = _make_features(file_hash="deadbeef")
        self._write_json(json_dir, f)

        db = CacheDB(tmp_path / "cache.db")
        first = migrate_json_cache(json_dir, db)
        second = migrate_json_cache(json_dir, db)
        assert first == 1
        assert second == 0  # already in DB, skipped

    def test_nonexistent_dir_returns_zero(self, tmp_path: Path) -> None:
        db = CacheDB(tmp_path / "cache.db")
        count = migrate_json_cache(tmp_path / "no_such_dir", db)
        assert count == 0

    def test_skips_invalid_json(self, tmp_path: Path) -> None:
        json_dir = tmp_path / "intensity"
        json_dir.mkdir()
        (json_dir / "badhash.json").write_text("not valid json {{")

        db = CacheDB(tmp_path / "cache.db")
        count = migrate_json_cache(json_dir, db)
        assert count == 0

    def test_migrates_multiple_files(self, tmp_path: Path) -> None:
        json_dir = tmp_path / "intensity"
        hashes = ["aaa111", "bbb222", "ccc333"]
        for h in hashes:
            self._write_json(json_dir, _make_features(file_hash=h))

        db = CacheDB(tmp_path / "cache.db")
        count = migrate_json_cache(json_dir, db)
        assert count == 3
        for h in hashes:
            assert db.get_intensity(h) is not None
