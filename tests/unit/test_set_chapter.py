"""Tests for SetChapter data model."""

from pathlib import Path

from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata
from playchitect.core.set_chapter import (
    ChapterManager,
    SetChapter,
    create_default_chapters,
)


class TestSetChapter:
    """Tests for SetChapter."""

    def test_chapter_creation(self) -> None:
        """Test creating a chapter."""
        chapter = SetChapter(id="test-1", name="Test Chapter")
        assert chapter.id == "test-1"
        assert chapter.name == "Test Chapter"
        assert chapter.get_track_count() == 0

    def test_add_track(self) -> None:
        """Test adding a track to chapter."""
        chapter = SetChapter(id="test", name="Test")
        track = Path("/music/track1.mp3")
        chapter.add_track(track)
        assert chapter.get_track_count() == 1
        assert track in chapter.tracks

    def test_add_duplicate_track(self) -> None:
        """Test that duplicate tracks are ignored."""
        chapter = SetChapter(id="test", name="Test")
        track = Path("/music/track1.mp3")
        chapter.add_track(track)
        chapter.add_track(track)
        assert chapter.get_track_count() == 1

    def test_remove_track(self) -> None:
        """Test removing a track from chapter."""
        chapter = SetChapter(id="test", name="Test")
        track = Path("/music/track1.mp3")
        chapter.add_track(track)
        chapter.remove_track(track)
        assert chapter.get_track_count() == 0

    def test_reorder_track_valid(self) -> None:
        """Test reordering tracks within chapter."""
        chapter = SetChapter(id="test", name="Test")
        chapter.tracks = [Path(f"/music/track{i}.mp3") for i in range(5)]

        result = chapter.reorder_track(0, 4)
        assert result is True
        assert chapter.tracks[0] == Path("/music/track1.mp3")
        assert chapter.tracks[4] == Path("/music/track0.mp3")

    def test_reorder_track_invalid_index(self) -> None:
        """Test reordering with invalid indices."""
        chapter = SetChapter(id="test", name="Test")
        chapter.tracks = [Path(f"/music/track{i}.mp3") for i in range(3)]

        result = chapter.reorder_track(-1, 1)
        assert result is False

        result = chapter.reorder_track(0, 10)
        assert result is False

    def test_calculate_total_duration(self) -> None:
        """Test calculating total duration."""
        chapter = SetChapter(id="test", name="Test")
        chapter.tracks = [
            Path("/music/track1.mp3"),
            Path("/music/track2.mp3"),
        ]

        metadata = {
            Path("/music/track1.mp3"): TrackMetadata(
                filepath=Path("/music/track1.mp3"), duration=300.0
            ),
            Path("/music/track2.mp3"): TrackMetadata(
                filepath=Path("/music/track2.mp3"), duration=240.0
            ),
        }

        duration = chapter.calculate_total_duration(metadata)
        assert duration == 540.0

    def test_calculate_total_duration_missing_metadata(self) -> None:
        """Test duration calculation with missing metadata."""
        chapter = SetChapter(id="test", name="Test")
        chapter.tracks = [Path("/music/track1.mp3")]

        metadata = {}  # Empty
        duration = chapter.calculate_total_duration(metadata)
        assert duration == 0.0

    def test_calculate_energy_range(self) -> None:
        """Test calculating energy range."""
        chapter = SetChapter(id="test", name="Test")
        chapter.tracks = [
            Path("/music/track1.mp3"),
            Path("/music/track2.mp3"),
        ]

        features = {
            Path("/music/track1.mp3"): IntensityFeatures(
                file_path=Path("/music/track1.mp3"),
                file_hash="hash1",
                rms_energy=0.3,
                brightness=0.5,
                sub_bass_energy=0.3,
                kick_energy=0.6,
                bass_harmonics=0.4,
                percussiveness=0.5,
                onset_strength=0.5,
                camelot_key="8B",
                key_index=0.0,
            ),
            Path("/music/track2.mp3"): IntensityFeatures(
                file_path=Path("/music/track2.mp3"),
                file_hash="hash2",
                rms_energy=0.7,
                brightness=0.5,
                sub_bass_energy=0.3,
                kick_energy=0.6,
                bass_harmonics=0.4,
                percussiveness=0.5,
                onset_strength=0.5,
                camelot_key="8B",
                key_index=0.0,
            ),
        }

        energy_min, energy_max = chapter.calculate_energy_range(features)
        assert energy_min == 0.3
        assert energy_max == 0.7

    def test_calculate_energy_range_empty(self) -> None:
        """Test energy range with empty chapter."""
        chapter = SetChapter(id="test", name="Test")
        energy_min, energy_max = chapter.calculate_energy_range({})
        assert energy_min == 0.0
        assert energy_max == 1.0

    def test_get_summary(self) -> None:
        """Test getting chapter summary."""
        chapter = SetChapter(id="test", name="Test")
        chapter.tracks = [Path("/music/track1.mp3")]

        metadata = {
            Path("/music/track1.mp3"): TrackMetadata(
                filepath=Path("/music/track1.mp3"), duration=300.0
            )
        }

        features = {
            Path("/music/track1.mp3"): IntensityFeatures(
                file_path=Path("/music/track1.mp3"),
                file_hash="hash",
                rms_energy=0.5,
                brightness=0.5,
                sub_bass_energy=0.3,
                kick_energy=0.6,
                bass_harmonics=0.4,
                percussiveness=0.5,
                onset_strength=0.5,
                camelot_key="8B",
                key_index=0.0,
            )
        }

        summary = chapter.get_summary(metadata, features)
        assert summary["name"] == "Test"
        assert summary["track_count"] == 1
        assert summary["total_duration"] == 300.0
        assert summary["energy_min"] == 0.5
        assert summary["energy_max"] == 0.5


class TestChapterManager:
    """Tests for ChapterManager."""

    def test_manager_creation(self) -> None:
        """Test creating a chapter manager."""
        manager = ChapterManager()
        assert manager.get_chapter_count() == 0

    def test_create_chapter(self) -> None:
        """Test creating a chapter via manager."""
        manager = ChapterManager()
        chapter = manager.create_chapter("My Chapter")
        assert chapter.name == "My Chapter"
        assert manager.get_chapter_count() == 1

    def test_create_chapter_auto_name(self) -> None:
        """Test creating chapter with auto-generated name."""
        manager = ChapterManager()
        chapter1 = manager.create_chapter()
        chapter2 = manager.create_chapter()
        assert chapter1.name == "Chapter 1"
        assert chapter2.name == "Chapter 2"

    def test_get_chapter(self) -> None:
        """Test getting a chapter by ID."""
        manager = ChapterManager()
        chapter = manager.create_chapter("Test")
        found = manager.get_chapter(chapter.id)
        assert found == chapter

    def test_get_chapter_not_found(self) -> None:
        """Test getting non-existent chapter."""
        manager = ChapterManager()
        found = manager.get_chapter("does-not-exist")
        assert found is None

    def test_remove_chapter(self) -> None:
        """Test removing a chapter."""
        manager = ChapterManager()
        chapter = manager.create_chapter("Test")
        result = manager.remove_chapter(chapter.id)
        assert result is True
        assert manager.get_chapter_count() == 0

    def test_remove_chapter_not_found(self) -> None:
        """Test removing non-existent chapter."""
        manager = ChapterManager()
        result = manager.remove_chapter("does-not-exist")
        assert result is False

    def test_reorder_chapter(self) -> None:
        """Test reordering chapters."""
        manager = ChapterManager()
        manager.create_chapter("Chapter 1")
        manager.create_chapter("Chapter 2")
        manager.create_chapter("Chapter 3")

        result = manager.reorder_chapter(0, 2)
        assert result is True

        chapters = manager.get_chapters()
        assert chapters[0].name == "Chapter 2"
        assert chapters[1].name == "Chapter 3"
        assert chapters[2].name == "Chapter 1"

    def test_reorder_chapter_invalid(self) -> None:
        """Test reordering with invalid indices."""
        manager = ChapterManager()
        manager.create_chapter("Chapter 1")

        result = manager.reorder_chapter(0, 5)
        assert result is False

    def test_clear(self) -> None:
        """Test clearing all chapters."""
        manager = ChapterManager()
        manager.create_chapter("Chapter 1")
        manager.create_chapter("Chapter 2")

        manager.clear()
        assert manager.get_chapter_count() == 0

    def test_update_chapter_name(self) -> None:
        """Test updating chapter name."""
        manager = ChapterManager()
        chapter = manager.create_chapter("Old Name")

        result = manager.update_chapter_name(chapter.id, "New Name")
        assert result is True
        assert chapter.name == "New Name"

    def test_get_chapters(self) -> None:
        """Test getting all chapters."""
        manager = ChapterManager()
        ch1 = manager.create_chapter()
        ch2 = manager.create_chapter()

        chapters = manager.get_chapters()
        assert chapters == [ch1, ch2]


class TestCreateDefaultChapters:
    """Tests for create_default_chapters helper."""

    def test_create_default_chapters(self) -> None:
        """Test creating default chapters."""
        chapters = create_default_chapters()
        assert len(chapters) == 4
        assert chapters[0].name == "Intro"
        assert chapters[1].name == "Build"
        assert chapters[2].name == "Peak"
        assert chapters[3].name == "Outro"

    def test_default_chapters_have_unique_ids(self) -> None:
        """Test default chapters have unique IDs."""
        chapters = create_default_chapters()
        ids = [c.id for c in chapters]
        assert len(ids) == len(set(ids))
