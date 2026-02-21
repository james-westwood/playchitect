"""
Unit tests for metadata_extractor module.
"""

from pathlib import Path

import numpy as np
import pytest

from playchitect.core.metadata_extractor import MetadataExtractor, TrackMetadata


class TestTrackMetadata:
    """Test TrackMetadata dataclass."""

    def test_track_metadata_initialization(self):
        """Test TrackMetadata initialization."""
        filepath = Path("/path/to/track.mp3")
        metadata = TrackMetadata(filepath=filepath)

        assert metadata.filepath == filepath
        assert metadata.bpm is None
        assert metadata.artist is None
        assert metadata.title is None
        assert metadata.album is None
        assert metadata.duration is None

    def test_track_metadata_with_values(self):
        """Test TrackMetadata with values."""
        filepath = Path("/path/to/track.mp3")
        metadata = TrackMetadata(
            filepath=filepath,
            bpm=128.0,
            artist="Test Artist",
            title="Test Track",
            album="Test Album",
            duration=180.5,
            year=2023,
            genre="Techno",
        )

        assert metadata.bpm == 128.0
        assert metadata.artist == "Test Artist"
        assert metadata.title == "Test Track"
        assert metadata.album == "Test Album"
        assert metadata.duration == 180.5
        assert metadata.year == 2023
        assert metadata.genre == "Techno"

    def test_track_metadata_to_dict(self):
        """Test conversion to dictionary."""
        filepath = Path("/path/to/track.mp3")
        metadata = TrackMetadata(filepath=filepath, bpm=128.0, artist="Test Artist")

        result = metadata.to_dict()

        assert isinstance(result, dict)
        assert result["filepath"] == str(filepath)
        assert result["bpm"] == 128.0
        assert result["artist"] == "Test Artist"
        assert result["title"] is None


class TestMetadataExtractor:
    """Test MetadataExtractor class."""

    def test_extractor_initialization(self):
        """Test extractor initialization."""
        extractor = MetadataExtractor(cache_enabled=True)
        assert extractor.cache_enabled is True

        extractor_no_cache = MetadataExtractor(cache_enabled=False)
        assert extractor_no_cache.cache_enabled is False

    def test_bpm_tags_defined(self):
        """Test that BPM tag list is defined."""
        assert "BPM" in MetadataExtractor.BPM_TAGS
        assert "bpm" in MetadataExtractor.BPM_TAGS
        assert "TBPM" in MetadataExtractor.BPM_TAGS
        assert "tempo" in MetadataExtractor.BPM_TAGS

    def test_extract_nonexistent_file(self):
        """Test extracting metadata from nonexistent file."""
        extractor = MetadataExtractor()
        filepath = Path("/nonexistent/track.mp3")

        # Should not raise exception, but return metadata with None values
        metadata = extractor.extract(filepath)

        assert metadata.filepath == filepath
        assert metadata.bpm is None

    def test_extract_returns_track_metadata(self, tmp_path):
        """Test that extract returns TrackMetadata object."""
        extractor = MetadataExtractor()

        # Create empty test file
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        metadata = extractor.extract(test_file)

        assert isinstance(metadata, TrackMetadata)
        assert metadata.filepath == test_file

    def test_parse_year_valid_formats(self):
        """Test _parse_year with valid year formats."""
        extractor = MetadataExtractor()

        assert extractor._parse_year("2023") == 2023
        assert extractor._parse_year("2023-05-10") == 2023
        assert extractor._parse_year("2000") == 2000
        assert extractor._parse_year("1990-01-01") == 1990

    def test_parse_year_invalid_formats(self):
        """Test _parse_year with invalid formats."""
        extractor = MetadataExtractor()

        assert extractor._parse_year("invalid") is None
        assert extractor._parse_year("20") is None
        assert extractor._parse_year("") is None
        assert extractor._parse_year("1800") is None  # Too old
        assert extractor._parse_year("2200") is None  # Too far in future

    def test_extract_batch_empty_list(self):
        """Test batch extraction with empty list."""
        extractor = MetadataExtractor()
        result = extractor.extract_batch([])

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_extract_batch_multiple_files(self, tmp_path):
        """Test batch extraction with multiple files."""
        extractor = MetadataExtractor()

        # Create test files
        files = []
        for i in range(3):
            file = tmp_path / f"track{i}.mp3"
            file.touch()
            files.append(file)

        result = extractor.extract_batch(files)

        assert len(result) == 3
        assert all(isinstance(m, TrackMetadata) for m in result.values())

    def test_cache_functionality(self, tmp_path):
        """Test that caching works."""
        extractor = MetadataExtractor(cache_enabled=True)

        test_file = tmp_path / "test.mp3"
        test_file.touch()

        # First extraction
        metadata1 = extractor.extract(test_file)

        # Second extraction should use cache
        metadata2 = extractor.extract(test_file)

        # Should be same object from cache
        assert metadata1 is metadata2

    def test_clear_cache(self, tmp_path):
        """Test clearing the cache."""
        extractor = MetadataExtractor(cache_enabled=True)

        test_file = tmp_path / "test.mp3"
        test_file.touch()

        # Extract and cache
        extractor.extract(test_file)
        assert len(extractor._cache) == 1

        # Clear cache
        extractor.clear_cache()
        assert len(extractor._cache) == 0

    def test_recalculate(self, tmp_path, monkeypatch):
        """Test forcing recalculation."""
        extractor = MetadataExtractor(cache_enabled=True)
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        # Initial mock BPM
        monkeypatch.setattr(MetadataExtractor, "calculate_bpm", lambda self, p: 120.0)
        metadata1 = extractor.extract(test_file)
        assert metadata1.bpm == 120.0

        # Change mock BPM and recalculate
        monkeypatch.setattr(MetadataExtractor, "calculate_bpm", lambda self, p: 130.0)
        metadata2 = extractor.recalculate(test_file)

        assert metadata2.bpm == 130.0
        assert metadata1 is not metadata2
        assert extractor.extract(test_file).bpm == 130.0

    def test_extract_without_mutagen(self, tmp_path, monkeypatch):
        """Test extraction when mutagen is not available."""
        # Mock mutagen as unavailable
        import playchitect.core.metadata_extractor as me_module

        monkeypatch.setattr(me_module, "MUTAGEN_AVAILABLE", False)

        # Mock calculate_bpm to return a fixed value
        monkeypatch.setattr(MetadataExtractor, "calculate_bpm", lambda self, p: 128.0)

        extractor = MetadataExtractor()
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        metadata = extractor.extract(test_file)

        # Should return metadata with calculated BPM
        assert metadata.filepath == test_file
        assert metadata.bpm == 128.0
        assert metadata.artist is None

    def test_is_bpm_suspicious(self):
        """Test BPM suspiciousness logic."""
        extractor = MetadataExtractor()

        # Whole numbers are not suspicious
        assert extractor.is_bpm_suspicious(128.0, "Techno") is False
        assert extractor.is_bpm_suspicious(120.0, None) is False

        # Non-whole numbers ARE suspicious
        assert extractor.is_bpm_suspicious(128.1, "Techno") is True
        assert extractor.is_bpm_suspicious(120.005, None) is True

        # Genre mismatches
        assert extractor.is_bpm_suspicious(70.0, "Techno") is True
        assert extractor.is_bpm_suspicious(80.0, "House") is True
        assert extractor.is_bpm_suspicious(87.0, "DnB") is True
        assert extractor.is_bpm_suspicious(174.0, "Drum & Bass") is False

    def test_calculate_bpm_mock(self, tmp_path, monkeypatch):
        """Test calculate_bpm method with mocked librosa."""
        import librosa

        # Create a dummy audio file
        test_file = tmp_path / "test.wav"
        test_file.touch()

        # Mock librosa.load with some signal
        monkeypatch.setattr(librosa, "load", lambda p, sr, duration: (np.random.rand(100), 22050))
        # Mock librosa.beat.beat_track
        monkeypatch.setattr(librosa.beat, "beat_track", lambda y, sr: (128.0, []))

        extractor = MetadataExtractor()
        bpm = extractor.calculate_bpm(test_file)

        assert bpm == 128.0

    def test_extract_suspicious_bpm_recalculates(self, tmp_path, monkeypatch):
        """Test that suspicious BPM in tags triggers recalculation."""

        # Mock MutagenFile to return an object with suspicious BPM
        class MockAudio:
            def __init__(self):
                self.tags = {"BPM": ["70.5"]}

            def __getitem__(self, key):
                return self.tags[key]

            def __contains__(self, key):
                return key in self.tags

        import mutagen

        monkeypatch.setattr(mutagen, "File", lambda p: MockAudio())

        # Mock calculate_bpm to return the "corrected" BPM
        monkeypatch.setattr(MetadataExtractor, "calculate_bpm", lambda self, p: 141.0)

        extractor = MetadataExtractor()
        test_file = tmp_path / "test.mp3"
        test_file.touch()

        # We need to ensure genre is set so we can test genre-based suspicion
        # But here 70.5 is already suspicious because it's not a whole number.
        metadata = extractor.extract(test_file)

        assert metadata.bpm == 141.0


class TestMetadataExtractionIntegration:
    """Integration tests with actual audio files (if available)."""

    @pytest.mark.skipif(
        not Path("/home/james/audio-management/scripts").exists(),
        reason="Test audio files not available",
    )
    def test_extract_from_real_audio_file(self):
        """Test extraction from real audio file if available."""
        # This test would require sample audio files with known metadata
        # Skip for now, but placeholder for future integration tests
        pytest.skip("Sample audio files with metadata not yet created")
