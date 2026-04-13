"""Unit tests for M3U chapter export functionality."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from playchitect.core.export import M3UExporter
from playchitect.core.metadata_extractor import TrackMetadata


@pytest.fixture
def temp_output_dir() -> Generator[Path]:
    """Create a temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_tracks() -> list[Path]:
    """Create sample track paths."""
    return [
        Path("/music/track1.mp3"),
        Path("/music/track2.mp3"),
        Path("/music/track3.mp3"),
        Path("/music/track4.mp3"),
        Path("/music/track5.mp3"),
    ]


@pytest.fixture
def sample_metadata(sample_tracks: list[Path]) -> dict[Path, TrackMetadata]:
    """Create sample metadata dictionary."""
    return {
        sample_tracks[0]: TrackMetadata(
            filepath=sample_tracks[0],
            title="Track 1",
            artist="Artist 1",
            duration=240.0,
            bpm=128.0,
        ),
        sample_tracks[1]: TrackMetadata(
            filepath=sample_tracks[1],
            title="Track 2",
            artist="Artist 2",
            duration=300.0,
            bpm=130.0,
        ),
        sample_tracks[2]: TrackMetadata(
            filepath=sample_tracks[2],
            title="Track 3",
            artist="Artist 3",
            duration=280.0,
            bpm=132.0,
        ),
        sample_tracks[3]: TrackMetadata(
            filepath=sample_tracks[3],
            title="Track 4",
            artist="Artist 4",
            duration=260.0,
            bpm=135.0,
        ),
        sample_tracks[4]: TrackMetadata(
            filepath=sample_tracks[4],
            title="Track 5",
            artist="Artist 5",
            duration=320.0,
            bpm=128.0,
        ),
    }


class TestM3UExporterPlaylistExport:
    """Tests for export_as_playlist method."""

    def test_export_as_playlist_single_file(
        self,
        temp_output_dir: Path,
        sample_tracks: list[Path],
        sample_metadata: dict[Path, TrackMetadata],
    ) -> None:
        """Test exporting all tracks as a single M3U file."""
        exporter = M3UExporter(temp_output_dir, playlist_prefix="Set")
        result = exporter.export_as_playlist(sample_tracks, sample_metadata)

        assert result.exists()
        content = result.read_text()

        assert "#EXTM3U" in content
        assert "#EXTINF:240,Artist 1 - Track 1" in content
        assert "#EXTINF:300,Artist 2 - Track 2" in content

    def test_export_as_playlist_with_chapter_markers(
        self,
        temp_output_dir: Path,
        sample_tracks: list[Path],
        sample_metadata: dict[Path, TrackMetadata],
    ) -> None:
        """Test exporting with chapter boundary markers."""
        exporter = M3UExporter(temp_output_dir, playlist_prefix="Set")
        chapter_boundaries = [(0, "Intro"), (2, "Build"), (4, "Cool Down")]

        result = exporter.export_as_playlist(
            sample_tracks, sample_metadata, chapter_boundaries, filename="set.m3u"
        )

        assert result.exists()
        content = result.read_text()

        assert "# --- Intro ---" in content
        assert "# --- Build ---" in content
        assert "# --- Cool Down ---" in content


class TestM3UExporterChapterExport:
    """Tests for export_as_chapters method."""

    def test_export_as_chapters_creates_multiple_files(
        self,
        temp_output_dir: Path,
        sample_tracks: list[Path],
        sample_metadata: dict[Path, TrackMetadata],
    ) -> None:
        """Test that exporting as chapters creates separate M3U files."""
        exporter = M3UExporter(temp_output_dir, playlist_prefix="Set")
        chapter_boundaries = [(0, "Intro"), (2, "Main"), (4, "Outro")]

        results = exporter.export_as_chapters(sample_tracks, sample_metadata, chapter_boundaries)

        assert len(results) == 3
        for path in results:
            assert path.exists()

    def test_export_as_chapters_correct_content_per_file(
        self,
        temp_output_dir: Path,
        sample_tracks: list[Path],
        sample_metadata: dict[Path, TrackMetadata],
    ) -> None:
        """Test that each chapter file contains correct tracks."""
        exporter = M3UExporter(temp_output_dir, playlist_prefix="Set")
        chapter_boundaries = [(0, "Intro"), (2, "Main")]

        results = exporter.export_as_chapters(sample_tracks, sample_metadata, chapter_boundaries)

        # First file: tracks 0-1 (Intro)
        intro_content = results[0].read_text()
        assert "Track 1" in intro_content
        assert "Track 2" in intro_content

        # Second file: tracks 2-4 (Main)
        main_content = results[1].read_text()
        assert "Track 3" in main_content
        assert "Track 4" in main_content
        assert "Track 5" in main_content

    def test_export_as_chapters_sequential_naming(
        self,
        temp_output_dir: Path,
        sample_tracks: list[Path],
        sample_metadata: dict[Path, TrackMetadata],
    ) -> None:
        """Test that chapter files are named with sequential numbering."""
        exporter = M3UExporter(temp_output_dir, playlist_prefix="Set")
        chapter_boundaries = [(0, "Intro"), (2, "Build")]

        results = exporter.export_as_chapters(sample_tracks, sample_metadata, chapter_boundaries)

        assert results[0].name == "01_Intro.m3u"
        assert results[1].name == "02_Build.m3u"

    def test_export_as_chapters_no_boundaries_falls_back_to_playlist(
        self,
        temp_output_dir: Path,
        sample_tracks: list[Path],
        sample_metadata: dict[Path, TrackMetadata],
    ) -> None:
        """Test that no boundaries falls back to single playlist export."""
        exporter = M3UExporter(temp_output_dir, playlist_prefix="Set")

        results = exporter.export_as_chapters(sample_tracks, sample_metadata, None)

        assert len(results) == 1
        assert results[0].name == "set.m3u"

    def test_export_as_chapters_sanitizes_filename(
        self,
        temp_output_dir: Path,
        sample_tracks: list[Path],
        sample_metadata: dict[Path, TrackMetadata],
    ) -> None:
        """Test that special characters in chapter names are sanitized."""
        exporter = M3UExporter(temp_output_dir, playlist_prefix="Set")
        chapter_boundaries = [(0, "Main/Build"), (2, "Cool-Down")]

        results = exporter.export_as_chapters(sample_tracks, sample_metadata, chapter_boundaries)

        # Should replace / and - with underscores or remove them
        names = [p.name for p in results]
        for name in names:
            assert "/" not in name

    def test_export_as_chapters_includes_header_comment(
        self,
        temp_output_dir: Path,
        sample_tracks: list[Path],
        sample_metadata: dict[Path, TrackMetadata],
    ) -> None:
        """Test that each chapter file includes chapter name as comment."""
        exporter = M3UExporter(temp_output_dir, playlist_prefix="Set")
        chapter_boundaries = [(0, "Intro")]

        results = exporter.export_as_chapters(sample_tracks, sample_metadata, chapter_boundaries)

        content = results[0].read_text()
        assert "# --- Intro ---" in content or "# Intro" in content
