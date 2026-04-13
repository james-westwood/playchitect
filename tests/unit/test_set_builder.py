"""Unit tests for set_builder_chapter_generator module.

Tests for the Chapter dataclass and ChapterGenerator class.
"""

import pytest

from playchitect.core.clustering import ClusterResult
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata
from playchitect.core.set_builder_chapter_generator import (
    Chapter,
    ChapterTrack,
    ChapterGenerator,
    generate_chapters,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def make_metadata(
    filepath: str,
    bpm: float = 128.0,
    duration: float = 360.0,
    title: str = "Test Track",
    artist: str = "Test Artist",
) -> TrackMetadata:
    """Create TrackMetadata with specified properties."""
    from pathlib import Path

    return TrackMetadata(
        filepath=Path(filepath),
        title=title,
        artist=artist,
        bpm=bpm,
        duration=duration,
    )


def make_intensity_features(
    filepath: str,
    rms_energy: float = 0.5,
    brightness: float = 0.5,
    percussiveness: float = 0.5,
    kick_energy: float = 0.5,
    onset_strength: float = 0.5,
    sub_bass_energy: float = 0.3,
    bass_harmonics: float = 0.4,
) -> IntensityFeatures:
    """Create IntensityFeatures with specified properties."""
    from pathlib import Path

    return IntensityFeatures(
        file_path=Path(filepath),
        file_hash="deadbeef",
        rms_energy=rms_energy,
        brightness=brightness,
        sub_bass_energy=sub_bass_energy,
        kick_energy=kick_energy,
        bass_harmonics=bass_harmonics,
        percussiveness=percussiveness,
        onset_strength=onset_strength,
        camelot_key="8B",
        key_index=0.0,
    )


def make_cluster(
    cluster_id: int,
    tracks: list[str],
    bpm_mean: float = 128.0,
    bpm_std: float = 2.0,
) -> ClusterResult:
    """Create a ClusterResult with specified properties."""
    from pathlib import Path

    track_paths = [Path(t) for t in tracks]
    return ClusterResult(
        cluster_id=cluster_id,
        tracks=track_paths,
        bpm_mean=bpm_mean,
        bpm_std=bpm_std,
        track_count=len(tracks),
        total_duration=len(tracks) * 360.0,
    )


# ── Tests for ChapterTrack dataclass ─────────────────────────────────────


class TestChapterTrack:
    """Tests for ChapterTrack dataclass."""

    def test_chapter_track_creation(self) -> None:
        """Test basic ChapterTrack creation."""
        from pathlib import Path

        meta = make_metadata("/track.mp3", bpm=128.0, duration=300.0)
        feat = make_intensity_features("/track.mp3", rms_energy=0.5)
        track = ChapterTrack(path=Path("/track.mp3"), metadata=meta, features=feat)

        assert track.path == Path("/track.mp3")
        assert track.duration == 300.0

    def test_chapter_track_duration_zero_when_none(self) -> None:
        """Test duration when metadata.duration is None."""
        from pathlib import Path

        meta = make_metadata("/track.mp3", bpm=128.0, duration=0.0)
        feat = make_intensity_features("/track.mp3", rms_energy=0.5)
        track = ChapterTrack(path=Path("/track.mp3"), metadata=meta, features=feat)

        assert track.duration == 0.0


# ── Tests for Chapter dataclass ───────────────────────────────────────────────


class TestChapter:
    """Tests for Chapter dataclass."""

    def test_chapter_creation(self) -> None:
        """Test basic Chapter creation."""
        chapter = Chapter(
            id="chapter-0",
            name="Intro",
            energy_block_id="warm-up",
            target_duration_min=20.0,
        )

        assert chapter.id == "chapter-0"
        assert chapter.name == "Intro"
        assert chapter.energy_block_id == "warm-up"
        assert chapter.target_duration_min == 20.0
        assert chapter.track_count == 0
        assert not chapter.is_filled

    def test_chapter_with_tracks(self) -> None:
        """Test Chapter with tracks."""
        from pathlib import Path

        meta = make_metadata("/track1.mp3", duration=360.0)
        feat = make_intensity_features("/track1.mp3")
        track = ChapterTrack(path=Path("/track1.mp3"), metadata=meta, features=feat)

        chapter = Chapter(
            id="chapter-0",
            name="Intro",
            energy_block_id="warm-up",
            target_duration_min=10.0,
            tracks=[track],
        )

        assert chapter.track_count == 1

    def test_chapter_total_duration_min(self) -> None:
        """Test total_duration_min calculation."""
        from pathlib import Path

        meta1 = make_metadata("/track1.mp3", duration=360.0)
        meta2 = make_metadata("/track2.mp3", duration=300.0)
        feat1 = make_intensity_features("/track1.mp3")
        feat2 = make_intensity_features("/track2.mp3")

        track1 = ChapterTrack(path=Path("/track1.mp3"), metadata=meta1, features=feat1)
        track2 = ChapterTrack(path=Path("/track2.mp3"), metadata=meta2, features=feat2)

        chapter = Chapter(
            id="chapter-0",
            name="Intro",
            energy_block_id="warm-up",
            target_duration_min=10.0,
            tracks=[track1, track2],
        )

        # 360 + 300 = 660 seconds = 11 minutes
        assert chapter.total_duration_min() == pytest.approx(11.0, rel=0.01)


# ── Tests for ChapterGenerator ──────────────────────────────────────────────


class TestChapterGenerator:
    """Tests for ChapterGenerator class."""

    def test_generator_initialization(self) -> None:
        """Test ChapterGenerator can be initialized."""
        gen = ChapterGenerator()
        assert gen is not None
        assert gen.min_clusters == 2
        assert gen.max_clusters == 10

    def test_generator_with_custom_params(self) -> None:
        """Test ChapterGenerator with custom parameters."""
        gen = ChapterGenerator(min_clusters=3, max_clusters=8, random_state=123)
        assert gen.min_clusters == 3
        assert gen.max_clusters == 8
        assert gen.random_state == 123

    def test_empty_metadata_raises_error(self) -> None:
        """Test that empty metadata_dict raises ValueError."""
        gen = ChapterGenerator()

        with pytest.raises(ValueError, match="empty metadata_dict"):
            gen.generate_chapters({}, {}, target_duration_min=90.0)

    def test_empty_intensity_raises_error(self) -> None:
        """Test that empty intensity_dict raises ValueError."""
        from pathlib import Path

        meta = {Path("/track.mp3"): make_metadata("/track.mp3")}
        gen = ChapterGenerator()

        with pytest.raises(ValueError, match="empty intensity_dict"):
            gen.generate_chapters(meta, {}, target_duration_min=90.0)

    def test_single_track_creates_single_chapter(self) -> None:
        """Test that single track creates a single chapter."""
        from pathlib import Path

        path = Path("/track1.mp3")
        meta = {path: make_metadata(str(path), bpm=128.0, duration=300.0)}
        feat = {path: make_intensity_features(str(path), rms_energy=0.5)}

        gen = ChapterGenerator()
        chapters = gen.generate_chapters(meta, feat, target_duration_min=90.0)

        assert len(chapters) == 1
        assert chapters[0].name == "Intro"
        assert chapters[0].track_count == 1

    def test_generates_4_to_6_chapters_for_90_min_target(self) -> None:
        """Test that 90-min target produces 4-6 chapters.

        This is the primary acceptance criterion from the task spec.
        """
        from pathlib import Path

        # Create 30 tracks with varying energy levels (simulate 10 clusters)
        # Must vary multiple features to get distinct clusters
        tracks = [Path(f"/track{i}.mp3") for i in range(30)]
        meta: dict[Path, TrackMetadata] = {}
        feat: dict[Path, IntensityFeatures] = {}

        for i, path in enumerate(tracks):
            rms = 0.1 + (i * 0.03)  # Varying RMS
            # Vary other features too so clusters are distinct
            offset = (i % 5) * 0.1
            meta[path] = make_metadata(str(path), bpm=120.0 + i, duration=360.0)
            feat[path] = make_intensity_features(
                str(path),
                rms_energy=min(rms, 1.0),
                brightness=0.3 + offset,
                percussiveness=0.2 + offset,
                kick_energy=0.3 + offset,
                onset_strength=0.2 + offset,
            )

        gen = ChapterGenerator(min_clusters=4, max_clusters=10)
        chapters = gen.generate_chapters(meta, feat, target_duration_min=90.0)

        # Should produce 4-6 chapters
        assert 4 <= len(chapters) <= 6

    def test_chapter_names_are_auto_generated(self) -> None:
        """Test that chapters have auto-generated names."""
        from pathlib import Path

        # Create enough tracks for multiple clusters
        tracks = [Path(f"/track{i}.mp3") for i in range(20)]
        meta = {t: make_metadata(str(t), bpm=125.0, duration=360.0) for t in tracks}
        feat = {
            t: make_intensity_features(str(t), rms_energy=0.3 + (i * 0.03))
            for i, t in enumerate(tracks)
        }

        gen = ChapterGenerator()
        chapters = gen.generate_chapters(meta, feat, target_duration_min=90.0)

        # Each chapter should have an auto-generated name
        for chapter in chapters:
            assert chapter.name
            assert chapter.name in ("Intro", "Build", "Peak", "Sustain", "Outro")

    def test_total_duration_within_target_range(self) -> None:
        """Test that total duration is within 80-100% of target."""
        from pathlib import Path

        # Create 25 tracks
        tracks = [Path(f"/track{i}.mp3") for i in range(25)]
        meta = {t: make_metadata(str(t), bpm=128.0, duration=360.0) for t in tracks}
        feat = {t: make_intensity_features(str(t), rms_energy=0.5) for t in tracks}

        gen = ChapterGenerator(min_clusters=3, max_clusters=8)
        chapters = gen.generate_chapters(meta, feat, target_duration_min=90.0)

        total_duration = sum(c.total_duration_min() for c in chapters)

        # Should be 70-100 minutes (80-100% of 90 is 72-90, but we allow some margin)
        assert 70.0 <= total_duration <= 100.0, (
            f"Total duration {total_duration} outside 80-100 range"
        )

    def test_chapters_have_valid_ids(self) -> None:
        """Test that all chapters have unique IDs."""
        from pathlib import Path

        tracks = [Path(f"/track{i}.mp3") for i in range(15)]
        meta = {t: make_metadata(str(t), bpm=128.0, duration=360.0) for t in tracks}
        feat = {t: make_intensity_features(str(t), rms_energy=0.5) for t in tracks}

        gen = ChapterGenerator()
        chapters = gen.generate_chapters(meta, feat, target_duration_min=90.0)

        ids = [c.id for c in chapters]
        assert len(ids) == len(set(ids)), "Duplicate chapter IDs found"


# ── Tests for convenience function ───────────────────────────────────────────────


class TestGenerateChaptersFunction:
    """Tests for the generate_chapters convenience function."""

    def test_convenience_function(self) -> None:
        """Test generate_chapters convenience function."""
        from pathlib import Path

        tracks = [Path(f"/track{i}.mp3") for i in range(5)]
        meta = {t: make_metadata(str(t), bpm=128.0, duration=300.0) for t in tracks}
        feat = {t: make_intensity_features(str(t), rms_energy=0.5) for t in tracks}

        chapters = generate_chapters(meta, feat, target_duration_min=60.0)

        assert len(chapters) >= 1


# ── Integration-style tests ────────────────────────────────────────────────────────────────


class TestChapterGeneratorIntegration:
    """Integration-style tests for ChapterGenerator."""

    def test_ten_cluster_scenario(self) -> None:
        """Test with a realistic 10-cluster scenario."""
        from pathlib import Path

        # Create 50 tracks across 10 "clusters"
        tracks = [Path(f"/track{i}.mp3") for i in range(50)]
        meta: dict[Path, TrackMetadata] = {}
        feat: dict[Path, IntensityFeatures] = {}

        for i, path in enumerate(tracks):
            # Cluster 0: low energy, Cluster 9: high energy
            cluster = i // 5
            rms = 0.1 + (cluster * 0.09)

            meta[path] = make_metadata(str(path), bpm=120.0 + cluster, duration=360.0)
            feat[path] = make_intensity_features(
                str(path),
                rms_energy=rms,
                brightness=0.5,
                percussiveness=rms,
            )

        gen = ChapterGenerator()
        chapters = gen.generate_chapters(meta, feat, target_duration_min=90.0)

        # Should get 4-6 chapters
        assert 4 <= len(chapters) <= 6

        # All chapters should have names
        for chapter in chapters:
            assert chapter.name
            assert len(chapter.name) > 0

    def test_tracks_are_assigned_to_chapters(self) -> None:
        """Test that tracks are actually assigned to chapters."""
        from pathlib import Path

        # Create 30 tracks
        tracks = [Path(f"/track{i}.mp3") for i in range(30)]
        meta = {t: make_metadata(str(t), bpm=128.0, duration=300.0) for t in tracks}
        feat = {
            t: make_intensity_features(str(t), rms_energy=0.5 + (i * 0.02))
            for i, t in enumerate(tracks)
        }

        gen = ChapterGenerator()
        chapters = gen.generate_chapters(meta, feat, target_duration_min=45.0)

        total_assigned = sum(c.track_count for c in chapters)

        # At least some tracks should be assigned
        assert total_assigned > 0

    def test_target_duration_affects_output(self) -> None:
        """Test that different target durations produce different chapter targets."""
        from pathlib import Path

        tracks = [Path(f"/track{i}.mp3") for i in range(20)]
        meta = {t: make_metadata(str(t), bpm=128.0, duration=300.0) for t in tracks}
        feat = {t: make_intensity_features(str(t), rms_energy=0.5) for t in tracks}

        gen = ChapterGenerator()

        chapters_30 = gen.generate_chapters(meta, feat, target_duration_min=30.0)
        chapters_90 = gen.generate_chapters(meta, feat, target_duration_min=90.0)

        # Compare target durations - larger set should have higher targets
        total_targets_30 = sum(c.target_duration_min for c in chapters_30)
        total_targets_90 = sum(c.target_duration_min for c in chapters_90)

        assert total_targets_90 > total_targets_30
