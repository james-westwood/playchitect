"""Unit tests for CueGenerator and CueSheet.

Tests cover:
- CueSheet.render() golden-file-style output matching
- CueGenerator.generate(): timing offsets, metadata fallback, title/performer defaults
- CueGenerator.generate_per_track(): one sheet per track at 00:00:00
- CUEExporter.export_cluster(): file written with correct content
- Missing metadata fallback paths
"""

from __future__ import annotations

from pathlib import Path

import pytest

from playchitect.core.clustering import ClusterResult
from playchitect.core.cue_generator import CueGenerator, CueSheet, CueTrack
from playchitect.core.export import CUEExporter
from playchitect.core.metadata_extractor import TrackMetadata

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_track_meta(
    filepath: Path,
    title: str = "Track",
    artist: str = "Artist",
    bpm: float = 128.0,
    duration: float = 300.0,
) -> TrackMetadata:
    return TrackMetadata(
        filepath=filepath,
        title=title,
        artist=artist,
        bpm=bpm,
        duration=duration,
    )


def _make_cluster(
    tracks: list[Path],
    bpm_mean: float = 128.0,
    bpm_std: float = 2.0,
    genre: str | None = None,
) -> ClusterResult:
    total = 0.0  # not used by CueGenerator; set to 0
    return ClusterResult(
        cluster_id=1,
        tracks=tracks,
        bpm_mean=bpm_mean,
        bpm_std=bpm_std,
        track_count=len(tracks),
        total_duration=total,
        genre=genre,
    )


# ── CueSheet.render() ─────────────────────────────────────────────────────────


class TestCueSheetRender:
    def test_basic_render(self) -> None:
        sheet = CueSheet(
            performer="Various Artists",
            title="Techno Mix 1",
            file_ref="playlist.m3u",
            file_type="MP3",
            tracks=[
                CueTrack(1, Path("/music/a.mp3"), "Alpha", "DJ A", 128.0, "00:00:00"),
                CueTrack(2, Path("/music/b.mp3"), "Beta", "DJ B", 130.0, "05:00:00"),
            ],
        )
        rendered = sheet.render()
        lines = rendered.strip().splitlines()

        assert lines[0] == 'PERFORMER "Various Artists"'
        assert lines[1] == 'TITLE "Techno Mix 1"'
        assert lines[2] == 'FILE "playlist.m3u" MP3'
        assert lines[3] == "  TRACK 01 AUDIO"
        assert lines[4] == '    TITLE "Alpha"'
        assert lines[5] == '    PERFORMER "DJ A"'
        assert lines[6] == "    REM BPM 128"
        assert lines[7] == "    INDEX 01 00:00:00"
        assert lines[8] == "  TRACK 02 AUDIO"
        assert lines[9] == '    TITLE "Beta"'
        assert lines[10] == '    PERFORMER "DJ B"'
        assert lines[11] == "    REM BPM 130"
        assert lines[12] == "    INDEX 01 05:00:00"

    def test_zero_bpm_omits_rem_line(self) -> None:
        sheet = CueSheet(
            performer="VA",
            title="Mix",
            file_ref="mix.m3u",
            file_type="MP3",
            tracks=[CueTrack(1, Path("/a.mp3"), "Untitled", "Unknown", 0.0, "00:00:00")],
        )
        assert "REM BPM" not in sheet.render()

    def test_track_numbers_zero_padded(self) -> None:
        tracks = [
            CueTrack(i, Path(f"/t{i}.mp3"), f"T{i}", "A", 128.0, "00:00:00") for i in range(1, 12)
        ]
        sheet = CueSheet("VA", "Mix", "mix.m3u", "MP3", tracks)
        rendered = sheet.render()
        assert "  TRACK 01 AUDIO" in rendered
        assert "  TRACK 11 AUDIO" in rendered

    def test_render_ends_with_newline(self) -> None:
        sheet = CueSheet("VA", "T", "f.m3u", "MP3", [])
        assert sheet.render().endswith("\n")


# ── CueGenerator.generate() ───────────────────────────────────────────────────


class TestCueGeneratorGenerate:
    def test_first_track_starts_at_zero(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.mp3"
        cluster = _make_cluster([p1])
        meta = {p1: _make_track_meta(p1, duration=300.0)}

        gen = CueGenerator()
        sheet = gen.generate(cluster, meta)

        assert sheet.tracks[0].index_time == "00:00:00"

    def test_second_track_offset(self, tmp_path: Path) -> None:
        p1, p2 = tmp_path / "a.mp3", tmp_path / "b.mp3"
        cluster = _make_cluster([p1, p2])
        meta = {
            p1: _make_track_meta(p1, duration=300.0),  # 5:00 → second track at 05:00:00
            p2: _make_track_meta(p2, duration=200.0),
        }

        gen = CueGenerator()
        sheet = gen.generate(cluster, meta)

        assert sheet.tracks[0].index_time == "00:00:00"
        assert sheet.tracks[1].index_time == "05:00:00"

    def test_fractional_seconds_converted_to_frames(self, tmp_path: Path) -> None:
        p1, p2 = tmp_path / "a.mp3", tmp_path / "b.mp3"
        cluster = _make_cluster([p1, p2])
        # 300.4 seconds → 300 * 75 + round(0.4 * 75) = 22500 + 30 = 22530 frames
        # = 300 s 30 frames → "05:00:30"
        meta = {
            p1: _make_track_meta(p1, duration=300.4),
            p2: _make_track_meta(p2, duration=200.0),
        }

        gen = CueGenerator()
        sheet = gen.generate(cluster, meta)

        assert sheet.tracks[1].index_time == "05:00:30"

    def test_missing_metadata_uses_filename_stem(self, tmp_path: Path) -> None:
        p1 = tmp_path / "my_awesome_track.mp3"
        cluster = _make_cluster([p1])
        meta: dict[Path, TrackMetadata] = {}

        gen = CueGenerator()
        sheet = gen.generate(cluster, meta)

        assert sheet.tracks[0].title == "my_awesome_track"
        assert sheet.tracks[0].performer == "Unknown"
        assert sheet.tracks[0].bpm == 0.0

    def test_default_title_uses_cluster_id(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.mp3"
        cluster = _make_cluster([p1])
        meta = {p1: _make_track_meta(p1)}

        gen = CueGenerator()
        sheet = gen.generate(cluster, meta)

        assert sheet.title == "Cluster 1"

    def test_custom_title_used(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.mp3"
        cluster = _make_cluster([p1])
        meta = {p1: _make_track_meta(p1)}

        gen = CueGenerator()
        sheet = gen.generate(cluster, meta, title="Techno Set A")

        assert sheet.title == "Techno Set A"

    def test_track_metadata_preserved(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.mp3"
        cluster = _make_cluster([p1])
        meta = {p1: _make_track_meta(p1, title="Dark Matter", artist="Surgeon", bpm=139.0)}

        gen = CueGenerator()
        sheet = gen.generate(cluster, meta)
        t = sheet.tracks[0]

        assert t.title == "Dark Matter"
        assert t.performer == "Surgeon"
        assert t.bpm == pytest.approx(139.0)

    def test_empty_cluster_produces_no_tracks(self, tmp_path: Path) -> None:
        cluster = _make_cluster([])
        gen = CueGenerator()
        sheet = gen.generate(cluster, {})
        assert sheet.tracks == []

    def test_track_count_matches_cluster(self, tmp_path: Path) -> None:
        paths = [tmp_path / f"t{i}.mp3" for i in range(5)]
        cluster = _make_cluster(paths)
        meta = {p: _make_track_meta(p, duration=240.0) for p in paths}

        gen = CueGenerator()
        sheet = gen.generate(cluster, meta)

        assert len(sheet.tracks) == 5

    def test_track_numbers_are_sequential(self, tmp_path: Path) -> None:
        paths = [tmp_path / f"t{i}.mp3" for i in range(3)]
        cluster = _make_cluster(paths)
        meta = {p: _make_track_meta(p) for p in paths}

        gen = CueGenerator()
        sheet = gen.generate(cluster, meta)

        assert [t.number for t in sheet.tracks] == [1, 2, 3]


# ── CueGenerator.generate_per_track() ────────────────────────────────────────


class TestCueGeneratorPerTrack:
    def test_one_sheet_per_track(self, tmp_path: Path) -> None:
        paths = [tmp_path / f"t{i}.mp3" for i in range(3)]
        cluster = _make_cluster(paths)
        meta = {p: _make_track_meta(p) for p in paths}

        gen = CueGenerator()
        sheets = gen.generate_per_track(cluster, meta)

        assert len(sheets) == 3

    def test_each_sheet_has_single_track_at_zero(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.flac"
        cluster = _make_cluster([p1])
        meta = {p1: _make_track_meta(p1, duration=300.0)}

        gen = CueGenerator()
        sheets = gen.generate_per_track(cluster, meta)

        assert len(sheets[0].tracks) == 1
        assert sheets[0].tracks[0].index_time == "00:00:00"
        assert sheets[0].tracks[0].number == 1

    def test_file_ref_is_track_filename(self, tmp_path: Path) -> None:
        p1 = tmp_path / "song.flac"
        cluster = _make_cluster([p1])
        meta = {p1: _make_track_meta(p1)}

        gen = CueGenerator()
        sheets = gen.generate_per_track(cluster, meta)

        assert sheets[0].file_ref == "song.flac"

    def test_flac_file_type_is_wave(self, tmp_path: Path) -> None:
        p1 = tmp_path / "song.flac"
        cluster = _make_cluster([p1])
        meta = {p1: _make_track_meta(p1)}

        gen = CueGenerator()
        sheets = gen.generate_per_track(cluster, meta)

        assert sheets[0].file_type == "WAVE"

    def test_mp3_file_type(self, tmp_path: Path) -> None:
        p1 = tmp_path / "song.mp3"
        cluster = _make_cluster([p1])
        meta = {p1: _make_track_meta(p1)}

        gen = CueGenerator()
        sheets = gen.generate_per_track(cluster, meta)

        assert sheets[0].file_type == "MP3"

    def test_missing_metadata_uses_stem_and_fallback_performer(self, tmp_path: Path) -> None:
        p1 = tmp_path / "lost_track.mp3"
        cluster = _make_cluster([p1])

        gen = CueGenerator()
        sheets = gen.generate_per_track(cluster, {}, performer="Various Artists")

        assert sheets[0].tracks[0].title == "lost_track"
        assert sheets[0].tracks[0].performer == "Various Artists"
        assert sheets[0].tracks[0].bpm == 0.0


# ── CUEExporter ───────────────────────────────────────────────────────────────


class TestCUEExporter:
    def test_creates_cue_file(self, tmp_path: Path) -> None:
        p1 = tmp_path / "music" / "a.mp3"
        p1.parent.mkdir()
        cluster = _make_cluster([p1])
        meta = {p1: _make_track_meta(p1, duration=300.0)}

        exporter = CUEExporter(tmp_path / "output")
        path = exporter.export_cluster(cluster, cluster_index=0, metadata_dict=meta)

        assert path.exists()
        assert path.suffix == ".cue"

    def test_cue_file_contains_performer_and_title(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.mp3"
        cluster = _make_cluster([p1])
        meta = {p1: _make_track_meta(p1, duration=300.0)}

        exporter = CUEExporter(tmp_path, playlist_prefix="Mix")
        path = exporter.export_cluster(cluster, cluster_index=0, metadata_dict=meta)

        content = path.read_text(encoding="utf-8")
        assert "PERFORMER" in content
        assert "TITLE" in content
        assert "INDEX 01 00:00:00" in content

    def test_export_clusters_returns_multiple_paths(self, tmp_path: Path) -> None:
        paths = [tmp_path / f"t{i}.mp3" for i in range(3)]
        clusters = [_make_cluster([p]) for p in paths]
        meta = {p: _make_track_meta(p, duration=200.0) for p in paths}

        exporter = CUEExporter(tmp_path)
        cue_paths = exporter.export_clusters(clusters, metadata_dict=meta)

        assert len(cue_paths) == 3
        assert all(p.exists() for p in cue_paths)

    def test_export_without_metadata(self, tmp_path: Path) -> None:
        p1 = tmp_path / "track.mp3"
        cluster = _make_cluster([p1])

        exporter = CUEExporter(tmp_path)
        path = exporter.export_cluster(cluster, metadata_dict=None)

        content = path.read_text(encoding="utf-8")
        # Filename stem used as title
        assert "track" in content

    def test_cue_filename_includes_bpm_range(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.mp3"
        cluster = _make_cluster([p1], bpm_mean=130.0, bpm_std=3.0)
        meta = {p1: _make_track_meta(p1)}

        exporter = CUEExporter(tmp_path, playlist_prefix="Set")
        path = exporter.export_cluster(cluster, cluster_index=0, metadata_dict=meta)

        assert "130-133bpm" in path.name

    def test_genre_label_in_filename(self, tmp_path: Path) -> None:
        p1 = tmp_path / "a.mp3"
        cluster = _make_cluster([p1], genre="techno")
        meta = {p1: _make_track_meta(p1)}

        exporter = CUEExporter(tmp_path)
        path = exporter.export_cluster(cluster, metadata_dict=meta)

        assert "techno" in path.name

    def test_second_track_has_nonzero_index(self, tmp_path: Path) -> None:
        p1, p2 = tmp_path / "a.mp3", tmp_path / "b.mp3"
        cluster = _make_cluster([p1, p2])
        meta = {
            p1: _make_track_meta(p1, duration=300.0),
            p2: _make_track_meta(p2, duration=200.0),
        }

        exporter = CUEExporter(tmp_path)
        path = exporter.export_cluster(cluster, metadata_dict=meta)

        content = path.read_text(encoding="utf-8")
        assert "INDEX 01 00:00:00" in content
        assert "INDEX 01 05:00:00" in content
