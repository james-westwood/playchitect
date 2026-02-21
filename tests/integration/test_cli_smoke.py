"""
CLI smoke test — runs a --dry-run scan against synthetic FLAC fixtures.

Uses Click's CliRunner (in-process) with tiny silent FLAC files created
via soundfile + mutagen. No real music or network access is required.

Sensible-output assertions:
  - Exit code 0
  - "DRY RUN" appears in output
  - All 20 synthetic tracks are found
  - At least 1 playlist is previewed
  - No "Error:" lines (those precede sys.exit(1))
  - No sentinel score -1.00 (indicates track missing from intensity_dict)
"""

import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner, Result
from mutagen.flac import FLAC

from playchitect.cli.commands import scan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 44100
_DURATION_S = 0.5  # short enough to keep fixture creation fast


def _write_flac(path: Path, bpm: float) -> None:
    """Write a 0.5-second silent stereo FLAC file with an embedded BPM tag."""
    samples = np.zeros((int(_SAMPLE_RATE * _DURATION_S), 2), dtype=np.float32)
    sf.write(str(path), samples, _SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    audio = FLAC(str(path))
    audio["bpm"] = str(int(round(bpm)))
    audio.save()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flac_music_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    20 synthetic FLAC files split into two clear BPM groups:
      - 10 tracks at 128 BPM  (group A)
      - 10 tracks at 140 BPM  (group B)

    Written once per module to keep the test suite fast.
    """
    music_dir = tmp_path_factory.mktemp("smoke_music")
    for i in range(10):
        _write_flac(music_dir / f"low_{i:02d}.flac", bpm=128.0)
    for i in range(10):
        _write_flac(music_dir / f"high_{i:02d}.flac", bpm=140.0)
    return music_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCliDryRun:
    """Smoke tests for `playchitect scan --dry-run`."""

    def _invoke(self, music_dir: Path, target_tracks: int = 25) -> Result:
        runner = CliRunner()
        args = [str(music_dir), "--dry-run", "--target-tracks", str(target_tracks)]
        return runner.invoke(scan, args)

    # --- basic sanity ---

    def test_exits_zero(self, flac_music_dir: Path) -> None:
        result = self._invoke(flac_music_dir)
        assert result.exit_code == 0, (
            f"CLI exited with code {result.exit_code}\nOutput:\n{result.output}"
        )

    def test_mentions_dry_run(self, flac_music_dir: Path) -> None:
        result = self._invoke(flac_music_dir)
        assert "DRY RUN" in result.output

    def test_finds_all_twenty_tracks(self, flac_music_dir: Path) -> None:
        result = self._invoke(flac_music_dir)
        assert "Found 20 audio files" in result.output

    def test_extracts_bpm_from_all_tracks(self, flac_music_dir: Path) -> None:
        result = self._invoke(flac_music_dir)
        assert "Extracted BPM from 20/20 tracks" in result.output

    # --- playlist output ---

    def test_creates_at_least_one_playlist(self, flac_music_dir: Path) -> None:
        result = self._invoke(flac_music_dir)
        match = re.search(r"Would create (\d+) playlists", result.output)
        assert match is not None, (
            f"'Would create N playlists' not found in output:\n{result.output}"
        )
        assert int(match.group(1)) >= 1

    def test_playlist_count_is_sensible(self, flac_music_dir: Path) -> None:
        """With 20 tracks and --target-tracks 25, we expect 1–4 playlists (not 20)."""
        result = self._invoke(flac_music_dir)
        match = re.search(r"Would create (\d+) playlists", result.output)
        assert match is not None
        n = int(match.group(1))
        assert 1 <= n <= 4, f"Unexpected playlist count: {n}\nOutput:\n{result.output}"

    # --- quality guards ---

    def test_no_error_lines(self, flac_music_dir: Path) -> None:
        """CLI must not print any 'Error:' lines — those precede sys.exit(1)."""
        result = self._invoke(flac_music_dir)
        error_lines = [ln for ln in result.output.splitlines() if ln.startswith("Error:")]
        assert not error_lines, "Unexpected Error line(s):\n" + "\n".join(error_lines)

    def test_no_negative_sentinel_scores(self, flac_music_dir: Path) -> None:
        """score: -1.00 is the sentinel for tracks missing from intensity_dict."""
        result = self._invoke(flac_music_dir)
        assert "score: -1.00" not in result.output

    # --- cluster splitting ---

    def test_cluster_splitting_with_small_target(self, flac_music_dir: Path) -> None:
        """With --target-tracks 5 and 20 tracks the split message must appear."""
        result = self._invoke(flac_music_dir, target_tracks=5)
        assert result.exit_code == 0, f"CLI exited {result.exit_code}\n{result.output}"
        # After splitting, we should have more playlists than natural BPM clusters
        match = re.search(r"Would create (\d+) playlists", result.output)
        assert match is not None
        assert int(match.group(1)) >= 2

    # --- sequencing ---

    def test_sequence_mode_ramp(
        self, flac_music_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that --sequence-mode ramp runs without error."""
        from playchitect.core.metadata_extractor import MetadataExtractor

        # Mock extract to avoid metadata extraction errors on tiny files
        original_extract = MetadataExtractor.extract

        def mock_extract(self, filepath):
            meta = original_extract(self, filepath)
            meta.bpm = 128.0
            return meta

        monkeypatch.setattr(MetadataExtractor, "extract", mock_extract)

        runner = CliRunner()
        args = [
            str(flac_music_dir),
            "--dry-run",
            "--target-tracks",
            "20",
            "--sequence-mode",
            "ramp",
        ]
        result = runner.invoke(scan, args)
        assert result.exit_code == 0
        assert "Sequencing tracks (mode: ramp)..." in result.output

    def test_target_duration_splitting(
        self, flac_music_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that --target-duration triggers cluster splitting."""
        from playchitect.core.metadata_extractor import MetadataExtractor

        # Mock extract to return fixed duration so splitting is predictable
        original_extract = MetadataExtractor.extract

        def mock_extract(self, filepath):
            meta = original_extract(self, filepath)
            meta.duration = 10.0  # 10s per track
            return meta

        monkeypatch.setattr(MetadataExtractor, "extract", mock_extract)

        runner = CliRunner()
        # 20 tracks, total 200s (3.33 min). Target 1 min should split each 10-track cluster into 2.
        args = [str(flac_music_dir), "--dry-run", "--target-duration", "1"]
        result = runner.invoke(scan, args)
        assert result.exit_code == 0
        assert "to meet target size" in result.output
        assert "Created 4 playlists" in result.output

    def test_bpm_only_clustering_no_intensity(
        self, flac_music_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify clustering works when intensity analysis is skipped (default mode)."""
        from playchitect.core.metadata_extractor import MetadataExtractor

        # Mock extract to avoid metadata extraction errors on tiny files
        original_extract = MetadataExtractor.extract

        def mock_extract(self, filepath):
            meta = original_extract(self, filepath)
            meta.bpm = 128.0
            return meta

        monkeypatch.setattr(MetadataExtractor, "extract", mock_extract)

        runner = CliRunner()
        # Default sequence-mode is now 'fixed', which skips intensity analysis
        args = [str(flac_music_dir), "--dry-run", "--target-tracks", "20"]
        result = runner.invoke(scan, args)
        assert result.exit_code == 0
        assert "Extracted BPM from 20/20 tracks" in result.output
        assert "Clustering tracks..." in result.output
        # Verify intensity analysis was NOT mentioned
        assert "Extracting audio intensity features..." not in result.output

    def test_clustering_failed_error(
        self, flac_music_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify CLI error message when clustering returns no results."""
        from playchitect.core.clustering import PlaylistClusterer

        monkeypatch.setattr(PlaylistClusterer, "cluster_by_bpm", lambda *args, **kwargs: [])

        runner = CliRunner()
        args = [str(flac_music_dir), "--dry-run", "--target-tracks", "20"]
        result = runner.invoke(scan, args)
        assert result.exit_code != 0
        assert "Error: Clustering failed" in result.output
