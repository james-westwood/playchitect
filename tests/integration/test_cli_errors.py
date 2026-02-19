"""
Integration tests for `playchitect scan` error paths.

Each test exercises a scenario that should produce a non-zero exit code and a
clear Error: message — verifying the CLI fails gracefully rather than with an
unhandled exception or a silent wrong result.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner, Result
from mutagen.flac import FLAC

from playchitect.cli.commands import scan

_SAMPLE_RATE = 44100


def _write_flac(path: Path, bpm: float | None = None) -> None:
    """Write a silent FLAC file, optionally with an embedded BPM tag."""
    samples = np.zeros((int(_SAMPLE_RATE * 0.5), 2), dtype=np.float32)
    sf.write(str(path), samples, _SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    if bpm is not None:
        audio = FLAC(str(path))
        audio["bpm"] = str(int(round(bpm)))
        audio.save()


@pytest.fixture()
def small_bpm_dir(tmp_path: Path) -> Path:
    """Five BPM-tagged FLACs — enough to satisfy the clusterer."""
    for i in range(5):
        _write_flac(tmp_path / f"track_{i:02d}.flac", bpm=128.0)
    return tmp_path


def _invoke(args: list[str]) -> Result:
    return CliRunner().invoke(scan, args)


class TestScanErrorPaths:
    def test_empty_directory_exits_nonzero(self, tmp_path: Path) -> None:
        result = _invoke([str(tmp_path), "--dry-run", "--target-tracks", "5"])
        assert result.exit_code != 0
        assert "No audio files found" in result.output

    def test_no_bpm_tags_exits_nonzero(self, tmp_path: Path) -> None:
        """FLAC files without BPM tags must raise a clear error, not crash."""
        for i in range(3):
            _write_flac(tmp_path / f"no_bpm_{i}.flac")  # bpm=None → no tag written
        result = _invoke([str(tmp_path), "--dry-run", "--target-tracks", "5"])
        assert result.exit_code != 0
        assert "No tracks with BPM metadata found" in result.output

    def test_both_target_flags_are_mutually_exclusive(self, small_bpm_dir: Path) -> None:
        result = _invoke(
            [
                str(small_bpm_dir),
                "--dry-run",
                "--target-tracks",
                "10",
                "--target-duration",
                "30",
            ]
        )
        assert result.exit_code != 0
        assert "Specify either" in result.output

    def test_nonexistent_path_rejected_by_click(self) -> None:
        """click.Path(exists=True) exits with code 2 before our code runs."""
        result = _invoke(["/nonexistent/totally/fake/path", "--dry-run", "--target-tracks", "5"])
        assert result.exit_code == 2

    def test_missing_music_path_without_use_test_path(self) -> None:
        """Omitting MUSIC_PATH without --use-test-path must fail cleanly."""
        result = _invoke(["--dry-run", "--target-tracks", "5"])
        assert result.exit_code != 0
        assert "MUSIC_PATH is required" in result.output


class TestScanPartialData:
    def test_partial_bpm_tags_clusters_tagged_subset(self, tmp_path: Path) -> None:
        """
        When only some tracks have BPM tags, clustering should proceed on the
        tagged subset rather than aborting, as long as ≥2 tagged tracks exist.
        """
        for i in range(10):
            _write_flac(tmp_path / f"tagged_{i:02d}.flac", bpm=128.0)
        for i in range(5):
            _write_flac(tmp_path / f"untagged_{i:02d}.flac")  # no BPM tag

        result = _invoke([str(tmp_path), "--dry-run", "--target-tracks", "25"])

        assert result.exit_code == 0
        assert "Extracted BPM from 10/15 tracks" in result.output
        # Clustering must still produce playlists from the 10 tagged tracks
        assert "Would create" in result.output
