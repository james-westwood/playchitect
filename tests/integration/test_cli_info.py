"""
Integration tests for the `playchitect info` command.

Covers text output, JSON output, empty directories, and the file-type
breakdown — ensuring the info command is usable for scripting (--format json)
as well as interactive inspection (default text mode).
"""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner, Result

from playchitect.cli.commands import info

_SAMPLE_RATE = 44100


def _write_flac(path: Path) -> None:
    """Write a minimal silent FLAC (no tags needed for info command)."""
    samples = np.zeros((int(_SAMPLE_RATE * 0.5), 2), dtype=np.float32)
    sf.write(str(path), samples, _SAMPLE_RATE, format="FLAC", subtype="PCM_16")


@pytest.fixture()
def audio_dir(tmp_path: Path) -> Path:
    """
    Four FLAC files plus a non-audio file that the scanner must ignore.
    Verifies file-type filtering as well as extension counting.
    """
    for i in range(4):
        _write_flac(tmp_path / f"track_{i:02d}.flac")
    (tmp_path / "cover.jpg").write_bytes(b"\xff\xd8fake jpeg")
    return tmp_path


def _invoke(args: list[str]) -> Result:
    return CliRunner().invoke(info, args)


class TestInfoTextOutput:
    def test_exits_zero(self, audio_dir: Path) -> None:
        result = _invoke([str(audio_dir)])
        assert result.exit_code == 0

    def test_shows_directory_path(self, audio_dir: Path) -> None:
        result = _invoke([str(audio_dir)])
        assert str(audio_dir) in result.output

    def test_shows_correct_file_count(self, audio_dir: Path) -> None:
        result = _invoke([str(audio_dir)])
        assert "Total audio files: 4" in result.output

    def test_shows_flac_extension(self, audio_dir: Path) -> None:
        result = _invoke([str(audio_dir)])
        assert ".flac" in result.output

    def test_non_audio_files_excluded(self, audio_dir: Path) -> None:
        """The .jpg cover file must not be counted as an audio file."""
        result = _invoke([str(audio_dir)])
        assert "Total audio files: 4" in result.output
        assert ".jpg" not in result.output

    def test_empty_directory_shows_zero(self, tmp_path: Path) -> None:
        result = _invoke([str(tmp_path)])
        assert result.exit_code == 0
        assert "Total audio files: 0" in result.output


class TestInfoJsonOutput:
    def test_exits_zero(self, audio_dir: Path) -> None:
        result = _invoke([str(audio_dir), "--format", "json"])
        assert result.exit_code == 0

    def test_output_is_valid_json(self, audio_dir: Path) -> None:
        result = _invoke([str(audio_dir), "--format", "json"])
        # Must not raise
        json.loads(result.output)

    def test_json_has_required_keys(self, audio_dir: Path) -> None:
        result = _invoke([str(audio_dir), "--format", "json"])
        data = json.loads(result.output)
        assert "path" in data
        assert "total_files" in data
        assert "files" in data

    def test_json_file_count_matches(self, audio_dir: Path) -> None:
        result = _invoke([str(audio_dir), "--format", "json"])
        data = json.loads(result.output)
        assert data["total_files"] == 4

    def test_json_files_are_strings(self, audio_dir: Path) -> None:
        result = _invoke([str(audio_dir), "--format", "json"])
        data = json.loads(result.output)
        assert isinstance(data["files"], list)
        assert all(isinstance(f, str) for f in data["files"])

    def test_json_path_matches_input(self, audio_dir: Path) -> None:
        result = _invoke([str(audio_dir), "--format", "json"])
        data = json.loads(result.output)
        assert data["path"] == str(audio_dir)

    def test_empty_directory_json(self, tmp_path: Path) -> None:
        result = _invoke([str(tmp_path), "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total_files"] == 0
        assert data["files"] == []

    def test_short_flag_format(self, audio_dir: Path) -> None:
        """-f json must be equivalent to --format json."""
        result = _invoke([str(audio_dir), "-f", "json"])
        data = json.loads(result.output)
        assert data["total_files"] == 4
