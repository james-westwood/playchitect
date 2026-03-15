"""
Integration tests for CLI weight flags (--weight-file and --learn-weights).

Covers:
  - --weight-file loads YAML weight overrides and exits 0
  - --no-learn-weights disables EWKM and exits 0
  - --help output includes both flags
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner, Result
from mutagen.flac import FLAC

from playchitect.cli.commands import scan

_SAMPLE_RATE = 44100
_DURATION_S = 0.5


def _write_flac(path: Path, bpm: float) -> None:
    """Write a 0.5-second silent stereo FLAC file with an embedded BPM tag."""
    samples = np.zeros((int(_SAMPLE_RATE * _DURATION_S), 2), dtype=np.float32)
    sf.write(str(path), samples, _SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    audio = FLAC(str(path))
    audio["bpm"] = str(int(round(bpm)))
    audio.save()


@pytest.fixture()
def flac_music_dir(tmp_path: Path) -> Path:
    """Create 8 synthetic FLAC files with varying BPMs."""
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    for i in range(8):
        bpm = 120.0 + i * 2  # 120, 122, 124, ..., 134
        _write_flac(music_dir / f"track_{i:02d}.flac", bpm)
    return music_dir


@pytest.fixture()
def weight_yaml_file(tmp_path: Path) -> Path:
    """Create a minimal YAML file with weight overrides."""
    yaml_path = tmp_path / "weights.yaml"
    yaml_content = """weights:
  bpm: 2.0
  rms_energy: 1.5
"""
    yaml_path.write_text(yaml_content)
    return yaml_path


class TestWeightFileFlag:
    """Tests for --weight-file option."""

    def _invoke(self, music_dir: Path, weight_file: Path | None = None) -> Result:
        runner = CliRunner()
        args = [str(music_dir), "--dry-run", "--target-tracks", "10"]
        if weight_file:
            args.extend(["--weight-file", str(weight_file)])
        return runner.invoke(scan, args)

    def test_weight_file_exits_zero(self, flac_music_dir: Path, weight_yaml_file: Path) -> None:
        """CLI with --weight-file should exit 0."""
        result = self._invoke(flac_music_dir, weight_yaml_file)
        assert result.exit_code == 0, (
            f"CLI exited with code {result.exit_code}\nOutput:\n{result.output}"
        )

    def test_weight_file_loads_message(self, flac_music_dir: Path, weight_yaml_file: Path) -> None:
        """CLI should indicate that weight overrides were loaded."""
        result = self._invoke(flac_music_dir, weight_yaml_file)
        assert "Loaded weight overrides" in result.output


class TestLearnWeightsFlag:
    """Tests for --learn-weights / --no-learn-weights option."""

    def _invoke(self, music_dir: Path, learn_weights: bool | None = None) -> Result:
        runner = CliRunner()
        args = [str(music_dir), "--dry-run", "--target-tracks", "10"]
        if learn_weights is False:
            args.append("--no-learn-weights")
        elif learn_weights is True:
            args.append("--learn-weights")
        return runner.invoke(scan, args)

    def test_no_learn_weights_exits_zero(self, flac_music_dir: Path) -> None:
        """CLI with --no-learn-weights should exit 0."""
        result = self._invoke(flac_music_dir, learn_weights=False)
        assert result.exit_code == 0, (
            f"CLI exited with code {result.exit_code}\nOutput:\n{result.output}"
        )

    def test_learn_weights_exits_zero(self, flac_music_dir: Path) -> None:
        """CLI with --learn-weights (explicit True) should exit 0."""
        result = self._invoke(flac_music_dir, learn_weights=True)
        assert result.exit_code == 0, (
            f"CLI exited with code {result.exit_code}\nOutput:\n{result.output}"
        )


class TestCombinedFlags:
    """Tests for combining --weight-file and --no-learn-weights."""

    def test_combined_flags_exits_zero(self, flac_music_dir: Path, weight_yaml_file: Path) -> None:
        """CLI with both --weight-file and --no-learn-weights should exit 0."""
        runner = CliRunner()
        args = [
            str(flac_music_dir),
            "--dry-run",
            "--target-tracks",
            "10",
            "--weight-file",
            str(weight_yaml_file),
            "--no-learn-weights",
        ]
        result = runner.invoke(scan, args)
        assert result.exit_code == 0, (
            f"CLI exited with code {result.exit_code}\nOutput:\n{result.output}"
        )


class TestHelpOutput:
    """Tests for --help output including new flags."""

    def test_help_includes_weight_file(self) -> None:
        """--help output should include --weight-file option."""
        runner = CliRunner()
        result = runner.invoke(scan, ["--help"])
        assert result.exit_code == 0
        assert "--weight-file" in result.output

    def test_help_includes_learn_weights(self) -> None:
        """--help output should include --learn-weights / --no-learn-weights option."""
        runner = CliRunner()
        result = runner.invoke(scan, ["--help"])
        assert result.exit_code == 0
        # Check for the positive flag (Click shows both when using / pattern)
        assert "--learn-weights" in result.output

    def test_help_includes_no_learn_weights(self) -> None:
        """--help output should include --no-learn-weights option."""
        runner = CliRunner()
        result = runner.invoke(scan, ["--help"])
        assert result.exit_code == 0
        assert "--no-learn-weights" in result.output
