"""
Integration tests for CLI weight-related flags (--weight-file, --learn-weights).

Tests that the scan command accepts and processes weight overrides from YAML
files and correctly handles the --learn-weights / --no-learn-weights toggle.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from click.testing import CliRunner, Result
from mutagen.flac import FLAC

from playchitect.cli.commands import scan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_RATE = 44100
_DURATION_S = 0.5


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
    10 synthetic FLAC files at 128 BPM for testing weight flags.
    """
    music_dir = tmp_path_factory.mktemp("weight_test_music")
    for i in range(10):
        _write_flac(music_dir / f"track_{i:02d}.flac", bpm=128.0)
    return music_dir


@pytest.fixture
def weight_yaml_file(tmp_path: Path) -> Path:
    """
    Create a minimal YAML weight overrides file.
    """
    weight_file = tmp_path / "weights.yaml"
    weight_data = {
        "weights": {
            "bpm": 2.0,
            "rms_energy": 1.5,
            "brightness": 1.0,
        }
    }
    with open(weight_file, "w", encoding="utf-8") as f:
        yaml.dump(weight_data, f)
    return weight_file


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCliWeightFlags:
    """Tests for --weight-file and --learn-weights flags."""

    def _invoke(
        self,
        music_dir: Path,
        target_tracks: int = 25,
        weight_file: Path | None = None,
        learn_weights: bool = True,
    ) -> Result:
        runner = CliRunner()
        args = [
            str(music_dir),
            "--dry-run",
            "--target-tracks",
            str(target_tracks),
            "--sequence-mode",
            "ramp",  # Force intensity analysis path to test weight flags
        ]
        if weight_file is not None:
            args.extend(["--weight-file", str(weight_file)])
        if not learn_weights:
            args.append("--no-learn-weights")
        return runner.invoke(scan, args)

    def test_weight_file_flag_exists(self, flac_music_dir: Path, weight_yaml_file: Path) -> None:
        """CLI should accept --weight-file flag and exit 0."""
        result = self._invoke(flac_music_dir, weight_file=weight_yaml_file)
        assert result.exit_code == 0, (
            f"CLI exited with code {result.exit_code}\nOutput:\n{result.output}"
        )

    def test_weight_file_loaded_message(self, flac_music_dir: Path, weight_yaml_file: Path) -> None:
        """CLI should report when weight overrides are loaded."""
        result = self._invoke(flac_music_dir, weight_file=weight_yaml_file)
        assert result.exit_code == 0
        assert "Loaded weight overrides from:" in result.output
        assert str(weight_yaml_file) in result.output

    def test_no_learn_weights_flag_exists(self, flac_music_dir: Path) -> None:
        """CLI should accept --no-learn-weights flag and exit 0."""
        result = self._invoke(flac_music_dir, learn_weights=False)
        assert result.exit_code == 0, (
            f"CLI exited with code {result.exit_code}\nOutput:\n{result.output}"
        )

    def test_both_flags_together(self, flac_music_dir: Path, weight_yaml_file: Path) -> None:
        """CLI should accept both --weight-file and --no-learn-weights together."""
        result = self._invoke(
            flac_music_dir,
            weight_file=weight_yaml_file,
            learn_weights=False,
        )
        assert result.exit_code == 0, (
            f"CLI exited with code {result.exit_code}\nOutput:\n{result.output}"
        )
        assert "Loaded weight overrides from:" in result.output

    def test_weight_file_invalid_path_fails(self, flac_music_dir: Path) -> None:
        """CLI should fail gracefully with non-existent weight file."""
        runner = CliRunner()
        args = [
            str(flac_music_dir),
            "--dry-run",
            "--target-tracks",
            "25",
            "--weight-file",
            "/nonexistent/path/weights.yaml",
        ]
        result = runner.invoke(scan, args)
        # Click should fail before our code runs due to exists=True validation
        assert result.exit_code != 0


def test_scan_help_includes_weight_flags() -> None:
    """Verify that --help output includes both new flags."""
    runner = CliRunner()
    result = runner.invoke(scan, ["--help"])
    assert result.exit_code == 0
    assert "--weight-file" in result.output
    assert "--learn-weights" in result.output or "--no-learn-weights" in result.output
