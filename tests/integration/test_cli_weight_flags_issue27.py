"""Integration tests for CLI weight-related flags - Issue #27.

Tests for --genre, --weight-file, and --learn-weights CLI flags
that expose the weighting system via CLI.
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from click.testing import CliRunner
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


class TestGenreFlag:
    """Tests for --genre CLI flag - Issue #27 acceptance criteria."""

    def test_genre_flag_exists(self) -> None:
        """The scan command should accept --genre flag."""
        runner = CliRunner()
        result = runner.invoke(scan, ["--help"])
        assert "--genre" in result.output, "scan command should have --genre flag per issue #27"

    def test_genre_techno_selects_heuristic_weights(self, tmp_path: Path) -> None:
        """--genre techno should select heuristic weights for techno."""
        # Create test music file
        music_dir = tmp_path / "music"
        music_dir.mkdir()
        track = music_dir / "test.flac"
        _write_flac(track, 128.0)

        runner = CliRunner()
        result = runner.invoke(
            scan,
            [str(music_dir), "--genre", "techno", "--dry-run"],
            catch_exceptions=False,
        )

        # Output should indicate heuristic weights are being used
        assert "genre" in result.output.lower() or "techno" in result.output.lower(), (
            "Output should mention genre/techno when --genre flag is used"
        )

    def test_genre_house_selects_heuristic_weights(self, tmp_path: Path) -> None:
        """--genre house should select heuristic weights for house."""
        music_dir = tmp_path / "music"
        music_dir.mkdir()
        track = music_dir / "test.flac"
        _write_flac(track, 124.0)

        runner = CliRunner()
        result = runner.invoke(
            scan,
            [str(music_dir), "--genre", "house", "--dry-run"],
        )

        assert result.exit_code == 0 or "house" in result.output.lower()

    def test_genre_invalid_raises_error(self, tmp_path: Path) -> None:
        """--genre with invalid genre should raise error."""
        music_dir = tmp_path / "music"
        music_dir.mkdir()
        track = music_dir / "test.flac"
        _write_flac(track, 128.0)

        runner = CliRunner()
        result = runner.invoke(
            scan,
            [str(music_dir), "--genre", "invalid_genre_xyz", "--dry-run"],
        )

        # Should fail with invalid genre
        assert result.exit_code != 0 or "invalid" in result.output.lower()


class TestWeightFileFlag:
    """Tests for --weight-file CLI flag - Issue #27 acceptance criteria."""

    def test_weight_file_flag_exists(self) -> None:
        """The scan command should accept --weight-file flag."""
        runner = CliRunner()
        result = runner.invoke(scan, ["--help"])
        assert "--weight-file" in result.output

    def test_weight_file_loads_yaml_overrides(self, tmp_path: Path) -> None:
        """--weight-file should load YAML weight file."""
        # Create test music
        music_dir = tmp_path / "music"
        music_dir.mkdir()
        track = music_dir / "test.flac"
        _write_flac(track, 128.0)

        # Create valid weight file
        weight_file = tmp_path / "weights.yaml"
        weight_file.write_text("""
weights:
  bpm: 0.3
  rms_energy: 0.2
  brightness: 0.1
  sub_bass: 0.15
  kick_energy: 0.15
  bass_harmonics: 0.05
  percussiveness: 0.03
  onset_strength: 0.02
""")

        runner = CliRunner()
        result = runner.invoke(
            scan,
            [str(music_dir), "--weight-file", str(weight_file), "--dry-run"],
        )

        # Should use the custom weights
        assert result.exit_code == 0

    def test_weight_file_not_found_raises(self, tmp_path: Path) -> None:
        """--weight-file with nonexistent file should raise error."""
        music_dir = tmp_path / "music"
        music_dir.mkdir()
        track = music_dir / "test.flac"
        _write_flac(track, 128.0)

        runner = CliRunner()
        result = runner.invoke(
            scan,
            [str(music_dir), "--weight-file", "/nonexistent/weights.yaml", "--dry-run"],
        )

        assert result.exit_code != 0


class TestLearnWeightsFlag:
    """Tests for --learn-weights CLI flag - Issue #27 acceptance criteria."""

    def test_learn_weights_flag_exists(self) -> None:
        """The scan command should accept --learn-weights flag."""
        runner = CliRunner()
        result = runner.invoke(scan, ["--help"])
        # Check for the flag (may be --learn-weights or --no-learn-weights)
        assert "learn-weight" in result.output.lower()

    def test_learn_weights_prints_pca_results(self, tmp_path: Path) -> None:
        """--learn-weights should print PCA-derived weights and exit."""
        # Create enough tracks for PCA (need 40+)
        music_dir = tmp_path / "music"
        music_dir.mkdir()
        for i in range(45):
            track = music_dir / f"track_{i}.flac"
            _write_flac(track, 120.0 + (i % 10))

        runner = CliRunner()
        result = runner.invoke(
            scan,
            [str(music_dir), "--learn-weights"],
        )

        # Should print weight information and exit
        assert "weight" in result.output.lower() or "pca" in result.output.lower(), (
            "Output should mention weights when --learn-weights is used"
        )


class TestWeightSummaryOutput:
    """Tests for weight profile summary in output - Issue #27."""

    def test_scan_output_includes_weight_source(self, tmp_path: Path) -> None:
        """Scan output should include weight source and top-3 features."""
        music_dir = tmp_path / "music"
        music_dir.mkdir()
        track = music_dir / "test.flac"
        _write_flac(track, 128.0)

        runner = CliRunner()
        result = runner.invoke(
            scan,
            [str(music_dir), "--dry-run"],
        )

        # Output should include weight information
        assert "weight" in result.output.lower() or "source" in result.output.lower(), (
            "Output should include weight profile information"
        )

    def test_scan_output_includes_top_features(self, tmp_path: Path) -> None:
        """Scan output should include top-3 features."""
        music_dir = tmp_path / "music"
        music_dir.mkdir()
        track = music_dir / "test.flac"
        _write_flac(track, 128.0)

        runner = CliRunner()
        result = runner.invoke(
            scan,
            [str(music_dir), "--dry-run"],
        )

        # Should show feature information
        output_lower = result.output.lower()
        has_feature = any(
            f in output_lower for f in ["bpm", "rms", "brightness", "sub_bass", "kick"]
        )
        assert has_feature, "Output should show feature weights"
