"""
Integration tests for the `playchitect playlist` CLI command.

These tests define the expected contract for the seed-playlist CLI interface
(TASK-07 skeleton, TASK-08 wiring). Until the `playlist` subcommand is
implemented, Click's CliRunner returns exit_code=2 with
"No such command 'playlist'".

Tests 1-6 exercise input validation — they should start passing once the
TASK-07 command skeleton is merged. Tests 7-8 mock the heavy pipeline
and will only pass once TASK-08 wires the command to
generate_playlist_from_seed().
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from playchitect.cli.commands import cli
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_synth_metadata(p: Path) -> TrackMetadata:
    """Return a TrackMetadata with synthetic but realistic values."""
    return TrackMetadata(filepath=p, bpm=128.0, duration=300.0, title=p.stem)


def make_synth_features(p: Path) -> IntensityFeatures:
    """Return an IntensityFeatures with synthetic but realistic values."""
    return IntensityFeatures(
        file_path=p,
        file_hash="abc123",  # pragma: allowlist secret
        rms_energy=0.5,
        brightness=0.5,
        sub_bass_energy=0.3,
        kick_energy=0.4,
        bass_harmonics=0.3,
        percussiveness=0.6,
        onset_strength=0.5,
        camelot_key="8B",
        key_index=0.0,
    )


def _create_seed_file(tmp_path: Path, name: str = "seed.flac") -> Path:
    """Create an empty file to serve as the --seed path (passes Click's exists=True)."""
    seed = tmp_path / name
    seed.touch()
    return seed


def _create_music_dir(tmp_path: Path, n_tracks: int = 20) -> Path:
    """Create a music directory with n_tracks empty .flac stubs."""
    music_dir = tmp_path / "music"
    music_dir.mkdir()
    for i in range(n_tracks):
        (music_dir / f"track_{i:02d}.flac").touch()
    return music_dir


# ---------------------------------------------------------------------------
# Tests 1-6: Validation (should pass once TASK-07 skeleton lands)
# ---------------------------------------------------------------------------


class TestPlaylistValidation:
    """Input validation tests for the `playchitect playlist` subcommand."""

    def test_playlist_help_shows_command(self) -> None:
        """`playchitect playlist --help` exits 0 and lists --seed and --duration options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["playlist", "--help"])
        assert result.exit_code == 0, (
            f"Expected exit_code 0, got {result.exit_code}\nOutput:\n{result.output}"
        )
        assert "--seed" in result.output, f"'--seed' not found in help output:\n{result.output}"
        assert "--duration" in result.output, (
            f"'--duration' not found in help output:\n{result.output}"
        )

    def test_missing_seed_fails(self, tmp_path: Path) -> None:
        """Invoking playlist without --seed must fail with a non-zero exit code."""
        music_dir = _create_music_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["playlist", "--music-dir", str(music_dir), "--duration", "60"],
        )
        assert result.exit_code != 0, (
            f"Expected non-zero exit code when --seed is missing, "
            f"got {result.exit_code}\nOutput:\n{result.output}"
        )
        # Click usually says "Missing option: --seed" or similar
        output_lower = result.output.lower()
        assert "seed" in output_lower, f"Expected error about missing --seed, got:\n{result.output}"

    def test_missing_music_dir_fails(self, tmp_path: Path) -> None:
        """Invoking playlist without --music-dir must fail with an error mentioning it."""
        seed = _create_seed_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["playlist", "--seed", str(seed), "--duration", "60"],
        )
        assert result.exit_code != 0, (
            f"Expected non-zero exit code when --music-dir is missing, "
            f"got {result.exit_code}\nOutput:\n{result.output}"
        )
        # The error message should reference the missing option
        assert "music" in result.output.lower(), (
            f"Expected error about missing --music-dir, got:\n{result.output}"
        )

    def test_missing_duration_fails(self, tmp_path: Path) -> None:
        """Invoking playlist without --duration must fail with an error mentioning duration."""
        seed = _create_seed_file(tmp_path)
        music_dir = _create_music_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["playlist", "--seed", str(seed), "--music-dir", str(music_dir)],
        )
        assert result.exit_code != 0, (
            f"Expected non-zero exit code when --duration is missing, "
            f"got {result.exit_code}\nOutput:\n{result.output}"
        )
        assert "duration" in result.output.lower(), (
            f"Expected 'duration' in error output, got:\n{result.output}"
        )

    def test_zero_duration_fails(self, tmp_path: Path) -> None:
        """Providing --duration 0 must fail with an error mentioning 'duration'."""
        seed = _create_seed_file(tmp_path)
        music_dir = _create_music_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["playlist", "--seed", str(seed), "--music-dir", str(music_dir), "--duration", "0"],
        )
        assert result.exit_code != 0, (
            f"Expected non-zero exit code for --duration 0, "
            f"got {result.exit_code}\nOutput:\n{result.output}"
        )
        assert "duration" in result.output.lower(), (
            f"Expected 'duration' in error output, got:\n{result.output}"
        )

    def test_nonexistent_seed_fails(self, tmp_path: Path) -> None:
        """Providing a --seed path that doesn't exist on disk must fail."""
        music_dir = _create_music_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "playlist",
                "--seed",
                "/does/not/exist.mp3",
                "--music-dir",
                str(music_dir),
                "--duration",
                "60",
            ],
        )
        assert result.exit_code != 0, (
            f"Expected non-zero exit code for nonexistent --seed, "
            f"got {result.exit_code}\nOutput:\n{result.output}"
        )
        # Click's Path(exists=True) produces an error mentioning the path
        assert "/does/not/exist.mp3" in result.output or "exist" in result.output.lower(), (
            f"Expected error about nonexistent seed path, got:\n{result.output}"
        )


# ---------------------------------------------------------------------------
# Tests 7-8: Mock-based full flow (need TASK-08 wiring to pass)
# ---------------------------------------------------------------------------


class TestPlaylistFullFlow:
    """End-to-end tests with mocked analysis pipeline.

    These mock MetadataExtractor.extract_batch and IntensityAnalyzer.analyze_batch
    to avoid needing real audio files, then verify the playlist command produces
    an .m3u output file with correct content.
    """

    @patch("playchitect.core.intensity_analyzer.IntensityAnalyzer")
    @patch("playchitect.core.metadata_extractor.MetadataExtractor")
    def test_valid_run_creates_m3u(
        self,
        mock_meta_cls: MagicMock,
        mock_intensity_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A valid invocation with a real seed path should create an .m3u file
        containing the seed track path."""
        seed = _create_seed_file(tmp_path, name="seed_track.flac")
        music_dir = _create_music_dir(tmp_path, n_tracks=20)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Build synthetic track paths (the 20 files in music_dir + the seed)
        all_paths = sorted(music_dir.glob("*.flac"))
        # Add the seed to the metadata/features dicts since it must be in candidate_features
        all_paths_with_seed = all_paths + [seed]

        # Configure MetadataExtractor.extract_batch mock
        metadata_dict = {p: make_synth_metadata(p) for p in all_paths_with_seed}
        mock_meta_instance = MagicMock()
        mock_meta_instance.extract_batch.return_value = metadata_dict
        mock_meta_cls.return_value = mock_meta_instance

        # Configure IntensityAnalyzer.analyze_batch mock
        features_dict = {p: make_synth_features(p) for p in all_paths_with_seed}
        mock_intensity_instance = MagicMock()
        mock_intensity_instance.analyze_batch.return_value = features_dict
        mock_intensity_cls.return_value = mock_intensity_instance

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "playlist",
                "--seed",
                str(seed),
                "--music-dir",
                str(music_dir),
                "--duration",
                "60",
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, (
            f"Expected exit_code 0, got {result.exit_code}\nOutput:\n{result.output}"
        )

        # Verify an .m3u file was created in the output directory
        m3u_files = list(output_dir.glob("*.m3u"))
        assert len(m3u_files) >= 1, (
            f"Expected at least one .m3u file in {output_dir}, found: {m3u_files}"
        )

        # Verify the seed track path appears in the M3U content
        m3u_content = m3u_files[0].read_text(encoding="utf-8")
        assert str(seed) in m3u_content, (
            f"Seed path '{seed}' not found in M3U content:\n{m3u_content}"
        )

    @patch("playchitect.core.intensity_analyzer.IntensityAnalyzer")
    @patch("playchitect.core.metadata_extractor.MetadataExtractor")
    def test_sequence_option_accepted(
        self,
        mock_meta_cls: MagicMock,
        mock_intensity_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """The --sequence build option should be accepted and exit 0."""
        seed = _create_seed_file(tmp_path, name="seed_track.flac")
        music_dir = _create_music_dir(tmp_path, n_tracks=20)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        all_paths = sorted(music_dir.glob("*.flac"))
        all_paths_with_seed = all_paths + [seed]

        metadata_dict = {p: make_synth_metadata(p) for p in all_paths_with_seed}
        mock_meta_instance = MagicMock()
        mock_meta_instance.extract_batch.return_value = metadata_dict
        mock_meta_cls.return_value = mock_meta_instance

        features_dict = {p: make_synth_features(p) for p in all_paths_with_seed}
        mock_intensity_instance = MagicMock()
        mock_intensity_instance.analyze_batch.return_value = features_dict
        mock_intensity_cls.return_value = mock_intensity_instance

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "playlist",
                "--seed",
                str(seed),
                "--music-dir",
                str(music_dir),
                "--duration",
                "60",
                "--sequence",
                "build",
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, (
            f"Expected exit_code 0 with --sequence build, "
            f"got {result.exit_code}\nOutput:\n{result.output}"
        )

    @patch("playchitect.cli.commands.AudioScanner")
    def test_no_audio_files_shows_error(self, mock_scanner_cls: MagicMock, tmp_path: Path) -> None:
        """Empty scan result should print an error and exit non-zero."""
        seed = _create_seed_file(tmp_path)
        music_dir = _create_music_dir(tmp_path, n_tracks=0)
        # Remove the dir so it's truly empty when scanned
        for f in music_dir.iterdir():
            f.unlink()

        mock_scanner = MagicMock()
        mock_scanner.scan.return_value = []
        mock_scanner_cls.return_value = mock_scanner

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "playlist",
                "--seed",
                str(seed),
                "--music-dir",
                str(music_dir),
                "--duration",
                "60",
            ],
        )
        assert result.exit_code != 0
        assert "no audio files" in result.output.lower()

    @patch("playchitect.core.seed_playlist.generate_playlist_from_seed")
    @patch("playchitect.core.intensity_analyzer.IntensityAnalyzer")
    @patch("playchitect.core.metadata_extractor.MetadataExtractor")
    def test_value_error_from_generate_is_handled(
        self,
        mock_meta_cls: MagicMock,
        mock_intensity_cls: MagicMock,
        mock_generate: MagicMock,
        tmp_path: Path,
    ) -> None:
        """ValueError from generate_playlist_from_seed should print error and exit non-zero."""
        seed = _create_seed_file(tmp_path)
        music_dir = _create_music_dir(tmp_path, n_tracks=5)

        all_paths = sorted(music_dir.glob("*.flac")) + [seed]
        mock_meta_instance = MagicMock()
        mock_meta_instance.extract_batch.return_value = {
            p: make_synth_metadata(p) for p in all_paths
        }
        mock_meta_cls.return_value = mock_meta_instance
        mock_intensity_instance = MagicMock()
        mock_intensity_instance.analyze_batch.return_value = {
            p: make_synth_features(p) for p in all_paths
        }
        mock_intensity_cls.return_value = mock_intensity_instance
        mock_generate.side_effect = ValueError("not enough tracks")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "playlist",
                "--seed",
                str(seed),
                "--music-dir",
                str(music_dir),
                "--duration",
                "60",
            ],
        )
        assert result.exit_code != 0
        assert "not enough tracks" in result.output

    @patch("playchitect.core.intensity_analyzer.IntensityAnalyzer")
    @patch("playchitect.core.metadata_extractor.MetadataExtractor")
    def test_output_contains_summary(
        self,
        mock_meta_cls: MagicMock,
        mock_intensity_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Successful run should print 'Playlist saved to:', track count, and duration."""
        seed = _create_seed_file(tmp_path)
        music_dir = _create_music_dir(tmp_path, n_tracks=10)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        all_paths = sorted(music_dir.glob("*.flac")) + [seed]
        mock_meta_instance = MagicMock()
        mock_meta_instance.extract_batch.return_value = {
            p: make_synth_metadata(p) for p in all_paths
        }
        mock_meta_cls.return_value = mock_meta_instance
        mock_intensity_instance = MagicMock()
        mock_intensity_instance.analyze_batch.return_value = {
            p: make_synth_features(p) for p in all_paths
        }
        mock_intensity_cls.return_value = mock_intensity_instance

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "playlist",
                "--seed",
                str(seed),
                "--music-dir",
                str(music_dir),
                "--duration",
                "30",
                "--output",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        assert "Playlist saved to:" in result.output
        assert "tracks" in result.output
        assert "min" in result.output
