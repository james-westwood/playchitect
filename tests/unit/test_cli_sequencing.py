"""
Unit tests for CLI sequencing integration in playchitect.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from playchitect.cli.commands import scan
from playchitect.core.clustering import ClusterResult
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata

FAKE_PATHS = [Path(f"/music/track_{i}.flac") for i in range(3)]


def make_fake_metadata(p: Path) -> TrackMetadata:
    """Create fake metadata for testing."""
    return TrackMetadata(filepath=p, bpm=128.0, duration=300.0)


def make_fake_features(p: Path) -> IntensityFeatures:
    """Create fake intensity features for testing."""
    return IntensityFeatures(
        filepath=p,
        file_hash="abc",
        rms_energy=0.5,
        brightness=0.5,
        sub_bass_energy=0.3,
        kick_energy=0.4,
        bass_harmonics=0.3,
        percussiveness=0.6,
        onset_strength=0.5,
    )


def make_fake_cluster() -> ClusterResult:
    """Create a fake cluster for testing."""
    return ClusterResult(
        cluster_id=0,
        tracks=FAKE_PATHS,
        bpm_mean=128.0,
        bpm_std=1.0,
        track_count=3,
        total_duration=900.0,
    )


class TestScanSequencing:
    """Tests for sequencing integration in the scan command."""

    @patch("playchitect.cli.commands.AudioScanner.scan")
    @patch("playchitect.cli.commands.MetadataExtractor.extract")
    @patch("playchitect.core.intensity_analyzer.IntensityAnalyzer")
    @patch("playchitect.cli.commands.PlaylistClusterer")
    @patch("playchitect.core.sequencer.Sequencer.sequence")
    def test_sequence_mode_ramp_calls_sequencer(
        self,
        mock_sequence: MagicMock,
        mock_clusterer_cls: MagicMock,
        mock_intensity_cls: MagicMock,
        mock_extract: MagicMock,
        mock_scan: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify that --sequence-mode ramp calls the Sequencer."""
        mock_scan.return_value = FAKE_PATHS
        mock_extract.side_effect = make_fake_metadata

        mock_intensity = mock_intensity_cls.return_value
        mock_intensity.analyze.side_effect = make_fake_features

        mock_clusterer = mock_clusterer_cls.return_value
        mock_clusterer.cluster_by_features.return_value = [make_fake_cluster()]
        mock_clusterer.split_cluster.side_effect = lambda c, t: [c]

        mock_sequence.return_value = FAKE_PATHS

        runner = CliRunner()
        result = runner.invoke(scan, [str(tmp_path), "--sequence-mode", "ramp", "--dry-run"])

        assert result.exit_code == 0
        assert "Sequencing tracks (mode: ramp)..." in result.output
        assert mock_sequence.called
        assert mock_sequence.call_count == 1

    @patch("playchitect.cli.commands.AudioScanner.scan")
    @patch("playchitect.cli.commands.MetadataExtractor.extract")
    @patch("playchitect.cli.commands.PlaylistClusterer")
    @patch("playchitect.core.sequencer.Sequencer.sequence")
    def test_sequence_mode_fixed_skips_sequencer(
        self,
        mock_sequence: MagicMock,
        mock_clusterer_cls: MagicMock,
        mock_extract: MagicMock,
        mock_scan: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify that --sequence-mode fixed (default) skips the Sequencer."""
        mock_scan.return_value = FAKE_PATHS
        mock_extract.side_effect = make_fake_metadata

        mock_clusterer = mock_clusterer_cls.return_value
        mock_clusterer.cluster_by_bpm.return_value = [make_fake_cluster()]
        mock_clusterer.split_cluster.side_effect = lambda c, t: [c]

        runner = CliRunner()
        # Default is fixed
        result = runner.invoke(scan, [str(tmp_path), "--dry-run"])

        assert result.exit_code == 0
        assert "Sequencing tracks" not in result.output
        assert not mock_sequence.called

    @patch("playchitect.cli.commands.AudioScanner.scan")
    @patch("playchitect.cli.commands.MetadataExtractor.extract")
    @patch("playchitect.core.intensity_analyzer.IntensityAnalyzer")
    @patch("playchitect.cli.commands.PlaylistClusterer")
    def test_sequence_mode_ramp_forces_intensity_analysis(
        self,
        mock_clusterer_cls: MagicMock,
        mock_intensity_cls: MagicMock,
        mock_extract: MagicMock,
        mock_scan: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Verify that --sequence-mode ramp forces intensity analysis."""
        mock_scan.return_value = FAKE_PATHS
        mock_extract.side_effect = make_fake_metadata

        mock_intensity = mock_intensity_cls.return_value
        mock_intensity.analyze.side_effect = make_fake_features

        mock_clusterer = mock_clusterer_cls.return_value
        mock_clusterer.cluster_by_features.return_value = [make_fake_cluster()]
        mock_clusterer.split_cluster.side_effect = lambda c, t: [c]

        runner = CliRunner()
        result = runner.invoke(scan, [str(tmp_path), "--sequence-mode", "ramp", "--dry-run"])

        assert result.exit_code == 0
        assert "Extracting audio intensity features..." in result.output
        assert mock_intensity.analyze.called
        assert mock_intensity.analyze.call_count == len(FAKE_PATHS)
