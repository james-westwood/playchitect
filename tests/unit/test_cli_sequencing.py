"""Unit tests for the CLI sequencing flags in playchitect.cli.commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from playchitect.cli.commands import cli
from playchitect.core.clustering import ClusterResult
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata

FAKE_PATHS = [Path("track_0.flac"), Path("track_1.flac"), Path("track_2.flac")]


def make_fake_metadata(p: Path) -> TrackMetadata:
    """Helper to build TrackMetadata for a given path."""
    return TrackMetadata(filepath=p, bpm=128.0, duration=300.0)


def make_fake_features(p: Path) -> IntensityFeatures:
    """Helper to build IntensityFeatures for a given path."""
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
    """Helper to build a ClusterResult containing FAKE_PATHS."""
    return ClusterResult(
        cluster_id=0,
        tracks=list(FAKE_PATHS),
        bpm_mean=128.0,
        bpm_std=1.0,
        track_count=3,
        total_duration=900.0,
    )


@pytest.fixture
def music_dir(tmp_path: Path) -> Path:
    """Create a temporary music directory."""
    d = tmp_path / "music"
    d.mkdir()
    for p in FAKE_PATHS:
        (d / p.name).touch()
    return d


@patch("playchitect.cli.commands.get_config")
@patch("playchitect.cli.commands.AudioScanner.scan")
@patch("playchitect.cli.commands.MetadataExtractor.extract")
@patch("playchitect.core.intensity_analyzer.IntensityAnalyzer")
@patch("playchitect.cli.commands.PlaylistClusterer.cluster_by_features")
@patch("playchitect.core.sequencer.Sequencer.sequence")
@patch("playchitect.cli.commands.M3UExporter.export_clusters")
def test_sequence_mode_ramp_calls_sequencer(
    mock_export: MagicMock,
    mock_sequence: MagicMock,
    mock_cluster: MagicMock,
    mock_analyze_cls: MagicMock,
    mock_extract: MagicMock,
    mock_scan: MagicMock,
    mock_config: MagicMock,
    music_dir: Path,
) -> None:
    """Verify Sequencer.sequence is called when mode is 'ramp'."""
    mock_scan.return_value = [music_dir / p.name for p in FAKE_PATHS]
    mock_extract.side_effect = make_fake_metadata
    mock_analyze_cls.return_value.analyze.side_effect = make_fake_features
    mock_cluster.return_value = [make_fake_cluster()]
    mock_sequence.side_effect = lambda c, *args, **kwargs: c.tracks
    mock_export.return_value = [Path("Playlist 1.m3u")]

    runner = CliRunner()
    result = runner.invoke(cli, ["scan", str(music_dir), "--sequence-mode", "ramp"])

    assert result.exit_code == 0, result.output
    assert "Sequencing tracks (mode: ramp)..." in result.output
    assert mock_sequence.called


@patch("playchitect.cli.commands.get_config")
@patch("playchitect.cli.commands.AudioScanner.scan")
@patch("playchitect.cli.commands.MetadataExtractor.extract")
@patch("playchitect.core.intensity_analyzer.IntensityAnalyzer")
@patch("playchitect.cli.commands.PlaylistClusterer.cluster_by_features")
@patch("playchitect.core.sequencer.Sequencer.sequence")
@patch("playchitect.cli.commands.M3UExporter.export_clusters")
def test_sequence_mode_fixed_skips_sequencer(
    mock_export: MagicMock,
    mock_sequence: MagicMock,
    mock_cluster: MagicMock,
    mock_analyze_cls: MagicMock,
    mock_extract: MagicMock,
    mock_scan: MagicMock,
    mock_config: MagicMock,
    music_dir: Path,
) -> None:
    """Verify Sequencer.sequence is NOT called when mode is 'fixed'."""
    mock_scan.return_value = [music_dir / p.name for p in FAKE_PATHS]
    mock_extract.side_effect = make_fake_metadata
    mock_analyze_cls.return_value.analyze.side_effect = make_fake_features
    mock_cluster.return_value = [make_fake_cluster()]
    mock_export.return_value = [Path("Playlist 1.m3u")]

    runner = CliRunner()
    result = runner.invoke(cli, ["scan", str(music_dir), "--sequence-mode", "fixed"])

    assert result.exit_code == 0, result.output
    assert "Sequencing tracks" not in result.output
    assert not mock_sequence.called


@patch("playchitect.cli.commands.get_config")
@patch("playchitect.cli.commands.AudioScanner.scan")
@patch("playchitect.cli.commands.MetadataExtractor.extract")
@patch("playchitect.core.intensity_analyzer.IntensityAnalyzer")
@patch("playchitect.cli.commands.PlaylistClusterer.cluster_by_features")
@patch("playchitect.cli.commands.M3UExporter.export_clusters")
def test_sequence_mode_ramp_forces_intensity_analysis(
    mock_export: MagicMock,
    mock_cluster: MagicMock,
    mock_analyze_cls: MagicMock,
    mock_extract: MagicMock,
    mock_scan: MagicMock,
    mock_config: MagicMock,
    music_dir: Path,
) -> None:
    """Verify IntensityAnalyzer.analyze is called when sequence_mode='ramp'."""
    mock_scan.return_value = [music_dir / p.name for p in FAKE_PATHS]
    mock_extract.side_effect = make_fake_metadata
    mock_analyze_cls.return_value.analyze.side_effect = make_fake_features
    mock_cluster.return_value = [make_fake_cluster()]
    mock_export.return_value = [Path("Playlist 1.m3u")]

    runner = CliRunner()
    # Use single-genre (default) which doesn't normally require intensity,
    # but ramp sequencing should force it.
    result = runner.invoke(cli, ["scan", str(music_dir), "--sequence-mode", "ramp"])

    assert result.exit_code == 0, result.output
    assert "Extracting audio intensity features..." in result.output
    assert mock_analyze_cls.return_value.analyze.called
