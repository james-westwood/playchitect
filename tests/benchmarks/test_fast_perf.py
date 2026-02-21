"""Fast performance benchmarks for CLI operations and core components."""

import json
import os
import random
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from playchitect.core.audio_scanner import AudioScanner
from playchitect.core.clustering import PlaylistClusterer
from playchitect.core.intensity_analyzer import IntensityAnalyzer
from playchitect.core.metadata_extractor import MetadataExtractor

# Assuming playchitect is installed in the system PATH or callable via uv run
CLI_COMMAND = "uv run playchitect"
TRACK_SUBSET_SIZE = 100


@pytest.fixture(scope="module")
def benchmark_target_library(
    tmp_path_factory: pytest.TempPathFactory,
    synthetic_library: Callable[[int], Path],
) -> Path:
    """
    Provides a small subset of audio files for benchmarking.
    Prioritises real music if PLAYCHITECT_BENCH_MUSIC_PATH is set,
    otherwise falls back to a synthetic library so CI always has a target.
    """
    real_music_path_str = os.environ.get("PLAYCHITECT_BENCH_MUSIC_PATH")
    if real_music_path_str:
        real_music_path = Path(real_music_path_str)
        if not real_music_path.exists():
            pytest.skip(f"Real music path not found: {real_music_path}")

        cmd = [
            "uv",
            "run",
            "playchitect",
            "info",
            str(real_music_path),
            "--format",
            "json",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info_data = json.loads(result.stdout)
        all_files = [Path(f) for f in info_data["files"]]

        if len(all_files) < TRACK_SUBSET_SIZE:
            pytest.skip(
                f"Not enough tracks in {real_music_path} for subset size {TRACK_SUBSET_SIZE}"
            )

        selected_files = random.sample(all_files, TRACK_SUBSET_SIZE)

        temp_dir = tmp_path_factory.mktemp("real_music_subset")
        for file_path in selected_files:
            shutil.copy(file_path, temp_dir / file_path.name)
        return temp_dir

    # Synthetic fallback: 10 tracks is enough to exercise the CLI path in CI
    return synthetic_library(10)


def run_cli_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Helper to run a playchitect CLI command."""
    full_cmd = command if command[0] == "uv" else CLI_COMMAND.split() + command
    return subprocess.run(full_cmd, capture_output=True, text=True, check=True, cwd=cwd)


class TestFastPerformanceChecks:
    """Benchmarks for fast CLI operations and core components."""

    def test_playchitect_info_cli(
        self, benchmark: BenchmarkFixture, benchmark_target_library: Path
    ):
        """Benchmark playchitect info command."""
        benchmark(run_cli_command, ["info", str(benchmark_target_library)])

    def test_playchitect_scan_dry_run_cli(
        self, benchmark: BenchmarkFixture, benchmark_target_library: Path
    ):
        """Benchmark playchitect scan --dry-run command."""
        benchmark(run_cli_command, ["scan", str(benchmark_target_library), "--dry-run"])

    def test_playchitect_scan_with_embeddings_dry_run_cli(
        self, benchmark: BenchmarkFixture, benchmark_target_library: Path
    ):
        """Benchmark playchitect scan --use-embeddings --dry-run command."""
        try:
            import importlib.util

            if importlib.util.find_spec("essentia.streaming") is None:
                raise ImportError
        except ImportError:
            pytest.skip("essentia-tensorflow not installed, skipping embeddings benchmark.")

        benchmark(
            run_cli_command,
            ["scan", str(benchmark_target_library), "--use-embeddings", "--dry-run"],
        )

    def test_audio_scanner_scan(
        self,
        benchmark: BenchmarkFixture,
        synthetic_library: Callable[[int], Path],
    ):
        """Benchmark AudioScanner.scan with a small synthetic library."""
        library_path = synthetic_library(50)
        scanner = AudioScanner()
        benchmark(scanner.scan, library_path)

    def test_metadata_extractor_extract_batch(
        self, benchmark: BenchmarkFixture, synthetic_library: Callable[[int], Path]
    ):
        """Benchmark MetadataExtractor.extract_batch with a small synthetic library."""
        library_path = synthetic_library(50)
        scanner = AudioScanner()
        all_files = scanner.scan(library_path)
        extractor = MetadataExtractor()
        benchmark(extractor.extract_batch, all_files)

    def test_intensity_analyzer_analyze(
        self, benchmark: BenchmarkFixture, synthetic_library: Callable[[int], Path]
    ):
        """Benchmark IntensityAnalyzer.analyze on a single synthetic audio file."""
        library_path = synthetic_library(1)
        audio_files = list(library_path.rglob("*.flac"))
        assert audio_files, "synthetic_library must produce at least one FLAC file"
        analyzer = IntensityAnalyzer()
        benchmark(analyzer.analyze, audio_files[0])

    def test_clustering_cluster_by_features(
        self, benchmark: BenchmarkFixture, mock_features_array_input: np.ndarray
    ):
        """Benchmark PlaylistClusterer.cluster_by_features with mocked in-memory data."""
        num_tracks = mock_features_array_input.shape[0]
        mock_metadata_dict = {
            Path(f"/path/to/track_{i}.flac"): MagicMock(bpm=120) for i in range(num_tracks)
        }
        mock_intensity_dict = {
            Path(f"/path/to/track_{i}.flac"): MagicMock(
                to_feature_vector=MagicMock(return_value=np.random.rand(7))
            )
            for i in range(num_tracks)
        }
        clustering = PlaylistClusterer(target_tracks_per_playlist=20)
        benchmark(
            clustering.cluster_by_features,
            mock_metadata_dict,
            mock_intensity_dict,
            embedding_dict=None,
            use_ewkm=False,
        )
