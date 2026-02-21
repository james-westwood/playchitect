"""Fast performance benchmarks for CLI operations and core components."""

import json
import os
import random
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from playchitect.core.audio_scanner import AudioScanner
from playchitect.core.clustering import ClusterResult, PlaylistClusterer
from playchitect.core.export import CUEExporter, M3UExporter
from playchitect.core.intensity_analyzer import IntensityAnalyzer, IntensityFeatures
from playchitect.core.metadata_extractor import MetadataExtractor, TrackMetadata

# Assuming playchitect is installed in the system PATH or callable via uv run
CLI_COMMAND = "uv run playchitect"
TRACK_SUBSET_SIZE = 100

# Performance thresholds (in seconds)
# These are used to fail CI if a regression is detected.
# Values are set to roughly 2-3x the local baseline to allow for CI runner variability.
THRESHOLD_AUDIO_SCANNER = 0.010  # 10ms for 50 tracks
THRESHOLD_METADATA_EXTRACTOR = 0.050  # 50ms for 50 tracks
THRESHOLD_INTENSITY_ANALYZER = 0.100  # 100ms for one 0.5s file
THRESHOLD_CLUSTERING = 0.800  # 800ms for 1000 tracks
THRESHOLD_M3U_EXPORT = 0.050  # 50ms for 50 tracks
THRESHOLD_CUE_EXPORT = 0.050  # 50ms for 50 tracks
THRESHOLD_CLI_INFO = 10.0  # 10s for 10 tracks (uv run overhead in CI)
THRESHOLD_CLI_SCAN = 12.0  # 12s for 10 tracks


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
        # Use type: ignore because ty doesn't know benchmark.stats structure
        assert benchmark.stats.stats.mean < THRESHOLD_CLI_INFO  # type: ignore

    def test_playchitect_scan_dry_run_cli(
        self, benchmark: BenchmarkFixture, benchmark_target_library: Path
    ):
        """Benchmark playchitect scan --dry-run command."""
        benchmark(run_cli_command, ["scan", str(benchmark_target_library), "--dry-run"])
        assert benchmark.stats.stats.mean < THRESHOLD_CLI_SCAN  # type: ignore

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
        # No threshold for embeddings yet as it's environment dependent

    def test_audio_scanner_scan(
        self,
        benchmark: BenchmarkFixture,
        synthetic_library: Callable[[int], Path],
    ):
        """Benchmark AudioScanner.scan with a small synthetic library."""
        library_path = synthetic_library(50)
        scanner = AudioScanner()
        benchmark(scanner.scan, library_path)
        assert benchmark.stats.stats.mean < THRESHOLD_AUDIO_SCANNER  # type: ignore

    def test_metadata_extractor_extract_batch(
        self, benchmark: BenchmarkFixture, synthetic_library: Callable[[int], Path]
    ):
        """Benchmark MetadataExtractor.extract_batch with a small synthetic library."""
        library_path = synthetic_library(50)
        scanner = AudioScanner()
        all_files = scanner.scan(library_path)
        # Disable cache to measure actual extraction performance
        extractor = MetadataExtractor(cache_enabled=False)
        benchmark(extractor.extract_batch, all_files)
        assert benchmark.stats.stats.mean < THRESHOLD_METADATA_EXTRACTOR  # type: ignore

    def test_intensity_analyzer_analyze(
        self, benchmark: BenchmarkFixture, synthetic_library: Callable[[int], Path]
    ):
        """Benchmark IntensityAnalyzer.analyze on a single synthetic audio file."""
        library_path = synthetic_library(1)
        audio_files = list(library_path.rglob("*.flac"))
        assert audio_files, "synthetic_library must produce at least one FLAC file"
        # Disable cache to measure actual analysis performance
        analyzer = IntensityAnalyzer(cache_enabled=False)
        benchmark(analyzer.analyze, audio_files[0])
        assert benchmark.stats.stats.mean < THRESHOLD_INTENSITY_ANALYZER  # type: ignore

    def test_clustering_cluster_by_features(self, benchmark: BenchmarkFixture):
        """Benchmark PlaylistClusterer.cluster_by_features with real data class instances."""
        num_tracks = 1000
        paths = [Path(f"/path/to/track_{i}.flac") for i in range(num_tracks)]
        metadata_dict = {p: TrackMetadata(filepath=p, bpm=120.0) for p in paths}
        intensity_dict = {
            p: IntensityFeatures(
                filepath=p,
                file_hash="fakehash",
                rms_energy=random.random(),
                brightness=random.random(),
                sub_bass_energy=random.random(),
                kick_energy=random.random(),
                bass_harmonics=random.random(),
                percussiveness=random.random(),
                onset_strength=random.random(),
            )
            for p in paths
        }
        clustering = PlaylistClusterer(target_tracks_per_playlist=20)
        benchmark(
            clustering.cluster_by_features,
            metadata_dict,
            intensity_dict,
            embedding_dict=None,
            use_ewkm=False,
        )
        assert benchmark.stats.stats.mean < THRESHOLD_CLUSTERING  # type: ignore

    def test_m3u_export(self, benchmark: BenchmarkFixture, tmp_path: Path):
        """Benchmark M3U playlist export performance."""
        num_tracks = 50
        paths = [Path(f"/path/to/track_{i}.flac") for i in range(num_tracks)]
        cluster = ClusterResult(
            cluster_id=0,
            tracks=paths,
            bpm_mean=120.0,
            bpm_std=2.0,
            track_count=num_tracks,
            total_duration=num_tracks * 180.0,
        )
        metadata_dict = {
            p: TrackMetadata(filepath=p, artist="Artist", title=f"Track {i}", duration=180.0)
            for i, p in enumerate(paths)
        }

        exporter = M3UExporter(output_dir=tmp_path)
        benchmark(exporter.export_clusters, [cluster], metadata_dict=metadata_dict)
        assert benchmark.stats.stats.mean < THRESHOLD_M3U_EXPORT  # type: ignore

    def test_cue_export(self, benchmark: BenchmarkFixture, tmp_path: Path):
        """Benchmark CUE sheet export performance."""
        num_tracks = 50
        paths = [Path(f"/path/to/track_{i}.flac") for i in range(num_tracks)]
        cluster = ClusterResult(
            cluster_id=0,
            tracks=paths,
            bpm_mean=120.0,
            bpm_std=2.0,
            track_count=num_tracks,
            total_duration=num_tracks * 180.0,
        )
        metadata_dict = {
            p: TrackMetadata(filepath=p, artist="Artist", title=f"Track {i}", duration=180.0)
            for i, p in enumerate(paths)
        }

        exporter = CUEExporter(output_dir=tmp_path)
        benchmark(exporter.export_clusters, [cluster], metadata_dict=metadata_dict)
        assert benchmark.stats.stats.mean < THRESHOLD_CUE_EXPORT  # type: ignore
