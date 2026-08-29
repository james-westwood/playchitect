"""Fast performance benchmarks for CLI operations and core components."""

import json
import os
import random
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
import soundfile as sf
from pytest_benchmark.fixture import BenchmarkFixture

import playchitect.core.embedding_extractor as emb_mod
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
THRESHOLD_INTENSITY_ANALYZER = (
    0.150  # 150ms for one 0.5s file (increased for structural+vocal features)
)
THRESHOLD_CLUSTERING = 1.500  # 1.5s for 1000 tracks (includes silhouette)
THRESHOLD_M3U_EXPORT = 0.050  # 50ms for 50 tracks
THRESHOLD_CUE_EXPORT = 0.050  # 50ms for 50 tracks
THRESHOLD_CLI_INFO = 10.0  # 10s for 10 tracks (uv run overhead in CI)
THRESHOLD_CLI_SCAN = 12.0  # 12s for 10 tracks

# ── Embedding benchmark fixture sizing (TASK-32) ─────────────────────────────
#
# MusiCNN patches audio into 187 mel frames (frameSize=512, hopSize=256 at
# 16 kHz, essentia's centred framing), so it needs at least 47361 samples =
# 2.9600625s of audio before it emits a single frame. Measured by binary
# search against the installed msd-musicnn-1.pb on 2026-08-29: 47360 samples
# -> 0 frames, 47361 -> 187 frames. Below that, mean-pooling zero frames
# silently yields NaN and the whole --use-embeddings run collapses.
#
# The shared synthetic_library default of 0.5s is therefore unusable here,
# but must NOT be changed: at 4.0s, IntensityAnalyzer.analyze takes ~0.349s
# against a 0.150s threshold (2.3x over) and MetadataExtractor.extract_batch
# lands within 6% of its threshold. So this benchmark gets its own longer
# library and every other benchmark keeps the 0.5s one.
#
# 4.0s gives ~35% headroom over the 2.96s floor (3.0s would leave only 1.3%,
# inside the rounding noise of int(22050 * D) plus the 22050 -> 16000
# resample) while staying inside the single-patch regime, so per-track
# extraction costs the same ~0.09s as a 3.0s clip. 5.0s would cross into a
# second patch (~20% dearer) for no extra safety.
EMBEDDING_BENCHMARK_TRACK_SECONDS = 4.0
# Cold-run cost is dominated by the one-time ~4-5s TensorFlow graph load, not
# per-track work, so trimming the track count barely helps and weakens the
# benchmark. Kept at 10, matching benchmark_target_library.
EMBEDDING_BENCHMARK_TRACK_COUNT = 10


@pytest.fixture(scope="module")
def benchmark_target_library(
    tmp_path_factory: pytest.TempPathFactory,
    synthetic_library: Callable[..., Path],
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


@pytest.fixture(scope="module")
def embedding_benchmark_library(
    benchmark_target_library: Path,
    synthetic_library: Callable[..., Path],
) -> Path:
    """
    Benchmark target for the MusiCNN embedding path specifically.

    Real music (PLAYCHITECT_BENCH_MUSIC_PATH) is already long enough, so that
    library is reused untouched. The synthetic fallback, however, must be
    generated at EMBEDDING_BENCHMARK_TRACK_SECONDS rather than the shared
    0.5s default -- see the constant's comment for why the default itself
    must stay at 0.5s.
    """
    if os.environ.get("PLAYCHITECT_BENCH_MUSIC_PATH"):
        return benchmark_target_library

    return synthetic_library(
        EMBEDDING_BENCHMARK_TRACK_COUNT,
        EMBEDDING_BENCHMARK_TRACK_SECONDS,
    )


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
        self, benchmark: BenchmarkFixture, embedding_benchmark_library: Path
    ):
        """
        Benchmark playchitect scan --use-embeddings --dry-run command.

        Uses embedding_benchmark_library (4.0s clips), not the shared 0.5s
        benchmark_target_library: MusiCNN emits zero frames below ~2.96s, so
        the 0.5s library makes this exit 1 rather than measure anything.
        """
        try:
            import importlib.util

            if importlib.util.find_spec("essentia.streaming") is None:
                raise ImportError
        except ImportError:
            pytest.skip("essentia-tensorflow not installed, skipping embeddings benchmark.")

        benchmark(
            run_cli_command,
            ["scan", str(embedding_benchmark_library), "--use-embeddings", "--dry-run"],
        )
        # No threshold for embeddings yet as it's environment dependent

    def test_audio_scanner_scan(
        self,
        benchmark: BenchmarkFixture,
        synthetic_library: Callable[..., Path],
    ):
        """Benchmark AudioScanner.scan with a small synthetic library."""
        library_path = synthetic_library(50)
        scanner = AudioScanner()
        benchmark(scanner.scan, library_path)
        assert benchmark.stats.stats.mean < THRESHOLD_AUDIO_SCANNER  # type: ignore

    def test_metadata_extractor_extract_batch(
        self, benchmark: BenchmarkFixture, synthetic_library: Callable[..., Path]
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
        self, benchmark: BenchmarkFixture, synthetic_library: Callable[..., Path]
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
                file_path=p,
                file_hash="fakehash",
                rms_energy=random.random(),
                brightness=random.random(),
                sub_bass_energy=random.random(),
                kick_energy=random.random(),
                bass_harmonics=random.random(),
                percussiveness=random.random(),
                onset_strength=random.random(),
                camelot_key="8B",
                key_index=0.0,
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


class TestBenchmarkLibraryDurations:
    """
    Guards on the audio length of the benchmark fixtures themselves.

    The embedding benchmark and every other benchmark pull from deliberately
    different libraries; these tests keep that split from silently collapsing
    in either direction.
    """

    def test_shared_synthetic_library_stays_short(
        self, synthetic_library: Callable[..., Path]
    ) -> None:
        """
        The shared synthetic_library default must stay at 0.5s. Lengthening it
        to 4.0s pushes IntensityAnalyzer.analyze to ~0.349s against the 0.150s
        THRESHOLD_INTENSITY_ANALYZER (2.3x over, a hard fail) and puts
        MetadataExtractor.extract_batch within 6% of its own threshold.
        Tracks needing longer audio must get their own fixture instead.
        """
        library_path = synthetic_library(1)
        clips = list(library_path.rglob("*.flac"))
        assert clips, "synthetic_library must produce at least one FLAC file"

        for clip in clips:
            info = sf.info(str(clip))
            assert info.duration == pytest.approx(0.5, abs=0.01), (
                f"{clip.name} is {info.duration}s; the shared synthetic_library "
                "default must remain 0.5s"
            )

    def test_embedding_benchmark_library_clears_the_musicnn_minimum(
        self, embedding_benchmark_library: Path
    ) -> None:
        """
        Every clip fed to the --use-embeddings benchmark must be comfortably
        longer than MusiCNN's measured minimum, or the model emits zero frames
        and the CLI exits 1 instead of producing a measurement.
        """
        if os.environ.get("PLAYCHITECT_BENCH_MUSIC_PATH"):
            pytest.skip("Real-music benchmark library; clip lengths are not ours to assert.")

        minimum = getattr(emb_mod, "_MUSICNN_MIN_AUDIO_SECONDS", None)
        assert minimum is not None, (
            "embedding_extractor must define a module-level "
            "_MUSICNN_MIN_AUDIO_SECONDS constant recording the measured "
            "MusiCNN minimum audio duration (2.9600625s at 16 kHz)"
        )

        clips = list(embedding_benchmark_library.rglob("*.flac"))
        assert len(clips) == EMBEDDING_BENCHMARK_TRACK_COUNT

        for clip in clips:
            duration = sf.info(str(clip)).duration
            # 20% headroom over the floor, so int(sample_rate * seconds)
            # rounding and the 22050 -> 16000 resample cannot drag a clip
            # back under it.
            assert duration >= float(minimum) * 1.2, (
                f"{clip.name} is {duration}s, too close to the {minimum}s MusiCNN minimum"
            )
