"""Fast performance benchmarks for CLI operations."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

# Assuming playchitect is installed in the system PATH or callable via uv run
CLI_COMMAND = "uv run playchitect"
TEST_MUSIC_PATH = Path(
    os.environ.get(
        "PLAYCHITECT_BENCH_MUSIC_PATH",
        "/mnt/1tb_ssd/Media/Music/Trying Before You Buying",
    )
)
TRACK_SUBSET_SIZE = 100


@pytest.fixture(scope="module")
def small_track_subset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Creates a temporary directory with a small subset of audio files
    from the TEST_MUSIC_PATH.
    """
    if not TEST_MUSIC_PATH.exists():
        pytest.skip(f"Test music path not found: {TEST_MUSIC_PATH}")

    # Use playchitect's audio scanner to find files
    cmd = [
        "uv",
        "run",
        "playchitect",
        "info",
        str(TEST_MUSIC_PATH),
        "--format",
        "json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info_data = json.loads(result.stdout)
    all_files = [Path(f) for f in info_data["files"]]

    if len(all_files) < TRACK_SUBSET_SIZE:
        pytest.skip(f"Not enough tracks in {TEST_MUSIC_PATH} for subset size {TRACK_SUBSET_SIZE}")

    selected_files = all_files[:TRACK_SUBSET_SIZE]

    # Create a temporary directory and copy the selected files
    temp_dir = tmp_path_factory.mktemp("small_music_library")
    for file_path in selected_files:
        # Recreate directory structure if needed, or flatten
        # For simplicity, let's flatten for now
        shutil.copy(file_path, temp_dir / file_path.name)

    return temp_dir


def run_cli_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Helper to run a playchitect CLI command."""
    full_cmd = command if command[0] == "uv" else CLI_COMMAND.split() + command
    return subprocess.run(full_cmd, capture_output=True, text=True, check=True, cwd=cwd)


class TestFastPerformanceChecks:
    """Benchmarks for fast CLI operations."""

    def test_playchitect_info(self, benchmark: BenchmarkFixture, small_track_subset: Path):
        """Benchmark playchitect info command."""
        benchmark(run_cli_command, ["info", str(small_track_subset)])

    def test_playchitect_scan_dry_run(self, benchmark: BenchmarkFixture, small_track_subset: Path):
        """Benchmark playchitect scan --dry-run command."""
        benchmark(run_cli_command, ["scan", str(small_track_subset), "--dry-run"])

    def test_playchitect_scan_with_embeddings_dry_run(
        self, benchmark: BenchmarkFixture, small_track_subset: Path
    ):
        """Benchmark playchitect scan --use-embeddings --dry-run command."""
        # This test requires essentia-tensorflow to be installed in the environment.
        # It might also require a dummy model file if the extractor tries to download.
        # For a true benchmark, the model should be present locally.
        try:
            # Check if essentia-tensorflow is available without importing the whole module
            import importlib.util

            if importlib.util.find_spec("essentia.streaming") is None:
                raise ImportError
        except ImportError:
            pytest.skip("essentia-tensorflow not installed, skipping embeddings benchmark.")

        # This will trigger download if model is not present, which is not ideal for benchmarking.
        # For CI, the model should be pre-cached.
        benchmark(
            run_cli_command, ["scan", str(small_track_subset), "--use-embeddings", "--dry-run"]
        )


# End of file - minor change to re-trigger CI
