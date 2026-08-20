"""
TDD red tests for TASK-16: make multi-dimensional clustering the default for scan.

These tests pin the new contract: intensity analysis and cluster_by_features run
by default, while a new --fast flag selects the old BPM-only path. All scan calls
use --dry-run and synthetic FLAC fixtures so nothing is written outside pytest tmp dirs.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner
from mutagen.flac import FLAC

from playchitect.cli.commands import scan
from playchitect.core.clustering import PlaylistClusterer
from playchitect.core.intensity_analyzer import IntensityFeatures

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


def _make_music_dir(
    root: Path,
    count: int = 20,
    fail_count: int = 0,
    low_bpm: float = 128.0,
    high_bpm: float = 140.0,
) -> Path:
    """Create a directory of BPM-tagged FLACs.

    First ``count - fail_count`` tracks are normal; the last ``fail_count`` tracks
    have ``fail`` in their filenames so the fake analyser can raise on them.
    """
    music_dir = root / "music"
    music_dir.mkdir()
    for i in range(count - fail_count):
        bpm = low_bpm if i < count // 2 else high_bpm
        _write_flac(music_dir / f"track_{i:02d}.flac", bpm=bpm)
    for j in range(fail_count):
        idx = count - fail_count + j
        _write_flac(music_dir / f"track_{idx:02d}_fail.flac", bpm=low_bpm)
    return music_dir


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeCacheDB:
    """Hermetic stand-in for CacheDB; the fake analyser never touches the DB."""

    def __init__(self, db_path: Path) -> None:
        pass


class FakeIntensityAnalyzer:
    """Record every call and return deterministic IntensityFeatures.

    Filenames containing ``fail`` trigger a RuntimeError, so failure handling can be
    tested without touching real librosa/audio decoding.
    """

    calls: list[Path] = []
    fail_on: Callable[[Path], bool] | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @classmethod
    def reset(cls) -> None:
        cls.calls.clear()
        cls.fail_on = None

    def analyze(self, path: Path) -> IntensityFeatures:
        self.__class__.calls.append(path)
        if self.__class__.fail_on is not None and self.__class__.fail_on(path):
            raise RuntimeError("analysis boom")
        # Vary RMS energy so clusters have something to separate on.
        match = re.search(r"(\d+)", path.stem)
        idx = int(match.group(1)) if match else 0
        return IntensityFeatures(
            file_path=path,
            file_hash="fakehash",
            rms_energy=0.3 + (idx % 10) / 20.0,
            brightness=0.5,
            sub_bass_energy=0.3,
            kick_energy=0.4,
            bass_harmonics=0.3,
            percussiveness=0.6,
            onset_strength=0.5,
            camelot_key="8B",
            key_index=0.0,
        )


# ---------------------------------------------------------------------------
# Spies
# ---------------------------------------------------------------------------


class ClusteringSpy:
    """Wrap the real PlaylistClusterer methods so assertions stay honest."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self._patch_method(monkeypatch, "cluster_by_features")
        self._patch_method(monkeypatch, "cluster_by_bpm")

    def _patch_method(self, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
        original = getattr(PlaylistClusterer, name)

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(PlaylistClusterer, name, wrapper)

    def count(self, name: str) -> int:
        return sum(1 for c, _, _ in self.calls if c == name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_fakes() -> None:
    """Ensure the fake intensity analyser's call list is empty for each test."""
    FakeIntensityAnalyzer.reset()


@pytest.fixture
def patched_scan(monkeypatch: pytest.MonkeyPatch) -> ClusteringSpy:
    """Patch CacheDB and IntensityAnalyzer at source and spy on clustering methods."""
    monkeypatch.setattr("playchitect.core.cache_db.CacheDB", FakeCacheDB)
    monkeypatch.setattr(
        "playchitect.core.intensity_analyzer.IntensityAnalyzer", FakeIntensityAnalyzer
    )
    return ClusteringSpy(monkeypatch)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScanDefaults:
    """Default scan path: intensity analysis + cluster_by_features."""

    def test_default_scan_runs_intensity_analysis(
        self, tmp_path: Path, patched_scan: ClusteringSpy
    ) -> None:
        """Default scan calls the intensity analyser once per audio file."""
        music_dir = _make_music_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            scan,
            [str(music_dir), "--dry-run", "--target-tracks", "25"],
        )
        assert result.exit_code == 0, result.output
        assert len(FakeIntensityAnalyzer.calls) == 20

    def test_default_scan_invokes_cluster_by_features(
        self, tmp_path: Path, patched_scan: ClusteringSpy
    ) -> None:
        """Default scan calls cluster_by_features exactly once and not cluster_by_bpm."""
        music_dir = _make_music_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            scan,
            [str(music_dir), "--dry-run", "--target-tracks", "25"],
        )
        assert result.exit_code == 0, result.output
        assert patched_scan.count("cluster_by_features") == 1
        assert patched_scan.count("cluster_by_bpm") == 0

    def test_weight_source_string_default_path(
        self, tmp_path: Path, patched_scan: ClusteringSpy
    ) -> None:
        """Default scan prints a weight-source line that is NOT the BPM-only one."""
        music_dir = _make_music_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            scan,
            [str(music_dir), "--dry-run", "--target-tracks", "25"],
        )
        assert result.exit_code == 0, result.output
        assert "Weight source:" in result.output
        assert "(BPM-only clustering)" not in result.output

    def test_failed_analysis_counted_in_summary(
        self, tmp_path: Path, patched_scan: ClusteringSpy
    ) -> None:
        """Failed intensity analyses are reported in the summary and exit stays 0."""
        music_dir = _make_music_dir(tmp_path, count=20, fail_count=3)
        FakeIntensityAnalyzer.fail_on = lambda p: "fail" in p.name
        runner = CliRunner()
        result = runner.invoke(
            scan,
            [str(music_dir), "--dry-run", "--target-tracks", "25"],
        )
        assert result.exit_code == 0, result.output
        assert "17 tracks analysed, 3 failed" in result.output

    def test_failed_tracks_remain_in_clustering(
        self, tmp_path: Path, patched_scan: ClusteringSpy
    ) -> None:
        """Failed tracks are still clustered so the total track count remains 20."""
        music_dir = _make_music_dir(tmp_path, count=20, fail_count=3)
        FakeIntensityAnalyzer.fail_on = lambda p: "fail" in p.name
        runner = CliRunner()
        result = runner.invoke(
            scan,
            [str(music_dir), "--dry-run", "--target-tracks", "25"],
        )
        assert result.exit_code == 0, result.output
        assert "17 tracks analysed, 3 failed" in result.output
        counts = [int(m.group(1)) for m in re.finditer(r"Cluster \d+: (\d+) tracks", result.output)]
        assert sum(counts) == 20


class TestScanFastPath:
    """--fast path: skip intensity analysis, use cluster_by_bpm."""

    def test_fast_flag_invokes_cluster_by_bpm(
        self, tmp_path: Path, patched_scan: ClusteringSpy
    ) -> None:
        """--fast skips intensity analysis and calls cluster_by_bpm exactly once."""
        music_dir = _make_music_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            scan,
            [str(music_dir), "--dry-run", "--target-tracks", "25", "--fast"],
        )
        assert result.exit_code == 0, result.output
        assert patched_scan.count("cluster_by_bpm") == 1
        assert patched_scan.count("cluster_by_features") == 0
        assert len(FakeIntensityAnalyzer.calls) == 0

    def test_fast_flag_appears_in_help(self) -> None:
        """scan --help lists the new --fast flag."""
        runner = CliRunner()
        result = runner.invoke(scan, ["--help"])
        assert result.exit_code == 0, result.output
        assert "--fast" in result.output

    def test_weight_source_string_fast_path(
        self, tmp_path: Path, patched_scan: ClusteringSpy
    ) -> None:
        """--fast prints the BPM-only weight-source string."""
        music_dir = _make_music_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            scan,
            [str(music_dir), "--dry-run", "--target-tracks", "25", "--fast"],
        )
        assert result.exit_code == 0, result.output
        assert "Weight source: uniform (BPM-only clustering)" in result.output

    def test_fast_and_use_embeddings_conflict(
        self, tmp_path: Path, patched_scan: ClusteringSpy
    ) -> None:
        """--fast with --use-embeddings is a usage error naming both flags."""
        music_dir = _make_music_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            scan,
            [str(music_dir), "--dry-run", "--fast", "--use-embeddings"],
        )
        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert "fast" in output_lower
        assert "embeddings" in output_lower

    def test_fast_and_per_genre_conflict(self, tmp_path: Path, patched_scan: ClusteringSpy) -> None:
        """--fast with --cluster-mode per-genre is a usage error."""
        music_dir = _make_music_dir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            scan,
            [
                str(music_dir),
                "--dry-run",
                "--fast",
                "--cluster-mode",
                "per-genre",
            ],
        )
        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert "fast" in output_lower
        assert "per-genre" in output_lower
