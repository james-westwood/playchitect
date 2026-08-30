"""
Integration tests for the `playchitect scan --use-embeddings` diagnostic path.

TASK-32. When the MusiCNN embedding step produces nothing at all, the user
currently sees only a generic "Error: Clustering failed" -- the real cause
(every clip being shorter than MusiCNN's ~2.96 s framing minimum) is buried in
a per-file WARNING log line. These tests pin the aggregate diagnostic the CLI
must emit instead.

The embedding model is faked out entirely, so nothing here needs
essentia-tensorflow installed and no model is downloaded.
"""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner, Result
from mutagen.flac import FLAC

import playchitect.core.embedding_extractor as emb_mod
from playchitect.cli.commands import scan
from playchitect.core.embedding_extractor import (
    AudioTooShortForEmbeddingError,
    EmbeddingFeatures,
)

_SAMPLE_RATE = 44100
_CLIP_SECONDS = 0.5
_TRACK_COUNT = 10

# Name of the production constant recording the measured MusiCNN minimum
# (47361 samples at 16 kHz = 2.9600625 s). See TestMusiCNNZeroFrameGuard in
# tests/unit/test_embedding_extractor.py for the measurement itself.
_MUSICNN_MIN_CONST_NAME = "_MUSICNN_MIN_AUDIO_SECONDS"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_flac(path: Path, bpm: float) -> None:
    """Write a short silent stereo FLAC with an embedded BPM tag."""
    samples = np.zeros((int(_SAMPLE_RATE * _CLIP_SECONDS), 2), dtype=np.float32)
    sf.write(str(path), samples, _SAMPLE_RATE, format="FLAC", subtype="PCM_16")
    audio = FLAC(str(path))
    audio["bpm"] = str(int(round(bpm)))
    audio.save()


def _make_features(path: Path, seed: int) -> EmbeddingFeatures:
    """A plausible EmbeddingFeatures the clusterer can actually consume."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(128).astype(np.float32)
    return EmbeddingFeatures(
        filepath=path,
        file_hash=f"hash{seed:04d}",
        embedding=vec / float(np.linalg.norm(vec)),
        top_tags=[("techno", 0.9), ("electronic", 0.6)],
        moods=[("Aggressive", 0.7), ("Passionate", 0.3)],
    )


def _too_short_error(path: Path) -> AudioTooShortForEmbeddingError:
    """The exception analyze() must raise for a sub-minimum clip."""
    return AudioTooShortForEmbeddingError(
        f"Audio file '{path.name}' is 0.500s long, which is too short for "
        "MusiCNN to produce a single embedding frame (requires approximately "
        "2.960s or more)."
    )


def _install_fake_extractor(
    monkeypatch: pytest.MonkeyPatch,
    analyze_fn: Callable[[Path], EmbeddingFeatures],
) -> None:
    """
    Replace EmbeddingExtractor with a stub driven by ``analyze_fn``.

    The CLI imports EmbeddingExtractor lazily from the module, so patching the
    module attribute is enough and keeps the test independent of whether
    essentia-tensorflow is installed.
    """

    class FakeEmbeddingExtractor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.analyzed: list[Path] = []

        def analyze(self, filepath: Path) -> EmbeddingFeatures:
            self.analyzed.append(filepath)
            return analyze_fn(filepath)

        def infer_genre(self, features: EmbeddingFeatures) -> str | None:
            return "techno"

    monkeypatch.setattr(emb_mod, "EmbeddingExtractor", FakeEmbeddingExtractor)


def _musicnn_min_seconds() -> float:
    """Return the production MusiCNN minimum constant, failing if absent."""
    value = getattr(emb_mod, _MUSICNN_MIN_CONST_NAME, None)
    assert value is not None, (
        f"embedding_extractor must define a module-level {_MUSICNN_MIN_CONST_NAME} "
        "constant so the CLI can cite the required minimum duration"
    )
    return float(value)


def _has_count(output: str, succeeded: int, attempted: int) -> bool:
    """True when the output states 'succeeded of attempted' in some form."""
    pattern = re.compile(
        rf"\b{succeeded}\b\s*(?:of|/|out of)\s*\b{attempted}\b",
        re.IGNORECASE,
    )
    return pattern.search(output) is not None


def _run_scan(music_dir: Path, analyze_fn: Callable[[Path], EmbeddingFeatures]) -> Result:
    with pytest.MonkeyPatch.context() as mp:
        _install_fake_extractor(mp, analyze_fn)
        return CliRunner().invoke(
            scan,
            [str(music_dir), "--dry-run", "--use-embeddings", "--target-tracks", "25"],
        )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def music_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Ten BPM-tagged clips in two tempo groups, all far below the MusiCNN minimum."""
    directory = tmp_path_factory.mktemp("embedding_diagnostics_music")
    for i in range(_TRACK_COUNT // 2):
        _write_flac(directory / f"aaa_{i:02d}.flac", bpm=128.0)
    for i in range(_TRACK_COUNT // 2):
        _write_flac(directory / f"bbb_{i:02d}.flac", bpm=140.0)
    return directory


@pytest.fixture(scope="module")
def zero_success_result(music_dir: Path) -> Result:
    """Every track fails with AudioTooShortForEmbeddingError -- the TASK-32 case."""

    def analyze(filepath: Path) -> EmbeddingFeatures:
        raise _too_short_error(filepath)

    return _run_scan(music_dir, analyze)


@pytest.fixture(scope="module")
def mixed_reason_result(music_dir: Path) -> Result:
    """
    Zero successes from two different causes: the five alphabetically-first
    tracks fail with a decode error, the other five with the too-short error.
    Ordering matters -- an implementation that merely echoes the *first*
    exception it saw would report the decode error and fail the assertions.
    """

    def analyze(filepath: Path) -> EmbeddingFeatures:
        if filepath.name.startswith("aaa_"):
            raise ValueError("libsndfile decode error 17")
        raise _too_short_error(filepath)

    return _run_scan(music_dir, analyze)


@pytest.fixture(scope="module")
def partial_success_result(music_dir: Path) -> Result:
    """Six of ten tracks embed successfully; the run must not abort."""

    def analyze(filepath: Path) -> EmbeddingFeatures:
        index = int(filepath.stem.split("_")[1])
        if filepath.name.startswith("aaa_") and index >= 1:
            raise _too_short_error(filepath)
        return _make_features(filepath, seed=hash(filepath.name) % 1000)

    return _run_scan(music_dir, analyze)


# ── Zero-success aggregate diagnostic ─────────────────────────────────────────


class TestZeroEmbeddingsAggregateError:
    """
    When --use-embeddings yields no embeddings at all, the CLI must fail at the
    embedding stage with a diagnostic that identifies the cause.
    """

    def test_exits_nonzero(self, zero_success_result: Result) -> None:
        assert zero_success_result.exit_code != 0

    def test_names_audio_length_as_the_cause(self, zero_success_result: Result) -> None:
        """Not just "empty" or "failed" -- the message must say *why*."""
        lowered = zero_success_result.output.lower()
        assert any(phrase in lowered for phrase in ("too short", "shorter than", "audio length")), (
            f"error must name the too-short cause:\n{zero_success_result.output}"
        )

    def test_reports_attempted_and_succeeded_counts(self, zero_success_result: Result) -> None:
        assert _has_count(zero_success_result.output, 0, _TRACK_COUNT), (
            "error must report how many of the attempted tracks embedded "
            f"(expected '0 of {_TRACK_COUNT}'):\n{zero_success_result.output}"
        )

    def test_cites_the_minimum_duration_constant(self, zero_success_result: Result) -> None:
        """The user needs the actual number to know how to fix their library."""
        minimum = _musicnn_min_seconds()
        candidates = (f"{minimum:.1f}", f"{minimum:.2f}", f"{minimum:.3f}", repr(minimum))
        assert any(candidate in zero_success_result.output for candidate in candidates), (
            f"error must cite the {minimum}s MusiCNN minimum:\n{zero_success_result.output}"
        )

    def test_suggests_rerunning_without_the_flag(self, zero_success_result: Result) -> None:
        output = zero_success_result.output
        assert "--use-embeddings" in output
        assert any(
            word in output.lower() for word in ("without", "omit", "drop", "re-run", "rerun")
        ), f"error must suggest retrying without --use-embeddings:\n{output}"

    def test_does_not_degrade_into_the_generic_clustering_error(
        self, zero_success_result: Result
    ) -> None:
        """
        The failure must be raised at the CLI's embedding stage. Falling
        through to the clusterer produces "Error: Clustering failed", which
        tells the user nothing about embeddings.
        """
        assert "Clustering failed" not in zero_success_result.output

    def test_names_the_dominant_failure_reason(self, mixed_reason_result: Result) -> None:
        """
        With five decode failures and five too-short failures, the summary must
        surface the too-short reason rather than only the first exception seen.
        """
        lowered = mixed_reason_result.output.lower()
        assert mixed_reason_result.exit_code != 0
        assert any(phrase in lowered for phrase in ("too short", "shorter than")), (
            "aggregate error must report the most common failure reason:\n"
            f"{mixed_reason_result.output}"
        )


# ── Partial success must not abort ────────────────────────────────────────────


class TestPartialEmbeddingSuccess:
    """The hard failure is reserved for the zero-success case."""

    def test_run_completes_successfully(self, partial_success_result: Result) -> None:
        assert partial_success_result.exit_code == 0, partial_success_result.output
        assert "Would create" in partial_success_result.output

    def test_no_aggregate_embedding_error_is_raised(self, partial_success_result: Result) -> None:
        assert "Clustering failed" not in partial_success_result.output
        assert "Error:" not in partial_success_result.output

    def test_reports_attempted_and_succeeded_counts(self, partial_success_result: Result) -> None:
        """Six of ten embedded — the user should be told, not left guessing."""
        assert _has_count(partial_success_result.output, 6, _TRACK_COUNT), (
            f"expected a '6 of {_TRACK_COUNT}' embedding summary:\n{partial_success_result.output}"
        )
