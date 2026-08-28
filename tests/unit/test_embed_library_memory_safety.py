"""
Regression tests for TASK-33: the embed_library.py OOM incident.

Incident summary: the first real batch run over the 1,730-track library
froze the workstation and crashed multiple applications, after writing only
7 rows. Root cause, confirmed: ``scripts/embed_library.py``'s production
(non-injected) embedding path constructed a fresh ``EmbeddingExtractor`` --
and therefore a fresh ~924MB TensorFlow/essentia graph and session -- once
PER TRACK instead of once per run. TF/essentia hold C++ resources that
Python's GC does not promptly reclaim, so RSS grew by roughly 1GB per track;
with ~4GB free the machine died after about 7 tracks. The visible symptom
was "TensorflowPredict: Successfully loaded graph file" repeating once per
track instead of once per run.

Existing coverage in ``tests/unit/test_embedding_cache.py::
TestEmbedLibraryScript`` only ever exercises ``embed_library()`` with a
cheap injected fake ``embed_fn`` -- the real, model-backed path
(``main()`` -> the production embed_fn wired from
``playchitect.core.embedding_extractor.EmbeddingExtractor``) had zero
coverage. That gap is the actual defect this file closes.

MEMORY SAFETY (read before touching this file): no test in this module may
construct a real ``EmbeddingExtractor`` or load a real TensorFlow model.
essentia IS installed on the machines that run this suite, so a careless
test here would actually allocate ~924MB and could repeat the incident.
Every test below patches
``playchitect.core.embedding_extractor.EmbeddingExtractor`` with a cheap
Python test double before invoking the production embedding path, and
never imports or references the real class directly.

Assumed interface (TDD phase 1 -- this module does not implement any of
this; it defines the contract the implementation must satisfy):

  * ``embed_library(directory, cache, embed_fn, log_every=..., \
        memory_ceiling_mb: float | None = ...)`` gains a new
    ``memory_ceiling_mb`` keyword argument. When resident memory (as
    reported by the module-level ``_get_rss_mb()`` hook below) exceeds
    this ceiling, the run must abort by raising
    ``MemoryCeilingExceededError`` rather than continuing to the next
    track.
  * ``_get_rss_mb() -> float`` is a module-level function in
    ``scripts/embed_library.py`` that returns current resident set size in
    MiB. It is the single seam these tests patch to simulate memory
    pressure without allocating anything -- whatever the real
    implementation uses internally (``resource.getrusage``, ``psutil``,
    etc.) must be reachable through this name so it stays mockable in
    exactly one place.
  * ``MemoryCeilingExceededError`` is a module-level exception class
    (``RuntimeError`` subclass, mirroring the existing
    ``EmbeddingDimensionMismatchError`` pattern) whose message includes
    both the measured RSS and the configured ceiling, so a human reading
    the crash has an actionable number rather than a bare abort.
  * Every-N-tracks progress log lines (the existing "Embedded %d new
    tracks so far (scanned %d/%d)" message) must additionally carry the
    current RSS figure, so memory growth is visible in the log stream
    during a long run.
  * Whatever internal restructuring fixes the per-track EmbeddingExtractor
    construction (factory function, callable class, lazily-initialised
    module-level singleton, etc.), the production entry point remains
    ``main()`` (the ``click`` command), invoked here via
    ``click.testing.CliRunner`` exactly as a real user would run
    ``uv run python scripts/embed_library.py``. Testing through that
    external boundary -- rather than guessing an internal factory
    function's name -- keeps this test valid regardless of which internal
    shape the implementer picks.
"""

from __future__ import annotations

import importlib.util
import logging
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from click.testing import CliRunner

from playchitect.core.embedding_cache import EmbeddingCache

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "embed_library.py"
_DISCOGS_EFFNET_DIM = 1280


# ── Helpers (mirrors tests/unit/test_embedding_cache.py's established pattern) ──


def _load_embed_library_module() -> types.ModuleType:
    """
    Dynamically load scripts/embed_library.py as a module.

    scripts/ is not an installed package, so the standalone script is loaded
    directly from disk by path, exactly like
    ``TestEmbedLibraryScript._load_embed_library_module`` in
    ``test_embedding_cache.py``.
    """
    spec = importlib.util.spec_from_file_location("embed_library", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_vector(seed: int, dim: int = _DISCOGS_EFFNET_DIM) -> np.ndarray:
    """Return a deterministic synthetic embedding vector -- never a real one."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def _make_fixture_library(tmp_path: Path, n_tracks: int) -> Path:
    library = tmp_path / "library"
    library.mkdir()
    for i in range(n_tracks):
        (library / f"track_{i:03d}.mp3").write_bytes(f"audio content {i}".encode() * 200)
    return library


class _InstanceCountingExtractor:
    """
    Test double for ``EmbeddingExtractor`` that mimics the real class's
    lazy-model-initialisation shape closely enough to catch two distinct
    ways the OOM bug can survive a refactor.

    The real ``EmbeddingExtractor.analyze_discogs_effnet`` builds its
    (expensive, ~924MB) TensorFlow graph lazily on the FIRST call, caching
    it on ``self._model_discogs_effnet`` and reusing it on every later call
    to *that same instance*. This double reproduces exactly that shape with
    two independent counters:

      * ``instantiation_count`` -- how many times ``EmbeddingExtractor()``
        itself was constructed. This is the literal bug reported in the
        incident (one construction per track instead of one per run).
      * ``model_build_count`` -- how many times the (simulated) expensive
        model graph was actually built. This catches a subtler variant: an
        implementation that reuses a single wrapper instance but has that
        instance rebuild its inner model on every call would leak exactly
        as badly, while ``instantiation_count`` alone would misreport it
        as fixed.
    """

    instantiation_count: int = 0
    model_build_count: int = 0

    def __init__(self, *args: object, **kwargs: object) -> None:
        type(self).instantiation_count += 1
        self._model_built = False

    def analyze_discogs_effnet(self, filepath: Path) -> np.ndarray:
        if not self._model_built:
            type(self).model_build_count += 1
            self._model_built = True
        # Deterministic, cheap, and dimensionally correct -- no real model,
        # no real audio decoding, no TensorFlow anywhere near this class.
        seed = abs(hash(str(filepath))) % (2**31)
        return _make_vector(seed=seed)

    @classmethod
    def reset(cls) -> None:
        cls.instantiation_count = 0
        cls.model_build_count = 0


# ── Category 1: the core regression -- instantiation count ─────────────────


class TestRealEmbeddingPathBuildsExtractorOnce:
    """
    TASK-33 regression: the production embedding path must build the
    (model-bearing) EmbeddingExtractor exactly once per `embed_library.py`
    run, never once per track.
    """

    def test_cli_run_constructs_extractor_exactly_once_for_five_tracks(
        self, tmp_path: Path
    ) -> None:
        _InstanceCountingExtractor.reset()
        n_tracks = 5
        library = _make_fixture_library(tmp_path, n_tracks=n_tracks)
        cache_path = tmp_path / "embeddings.parquet"

        with patch(
            "playchitect.core.embedding_extractor.EmbeddingExtractor",
            _InstanceCountingExtractor,
        ):
            mod = _load_embed_library_module()
            runner = CliRunner()
            result = runner.invoke(
                mod.main,
                [str(library), "--cache-path", str(cache_path)],
            )

        assert result.exit_code == 0, (
            f"CLI run failed unexpectedly: {result.output}\n{result.exception}"
        )

        assert _InstanceCountingExtractor.instantiation_count == 1, (
            "TASK-33 regression: EmbeddingExtractor (the ~924MB TensorFlow/"
            "essentia model wrapper) was constructed "
            f"{_InstanceCountingExtractor.instantiation_count} time(s) for a "
            f"{n_tracks}-track run; it must be constructed EXACTLY ONCE per "
            "run and reused for every track. This is the precise defect "
            "that froze the workstation and crashed multiple applications "
            "after writing only 7 rows in the real 1,730-track library run "
            "(each extra instantiation builds a fresh, never-released "
            "TensorFlow graph costing ~924MB resident)."
        )

        # The extractor could in principle be reused as a wrapper while
        # still rebuilding its expensive inner model on every call -- that
        # would leak identically. Guard the property that actually matters.
        assert _InstanceCountingExtractor.model_build_count == 1, (
            "TASK-33 regression: the underlying discogs-effnet model graph "
            f"was (simulated as) built {_InstanceCountingExtractor.model_build_count} "
            "time(s) instead of once, even though the wrapper instantiation "
            "count may look correct. A single EmbeddingExtractor instance "
            "that rebuilds its model per call leaks exactly as badly as "
            "constructing a fresh instance per track -- one model build per "
            "run is the property that prevents the incident, not merely one "
            "wrapper object existing."
        )

        # Non-hollow: confirm the mocked path actually ran to completion and
        # wrote real cache rows, not just that the CLI exited 0.
        cache = EmbeddingCache(cache_path=cache_path)
        for track_path in sorted(library.glob("*.mp3")):
            vector = cache.get(track_path)
            assert vector is not None, f"expected {track_path.name} to be cached"
            assert vector.shape == (_DISCOGS_EFFNET_DIM,)

    def test_cli_run_with_more_tracks_still_constructs_extractor_exactly_once(
        self, tmp_path: Path
    ) -> None:
        """
        A second, larger run guards against an implementation that happens
        to pass the 5-track test by coincidence (e.g. some off-by-one
        "construct once per batch of 5" scheme). 12 tracks, still exactly
        one instantiation.
        """
        _InstanceCountingExtractor.reset()
        n_tracks = 12
        library = _make_fixture_library(tmp_path, n_tracks=n_tracks)
        cache_path = tmp_path / "embeddings.parquet"

        with patch(
            "playchitect.core.embedding_extractor.EmbeddingExtractor",
            _InstanceCountingExtractor,
        ):
            mod = _load_embed_library_module()
            runner = CliRunner()
            result = runner.invoke(
                mod.main,
                [str(library), "--cache-path", str(cache_path), "--log-every", "1"],
            )

        assert result.exit_code == 0, (
            f"CLI run failed unexpectedly: {result.output}\n{result.exception}"
        )
        assert _InstanceCountingExtractor.instantiation_count == 1, (
            f"Expected exactly 1 EmbeddingExtractor construction for a "
            f"{n_tracks}-track run, got "
            f"{_InstanceCountingExtractor.instantiation_count}."
        )


# ── Category 2: memory ceiling ──────────────────────────────────────────────


class TestMemoryCeiling:
    """
    TASK-33: a future leak (even a smaller, slower one than the incident)
    must abort the run with a clear, specifically-named error once
    resident memory crosses a configurable ceiling, rather than silently
    growing until the OS kills the machine.
    """

    def test_embed_library_aborts_with_named_error_when_rss_exceeds_ceiling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _load_embed_library_module()
        library = _make_fixture_library(tmp_path, n_tracks=5)
        cache = EmbeddingCache(cache_path=tmp_path / "embeddings.parquet")

        # Simulate RSS already far above any reasonable ceiling -- no real
        # memory is allocated, the reporting function is simply stubbed.
        monkeypatch.setattr(mod, "_get_rss_mb", lambda: 5000.0)

        call_count = 0

        def fake_embed(path: Path) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return _make_vector(seed=call_count, dim=32)

        with pytest.raises(mod.MemoryCeilingExceededError) as exc_info:
            # log_every=1 makes the abort point independent of whichever
            # check-frequency the implementation picks (per-track vs.
            # per-log_every); see module docstring for the assumed contract.
            mod.embed_library(library, cache, fake_embed, log_every=1, memory_ceiling_mb=100.0)

        message = str(exc_info.value)
        assert "5000" in message, (
            f"error message must report the measured RSS so the abort is "
            f"actionable, got: {message!r}"
        )
        assert "100" in message, (
            f"error message must report the configured ceiling, got: {message!r}"
        )
        assert "memory" in message.lower(), (
            f"error message must clearly name memory/RSS as the cause, got: {message!r}"
        )

        # The run must actually abort partway through, not merely raise
        # after silently finishing the whole library.
        assert call_count < 5, (
            "embed_fn was called for the full library before the memory "
            f"ceiling error was raised (call_count={call_count}); the run "
            "must stop invoking embed_fn as soon as the ceiling is crossed, "
            "not merely raise cosmetically at the end."
        )

    def test_embed_library_generous_ceiling_does_not_abort(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = _load_embed_library_module()
        library = _make_fixture_library(tmp_path, n_tracks=5)
        cache = EmbeddingCache(cache_path=tmp_path / "embeddings.parquet")

        monkeypatch.setattr(mod, "_get_rss_mb", lambda: 200.0)

        call_count = 0

        def fake_embed(path: Path) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return _make_vector(seed=call_count, dim=32)

        count = mod.embed_library(
            library, cache, fake_embed, log_every=1, memory_ceiling_mb=100_000.0
        )

        assert count == 5
        assert call_count == 5

    def test_memory_ceiling_is_configurable_not_hardcoded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        The identical (mocked) RSS reading must produce different outcomes
        depending solely on the caller-supplied ceiling -- proving the
        ceiling is a real parameter, not a hardcoded internal constant.
        """
        mod = _load_embed_library_module()
        library = _make_fixture_library(tmp_path, n_tracks=5)
        monkeypatch.setattr(mod, "_get_rss_mb", lambda: 1000.0)

        cache_low = EmbeddingCache(cache_path=tmp_path / "low.parquet")
        with pytest.raises(mod.MemoryCeilingExceededError):
            mod.embed_library(
                library,
                cache_low,
                lambda p: _make_vector(seed=1, dim=32),
                log_every=1,
                memory_ceiling_mb=500.0,
            )

        cache_high = EmbeddingCache(cache_path=tmp_path / "high.parquet")
        count_high = mod.embed_library(
            library,
            cache_high,
            lambda p: _make_vector(seed=1, dim=32),
            log_every=1,
            memory_ceiling_mb=2000.0,
        )
        assert count_high == 5


# ── Category 3: RSS logged with progress ────────────────────────────────────


class TestProgressLogIncludesMemory:
    """
    TASK-33: every-N-tracks progress lines must carry the current RSS, so
    a leak's growth is visible in the log stream of a long run instead of
    only becoming apparent once the OS kills the process.
    """

    def test_progress_log_message_includes_rss_figure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        mod = _load_embed_library_module()
        n_tracks = 10
        library = _make_fixture_library(tmp_path, n_tracks=n_tracks)
        cache = EmbeddingCache(cache_path=tmp_path / "embeddings.parquet")

        monkeypatch.setattr(mod, "_get_rss_mb", lambda: 512.5)

        def fake_embed(path: Path) -> np.ndarray:
            return _make_vector(seed=1, dim=32)

        with caplog.at_level(logging.INFO):
            count = mod.embed_library(library, cache, fake_embed, log_every=5)

        assert count == n_tracks

        progress_records = [
            r.getMessage()
            for r in caplog.records
            if "Embedded" in r.getMessage() and "so far" in r.getMessage()
        ]
        assert len(progress_records) >= 1, (
            "expected at least one every-N-tracks progress log line for a "
            f"{n_tracks}-track run with log_every=5, got none. All records: "
            f"{[r.getMessage() for r in caplog.records]}"
        )
        assert any("512.5" in msg for msg in progress_records), (
            "progress log lines must include the current RSS (MB) so memory "
            f"growth is visible during a long run; got: {progress_records}"
        )

    def test_progress_log_rss_figure_changes_between_calls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        The logged figure must be a live reading, not a value captured once
        at start-of-run and stamped onto every progress line thereafter --
        that would defeat the entire purpose of watching for growth.
        """
        mod = _load_embed_library_module()
        n_tracks = 10
        library = _make_fixture_library(tmp_path, n_tracks=n_tracks)
        cache = EmbeddingCache(cache_path=tmp_path / "embeddings.parquet")

        rss_readings = iter([100.0, 100.0, 100.0, 100.0, 900.0, 900.0, 900.0, 900.0, 900.0, 900.0])
        monkeypatch.setattr(mod, "_get_rss_mb", lambda: next(rss_readings, 900.0))

        def fake_embed(path: Path) -> np.ndarray:
            return _make_vector(seed=1, dim=32)

        with caplog.at_level(logging.INFO):
            mod.embed_library(library, cache, fake_embed, log_every=5)

        progress_records = [
            r.getMessage()
            for r in caplog.records
            if "Embedded" in r.getMessage() and "so far" in r.getMessage()
        ]
        assert len(progress_records) >= 2, (
            f"expected at least two progress lines (log_every=5 over "
            f"{n_tracks} tracks), got: {progress_records}"
        )
        # The first progress line (at track 5) should reflect the lower
        # early reading; a later line should reflect the higher one -- if
        # the implementation snapshots RSS once, both lines would carry the
        # same figure.
        assert "100.0" in progress_records[0] or "100" in progress_records[0], (
            f"first progress line should reflect the early (lower) RSS "
            f"reading, got: {progress_records[0]!r}"
        )
        assert any("900" in msg for msg in progress_records[1:]), (
            "a later progress line should reflect the later (higher) RSS "
            f"reading, showing growth over time; got: {progress_records}"
        )
