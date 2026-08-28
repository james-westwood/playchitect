"""
Unit tests for the pre-batch embedding smoke check (TASK-31, rail 2).

This gates the library-wide batch embedding run over ~1,726 tracks. The
real failure mode of pinning a dev-line essentia-tensorflow wheel is not
"it won't install" -- that already fails loudly at `import essentia` --
it's "it installs and silently produces garbage": a differently-shaped,
NaN-filled, or non-reproducible embedding vector would poison the entire
cache without ever raising an exception. This smoke check exists to catch
that before the batch run.

Interface under test (PROPOSED by the test writer -- TASK-31 only locks
down that *some* invocable callable must exist; these exact names/shapes
are a proposal, not a locked contract):

    playchitect.core.embedding_extractor._DISCOGS_EFFNET_EMBEDDING_DIM: int
        = 1280. Named constant for discogs-effnet's embedding
        dimensionality, taken from the model card schema at
        https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.json
        (declares outputs predictions [64, 400] and embeddings [64, 1280]),
        verified 2026-08-27.

    EmbeddingExtractor.analyze_discogs_effnet(filepath: Path) -> np.ndarray
        Mirrors the existing analyze() MusiCNN pathway, but runs the
        discogs-effnet model and returns the raw mean-pooled 1280-dim
        vector uncached (this is a diagnostic/smoke path, distinct from the
        production PCA-reduced embedding-cache path in embedding_cache.py).
        Raises FileNotFoundError for a missing file, matching analyze().

    playchitect.core.embedding_extractor.EmbeddingSmokeCheckError(RuntimeError)
        Raised by run_embedding_smoke_check() on any failed assertion, with
        a message specific enough to tell an operator which check failed
        (dimensionality / finiteness / norm band / reproducibility).

    playchitect.core.embedding_extractor.run_embedding_smoke_check(
        fixture_path: Path,
    ) -> None
        Runs `fixture_path` through TWO independent EmbeddingExtractor
        instances (cache disabled, so a synthetic smoke fixture never
        pollutes the production embedding cache) and raises
        EmbeddingSmokeCheckError if dimensionality, finiteness, L2-norm
        sanity, or cross-instance reproducibility fail. Returns None (no
        exception) on success. Must be callable standalone -- e.g. from
        scripts/embed_library.py as a pre-flight gate -- without any pytest
        fixture.

Because essentia-tensorflow is NOT installed in the default dev environment
(and is expected to stay that way -- see the TASK-19/TASK-31 notes in
prd.json), every test in TestSmokeCheckAgainstRealModel below is guarded by
`@pytest.mark.skipif(not emb_mod._ESSENTIA_AVAILABLE, ...)`, checked against
the flag that already exists in the shipped module today. Crucially, this
file never does `from playchitect.core.embedding_extractor import
run_embedding_smoke_check` at module level -- every not-yet-implemented
symbol is looked up lazily as `emb_mod.<name>` *inside* each test body, so
the module always imports cleanly and the skip decision is made before any
of those symbols are ever touched. That is what lets
TestSmokeCheckAgainstRealModel report a clean, correctly-reasoned SKIP on
this machine right now, rather than a collection error.

The other test classes below mock EmbeddingExtractor.analyze_discogs_effnet
directly (no real essentia involved) and exercise the smoke check's own
threshold/error logic -- dimension mismatch, NaN, degenerate norm,
non-reproducibility. These are expected to run (not skip) right now, and
fail with AttributeError because the symbols above do not exist yet.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import playchitect.core.embedding_extractor as emb_mod
from playchitect.core.embedding_extractor import EmbeddingExtractor

# Deliberately NOT imported from the module under test -- this file asserts
# the real number from the model card even if the implementation's own
# constant is wrong or missing.
_EXPECTED_DIM = 1280


def _write_synthetic_wav(
    path: Path, duration_seconds: float = 5.0, sample_rate: int = 16000
) -> None:
    """
    Write a short synthetic sine-wave WAV fixture.

    The smoke check only cares about the model's output shape/finiteness/
    reproducibility, not musical content, so a synthetic tone is sufficient
    and keeps this test file free of any committed binary fixture.

    Default duration is 5.0s, comfortably above discogs-effnet's ~2.048s
    minimum (128 mel frames * 256-sample hop / 16 kHz). Below that
    threshold the model returns zero frames, and mean-pooling an empty
    array silently collapses to a NaN scalar rather than a (1280,) vector
    -- which would fail these tests for a reason unrelated to what they are
    actually testing. Kept well clear of the cliff edge in case the
    model's framing shifts slightly.
    """
    import soundfile as sf  # noqa: PLC0415

    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    sf.write(str(path), audio, sample_rate)


# ── TestSmokeCheckAgainstRealModel ──────────────────────────────────────────


@pytest.mark.skipif(
    not emb_mod._ESSENTIA_AVAILABLE,
    reason=(
        "essentia-tensorflow is not installed in this environment. The "
        "pre-batch smoke check (TASK-31 rail 2) can only meaningfully "
        "exercise the real discogs-effnet model on a machine where "
        "essentia-tensorflow==2.1b6.dev1389 is installed; this is expected "
        "to be the case for the default dev environment and CI, per "
        "prd.json TASK-31."
    ),
)
class TestSmokeCheckAgainstRealModel:
    """
    Genuine, unmocked exercise of the smoke check against the real
    discogs-effnet model.

    Must NOT be mocked: a mocked "smoke test" would defeat the entire
    purpose of this rail, which is to catch a real dev-line wheel silently
    misbehaving. On a machine where essentia is installed, these tests
    actually load the model and run inference.
    """

    def test_smoke_check_passes_silently_on_valid_fixture(self, tmp_path: Path) -> None:
        fixture = tmp_path / "smoke_fixture.wav"
        _write_synthetic_wav(fixture)

        # Must not raise -- this is the actual gate the batch script relies on.
        emb_mod.run_embedding_smoke_check(fixture)  # ty: ignore[unresolved-attribute]

    def test_smoke_check_produces_expected_dimensionality_and_finiteness(
        self, tmp_path: Path
    ) -> None:
        fixture = tmp_path / "smoke_fixture.wav"
        _write_synthetic_wav(fixture)

        extractor = EmbeddingExtractor(cache_enabled=False)
        vector = extractor.analyze_discogs_effnet(fixture)  # ty: ignore[unresolved-attribute]

        assert vector.shape == (_EXPECTED_DIM,)
        assert np.all(np.isfinite(vector))

    def test_smoke_check_is_reproducible_across_separate_extractor_instances(
        self, tmp_path: Path
    ) -> None:
        fixture = tmp_path / "smoke_fixture.wav"
        _write_synthetic_wav(fixture)

        vector_a = EmbeddingExtractor(cache_enabled=False).analyze_discogs_effnet(fixture)  # ty: ignore[unresolved-attribute]
        vector_b = EmbeddingExtractor(cache_enabled=False).analyze_discogs_effnet(fixture)  # ty: ignore[unresolved-attribute]

        np.testing.assert_allclose(vector_a, vector_b, atol=1e-5)

    def test_missing_fixture_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            emb_mod.run_embedding_smoke_check(tmp_path / "does_not_exist.wav")  # ty: ignore[unresolved-attribute]


# ── TestAnalyzeDiscogsEffnetContract ────────────────────────────────────────


class TestAnalyzeDiscogsEffnetContract:
    """
    Minimal, always-run contract checks for analyze_discogs_effnet() that do
    not require a real model, mirroring
    TestAnalyzeAdditional.test_analyze_nonexistent_file_raises for the
    existing MusiCNN analyze() method in test_embedding_extractor.py.
    """

    @pytest.fixture()
    def extractor(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EmbeddingExtractor:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)
        return EmbeddingExtractor(
            cache_enabled=False,
            model_path=tmp_path / "fake.pb",
            discogs_effnet_model_path=tmp_path / "fake_discogs.pb",
        )

    def test_missing_file_raises_file_not_found(
        self, extractor: EmbeddingExtractor, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            extractor.analyze_discogs_effnet(tmp_path / "does_not_exist.mp3")  # ty: ignore[unresolved-attribute]


# ── TestRunEmbeddingSmokeCheckLogic ─────────────────────────────────────────


class TestRunEmbeddingSmokeCheckLogic:
    """
    Exercises run_embedding_smoke_check()'s own threshold/error logic by
    monkeypatching EmbeddingExtractor.analyze_discogs_effnet to return
    controlled vectors.

    This tests the smoke check's real arithmetic (shape / finite /
    norm-band / reproducibility comparisons) -- it is deliberately NOT a
    substitute for TestSmokeCheckAgainstRealModel above, which is the only
    class that exercises the genuine model.
    """

    @pytest.fixture()
    def fixture_path(self, tmp_path: Path) -> Path:
        p = tmp_path / "fixture.wav"
        p.write_bytes(b"\x00" * 100)
        return p

    def _patch_discogs_effnet(
        self,
        monkeypatch: pytest.MonkeyPatch,
        side_effect: Callable[[], np.ndarray],
    ) -> None:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)

        def fake_analyze(self: EmbeddingExtractor, filepath: Path) -> np.ndarray:
            if not filepath.exists():
                raise FileNotFoundError(filepath)
            return side_effect()

        monkeypatch.setattr(
            EmbeddingExtractor, "analyze_discogs_effnet", fake_analyze, raising=False
        )

    def test_passes_for_well_formed_vector(
        self, monkeypatch: pytest.MonkeyPatch, fixture_path: Path
    ) -> None:
        vec = np.full(1280, 0.05, dtype=np.float32)
        self._patch_discogs_effnet(monkeypatch, lambda: vec.copy())

        # Must not raise.
        emb_mod.run_embedding_smoke_check(fixture_path)  # ty: ignore[unresolved-attribute]

    def test_raises_on_wrong_dimensionality(
        self, monkeypatch: pytest.MonkeyPatch, fixture_path: Path
    ) -> None:
        wrong_dim_vec = np.ones(128, dtype=np.float32)  # MusiCNN dim, not discogs-effnet
        self._patch_discogs_effnet(monkeypatch, lambda: wrong_dim_vec.copy())

        with pytest.raises(emb_mod.EmbeddingSmokeCheckError, match="1280"):  # ty: ignore[unresolved-attribute]
            emb_mod.run_embedding_smoke_check(fixture_path)  # ty: ignore[unresolved-attribute]

    def test_raises_on_nan_values(
        self, monkeypatch: pytest.MonkeyPatch, fixture_path: Path
    ) -> None:
        vec = np.full(1280, 0.05, dtype=np.float32)
        vec[3] = np.nan
        self._patch_discogs_effnet(monkeypatch, lambda: vec.copy())

        with pytest.raises(emb_mod.EmbeddingSmokeCheckError, match="(?i)nan|finite"):  # ty: ignore[unresolved-attribute]
            emb_mod.run_embedding_smoke_check(fixture_path)  # ty: ignore[unresolved-attribute]

    def test_raises_on_inf_values(
        self, monkeypatch: pytest.MonkeyPatch, fixture_path: Path
    ) -> None:
        vec = np.full(1280, 0.05, dtype=np.float32)
        vec[10] = np.inf
        self._patch_discogs_effnet(monkeypatch, lambda: vec.copy())

        with pytest.raises(emb_mod.EmbeddingSmokeCheckError, match="(?i)inf|finite"):  # ty: ignore[unresolved-attribute]
            emb_mod.run_embedding_smoke_check(fixture_path)  # ty: ignore[unresolved-attribute]

    def test_raises_on_near_zero_norm(
        self, monkeypatch: pytest.MonkeyPatch, fixture_path: Path
    ) -> None:
        """
        A near-zero-everywhere embedding is the classic 'model loaded but
        produced garbage' failure mode (e.g. a mis-wired graph output layer
        that returns an all-but-empty tensor).
        """
        vec = np.full(1280, 1e-9, dtype=np.float32)
        self._patch_discogs_effnet(monkeypatch, lambda: vec.copy())

        with pytest.raises(emb_mod.EmbeddingSmokeCheckError, match="(?i)norm"):  # ty: ignore[unresolved-attribute]
            emb_mod.run_embedding_smoke_check(fixture_path)  # ty: ignore[unresolved-attribute]

    def test_raises_on_absurdly_large_norm(
        self, monkeypatch: pytest.MonkeyPatch, fixture_path: Path
    ) -> None:
        vec = np.full(1280, 1e8, dtype=np.float32)
        self._patch_discogs_effnet(monkeypatch, lambda: vec.copy())

        with pytest.raises(emb_mod.EmbeddingSmokeCheckError, match="(?i)norm"):  # ty: ignore[unresolved-attribute]
            emb_mod.run_embedding_smoke_check(fixture_path)  # ty: ignore[unresolved-attribute]

    def test_raises_on_non_reproducible_output(
        self, monkeypatch: pytest.MonkeyPatch, fixture_path: Path
    ) -> None:
        """
        Two calls returning meaningfully different vectors must fail -- this
        is what a buggy wheel leaving e.g. dropout/batchnorm in training
        mode would look like: plausible-looking output that silently
        changes between runs.
        """
        calls = {"n": 0}

        def flaky() -> np.ndarray:
            calls["n"] += 1
            base = np.full(1280, 0.05, dtype=np.float32)
            if calls["n"] == 1:
                return base
            return base + 5.0  # far outside any reasonable tolerance

        self._patch_discogs_effnet(monkeypatch, flaky)

        with pytest.raises(emb_mod.EmbeddingSmokeCheckError, match="(?i)reproduc"):  # ty: ignore[unresolved-attribute]
            emb_mod.run_embedding_smoke_check(fixture_path)  # ty: ignore[unresolved-attribute]

    def test_missing_fixture_raises_file_not_found_not_smoke_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A missing fixture is a caller error, not a model-quality failure --
        it must surface as FileNotFoundError, not be swallowed into a
        generic EmbeddingSmokeCheckError."""
        self._patch_discogs_effnet(monkeypatch, lambda: np.full(1280, 0.05, dtype=np.float32))

        with pytest.raises(FileNotFoundError):
            emb_mod.run_embedding_smoke_check(tmp_path / "does_not_exist.wav")  # ty: ignore[unresolved-attribute]
