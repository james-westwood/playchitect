"""
Unit tests for the content-addressed embedding cache (TASK-19).

Covers:
  * EmbeddingCache.put / EmbeddingCache.get round-trip semantics
  * Content-hash keying: sha256(first 1 MB) + file size, NOT the filesystem path
  * scripts/embed_library.py idempotency (second run embeds 0 new tracks)
  * PCA(n_components=64, whiten=True) fit / persist / reload contract

The EmbeddingCache/EmbeddingFeatures/PCA plumbing does not exist yet (TDD phase 1).
These tests are expected to fail at COLLECTION time with
``ModuleNotFoundError: No module named 'playchitect.core.embedding_cache'``
until the module is implemented. That failure mode is the intended "red" state.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import types
from pathlib import Path

import numpy as np
import pytest

# NOTE: this import is expected to fail right now (module does not exist yet).
# Keeping it as the *first* import in the file means the collection error
# reported by pytest points straight at the missing module, not at a
# secondary missing dependency (e.g. pyarrow).
from playchitect.core.embedding_cache import EmbeddingCache, fit_and_save_pca

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "embed_library.py"

_EMBEDDING_DIM = 1280  # discogs-effnet embedding dimensionality


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_vector(seed: int, dim: int = _EMBEDDING_DIM) -> np.ndarray:
    """Return a deterministic synthetic embedding vector."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


def _load_embed_library_module() -> types.ModuleType:
    """
    Dynamically load scripts/embed_library.py as a module.

    scripts/ is not an installed package, so the standalone script is loaded
    directly from disk by path. Raises FileNotFoundError until the script
    exists (acceptable red-phase failure).
    """
    spec = importlib.util.spec_from_file_location("embed_library", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ── TestEmbeddingCachePutGet ─────────────────────────────────────────────────


class TestEmbeddingCachePutGet:
    """put()/get() round-trip contract."""

    def test_roundtrip_preserves_vector_values_and_dtype(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(cache_path=tmp_path / "embeddings.parquet")
        track = tmp_path / "track.mp3"
        track.write_bytes(b"synthetic audio bytes " * 2000)
        vector = _make_vector(seed=1)

        cache.put(track, vector)
        result = cache.get(track)

        assert result is not None
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, vector)

    def test_get_returns_none_for_unknown_file(self, tmp_path: Path) -> None:
        """A file that was never put() into a populated cache is a cache miss."""
        cache = EmbeddingCache(cache_path=tmp_path / "embeddings.parquet")
        known = tmp_path / "known.mp3"
        known.write_bytes(b"known content " * 500)
        cache.put(known, _make_vector(seed=2))

        unknown = tmp_path / "unknown.mp3"
        unknown.write_bytes(b"never cached content " * 500)

        assert cache.get(unknown) is None

    def test_empty_cache_get_returns_none_without_error(self, tmp_path: Path) -> None:
        """
        A brand-new cache (backing parquet file does not exist on disk yet)
        must handle get() gracefully rather than raising e.g. FileNotFoundError.
        """
        cache = EmbeddingCache(cache_path=tmp_path / "embeddings.parquet")
        f = tmp_path / "track.mp3"
        f.write_bytes(b"some audio bytes")

        assert cache.get(f) is None

    def test_put_overwrites_existing_entry_for_same_content(self, tmp_path: Path) -> None:
        """Putting a new vector for the same content hash replaces the old one."""
        cache = EmbeddingCache(cache_path=tmp_path / "embeddings.parquet")
        track = tmp_path / "track.mp3"
        track.write_bytes(b"stable content " * 500)

        first_vector = _make_vector(seed=10)
        second_vector = _make_vector(seed=11)

        cache.put(track, first_vector)
        cache.put(track, second_vector)

        result = cache.get(track)
        assert result is not None
        np.testing.assert_array_equal(result, second_vector)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(result, first_vector)


# ── TestContentHashKeying ────────────────────────────────────────────────────


class TestContentHashKeying:
    """
    Cache keying must be by content hash (sha256 of first 1 MB + file size),
    not filesystem path.
    """

    def test_get_succeeds_after_file_renamed_same_bytes(self, tmp_path: Path) -> None:
        """Same bytes, different path -> cache hit (proves keying is by content)."""
        cache = EmbeddingCache(cache_path=tmp_path / "embeddings.parquet")
        data = b"identical audio payload " * 3000

        original = tmp_path / "original_name.mp3"
        original.write_bytes(data)
        vector = _make_vector(seed=20)
        cache.put(original, vector)

        moved_dir = tmp_path / "renamed_subdir"
        moved_dir.mkdir()
        moved = moved_dir / "completely_different_name.mp3"
        moved.write_bytes(data)  # exact same bytes, new location + new filename

        result = cache.get(moved)
        assert result is not None
        np.testing.assert_array_equal(result, vector)

    def test_get_does_not_return_stale_vector_for_changed_content(self, tmp_path: Path) -> None:
        """
        Different bytes at the SAME path must NOT return the previously cached
        vector (proves keying isn't just "ignore content, key by path").
        """
        cache = EmbeddingCache(cache_path=tmp_path / "embeddings.parquet")
        path = tmp_path / "track.mp3"

        path.write_bytes(b"version one of the audio file " * 3000)
        vector_v1 = _make_vector(seed=30)
        cache.put(path, vector_v1)

        # Overwrite the SAME path with materially different content, without
        # re-calling put() for the new content.
        path.write_bytes(b"a totally different re-encoded audio file " * 3000)

        result = cache.get(path)
        assert result is None

    def test_same_1mb_prefix_but_different_size_is_a_different_entry(self, tmp_path: Path) -> None:
        """
        Spec: content hash = sha256(first 1 MB) + file size. Two files sharing
        an identical <1MB prefix but differing in total size must hash
        differently -- otherwise a truncated/extended file would silently
        alias onto a different track's cached embedding.
        """
        cache = EmbeddingCache(cache_path=tmp_path / "embeddings.parquet")
        shared_prefix = b"shared audio prefix data " * 4000  # well under 1 MB

        short = tmp_path / "short.mp3"
        short.write_bytes(shared_prefix)
        cache.put(short, _make_vector(seed=40))

        long = tmp_path / "long.mp3"
        long.write_bytes(shared_prefix + b"extra trailing bytes that change size only")

        assert cache.get(long) is None

    def test_small_file_under_1mb_still_hashes_and_caches(self, tmp_path: Path) -> None:
        """A file far smaller than the 1 MB hashing window must still work."""
        cache = EmbeddingCache(cache_path=tmp_path / "embeddings.parquet")
        tiny = tmp_path / "tiny.mp3"
        tiny.write_bytes(b"tiny")  # 4 bytes, nowhere near 1 MB
        vector = _make_vector(seed=50, dim=64)

        cache.put(tiny, vector)
        result = cache.get(tiny)

        assert result is not None
        np.testing.assert_array_equal(result, vector)


# ── TestEmbeddingCacheSchema ─────────────────────────────────────────────────


class TestEmbeddingCacheSchema:
    """Verify the on-disk parquet schema matches the documented contract."""

    def test_persisted_parquet_has_documented_columns(self, tmp_path: Path) -> None:
        pytest.importorskip("pyarrow")
        import pyarrow.parquet as pq  # noqa: PLC0415

        cache_path = tmp_path / "embeddings.parquet"
        cache = EmbeddingCache(cache_path=cache_path, model_version="discogs-effnet-1")
        track = tmp_path / "track.mp3"
        track.write_bytes(b"schema check audio bytes " * 500)
        cache.put(track, _make_vector(seed=60))

        assert cache_path.exists()
        table = pq.read_table(cache_path)
        columns = set(table.column_names)
        expected = {"content_hash", "path", "model_version", "embedding", "created_at"}
        assert expected.issubset(columns)

        model_versions = table.column("model_version").to_pylist()
        assert "discogs-effnet-1" in model_versions


# ── TestEmbedLibraryScript ───────────────────────────────────────────────────


class TestEmbedLibraryScript:
    """
    scripts/embed_library.py: walks a library dir, skips already-cached
    tracks, embeds the rest, logs progress every 50 tracks, and is
    idempotent (a second run over the same directory embeds 0 new tracks).

    The embedding function itself is injected (embed_fn) so these tests never
    require essentia/librosa -- they exercise only the cache/skip/idempotency
    logic that scripts/embed_library.py is responsible for.
    """

    def _make_fixture_library(self, tmp_path: Path, n_tracks: int) -> Path:
        library = tmp_path / "library"
        library.mkdir()
        for i in range(n_tracks):
            (library / f"track_{i:03d}.mp3").write_bytes(f"audio content {i}".encode() * 200)
        return library

    def test_first_run_embeds_every_track_in_fixture_dir(self, tmp_path: Path) -> None:
        mod = _load_embed_library_module()
        library = self._make_fixture_library(tmp_path, n_tracks=5)
        cache = EmbeddingCache(cache_path=tmp_path / "embeddings.parquet")

        call_count = 0

        def fake_embed(path: Path) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return _make_vector(seed=call_count, dim=32)

        newly_embedded = mod.embed_library(library, cache, fake_embed)

        assert newly_embedded == 5
        assert call_count == 5

    def test_second_run_embeds_zero_new_tracks(self, tmp_path: Path) -> None:
        mod = _load_embed_library_module()
        library = self._make_fixture_library(tmp_path, n_tracks=5)
        cache = EmbeddingCache(cache_path=tmp_path / "embeddings.parquet")

        call_count = 0

        def fake_embed(path: Path) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return _make_vector(seed=call_count, dim=32)

        first_run_count = mod.embed_library(library, cache, fake_embed)
        calls_after_first_run = call_count

        second_run_count = mod.embed_library(library, cache, fake_embed)

        assert first_run_count == 5
        assert second_run_count == 0
        # The embed function must not have been invoked again on the second run.
        assert call_count == calls_after_first_run

    def test_adding_one_new_track_embeds_exactly_one_on_third_run(self, tmp_path: Path) -> None:
        """Idempotency must be per-track, not all-or-nothing for the whole dir."""
        mod = _load_embed_library_module()
        library = self._make_fixture_library(tmp_path, n_tracks=5)
        cache = EmbeddingCache(cache_path=tmp_path / "embeddings.parquet")

        call_count = 0

        def fake_embed(path: Path) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return _make_vector(seed=call_count, dim=32)

        mod.embed_library(library, cache, fake_embed)  # run 1: embeds 5
        mod.embed_library(library, cache, fake_embed)  # run 2: embeds 0

        (library / "track_new.mp3").write_bytes(b"brand new track content " * 200)
        third_run_count = mod.embed_library(library, cache, fake_embed)

        assert third_run_count == 1
        assert call_count == 6

    def test_logs_progress_every_50_tracks(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        mod = _load_embed_library_module()
        n_tracks = 120
        library = self._make_fixture_library(tmp_path, n_tracks=n_tracks)
        cache = EmbeddingCache(cache_path=tmp_path / "embeddings.parquet")

        def fake_embed(path: Path) -> np.ndarray:
            return _make_vector(seed=1, dim=32)

        with caplog.at_level(logging.INFO):
            count = mod.embed_library(library, cache, fake_embed, log_every=50)

        assert count == n_tracks

        progress_messages = [
            r.getMessage() for r in caplog.records if re.search(r"\b(50|100)\b", r.getMessage())
        ]
        # Expect at least one progress log at the 50-track mark and one at 100.
        assert len(progress_messages) >= 2

    def test_empty_library_dir_embeds_zero_tracks(self, tmp_path: Path) -> None:
        mod = _load_embed_library_module()
        library = tmp_path / "empty_library"
        library.mkdir()
        cache = EmbeddingCache(cache_path=tmp_path / "embeddings.parquet")

        def fake_embed(path: Path) -> np.ndarray:
            raise AssertionError("embed_fn must not be called for an empty directory")

        count = mod.embed_library(library, cache, fake_embed)
        assert count == 0


# ── TestPCAFitAndPersist ─────────────────────────────────────────────────────


class TestPCAFitAndPersist:
    """
    PCA(n_components=64, whiten=True) fit on the embedding matrix, persisted
    to a joblib file, and reloadable with identical components.
    """

    def _fit_source_matrix(self, n_samples: int = 200, dim: int = _EMBEDDING_DIM) -> np.ndarray:
        rng = np.random.default_rng(99)
        return rng.standard_normal((n_samples, dim)).astype(np.float32)

    def test_pca_reloads_with_identical_components(self, tmp_path: Path) -> None:
        import joblib  # noqa: PLC0415

        embeddings = self._fit_source_matrix()
        output_path = tmp_path / "embedding_pca.joblib"

        pca = fit_and_save_pca(embeddings, output_path, n_components=64)

        assert output_path.exists()
        reloaded = joblib.load(output_path)

        np.testing.assert_allclose(reloaded.components_, pca.components_)
        np.testing.assert_allclose(reloaded.mean_, pca.mean_)

    def test_pca_whiten_true_survives_roundtrip(self, tmp_path: Path) -> None:
        """
        whiten=True is load-bearing for TASK-27's diagonal Mahalanobis weights
        to be interpretable as feature importance. A future refactor that
        silently drops whiten must fail this test.
        """
        import joblib  # noqa: PLC0415

        embeddings = self._fit_source_matrix()
        output_path = tmp_path / "embedding_pca.joblib"

        fit_and_save_pca(embeddings, output_path, n_components=64)
        reloaded = joblib.load(output_path)

        assert reloaded.whiten is True

    def test_pca_n_components_is_64_by_default(self, tmp_path: Path) -> None:
        import joblib  # noqa: PLC0415

        embeddings = self._fit_source_matrix()
        output_path = tmp_path / "embedding_pca.joblib"

        pca = fit_and_save_pca(embeddings, output_path)  # default n_components

        assert pca.n_components == 64
        assert pca.components_.shape == (64, _EMBEDDING_DIM)

        reloaded = joblib.load(output_path)
        assert reloaded.n_components == 64
        assert reloaded.components_.shape == (64, _EMBEDDING_DIM)

    def test_pca_transform_output_is_whitened_unit_variance(self, tmp_path: Path) -> None:
        """
        Sanity-check the actual whitening behaviour (not just the flag):
        transformed components should have approximately unit variance.
        """
        embeddings = self._fit_source_matrix(n_samples=500)
        output_path = tmp_path / "embedding_pca.joblib"

        pca = fit_and_save_pca(embeddings, output_path, n_components=64)
        transformed = pca.transform(embeddings)

        variances = np.var(transformed, axis=0)
        np.testing.assert_allclose(variances, np.ones(64), atol=0.15)
