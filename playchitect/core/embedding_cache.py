"""
Content-addressed cache for semantic audio embeddings.

Embeddings are expensive to compute (a full library pass with discogs-effnet
can take hours), so this module persists them to a parquet file keyed by the
*content* of the audio file rather than its filesystem path. Renaming or
moving a track therefore preserves its cache entry, while re-encoding or
otherwise mutating the file's bytes correctly invalidates it.

Also provides ``fit_and_save_pca``, which reduces the raw embedding
dimensionality (e.g. 1280-D discogs-effnet vectors) down to a smaller,
whitened PCA space suitable for downstream clustering/weighting work
(see TASK-27).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# Number of bytes read from the start of the file when computing the content
# hash. Reading the whole file for large audio files would be slow; the first
# 1 MB combined with the total file size is sufficient to distinguish tracks
# in practice while staying fast.
_HASH_WINDOW_BYTES: int = 1024 * 1024

_DEFAULT_MODEL_VERSION: str = "discogs-effnet-1"


def _compute_content_hash(track_path: Path) -> str:
    """
    Compute a content-addressed cache key for an audio file.

    The key is ``sha256(first 1 MB of file bytes)`` combined with the total
    file size, so that files sharing an identical sub-1MB prefix but
    differing in overall length still hash differently.

    Args:
        track_path: Path to the audio file to hash.

    Returns:
        Hex-encoded content hash string.
    """
    hasher = hashlib.sha256()
    with track_path.open("rb") as f:
        hasher.update(f.read(_HASH_WINDOW_BYTES))
    file_size = track_path.stat().st_size
    hasher.update(str(file_size).encode("ascii"))
    return hasher.hexdigest()


class EmbeddingCache:
    """
    Parquet-backed, content-addressed cache of semantic embedding vectors.

    Entries are keyed by content hash (see ``_compute_content_hash``), not by
    filesystem path, so a renamed or moved file with identical bytes still
    hits the cache. Re-``put``-ing the same content upserts (replaces) the
    prior row rather than appending a duplicate.
    """

    def __init__(self, cache_path: Path, model_version: str = _DEFAULT_MODEL_VERSION) -> None:
        """
        Initialise the cache.

        Args:
            cache_path: Path to the backing parquet file. Does not need to
                exist yet — it is created on the first ``put``.
            model_version: Identifier for the embedding model that produced
                the cached vectors (e.g. "discogs-effnet-1"), stored alongside
                each entry so a future model change can be distinguished from
                stale data.
        """
        self.cache_path = Path(cache_path)
        self.model_version = model_version

    def put(self, track_path: Path, vector: np.ndarray) -> None:
        """
        Store (or replace) the embedding vector for a track's content.

        Args:
            track_path: Path to the audio file the vector was computed from.
                Only its content (first 1 MB + size) is used as the cache
                key; the path itself is stored for diagnostics only.
            vector: Embedding vector to cache. Stored as float32.
        """
        content_hash = _compute_content_hash(track_path)
        table = self._read_table()

        if table is not None:
            # pyarrow.compute functions are generated dynamically at import time,
            # so ty's stubs don't see `not_equal`/`equal` even though they exist
            # at runtime (exercised by the passing put/get round-trip tests).
            hash_col = table.column("content_hash")
            keep_mask = pc.not_equal(hash_col, content_hash)  # ty: ignore[unresolved-attribute]
            table = table.filter(keep_mask)

        new_row = pa.table(
            {
                "content_hash": [content_hash],
                "path": [str(track_path)],
                "model_version": [self.model_version],
                "embedding": [vector.astype(np.float32).tolist()],
                "created_at": [datetime.now(UTC).isoformat()],
            }
        )

        combined = new_row if table is None else pa.concat_tables([table, new_row])
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(combined, self.cache_path)
        logger.debug("Cached embedding for content_hash=%s", content_hash[:12])

    def get(self, track_path: Path) -> np.ndarray | None:
        """
        Look up the cached embedding vector for a track's content.

        Args:
            track_path: Path to the audio file to look up.

        Returns:
            The cached float32 vector, or ``None`` on a cache miss (unknown
            content or an empty/absent cache file).
        """
        content_hash = _compute_content_hash(track_path)
        table = self._read_table()
        if table is None:
            return None

        # See the matching ty: ignore note in put() above.
        hash_col = table.column("content_hash")
        matches = table.filter(pc.equal(hash_col, content_hash))  # ty: ignore[unresolved-attribute]
        if matches.num_rows == 0:
            return None

        embedding = matches.column("embedding")[0].as_py()
        return np.array(embedding, dtype=np.float32)

    def _read_table(self) -> pa.Table | None:
        """Read the backing parquet file, returning None if absent/empty."""
        if not self.cache_path.exists():
            return None
        table = pq.read_table(self.cache_path)
        if table.num_rows == 0:
            return None
        return table


def fit_and_save_pca(embeddings: np.ndarray, output_path: Path, n_components: int = 64) -> PCA:
    """
    Fit a whitened PCA on a matrix of embeddings and persist it via joblib.

    ``whiten=True`` is intentional and load-bearing: it rescales each
    principal component to unit variance so that downstream diagonal
    Mahalanobis feature weighting (TASK-27) reads as pure feature importance
    rather than re-deriving the embedding's original variance profile.

    Args:
        embeddings: Array of shape (n_samples, n_features) of raw embedding
            vectors to fit the PCA on.
        output_path: Destination path for the joblib-persisted PCA model.
        n_components: Number of principal components to retain.

    Returns:
        The fitted ``sklearn.decomposition.PCA`` instance.
    """
    import joblib  # noqa: PLC0415

    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(embeddings)

    explained = float(np.sum(pca.explained_variance_ratio_))
    logger.info(
        "Fitted PCA: n_components=%d, explained_variance_ratio_sum=%.4f",
        n_components,
        explained,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, output_path)
    logger.debug("Persisted PCA model to %s", output_path)

    return pca
