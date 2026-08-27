#!/usr/bin/env python3
"""
ETL script: walk a music library, compute semantic embeddings for every
track not already present in the content-addressed EmbeddingCache, and
persist them.

Idempotent per track: re-running over a directory that has already been
fully embedded embeds zero new tracks; adding a single new file embeds
exactly that one file on the next run.

Usage:
    uv run python scripts/embed_library.py /path/to/music/library \
        --cache-path data/embeddings.parquet
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import click
import numpy as np

from playchitect.core.audio_scanner import AudioScanner
from playchitect.core.embedding_cache import _DEFAULT_MODEL_VERSION, EmbeddingCache

logger = logging.getLogger(__name__)

_DEFAULT_LOG_EVERY: int = 50
_DEFAULT_CACHE_PATH: Path = Path("data") / "embeddings.parquet"


class EmbeddingDimensionMismatchError(RuntimeError):
    """
    Raised when a computed embedding's dimensionality does not match what
    its declared ``model_version`` promises.

    The EmbeddingCache's ``model_version`` column is what makes the model
    choice reversible later -- a column that lies (e.g. a MusiCNN 200-D
    vector silently stored under the "discogs-effnet-1" label) poisons the
    cache in a way that is hard to detect afterwards. Raising this instead
    of writing the mismatched row keeps that guarantee intact.
    """


def embed_library(
    directory: Path,
    cache: EmbeddingCache,
    embed_fn: Callable[[Path], np.ndarray],
    log_every: int = _DEFAULT_LOG_EVERY,
) -> int:
    """
    Walk ``directory`` and embed every track not already in ``cache``.

    Tracks already present in the cache (matched by content hash) are
    skipped without invoking ``embed_fn``, so this function is safe to
    re-run repeatedly over a growing library: only newly-added or changed
    tracks incur the (expensive) embedding computation.

    Args:
        directory: Root directory to scan for audio files.
        cache: EmbeddingCache instance to check for existing entries and
            store newly-computed vectors into.
        embed_fn: Callable that computes an embedding vector for a single
            track path. Injected so callers can use the real extractor in
            production and a cheap fake in tests.
        log_every: Log a progress message every this many *newly embedded*
            tracks.

    Returns:
        The count of tracks that were newly embedded (cache misses).
    """
    scanner = AudioScanner()
    tracks = scanner.scan(directory)

    newly_embedded = 0
    for track_path in tracks:
        if cache.get(track_path) is not None:
            logger.debug("Skipping already-cached track: %s", track_path.name)
            continue

        vector = embed_fn(track_path)
        cache.put(track_path, vector)
        newly_embedded += 1

        if newly_embedded % log_every == 0:
            logger.info(
                "Embedded %d new tracks so far (scanned %d/%d)",
                newly_embedded,
                newly_embedded,
                len(tracks),
            )

    logger.info(
        "Finished embedding library: %d newly embedded, %d total tracks scanned",
        newly_embedded,
        len(tracks),
    )
    return newly_embedded


def _real_embed_fn(track_path: Path) -> np.ndarray:
    """
    Compute a discogs-effnet embedding for a single track.

    Lazily imports EmbeddingExtractor so this module (and embed_library())
    can be imported/used in tests without requiring essentia-tensorflow to
    be installed.

    Args:
        track_path: Path to the audio file to embed.

    Returns:
        The computed discogs-effnet embedding vector as a float32 numpy
        array, matching the dimensionality declared by the cache's
        ``model_version`` ("discogs-effnet-1").

    Raises:
        EmbeddingDimensionMismatchError: If the computed vector's shape does
            not match ``_DISCOGS_EFFNET_EMBEDDING_DIM`` -- writing a
            mismatched vector under the "discogs-effnet-1" label would
            silently poison the cache, so this fails loudly instead.
    """
    from playchitect.core.embedding_extractor import (  # noqa: PLC0415
        _DISCOGS_EFFNET_EMBEDDING_DIM,
        EmbeddingExtractor,
    )

    extractor = EmbeddingExtractor()
    vector = extractor.analyze_discogs_effnet(track_path)

    if vector.shape != (_DISCOGS_EFFNET_EMBEDDING_DIM,):
        raise EmbeddingDimensionMismatchError(
            f"Embedding computed for {track_path} has shape {vector.shape}, "
            f"expected ({_DISCOGS_EFFNET_EMBEDDING_DIM},) to match the "
            f"'{_DEFAULT_MODEL_VERSION}' model_version recorded by the "
            "cache. Refusing to write a mismatched row."
        )

    return vector


@click.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--cache-path",
    type=click.Path(path_type=Path),
    default=_DEFAULT_CACHE_PATH,
    show_default=True,
    help="Path to the embeddings parquet cache file.",
)
@click.option(
    "--log-every",
    type=int,
    default=_DEFAULT_LOG_EVERY,
    show_default=True,
    help="Log a progress message every N newly-embedded tracks.",
)
def main(directory: Path, cache_path: Path, log_every: int) -> None:
    """Embed every track in DIRECTORY not already present in the cache."""
    logging.basicConfig(level=logging.INFO)
    cache = EmbeddingCache(cache_path=cache_path)
    count = embed_library(directory, cache, _real_embed_fn, log_every=log_every)
    click.echo(f"Newly embedded {count} track(s) into {cache_path}")


if __name__ == "__main__":
    main()
