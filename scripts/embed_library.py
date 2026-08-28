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
import resource
import sys
from collections.abc import Callable
from pathlib import Path

import click
import numpy as np

from playchitect.core.audio_scanner import AudioScanner
from playchitect.core.embedding_cache import _DEFAULT_MODEL_VERSION, EmbeddingCache

logger = logging.getLogger(__name__)

_DEFAULT_LOG_EVERY: int = 50
_DEFAULT_CACHE_PATH: Path = Path("data") / "embeddings.parquet"

# TASK-33: default resident-memory ceiling for a single run of this script.
# The incident that motivated this constant grew RSS by ~1GB PER TRACK
# (a fresh ~924MB TensorFlow/essentia graph built per track instead of once
# per run) and froze a machine with ~4GB free after about 7 tracks. 4096 MiB
# is a conservative default for a single-model, single-process run on a
# typical development workstation; callers with tighter or looser
# constraints (e.g. a hard cgroup cap) should override it via
# ``--memory-ceiling-mb`` / the ``memory_ceiling_mb`` argument rather than
# editing this constant.
_DEFAULT_MEMORY_CEILING_MB: float = 4096.0


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


class MemoryCeilingExceededError(RuntimeError):
    """
    Raised when resident memory (RSS) exceeds the configured ceiling during
    an ``embed_library()`` run.

    TASK-33: the incident this guards against grew RSS by roughly 1GB per
    track (a fresh TensorFlow/essentia graph built per track instead of once
    per run) and froze the workstation after only 7 tracks. This is a
    defence-in-depth backstop for *any* future leak, however it arises --
    the primary fix is constructing the model-bearing extractor exactly once
    per run (see ``_make_real_embed_fn`` below) -- so that a regression aborts
    loudly, with an actionable number, instead of silently growing until the
    OS kills the machine.
    """


def _get_rss_mb() -> float:
    """
    Return this process's current resident set size (RSS) in MiB.

    Deliberately dependency-free (uses the stdlib ``resource`` module rather
    than adding ``psutil``) and kept as the sole module-level seam so tests
    can simulate memory pressure by monkeypatching this one function without
    allocating anything real.

    Returns:
        Current RSS in mebibytes. ``ru_maxrss`` is reported in kilobytes on
        Linux and in bytes on macOS/BSD; this normalises both to MiB.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def embed_library(
    directory: Path,
    cache: EmbeddingCache,
    embed_fn: Callable[[Path], np.ndarray],
    log_every: int = _DEFAULT_LOG_EVERY,
    memory_ceiling_mb: float | None = None,
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
        memory_ceiling_mb: Abort the run with ``MemoryCeilingExceededError``
            if resident memory (as reported by ``_get_rss_mb()``) exceeds
            this many MiB. Checked before every track that will actually be
            embedded (not merely every ``log_every`` tracks), since the
            TASK-33 incident grew RSS by roughly 1GB per track -- a
            per-log-interval check with the default ``log_every=50`` would
            let tens of gigabytes accumulate before noticing. Defaults to
            ``None`` (no limit) at this library-API level so existing/direct
            callers are unaffected; the ``main()`` CLI entry point wires up
            a real, CLI-configurable ceiling for production runs.

    Returns:
        The count of tracks that were newly embedded (cache misses).

    Raises:
        MemoryCeilingExceededError: If resident memory exceeds
            ``memory_ceiling_mb`` before a track can be embedded.
    """
    scanner = AudioScanner()
    tracks = scanner.scan(directory)

    newly_embedded = 0
    # Tracks the most recently observed RSS reading so the ceiling check and
    # the progress log can share a single, cheap-to-reuse figure instead of
    # each calling _get_rss_mb() independently. Refreshed once at the end of
    # every embedded track, so it always reflects memory as measured
    # immediately after the most recently completed track -- current enough
    # to catch the ~1GB-per-track growth pattern from the TASK-33 incident
    # within a track or two, without adding redundant syscalls mid-track.
    last_rss_mb: float | None = None

    for track_path in tracks:
        if cache.get(track_path) is not None:
            logger.debug("Skipping already-cached track: %s", track_path.name)
            continue

        if memory_ceiling_mb is not None:
            if last_rss_mb is None:
                last_rss_mb = _get_rss_mb()
            if last_rss_mb > memory_ceiling_mb:
                raise MemoryCeilingExceededError(
                    f"Aborting embed_library run: resident memory {last_rss_mb:.1f}MB "
                    f"exceeds the configured ceiling of {memory_ceiling_mb:.1f}MB. "
                    f"Stopped before embedding {track_path} "
                    f"({newly_embedded} track(s) embedded so far this run)."
                )

        vector = embed_fn(track_path)
        cache.put(track_path, vector)
        newly_embedded += 1

        if newly_embedded % log_every == 0:
            rss_for_log = last_rss_mb if last_rss_mb is not None else _get_rss_mb()
            logger.info(
                "Embedded %d new tracks so far (scanned %d/%d, RSS %.1fMB)",
                newly_embedded,
                newly_embedded,
                len(tracks),
                rss_for_log,
            )

        last_rss_mb = _get_rss_mb()

    logger.info(
        "Finished embedding library: %d newly embedded, %d total tracks scanned",
        newly_embedded,
        len(tracks),
    )
    return newly_embedded


def _make_real_embed_fn() -> Callable[[Path], np.ndarray]:
    """
    Build a discogs-effnet ``embed_fn`` backed by exactly one
    ``EmbeddingExtractor`` instance, reused for every track in the run.

    TASK-33 root cause: the previous implementation constructed a fresh
    ``EmbeddingExtractor()`` *inside* the per-track embed function, so every
    single call built a brand-new ~924MB TensorFlow/essentia graph and
    session that was never released (TF/essentia hold C++ resources that
    Python's GC does not promptly reclaim). RSS grew by roughly 1GB per
    track and froze the workstation after about 7 tracks.

    The fix: construct ``EmbeddingExtractor()`` exactly once, here, and
    close over that single instance in the returned closure. The extractor's
    own lazy model cache (``self._model_discogs_effnet``, built on first use
    and reused thereafter) then only ever builds the model once per run,
    because the extractor instance itself persists across every track.

    Do not reintroduce ``EmbeddingExtractor()`` construction inside the
    per-track closure or inside ``embed_library()``'s loop body -- that is
    precisely the regression this function exists to prevent.

    Returns:
        A callable computing a discogs-effnet embedding for a single track,
        suitable for passing as ``embed_library()``'s ``embed_fn``.
    """
    from playchitect.core.embedding_extractor import (  # noqa: PLC0415
        _DISCOGS_EFFNET_EMBEDDING_DIM,
        EmbeddingExtractor,
    )

    extractor = EmbeddingExtractor()

    def _real_embed_fn(track_path: Path) -> np.ndarray:
        """
        Compute a discogs-effnet embedding for a single track using the
        single, run-scoped ``EmbeddingExtractor`` closed over above.

        Args:
            track_path: Path to the audio file to embed.

        Returns:
            The computed discogs-effnet embedding vector as a float32 numpy
            array, matching the dimensionality declared by the cache's
            ``model_version`` ("discogs-effnet-1").

        Raises:
            EmbeddingDimensionMismatchError: If the computed vector's shape
                does not match ``_DISCOGS_EFFNET_EMBEDDING_DIM`` -- writing
                a mismatched vector under the "discogs-effnet-1" label would
                silently poison the cache, so this fails loudly instead.
        """
        vector = extractor.analyze_discogs_effnet(track_path)

        if vector.shape != (_DISCOGS_EFFNET_EMBEDDING_DIM,):
            raise EmbeddingDimensionMismatchError(
                f"Embedding computed for {track_path} has shape {vector.shape}, "
                f"expected ({_DISCOGS_EFFNET_EMBEDDING_DIM},) to match the "
                f"'{_DEFAULT_MODEL_VERSION}' model_version recorded by the "
                "cache. Refusing to write a mismatched row."
            )

        return vector

    return _real_embed_fn


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
@click.option(
    "--memory-ceiling-mb",
    type=float,
    default=_DEFAULT_MEMORY_CEILING_MB,
    show_default=True,
    help=(
        "Abort the run if resident memory (RSS) exceeds this many MiB. "
        "TASK-33: protects against a leaking embed_fn freezing the machine. "
        "Pass a very large value (or edit the source) to disable."
    ),
)
def main(directory: Path, cache_path: Path, log_every: int, memory_ceiling_mb: float) -> None:
    """Embed every track in DIRECTORY not already present in the cache."""
    logging.basicConfig(level=logging.INFO)
    cache = EmbeddingCache(cache_path=cache_path)
    real_embed_fn = _make_real_embed_fn()
    count = embed_library(
        directory,
        cache,
        real_embed_fn,
        log_every=log_every,
        memory_ceiling_mb=memory_ceiling_mb,
    )
    click.echo(f"Newly embedded {count} track(s) into {cache_path}")


if __name__ == "__main__":
    main()
