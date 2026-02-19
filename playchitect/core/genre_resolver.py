"""
Genre resolution for clustering: manual override, metadata, and embedding inference.

Resolves per-track genre using priority:
  1. Manual assignments from genre_map YAML
  2. Metadata (ID3/ etc.) genre tag
  3. MusiCNN infer_genre when embeddings are available
  4. Fallback to "unknown" for unassigned tracks
"""

import logging
from pathlib import Path
from typing import Any

from playchitect.core.metadata_extractor import TrackMetadata
from playchitect.core.weighting import SUPPORTED_GENRES

logger = logging.getLogger(__name__)

# Normalised genre labels we use internally (lowercase, consistent with weighting)
_KNOWN_GENRES = {g.lower() for g in SUPPORTED_GENRES}


def load_genre_map(path: Path) -> dict[str, str]:
    """
    Load manual genre assignments from a YAML file.

    Expected format:
        manual_assignments:
          "track1.mp3": "techno"
          "/full/path/to/track2.mp3": "house"

    Keys may be filename-only (matched against any track with that name) or
    absolute/relative paths. Values are normalised to lowercase and must be
    in SUPPORTED_GENRES.

    Args:
        path: Path to the YAML file.

    Returns:
        Dict mapping path key (as provided) -> genre string.
        Empty dict if file not found or invalid.
    """
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed; --genre-map will be ignored")
        return {}

    if not path.exists():
        logger.warning("Genre map file not found: %s", path)
        return {}

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Failed to load genre map %s: %s", path, exc)
        return {}

    raw = data.get("manual_assignments")
    if not isinstance(raw, dict):
        return {}

    result: dict[str, str] = {}
    for key, val in raw.items():
        if not isinstance(key, str) or not isinstance(val, str):
            continue
        genre = val.strip().lower()
        if genre in _KNOWN_GENRES:
            result[key] = genre
        else:
            logger.warning("Genre map: unknown genre '%s' for %s; skipping", val, key)
    return result


def _match_override(
    track_path: Path,
    genre_map: dict[str, str],
    music_root: Path | None,
) -> str | None:
    """
    Check if track matches any manual override key.

    Matching order: exact path, path relative to music_root, filename only.
    """
    path_str = str(track_path)
    path_resolved = str(track_path.resolve())

    # Exact matches
    if path_str in genre_map:
        return genre_map[path_str]
    if path_resolved in genre_map:
        return genre_map[path_resolved]

    # Relative to music root
    if music_root:
        try:
            rel = str(track_path.relative_to(music_root))
            if rel in genre_map:
                return genre_map[rel]
        except ValueError:
            pass

    # Filename only
    name = track_path.name
    if name in genre_map:
        return genre_map[name]

    return None


def resolve_genres(
    metadata_dict: dict[Path, TrackMetadata],
    embedding_dict: dict[Path, Any] | None,
    genre_map: dict[str, str],
    music_root: Path | None,
    infer_genre_fn: Any = None,
) -> dict[Path, str]:
    """
    Resolve per-track genre from overrides, metadata, and embeddings.

    Priority:
      1. Manual override from genre_map (matched by path/filename)
      2. metadata.genre if in SUPPORTED_GENRES
      3. infer_genre_fn(embedding_dict[path]) when embeddings available
      4. "unknown"

    Args:
        metadata_dict: Mapping path -> TrackMetadata
        embedding_dict: Optional mapping path -> EmbeddingFeatures
        genre_map: Manual assignments from load_genre_map()
        music_root: Base path for relative matching in genre_map
        infer_genre_fn: Optional callable(features) -> str | None;
            e.g. EmbeddingExtractor.infer_genre

    Returns:
        Dict mapping Path -> genre string (lowercase, or "unknown")
    """
    result: dict[Path, str] = {}
    for path in metadata_dict:
        # 1. Manual override
        override = _match_override(path, genre_map, music_root)
        if override:
            result[path] = override
            continue

        # 2. Metadata genre
        meta = metadata_dict[path]
        if meta.genre and meta.genre.strip().lower() in _KNOWN_GENRES:
            result[path] = meta.genre.strip().lower()
            continue

        # 3. Embedding inference
        if embedding_dict and path in embedding_dict and infer_genre_fn:
            try:
                g = infer_genre_fn(embedding_dict[path])
                if g and g.lower() in _KNOWN_GENRES:
                    result[path] = g.lower()
                    continue
            except Exception as exc:
                logger.debug("Genre inference failed for %s: %s", path.name, exc)

        # 4. Fallback
        result[path] = "unknown"
    return result
