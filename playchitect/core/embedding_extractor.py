"""
MusiCNN semantic embedding extraction using Essentia TensorFlow.

Extracts 128-dimensional audio embeddings via the MSD-MusiCNN model
(~50 MB, auto-downloaded on first use to ~/.cache/playchitect/models/).

When essentia-tensorflow is not installed, importing this module succeeds but
instantiating EmbeddingExtractor raises RuntimeError.  All existing
functionality works unchanged when this module is not used.
"""

import hashlib
import json
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Model constants ───────────────────────────────────────────────────────────

_MSD_MUSICNN_URL = "https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb"
_MSD_MUSICNN_META = "https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.json"
_EMBEDDING_SAMPLE_RATE: int = 16000
_EMBEDDING_DIM: int = 128
_DEFAULT_MODEL_DIR: Path = Path.home() / ".cache" / "playchitect" / "models"

# TensorFlow graph output layer names within the MusiCNN model
_EMB_OUTPUT_LAYER: str = "model/Bottleneck/Squeeze"
_TAG_OUTPUT_LAYER: str = "model/dense/Sigmoid"

# MSD tags → our genre vocabulary (used by infer_genre)
_TAG_GENRE_MAP: dict[str, str] = {
    "techno": "techno",
    "house": "house",
    "ambient": "ambient",
    "drum and bass": "dnb",
    "dnb": "dnb",
    "jungle": "dnb",
    "electronic": "techno",
    "electronica": "techno",
    "deep house": "house",
}

# ── Optional Essentia import ──────────────────────────────────────────────────

try:
    from essentia.standard import (  # type: ignore[import-untyped,unresolved-import]
        TensorflowPredictMusiCNN as _EssentiaModel,
    )

    _ESSENTIA_AVAILABLE = True
except ImportError:
    _EssentiaModel = None
    _ESSENTIA_AVAILABLE = False

# ── Dataclass ─────────────────────────────────────────────────────────────────


@dataclass
class EmbeddingFeatures:
    """Container for MusiCNN embedding features."""

    filepath: Path
    file_hash: str
    embedding: np.ndarray  # shape (128,), float32
    top_tags: list[tuple[str, float]]  # [(tag, confidence), …] sorted descending

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict (embedding stored as list)."""
        return {
            "filepath": str(self.filepath),
            "file_hash": self.file_hash,
            "embedding": self.embedding.tolist(),
            "top_tags": [[tag, conf] for tag, conf in self.top_tags],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingFeatures":
        """Deserialise from dict (inverse of to_dict)."""
        return cls(
            filepath=Path(data["filepath"]),
            file_hash=data["file_hash"],
            embedding=np.array(data["embedding"], dtype=np.float32),
            top_tags=[(str(t[0]), float(t[1])) for t in data["top_tags"]],
        )


# ── Extractor ─────────────────────────────────────────────────────────────────


class EmbeddingExtractor:
    """
    Extracts MusiCNN semantic embeddings from audio files.

    Requires essentia-tensorflow; raises RuntimeError on instantiation if
    the package is not installed.

    Model auto-downloads (~50 MB) on first use to:
        ~/.cache/playchitect/models/msd-musicnn-1.pb
    """

    def __init__(
        self,
        model_path: Path | None = None,
        cache_dir: Path | None = None,
        cache_enabled: bool = True,
        sample_rate: int = _EMBEDDING_SAMPLE_RATE,
    ):
        """
        Initialise the extractor.

        Args:
            model_path: Path to msd-musicnn-1.pb.  None → auto-download.
            cache_dir:  Directory for per-track embedding cache.
            cache_enabled: Whether to cache results to disk.
            sample_rate: Audio sample rate required by MusiCNN (16 000 Hz).

        Raises:
            RuntimeError: When essentia-tensorflow is not installed.
        """
        if not _ESSENTIA_AVAILABLE:
            raise RuntimeError(
                "essentia-tensorflow is required for embedding analysis. "
                "Install with: uv pip install 'playchitect[embeddings]'"
            )

        self.model_path = model_path or (_DEFAULT_MODEL_DIR / "msd-musicnn-1.pb")
        self.cache_enabled = cache_enabled
        self.sample_rate = sample_rate

        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "playchitect" / "embeddings"
        self.cache_dir = Path(cache_dir)

        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-initialised model instances
        self._model_emb: Any = None
        self._model_tags: Any = None
        self._tag_labels: list[str] | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, filepath: Path) -> EmbeddingFeatures:
        """
        Extract embedding and top tags from an audio file.

        The model is downloaded and initialised on the first call.

        Args:
            filepath: Path to the audio file.

        Returns:
            EmbeddingFeatures with a 128-dim embedding and top MSD tags.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        file_hash = self._compute_file_hash(filepath)

        if self.cache_enabled:
            cached = self._load_from_cache(file_hash)
            if cached is not None:
                logger.debug("Using cached embedding for: %s", filepath.name)
                cached.filepath = filepath
                return cached

        logger.debug("Extracting embedding: %s", filepath.name)

        import librosa  # noqa: PLC0415

        y, _ = librosa.load(filepath, sr=self.sample_rate, mono=True)

        self._ensure_model()

        # Frame-level embeddings → (N_frames, 128); mean-pool → (128,)
        emb_frames = self._model_emb(y)
        embedding = np.mean(emb_frames, axis=0).astype(np.float32)

        # Frame-level sigmoid activations → (N_frames, 50); mean → (50,)
        tag_activations = np.mean(self._model_tags(y), axis=0)
        top_tags = self._build_top_tags(tag_activations)

        features = EmbeddingFeatures(
            filepath=filepath,
            file_hash=file_hash,
            embedding=embedding,
            top_tags=top_tags,
        )

        if self.cache_enabled:
            self._save_to_cache(file_hash, features)

        return features

    def analyze_batch(self, filepaths: list[Path]) -> dict[Path, EmbeddingFeatures]:
        """
        Analyze a batch of files.

        Files that fail analysis are skipped and logged at WARNING level.

        Args:
            filepaths: List of audio file paths.

        Returns:
            Dict mapping successfully-analyzed paths → EmbeddingFeatures.
        """
        results: dict[Path, EmbeddingFeatures] = {}
        for fp in filepaths:
            try:
                results[fp] = self.analyze(fp)
            except Exception as exc:
                logger.warning("Embedding extraction failed for %s: %s", fp.name, exc)
        return results

    def infer_genre(self, features: EmbeddingFeatures) -> str | None:
        """
        Infer genre from top_tags using the MSD → genre vocabulary map.

        Tags are evaluated in descending confidence order; the first match in
        _TAG_GENRE_MAP is returned.  Returns None when no known genre tag is
        found.

        Args:
            features: EmbeddingFeatures instance with populated top_tags.

        Returns:
            Genre string ('techno', 'house', 'ambient', 'dnb'), or None.
        """
        for tag, _confidence in features.top_tags:
            genre = _TAG_GENRE_MAP.get(tag.lower())
            if genre is not None:
                return genre
        return None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        """Lazy-initialise both model instances, downloading .pb if absent."""
        if not self.model_path.exists():
            self._download_model(self.model_path)

        # _EssentiaModel is always defined when _ESSENTIA_AVAILABLE is True,
        # which is guaranteed by __init__'s guard.
        assert _EssentiaModel is not None

        if self._model_emb is None:
            self._model_emb = _EssentiaModel(
                graphFilename=str(self.model_path),
                output=_EMB_OUTPUT_LAYER,
            )
        if self._model_tags is None:
            self._model_tags = _EssentiaModel(
                graphFilename=str(self.model_path),
                output=_TAG_OUTPUT_LAYER,
            )

    def _download_model(self, target: Path) -> None:
        """
        Download model .pb and companion metadata .json to target path.

        Prints a one-time notice; also fetches the metadata JSON used for
        tag labels.

        Args:
            target: Destination path for msd-musicnn-1.pb.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        meta_path = target.with_suffix(".json")

        print(f"Downloading MusiCNN model to {target} (~50 MB, one-time download)...")
        logger.info("Downloading MSD-MusiCNN model to %s", target)
        urllib.request.urlretrieve(_MSD_MUSICNN_URL, target)

        logger.info("Downloading MSD-MusiCNN metadata to %s", meta_path)
        urllib.request.urlretrieve(_MSD_MUSICNN_META, meta_path)

    def _build_top_tags(self, activations: np.ndarray) -> list[tuple[str, float]]:
        """Convert activation vector to sorted (tag, confidence) list (descending)."""
        labels = self._load_tag_labels()
        if labels is None or len(labels) != len(activations):
            # Fallback: numeric string labels
            labels = [str(i) for i in range(len(activations))]

        pairs = sorted(
            zip(labels, activations.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(tag, float(conf)) for tag, conf in pairs]

    def _load_tag_labels(self) -> list[str] | None:
        """Load tag label list from companion metadata JSON (cached in memory)."""
        if self._tag_labels is not None:
            return self._tag_labels

        meta_path = self.model_path.with_suffix(".json")
        if not meta_path.exists():
            return None

        try:
            with open(meta_path) as f:
                meta = json.load(f)
            self._tag_labels = meta.get("classes", [])
            return self._tag_labels
        except Exception as exc:
            logger.warning("Failed to load tag labels: %s", exc)
            return None

    def _compute_file_hash(self, filepath: Path) -> str:
        """MD5 hash of first 1 MB for cache keying."""
        md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            md5.update(f.read(1024 * 1024))
        return md5.hexdigest()

    def _get_cache_path(self, file_hash: str) -> Path:
        """Return the .npy embedding cache path for the given hash."""
        return self.cache_dir / f"{file_hash}.npy"

    def _save_to_cache(self, file_hash: str, feat: EmbeddingFeatures) -> None:
        """Save embedding → {hash}.npy, top_tags → {hash}_tags.json."""
        try:
            np.save(str(self._get_cache_path(file_hash)), feat.embedding)
            tags_path = self.cache_dir / f"{file_hash}_tags.json"
            with open(tags_path, "w") as f:
                json.dump([[tag, conf] for tag, conf in feat.top_tags], f)
            logger.debug("Cached embedding: %s", file_hash[:8])
        except Exception as exc:
            logger.warning("Failed to cache embedding: %s", exc)

    def _load_from_cache(self, file_hash: str) -> EmbeddingFeatures | None:
        """Load embedding from .npy and tags from _tags.json."""
        emb_path = self._get_cache_path(file_hash)
        tags_path = self.cache_dir / f"{file_hash}_tags.json"

        if not emb_path.exists() or not tags_path.exists():
            return None

        try:
            embedding = np.load(str(emb_path)).astype(np.float32)
            with open(tags_path) as f:
                raw_tags = json.load(f)
            top_tags = [(str(t[0]), float(t[1])) for t in raw_tags]
            return EmbeddingFeatures(
                filepath=Path(""),  # Caller updates this
                file_hash=file_hash,
                embedding=embedding,
                top_tags=top_tags,
            )
        except Exception as exc:
            logger.warning("Failed to load cached embedding: %s", exc)
            return None
