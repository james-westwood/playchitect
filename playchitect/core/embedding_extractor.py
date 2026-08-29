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
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

try:
    # Essentia imports are optional. essentia-tensorflow builds its
    # `essentia.standard` algorithm classes (TensorflowPredictMusiCNN,
    # TensorflowPredictEffnetDiscogs, ...) dynamically at runtime via its SWIG
    # bindings, so there are no static attributes for ty to resolve even when
    # the package is installed -- unlike a plain "no stubs" situation, this
    # can never be fixed by adding stubs. Suppressed centrally via the
    # `allowed-unresolved-imports` analysis setting in pyproject.toml instead
    # of a local ignore comment, because a local ignore comment would itself
    # be reported as unused when essentia is absent and the whole module
    # fails to resolve differently -- see the comment in pyproject.toml for
    # why this must stay stable across both states.
    from essentia.standard import TensorflowPredict2D
    from essentia.standard import TensorflowPredictEffnetDiscogs as _DiscogsEffnetModel
    from essentia.standard import TensorflowPredictMusiCNN as _EssentiaModel

    _ESSENTIA_AVAILABLE = True
except ImportError:
    # Use Any for type checking when Essentia is missing
    TensorflowPredict2D = Any
    _EssentiaModel = Any
    _DiscogsEffnetModel = Any
    _ESSENTIA_AVAILABLE = False

# ── Model constants ───────────────────────────────────────────────────────────

_MSD_MUSICNN_URL = "https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.pb"
_MSD_MUSICNN_META = "https://essentia.upf.edu/models/feature-extractors/musicnn/msd-musicnn-1.json"

_MIREX_MOODS_URL = (
    "https://essentia.upf.edu/models/classification-heads/moods_mirex/moods_mirex-msd-musicnn-1.pb"
)
_MIREX_MOODS_META = "https://essentia.upf.edu/models/classification-heads/moods_mirex/moods_mirex-msd-musicnn-1.json"

# discogs-effnet: alternative embedding model used by the TASK-19 embedding
# cache ETL pipeline (playchitect/core/embedding_cache.py,
# scripts/embed_library.py). Produces 1280-dimensional embeddings.
_DISCOGS_EFFNET_URL = (
    "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb"
)
_DISCOGS_EFFNET_META = (
    "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.json"
)

# ── Model integrity pins (TASK-31, rail 3) ──────────────────────────────────
#
# essentia.upf.edu publishes NO checksums for these .pb files -- the model
# metadata JSON only carries name/type/link/version/description/author/
# release_date, no digest field. These sha256 pins are therefore necessarily
# trust-on-first-use: they do NOT protect the very first download of a
# model. What they DO protect is the model file silently changing under us
# on a *later* download -- altering embedding semantics while the code pin
# and the embedding cache's model_version column both still claim nothing
# moved.
#
# Digests below were downloaded and hashed by the project owner from the
# currently-published model files on 2026-08-27, and verified stable across
# repeated downloads. If a legitimate upstream re-release changes a model
# file, re-download it, recompute its sha256, and update the constant here
# deliberately -- do not silently widen the check to tolerate mismatches.
# These are public sha256 digests of downloadable model files, not
# credentials -- allowlisted below to keep detect-secrets quiet.
_MSD_MUSICNN_SHA256 = (
    "cdea0722bcee7f731286843f2233e3aa69887bb5c3e2dce011eff55f38d04f3e"  # pragma: allowlist secret
)
_MIREX_MOODS_SHA256 = (
    "d90d3020fab17ad641857a7272c94fd27a0d5c11aa92aa130dcc807c58ee6fab"  # pragma: allowlist secret
)
_DISCOGS_EFFNET_SHA256 = (
    "3ed9af50d5367c0b9c795b294b00e7599e4943244f4cbd376869f3bfc87721b1"  # pragma: allowlist secret
)

_EMBEDDING_SAMPLE_RATE: int = 16000
_EMBEDDING_DIM: int = 128
_DISCOGS_EFFNET_EMBEDDING_DIM: int = 1280
_DEFAULT_MODEL_DIR: Path = Path.home() / ".cache" / "playchitect" / "models"

# TensorFlow graph output layer names
_EMB_OUTPUT_LAYER: str = "model/Squeeze"
_TAG_OUTPUT_LAYER: str = "model/Sigmoid"
_MIREX_FEATS_LAYER: str = "model/dense/BiasAdd"  # 200D layer for MIREX

# MIREX model node names
_MOOD_INPUT_LAYER: str = "serving_default_model_Placeholder"
_MOOD_OUTPUT_LAYER: str = "PartitionedCall"

# discogs-effnet graph output node for the 1280-D embedding layer (see the
# model card referenced above; "PartitionedCall:1" is the embeddings output,
# distinct from "PartitionedCall:0" which carries the 400-class predictions).
_DISCOGS_EFFNET_EMB_OUTPUT_LAYER: str = "PartitionedCall:1"

# discogs-effnet consumes patches of 128 mel frames at a 256-sample hop over
# 16 kHz audio -- roughly 2.048s of audio minimum (128 * 256 / 16000). Below
# that threshold the model returns zero frames and mean-pooling an empty
# array silently collapses to a NaN scalar rather than raising, so this is
# used as an explicit pre-flight guard in analyze_discogs_effnet(). Measured
# directly against the installed model on 2026-08-27: 2.0s -> zero frames,
# 2.5s -> a valid (1280,) vector.
_DISCOGS_EFFNET_MIN_AUDIO_SECONDS: float = 2.048

# MusiCNN frames audio differently from discogs-effnet and needs a longer
# clip: it consumes patches of 187 mel frames, and TensorflowInputMusiCNN
# frames 16 kHz audio with frameSize=512 / hopSize=256 using essentia's
# centred framing (startFromZero=False), giving
# frames = floor((N - 1) / 256) + 2. Solving frames >= 187 gives
# N >= 185 * 256 + 1 = 47361 samples = 2.9600625s. Measured by binary search
# against the installed msd-musicnn-1.pb: 47360 samples -> 0 frames, 47361
# samples -> 187 frames. The nominal 187 * 256 / 16000 = 2.992s figure
# overstates the requirement, because centred framing supplies the first
# frames from zero-padding. Below this threshold the model returns zero
# frames, and mean-pooling them silently yields NaN instead of raising, so
# analyze() uses this as an explicit post-inference guard.
_MUSICNN_MIN_AUDIO_SECONDS: float = 2.9600625

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

# ── Exceptions ────────────────────────────────────────────────────────────────


class ModelIntegrityError(RuntimeError):
    """
    Raised when a downloaded model file's sha256 does not match its pin.

    essentia.upf.edu publishes no checksums for its model files, so this
    check is trust-on-first-use: it cannot validate the very first download,
    only detect the file changing under us on a later one. A mismatch is
    therefore not automatically an attack -- it may be a legitimate upstream
    re-release. See this exception's message for expected/actual digests and
    guidance on updating the pin if that's the case.
    """


class EmbeddingSmokeCheckError(RuntimeError):
    """
    Raised by ``run_embedding_smoke_check`` when the discogs-effnet output
    fails a dimensionality, finiteness, norm-band, or reproducibility check.

    Distinct from ``FileNotFoundError`` (a caller error, raised separately)
    -- this exception specifically signals that the model produced output
    that does not look like a valid embedding.
    """


class AudioTooShortForEmbeddingError(RuntimeError):
    """
    Raised when an audio clip is too short for an embedding model to produce
    a single output frame.

    Both supported models patch audio into fixed-length mel-frame windows at
    a 256-sample hop over 16 kHz audio, and each has its own minimum:
    discogs-effnet needs ``_DISCOGS_EFFNET_MIN_AUDIO_SECONDS``, MusiCNN the
    longer ``_MUSICNN_MIN_AUDIO_SECONDS``. Below its threshold a model
    returns zero frames, and mean-pooling zero frames silently collapses to
    a NaN scalar rather than raising, so ``analyze`` and
    ``analyze_discogs_effnet`` raise this instead of returning (or caching)
    malformed output.
    """


# ── Dataclass ─────────────────────────────────────────────────────────────────


@dataclass
class EmbeddingFeatures:
    """Container for MusiCNN embedding features and MIREX moods."""

    filepath: Path
    file_hash: str
    embedding: np.ndarray  # shape (128,), float32
    top_tags: list[tuple[str, float]]  # [(tag, confidence), …] sorted descending
    moods: list[tuple[str, float]]  # [(mood, probability), …] sorted descending

    @property
    def primary_mood(self) -> str | None:
        """Return the mood with highest probability, or None if empty."""
        return self.moods[0][0] if self.moods else None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict (embedding stored as list)."""
        return {
            "filepath": str(self.filepath),
            "file_hash": self.file_hash,
            "embedding": self.embedding.tolist(),
            "top_tags": [[tag, conf] for tag, conf in self.top_tags],
            "moods": [[mood, prob] for mood, prob in self.moods],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EmbeddingFeatures":
        """Deserialise from dict (inverse of to_dict)."""
        return cls(
            filepath=Path(data["filepath"]),
            file_hash=data["file_hash"],
            embedding=np.array(data["embedding"], dtype=np.float32),
            top_tags=[(str(t[0]), float(t[1])) for t in data["top_tags"]],
            moods=[(str(m[0]), float(m[1])) for m in data.get("moods", [])],
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
        mood_model_path: Path | None = None,
        cache_dir: Path | None = None,
        cache_enabled: bool = True,
        sample_rate: int = _EMBEDDING_SAMPLE_RATE,
        cache_db: Any | None = None,  # CacheDB type hinted as Any to avoid circular import
        discogs_effnet_model_path: Path | None = None,
    ):
        """
        Initialise the extractor.

        Args:
            model_path: Path to msd-musicnn-1.pb.  None → auto-download.
            mood_model_path: Path to moods_mirex-msd-musicnn-1.pb. None → auto-download.
            cache_dir:  Directory for per-track embedding cache.
            cache_enabled: Whether to cache results to disk.
            sample_rate: Audio sample rate required by MusiCNN (16 000 Hz).
            cache_db:   Optional SQLite-backed CacheDB instance for persisting moods.
            discogs_effnet_model_path: Path to discogs-effnet-bs64-1.pb, used by
                the TASK-19 embedding cache ETL pipeline. None → auto-download.

        Raises:
            RuntimeError: When essentia-tensorflow is not installed.
        """
        if not _ESSENTIA_AVAILABLE:
            raise RuntimeError(
                "essentia-tensorflow is required for embedding analysis. "
                "Install with: uv pip install 'playchitect[embeddings]'"
            )

        self.model_path = model_path or (_DEFAULT_MODEL_DIR / "msd-musicnn-1.pb")
        self.mood_model_path = mood_model_path or (
            _DEFAULT_MODEL_DIR / "moods_mirex-msd-musicnn-1.pb"
        )
        self.discogs_effnet_model_path = discogs_effnet_model_path or (
            _DEFAULT_MODEL_DIR / "discogs-effnet-bs64-1.pb"
        )
        self.cache_enabled = cache_enabled
        self.sample_rate = sample_rate
        self.cache_db = cache_db

        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "playchitect" / "embeddings"
        self.cache_dir = Path(cache_dir)

        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-initialised model instances
        self._model_emb: Any = None
        self._model_tags: Any = None
        self._model_mirex_feats: Any = None
        self._model_moods: Any = None
        self._model_discogs_effnet: Any = None
        self._tag_labels: list[str] | None = None
        self._mood_labels: list[str] | None = None

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
            AudioTooShortForEmbeddingError: If the audio is shorter than
                ``_MUSICNN_MIN_AUDIO_SECONDS``, in which case the model
                produces zero frames and there is nothing valid to
                mean-pool. Raised before any cache write, so a NaN
                embedding is never persisted.
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
        # len() rather than .size: the essentia bindings return a plain empty
        # list for zero frames in some builds and a (0, 128) ndarray in
        # others, and only len() covers both.
        if len(emb_frames) == 0:
            duration_seconds = len(y) / self.sample_rate
            logger.error(
                "MusiCNN produced zero frames for %s (%.3fs, below the ~%.3fs minimum)",
                filepath.name,
                duration_seconds,
                _MUSICNN_MIN_AUDIO_SECONDS,
            )
            raise AudioTooShortForEmbeddingError(
                f"Audio file '{filepath.name}' is {duration_seconds:.3f}s long, "
                "which is too short for MusiCNN to produce a single embedding "
                f"frame (requires approximately {_MUSICNN_MIN_AUDIO_SECONDS:.3f}s "
                "or more). The model returned zero frames; mean-pooling them "
                "would silently produce a NaN embedding."
            )
        embedding = np.mean(emb_frames, axis=0).astype(np.float32)

        # Frame-level sigmoid activations → (N_frames, 50); mean → (50,)
        tag_activations = np.mean(self._model_tags(y), axis=0)
        top_tags = self._build_top_tags(tag_activations)

        # MIREX pipeline:
        # 1. Get 200D features from MusiCNN (N_frames, 200)
        mirex_feats = self._model_mirex_feats(y)
        # 2. Feed 200D features into MIREX head (N_frames, 5)
        mood_results = self._model_moods(mirex_feats)
        mood_activations = np.mean(mood_results, axis=0)
        moods = self._build_moods(mood_activations)

        features = EmbeddingFeatures(
            filepath=filepath,
            file_hash=file_hash,
            embedding=embedding,
            top_tags=top_tags,
            moods=moods,
        )

        if self.cache_enabled:
            self._save_to_cache(file_hash, features)

        if self.cache_db is not None:
            self.cache_db.put_moods(file_hash, features.moods, features.primary_mood or "Unknown")

        return features

    def analyze_batch(self, filepaths: list[Path]) -> dict[Path, EmbeddingFeatures]:
        """
        Analyze a batch of files.

        Files that fail analysis are skipped and logged at WARNING level;
        clips rejected as too short for the model are named as such rather
        than reported as a generic failure.

        Args:
            filepaths: List of audio file paths.

        Returns:
            Dict mapping successfully-analyzed paths → EmbeddingFeatures.
        """
        results: dict[Path, EmbeddingFeatures] = {}

        for fp in filepaths:
            try:
                # Pre-flight check: if we have it in CacheDB but not in file cache,
                # we still need to run analyze() to get the embedding and tags
                # because they aren't in CacheDB yet.
                # Actually, analyze() checks file cache first.
                feat = self.analyze(fp)

                # If we just computed it and have a DB, it's already put in analyze()
                results[fp] = feat
            except AudioTooShortForEmbeddingError as exc:
                # Named separately from the generic handler so the log line
                # states the real cause rather than a downstream symptom.
                logger.warning("Skipping %s: audio too short for MusiCNN — %s", fp.name, exc)
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

    def analyze_discogs_effnet(self, filepath: Path) -> np.ndarray:
        """
        Extract a raw, mean-pooled discogs-effnet embedding from an audio file.

        Mirrors ``analyze()``'s MusiCNN pathway, but runs the discogs-effnet
        model instead and returns the raw 1280-dim vector uncached. This is
        a diagnostic/smoke path (see ``run_embedding_smoke_check``) and is
        distinct from the production PCA-reduced embedding-cache pathway in
        ``embedding_cache.py`` / ``scripts/embed_library.py``, which calls
        this method and then persists its output via ``EmbeddingCache``.

        The model is downloaded and initialised on the first call.

        Args:
            filepath: Path to the audio file.

        Returns:
            A float32 numpy array of shape (1280,) — the mean-pooled
            discogs-effnet embedding.

        Raises:
            FileNotFoundError: If the file does not exist.
            AudioTooShortForEmbeddingError: If the audio is shorter than
                ``_DISCOGS_EFFNET_MIN_AUDIO_SECONDS``, in which case the
                model produces zero frames and there is nothing valid to
                mean-pool.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        logger.debug("Extracting discogs-effnet embedding: %s", filepath.name)

        import librosa  # noqa: PLC0415

        y, _ = librosa.load(filepath, sr=self.sample_rate, mono=True)

        self._ensure_discogs_effnet_model()

        # _DiscogsEffnetModel is always defined when _ESSENTIA_AVAILABLE is
        # True, which is guaranteed by __init__'s guard.
        assert _DiscogsEffnetModel is not None

        if self._model_discogs_effnet is None:
            self._model_discogs_effnet = cast(Any, _DiscogsEffnetModel)(
                graphFilename=str(self.discogs_effnet_model_path),
                output=_DISCOGS_EFFNET_EMB_OUTPUT_LAYER,
            )

        # Frame-level embeddings → (N_frames, 1280); mean-pool → (1280,)
        emb_frames = self._model_discogs_effnet(y)
        if emb_frames.size == 0:
            duration_seconds = len(y) / self.sample_rate
            logger.error(
                "discogs-effnet produced zero frames for %s (%.3fs, below the ~%.3fs minimum)",
                filepath.name,
                duration_seconds,
                _DISCOGS_EFFNET_MIN_AUDIO_SECONDS,
            )
            raise AudioTooShortForEmbeddingError(
                f"Audio file '{filepath.name}' is {duration_seconds:.3f}s long, "
                "which is too short for discogs-effnet to produce a single "
                f"embedding frame (requires approximately "
                f"{_DISCOGS_EFFNET_MIN_AUDIO_SECONDS:.3f}s or more). The model "
                "returned zero frames; mean-pooling them would silently "
                "produce a NaN scalar instead of a (1280,) vector."
            )

        return np.mean(emb_frames, axis=0).astype(np.float32)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        """Lazy-initialise both model instances, downloading .pb if absent."""
        if not self.model_path.exists():
            self._download_model(
                self.model_path,
                _MSD_MUSICNN_URL,
                _MSD_MUSICNN_META,
                expected_sha256=_MSD_MUSICNN_SHA256,
            )

        if not self.mood_model_path.exists():
            self._download_model(
                self.mood_model_path,
                _MIREX_MOODS_URL,
                _MIREX_MOODS_META,
                expected_sha256=_MIREX_MOODS_SHA256,
            )

        # _EssentiaModel is always defined when _ESSENTIA_AVAILABLE is True,
        # which is guaranteed by __init__'s guard.
        assert _EssentiaModel is not None

        if self._model_emb is None:
            self._model_emb = cast(Any, _EssentiaModel)(
                graphFilename=str(self.model_path),
                output=_EMB_OUTPUT_LAYER,
            )
        if self._model_tags is None:
            self._model_tags = cast(Any, _EssentiaModel)(
                graphFilename=str(self.model_path),
                output=_TAG_OUTPUT_LAYER,
            )
        if self._model_mirex_feats is None:
            self._model_mirex_feats = cast(Any, _EssentiaModel)(
                graphFilename=str(self.model_path),
                output=_MIREX_FEATS_LAYER,
            )
        if self._model_moods is None:
            self._model_moods = cast(Any, TensorflowPredict2D)(
                graphFilename=str(self.mood_model_path),
                input=_MOOD_INPUT_LAYER,
                output=_MOOD_OUTPUT_LAYER,
            )

    def _ensure_discogs_effnet_model(self) -> None:
        """Download the discogs-effnet model .pb if it is not already present."""
        if not self.discogs_effnet_model_path.exists():
            self._download_model(
                self.discogs_effnet_model_path,
                _DISCOGS_EFFNET_URL,
                _DISCOGS_EFFNET_META,
                expected_sha256=_DISCOGS_EFFNET_SHA256,
            )

    def _download_model(
        self, target: Path, pb_url: str, meta_url: str, *, expected_sha256: str
    ) -> None:
        """
        Download model .pb and companion metadata .json to target path.

        After downloading the .pb, verifies its sha256 against
        ``expected_sha256`` before fetching the metadata JSON. This is
        trust-on-first-use: essentia.upf.edu publishes no checksums, so this
        cannot validate the very first download, only detect the model file
        changing under us on a later one.

        Args:
            target: Destination path for .pb file.
            pb_url: URL to the .pb model file.
            meta_url: URL to the .json metadata file.
            expected_sha256: Pinned sha256 hex digest the downloaded .pb must
                match. Required with no default so a call site cannot
                accidentally skip verification.

        Raises:
            ModelIntegrityError: If the downloaded .pb's sha256 does not
                match ``expected_sha256``. The mismatched file is deleted
                from disk so a later ``_ensure_model()`` /
                ``_ensure_discogs_effnet_model()`` call does not mistake it
                for a valid cached download.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        meta_path = target.with_suffix(".json")

        logger.info("Downloading model to %s", target)
        urllib.request.urlretrieve(pb_url, target)

        actual_sha256 = self._compute_sha256(target)
        if actual_sha256 != expected_sha256:
            target.unlink(missing_ok=True)
            raise ModelIntegrityError(
                f"Integrity check failed for model downloaded from {pb_url}: "
                f"expected sha256={expected_sha256}, actual sha256={actual_sha256}. "
                "essentia.upf.edu publishes no checksums, so this may be a "
                "legitimate upstream re-release rather than corruption or "
                "tampering. If you have independently verified the new file "
                "is trustworthy, update the corresponding *_SHA256 constant "
                "in playchitect/core/embedding_extractor.py to re-pin it."
            )

        logger.info("Downloading metadata to %s", meta_path)
        urllib.request.urlretrieve(meta_url, meta_path)

    def _compute_sha256(self, path: Path) -> str:
        """Compute the sha256 hex digest of a file's full contents."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

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

    def _build_moods(self, activations: np.ndarray) -> list[tuple[str, float]]:
        """Convert mood activation vector to sorted (mood, probability) list."""
        labels = self._load_mood_labels()
        if labels is None or len(labels) != len(activations):
            labels = [str(i) for i in range(len(activations))]

        pairs = sorted(
            zip(labels, activations.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(mood, float(prob)) for mood, prob in pairs]

    def _load_tag_labels(self) -> list[str] | None:
        """Load tag label list from companion metadata JSON (cached in memory)."""
        if self._tag_labels is not None:
            return self._tag_labels

        meta_path = self.model_path.with_suffix(".json")
        self._tag_labels = self._load_labels_from_json(meta_path)
        return self._tag_labels

    def _load_mood_labels(self) -> list[str] | None:
        """Load mood label list from companion metadata JSON (cached in memory)."""
        if self._mood_labels is not None:
            return self._mood_labels

        meta_path = self.mood_model_path.with_suffix(".json")
        self._mood_labels = self._load_labels_from_json(meta_path)
        return self._mood_labels

    def _load_labels_from_json(self, meta_path: Path) -> list[str] | None:
        """Helper to load 'classes' from an Essentia metadata JSON."""
        if not meta_path.exists():
            return None

        try:
            with open(meta_path) as f:
                meta = json.load(f)
            return meta.get("classes", [])
        except Exception as exc:
            logger.warning("Failed to load labels from %s: %s", meta_path.name, exc)
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
        """Save embedding → {hash}.npy, tags/moods → {hash}_metadata.json."""
        try:
            np.save(str(self._get_cache_path(file_hash)), feat.embedding)
            meta_path = self.cache_dir / f"{file_hash}_metadata.json"
            meta_data = {
                "top_tags": feat.top_tags,
                "moods": feat.moods,
            }
            with open(meta_path, "w") as f:
                json.dump(meta_data, f)
            logger.debug("Cached embedding and moods: %s", file_hash[:8])
        except Exception as exc:
            logger.warning("Failed to cache embedding: %s", exc)

    def _load_from_cache(self, file_hash: str) -> EmbeddingFeatures | None:
        """Load embedding from .npy and metadata from _metadata.json."""
        emb_path = self._get_cache_path(file_hash)
        meta_path = self.cache_dir / f"{file_hash}_metadata.json"

        # Backward compatibility: check for old _tags.json
        old_tags_path = self.cache_dir / f"{file_hash}_tags.json"

        if not emb_path.exists():
            return None

        try:
            embedding = np.load(str(emb_path)).astype(np.float32)
            top_tags = []
            moods = []

            if meta_path.exists():
                with open(meta_path) as f:
                    meta_data = json.load(f)
                top_tags = [(str(t[0]), float(t[1])) for t in meta_data.get("top_tags", [])]
                moods = [(str(m[0]), float(m[1])) for m in meta_data.get("moods", [])]
            elif old_tags_path.exists():
                with open(old_tags_path) as f:
                    raw_tags = json.load(f)
                top_tags = [(str(t[0]), float(t[1])) for t in raw_tags]
            else:
                # No metadata found, cache is incomplete
                return None

            return EmbeddingFeatures(
                filepath=Path(""),  # Caller updates this
                file_hash=file_hash,
                embedding=embedding,
                top_tags=top_tags,
                moods=moods,
            )
        except Exception as exc:
            logger.warning("Failed to load cached embedding: %s", exc)
            return None


# ── Pre-batch smoke check (TASK-31, rail 2) ─────────────────────────────────
#
# Gates the library-wide batch embedding run (scripts/embed_library.py) over
# a pinned dev-line essentia-tensorflow wheel. The real failure mode of that
# pin is not "it won't install" -- that already fails loudly at
# `import essentia` -- it's "it installs and silently produces garbage": a
# differently-shaped, NaN-filled, or non-reproducible embedding vector would
# poison the entire cache without ever raising an exception. This check
# exists to catch that before the batch run, so it must never be mocked out
# on a machine that actually has essentia-tensorflow installed.

# L2-norm sanity band for a discogs-effnet mean-pooled embedding. Bounds are
# deliberately generous -- this is a coarse "did the model produce something
# that looks like an embedding at all" check, not a precise statistical
# bound, so it tolerates ordinary track-to-track variation while still
# catching a mis-wired graph output (near-zero) or a garbage/unnormalised
# tensor (absurdly large).
_SMOKE_CHECK_MIN_NORM: float = 1e-3
_SMOKE_CHECK_MAX_NORM: float = 1e6

# Absolute tolerance for cross-instance reproducibility comparison. Two
# independent EmbeddingExtractor instances analysing the same fixture should
# produce numerically identical (deterministic inference) or near-identical
# (floating point ordering) output; anything beyond this tolerance indicates
# non-deterministic behaviour (e.g. dropout/batchnorm left in training mode).
_SMOKE_CHECK_REPRODUCIBILITY_ATOL: float = 1e-4


def run_embedding_smoke_check(fixture_path: Path) -> None:
    """
    Run a small audio fixture through discogs-effnet twice and sanity-check
    the output before trusting the model for a full batch run.

    Runs ``fixture_path`` through TWO independent ``EmbeddingExtractor``
    instances (cache disabled, so a synthetic smoke fixture never pollutes
    the production embedding cache) and validates dimensionality,
    finiteness, L2-norm sanity, and cross-instance reproducibility.

    Must be callable standalone -- e.g. from ``scripts/embed_library.py`` as
    a pre-flight gate -- without any pytest fixture.

    Args:
        fixture_path: Path to a short audio fixture used purely to exercise
            the model; its musical content is irrelevant, only the shape/
            finiteness/reproducibility of the resulting embedding matters.

    Returns:
        None, on success.

    Raises:
        FileNotFoundError: If ``fixture_path`` does not exist. This is a
            caller error, not a model-quality failure, so it is not wrapped
            into ``EmbeddingSmokeCheckError``.
        EmbeddingSmokeCheckError: If the embedding fails a dimensionality,
            finiteness, norm-band, or reproducibility check.
    """
    extractor_a = EmbeddingExtractor(cache_enabled=False)
    extractor_b = EmbeddingExtractor(cache_enabled=False)

    vector_a = extractor_a.analyze_discogs_effnet(fixture_path)
    vector_b = extractor_b.analyze_discogs_effnet(fixture_path)

    for label, vector in (("first", vector_a), ("second", vector_b)):
        if vector.shape != (_DISCOGS_EFFNET_EMBEDDING_DIM,):
            raise EmbeddingSmokeCheckError(
                f"{label} discogs-effnet embedding has shape {vector.shape}, "
                f"expected ({_DISCOGS_EFFNET_EMBEDDING_DIM},). The model may "
                "be mis-wired or the pinned wheel may not match the pinned "
                "model file."
            )
        if not np.all(np.isfinite(vector)):
            raise EmbeddingSmokeCheckError(
                f"{label} discogs-effnet embedding contains non-finite "
                "values (NaN or Inf) -- the model produced garbage output."
            )
        norm = float(np.linalg.norm(vector))
        if not (_SMOKE_CHECK_MIN_NORM <= norm <= _SMOKE_CHECK_MAX_NORM):
            raise EmbeddingSmokeCheckError(
                f"{label} discogs-effnet embedding L2 norm ({norm}) is "
                f"outside the sanity band [{_SMOKE_CHECK_MIN_NORM}, "
                f"{_SMOKE_CHECK_MAX_NORM}] -- the model likely produced a "
                "degenerate (near-zero) or unnormalised (absurdly large) "
                "output."
            )

    if not np.allclose(vector_a, vector_b, atol=_SMOKE_CHECK_REPRODUCIBILITY_ATOL):
        raise EmbeddingSmokeCheckError(
            "discogs-effnet embedding is not reproducible: two independent "
            "EmbeddingExtractor instances produced meaningfully different "
            "vectors for the same fixture (e.g. dropout/batchnorm left in "
            "training mode)."
        )

    logger.info("Embedding smoke check passed for %s", fixture_path.name)
