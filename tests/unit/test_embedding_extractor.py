"""
Unit tests for embedding_extractor module.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import playchitect.core.embedding_extractor as emb_mod
from playchitect.core.embedding_extractor import (
    _EMB_OUTPUT_LAYER,
    _MSD_MUSICNN_URL,
    EmbeddingExtractor,
    EmbeddingFeatures,
)

# ── Helper ────────────────────────────────────────────────────────────────────


def make_embedding(path: Path, seed: int = 0) -> EmbeddingFeatures:
    """Create a synthetic EmbeddingFeatures for testing."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(128).astype(np.float32)
    return EmbeddingFeatures(
        filepath=path,
        file_hash="abc123",
        embedding=vec / np.linalg.norm(vec),
        top_tags=[("techno", 0.9), ("electronic", 0.7)],
    )


# ── TestEmbeddingFeatures ────────────────────────────────────────────────────


class TestEmbeddingFeatures:
    """Test EmbeddingFeatures dataclass."""

    def test_construction(self) -> None:
        feat = make_embedding(Path("track.mp3"))
        assert feat.filepath == Path("track.mp3")
        assert feat.file_hash == "abc123"
        assert feat.embedding.shape == (128,)
        assert feat.top_tags[0] == ("techno", 0.9)

    def test_to_dict_round_trip(self) -> None:
        original = make_embedding(Path("track.mp3"), seed=7)
        data = original.to_dict()

        assert data["filepath"] == "track.mp3"
        assert data["file_hash"] == "abc123"
        assert isinstance(data["embedding"], list)
        assert len(data["embedding"]) == 128
        assert data["top_tags"] == [["techno", 0.9], ["electronic", 0.7]]

        restored = EmbeddingFeatures.from_dict(data)

        assert restored.filepath == original.filepath
        assert restored.file_hash == original.file_hash
        np.testing.assert_array_almost_equal(restored.embedding, original.embedding)
        assert restored.top_tags == original.top_tags

    def test_from_dict_handles_list_tags(self) -> None:
        """from_dict should accept JSON list-of-lists for top_tags."""
        data = {
            "filepath": "foo.mp3",
            "file_hash": "deadbeef",
            "embedding": [0.0] * 128,
            "top_tags": [["house", 0.8], ["ambient", 0.3]],
        }
        feat = EmbeddingFeatures.from_dict(data)
        assert feat.top_tags == [("house", 0.8), ("ambient", 0.3)]

    def test_embedding_dtype(self) -> None:
        feat = make_embedding(Path("t.mp3"))
        assert feat.embedding.dtype == np.float32


# ── TestEmbeddingExtractorGenre ──────────────────────────────────────────────


class TestEmbeddingExtractorGenre:
    """Test infer_genre() without loading any model."""

    @pytest.fixture()
    def extractor(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EmbeddingExtractor:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)
        return EmbeddingExtractor(cache_enabled=False, model_path=tmp_path / "fake.pb")

    def test_known_tag_returns_genre(self, extractor: EmbeddingExtractor) -> None:
        feat = EmbeddingFeatures(
            filepath=Path("t.mp3"),
            file_hash="x",
            embedding=np.zeros(128, dtype=np.float32),
            top_tags=[("techno", 0.95), ("electronic", 0.6)],
        )
        assert extractor.infer_genre(feat) == "techno"

    def test_ambient_tag(self, extractor: EmbeddingExtractor) -> None:
        feat = EmbeddingFeatures(
            filepath=Path("t.mp3"),
            file_hash="x",
            embedding=np.zeros(128, dtype=np.float32),
            top_tags=[("ambient", 0.85)],
        )
        assert extractor.infer_genre(feat) == "ambient"

    def test_dnb_aliases(self, extractor: EmbeddingExtractor) -> None:
        for tag in ("drum and bass", "dnb", "jungle"):
            feat = EmbeddingFeatures(
                filepath=Path("t.mp3"),
                file_hash="x",
                embedding=np.zeros(128, dtype=np.float32),
                top_tags=[(tag, 0.9)],
            )
            assert extractor.infer_genre(feat) == "dnb"

    def test_unknown_tags_return_none(self, extractor: EmbeddingExtractor) -> None:
        feat = EmbeddingFeatures(
            filepath=Path("t.mp3"),
            file_hash="x",
            embedding=np.zeros(128, dtype=np.float32),
            top_tags=[("jazz", 0.9), ("classical", 0.7)],
        )
        assert extractor.infer_genre(feat) is None

    def test_highest_confidence_wins(self, extractor: EmbeddingExtractor) -> None:
        """When multiple genre tags are present the first (highest conf) wins."""
        feat = EmbeddingFeatures(
            filepath=Path("t.mp3"),
            file_hash="x",
            embedding=np.zeros(128, dtype=np.float32),
            top_tags=[("house", 0.95), ("techno", 0.85)],
        )
        assert extractor.infer_genre(feat) == "house"

    def test_case_insensitive(self, extractor: EmbeddingExtractor) -> None:
        feat = EmbeddingFeatures(
            filepath=Path("t.mp3"),
            file_hash="x",
            embedding=np.zeros(128, dtype=np.float32),
            top_tags=[("Techno", 0.9)],
        )
        assert extractor.infer_genre(feat) == "techno"


# ── TestEmbeddingExtractorCache ───────────────────────────────────────────────


class TestEmbeddingExtractorCache:
    """Test cache helpers without invoking the model."""

    @pytest.fixture()
    def extractor(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EmbeddingExtractor:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)
        return EmbeddingExtractor(
            cache_dir=tmp_path / "cache",
            cache_enabled=True,
            model_path=tmp_path / "fake.pb",
        )

    def test_cache_miss_returns_none(self, extractor: EmbeddingExtractor) -> None:
        assert extractor._load_from_cache("nonexistent_hash") is None

    def test_save_and_load_round_trip(self, extractor: EmbeddingExtractor, tmp_path: Path) -> None:
        feat = make_embedding(Path("track.mp3"), seed=3)
        extractor._save_to_cache(feat.file_hash, feat)

        loaded = extractor._load_from_cache(feat.file_hash)
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded.embedding, feat.embedding)
        assert loaded.top_tags == feat.top_tags
        assert loaded.file_hash == feat.file_hash

    def test_npy_file_created(self, extractor: EmbeddingExtractor) -> None:
        feat = make_embedding(Path("track.mp3"))
        extractor._save_to_cache(feat.file_hash, feat)

        npy_files = list(extractor.cache_dir.glob("*.npy"))
        assert len(npy_files) == 1

    def test_tags_json_created(self, extractor: EmbeddingExtractor) -> None:
        feat = make_embedding(Path("track.mp3"))
        extractor._save_to_cache(feat.file_hash, feat)

        json_files = list(extractor.cache_dir.glob("*_tags.json"))
        assert len(json_files) == 1

    def test_partial_cache_miss_returns_none(self, extractor: EmbeddingExtractor) -> None:
        """If only .npy exists (no _tags.json), load returns None."""
        feat = make_embedding(Path("track.mp3"))
        # Only write the .npy, not the tags JSON
        np.save(str(extractor._get_cache_path(feat.file_hash)), feat.embedding)
        assert extractor._load_from_cache(feat.file_hash) is None


# ── TestEmbeddingExtractorAnalysis ────────────────────────────────────────────


class TestEmbeddingExtractorAnalysis:
    """Test analyze() with a fully mocked Essentia model."""

    _N_FRAMES = 5
    _N_TAGS = 50

    def _make_mock_model_class(self) -> type:
        """Return a mock _EssentiaModel class whose instances return synthetic arrays."""
        n_frames = self._N_FRAMES
        n_tags = self._N_TAGS

        class MockModel:
            def __init__(self, graphFilename: str, output: str = "", **kwargs: object):
                self.output = output

            def __call__(self, audio: np.ndarray) -> np.ndarray:
                rng = np.random.default_rng(0)
                if self.output == _EMB_OUTPUT_LAYER:
                    return rng.standard_normal((n_frames, 128)).astype(np.float32)
                # TAG_OUTPUT_LAYER — sigmoid activations in [0, 1]
                return np.abs(rng.standard_normal((n_frames, n_tags))).astype(np.float32)

        return MockModel

    @pytest.fixture()
    def extractor_with_mock_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> EmbeddingExtractor:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)
        monkeypatch.setattr(emb_mod, "_EssentiaModel", self._make_mock_model_class())

        # Create a fake .pb so _ensure_model skips download
        model_file = tmp_path / "fake.pb"
        model_file.write_bytes(b"fake_model")

        # Create a fake metadata JSON with 50 class labels
        meta_file = tmp_path / "fake.json"
        labels = [f"tag_{i}" for i in range(self._N_TAGS)]
        meta_file.write_text(json.dumps({"classes": labels}))

        return EmbeddingExtractor(
            model_path=model_file,
            cache_dir=tmp_path / "cache",
            cache_enabled=True,
        )

    def test_analyze_returns_embedding_shape(
        self,
        extractor_with_mock_model: EmbeddingExtractor,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        audio_file = tmp_path / "track.wav"
        audio_file.write_bytes(b"\x00" * 100)

        # Patch librosa.load to return synthetic audio
        monkeypatch.setattr(
            "playchitect.core.embedding_extractor.librosa",
            MagicMock(load=MagicMock(return_value=(np.zeros(16000, dtype=np.float32), 16000))),
            raising=False,
        )
        # librosa is a lazy import inside analyze(); patch at the module level
        import librosa as _librosa  # noqa: PLC0415

        monkeypatch.setattr(
            _librosa, "load", lambda *a, **kw: (np.zeros(16000, dtype=np.float32), 16000)
        )

        feat = extractor_with_mock_model.analyze(audio_file)

        assert feat.embedding.shape == (128,)
        assert feat.embedding.dtype == np.float32

    def test_analyze_produces_top_tags(
        self,
        extractor_with_mock_model: EmbeddingExtractor,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        audio_file = tmp_path / "track.wav"
        audio_file.write_bytes(b"\x00" * 100)

        import librosa as _librosa  # noqa: PLC0415

        monkeypatch.setattr(
            _librosa, "load", lambda *a, **kw: (np.zeros(16000, dtype=np.float32), 16000)
        )

        feat = extractor_with_mock_model.analyze(audio_file)

        assert len(feat.top_tags) == self._N_TAGS
        # Tags should be sorted descending by confidence
        confs = [c for _, c in feat.top_tags]
        assert confs == sorted(confs, reverse=True)

    def test_analyze_writes_cache(
        self,
        extractor_with_mock_model: EmbeddingExtractor,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        audio_file = tmp_path / "track.wav"
        audio_file.write_bytes(b"\x00" * 100)

        import librosa as _librosa  # noqa: PLC0415

        monkeypatch.setattr(
            _librosa, "load", lambda *a, **kw: (np.zeros(16000, dtype=np.float32), 16000)
        )

        extractor_with_mock_model.analyze(audio_file)

        npy_files = list(extractor_with_mock_model.cache_dir.glob("*.npy"))
        assert len(npy_files) == 1

    def test_mean_pooling_is_correct(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify that frame-level arrays are mean-pooled to (128,)."""
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)

        n_frames = 4
        fixed_frames = np.arange(n_frames * 128, dtype=np.float32).reshape(n_frames, 128)
        expected_mean = fixed_frames.mean(axis=0)

        class FixedModel:
            def __init__(self, graphFilename: str, output: str = "", **kwargs: object):
                self.output = output

            def __call__(self, audio: np.ndarray) -> np.ndarray:
                if self.output == _EMB_OUTPUT_LAYER:
                    return fixed_frames
                return np.zeros((n_frames, self._N_TAGS), dtype=np.float32)

            _N_TAGS = 50

        monkeypatch.setattr(emb_mod, "_EssentiaModel", FixedModel)

        model_file = tmp_path / "fake.pb"
        model_file.write_bytes(b"x")
        meta_file = tmp_path / "fake.json"
        meta_file.write_text(json.dumps({"classes": [str(i) for i in range(50)]}))

        extractor = EmbeddingExtractor(
            model_path=model_file,
            cache_enabled=False,
        )

        audio_file = tmp_path / "t.wav"
        audio_file.write_bytes(b"\x00" * 100)

        import librosa as _librosa  # noqa: PLC0415

        monkeypatch.setattr(
            _librosa, "load", lambda *a, **kw: (np.zeros(16000, dtype=np.float32), 16000)
        )

        feat = extractor.analyze(audio_file)
        np.testing.assert_array_almost_equal(feat.embedding, expected_mean)


# ── TestEmbeddingExtractorMissingEssentia ─────────────────────────────────────


class TestEmbeddingExtractorMissingEssentia:
    """Confirm RuntimeError when essentia-tensorflow is absent."""

    def test_init_raises_without_essentia(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", False)

        with pytest.raises(RuntimeError, match="essentia-tensorflow"):
            EmbeddingExtractor(model_path=tmp_path / "fake.pb")


# ── TestEmbeddingExtractorDownload ────────────────────────────────────────────


class TestEmbeddingExtractorDownload:
    """Verify _download_model calls urlretrieve with the correct URL."""

    def test_download_uses_correct_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)

        target = tmp_path / "models" / "msd-musicnn-1.pb"
        calls: list[tuple[str, Path]] = []

        def fake_urlretrieve(url: str, dest: object) -> None:
            calls.append((url, Path(str(dest))))

        monkeypatch.setattr(
            "playchitect.core.embedding_extractor.urllib.request.urlretrieve",
            fake_urlretrieve,
        )

        extractor = EmbeddingExtractor(model_path=target, cache_enabled=False)
        extractor._download_model(target)

        # First call must be the .pb model URL
        assert len(calls) >= 1
        pb_call = calls[0]
        assert pb_call[0] == _MSD_MUSICNN_URL
        assert pb_call[1] == target

    def test_download_target_path_matches_model_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)

        custom_path = tmp_path / "my_models" / "msd-musicnn-1.pb"
        retrieved: list[Path] = []

        def fake_urlretrieve(url: str, dest: object) -> None:
            retrieved.append(Path(str(dest)))

        monkeypatch.setattr(
            "playchitect.core.embedding_extractor.urllib.request.urlretrieve",
            fake_urlretrieve,
        )

        extractor = EmbeddingExtractor(model_path=custom_path, cache_enabled=False)
        extractor._download_model(custom_path)

        assert retrieved[0] == custom_path
