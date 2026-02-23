"""
Unit tests for embedding_extractor module.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import playchitect.core.embedding_extractor as emb_mod
from playchitect.core.embedding_extractor import (
    _EMB_OUTPUT_LAYER,
    _MIREX_FEATS_LAYER,
    _MOOD_OUTPUT_LAYER,
    _MSD_MUSICNN_URL,
    _TAG_OUTPUT_LAYER,
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
        moods=[("Aggressive", 0.8), ("Passionate", 0.2)],
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
        assert feat.moods[0] == ("Aggressive", 0.8)

    def test_primary_mood(self) -> None:
        feat = make_embedding(Path("track.mp3"))
        assert feat.primary_mood == "Aggressive"

    def test_primary_mood_empty(self) -> None:
        feat = EmbeddingFeatures(
            filepath=Path("t.mp3"),
            file_hash="x",
            embedding=np.zeros(128, dtype=np.float32),
            top_tags=[],
            moods=[],
        )
        assert feat.primary_mood is None

    def test_to_dict_round_trip(self) -> None:
        original = make_embedding(Path("track.mp3"), seed=7)
        data = original.to_dict()

        assert data["filepath"] == "track.mp3"
        assert data["file_hash"] == "abc123"
        assert isinstance(data["embedding"], list)
        assert len(data["embedding"]) == 128
        assert data["top_tags"] == [["techno", 0.9], ["electronic", 0.7]]
        assert data["moods"] == [["Aggressive", 0.8], ["Passionate", 0.2]]

        restored = EmbeddingFeatures.from_dict(data)

        assert restored.filepath == original.filepath
        assert restored.file_hash == original.file_hash
        np.testing.assert_array_almost_equal(restored.embedding, original.embedding)
        assert restored.top_tags == original.top_tags
        assert restored.moods == original.moods

    def test_from_dict_handles_list_tags(self) -> None:
        """from_dict should accept JSON list-of-lists for top_tags and moods."""
        data = {
            "filepath": "foo.mp3",
            "file_hash": "deadbeef",
            "embedding": [0.0] * 128,
            "top_tags": [["house", 0.8], ["ambient", 0.3]],
            "moods": [["Cheerful", 0.9]],
        }
        feat = EmbeddingFeatures.from_dict(data)
        assert feat.top_tags == [("house", 0.8), ("ambient", 0.3)]
        assert feat.moods == [("Cheerful", 0.9)]

    def test_embedding_dtype(self) -> None:
        feat = make_embedding(Path("t.mp3"))
        assert feat.embedding.dtype == np.float32


# ── TestEmbeddingExtractorGenre ──────────────────────────────────────────────


class TestEmbeddingExtractorGenre:
    """Test infer_genre() without loading any model."""

    @pytest.fixture()
    def extractor(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EmbeddingExtractor:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)
        return EmbeddingExtractor(
            cache_enabled=False,
            model_path=tmp_path / "fake.pb",
            mood_model_path=tmp_path / "fake_mood.pb",
        )

    def test_known_tag_returns_genre(self, extractor: EmbeddingExtractor) -> None:
        feat = EmbeddingFeatures(
            filepath=Path("t.mp3"),
            file_hash="x",
            embedding=np.zeros(128, dtype=np.float32),
            top_tags=[("techno", 0.95), ("electronic", 0.6)],
            moods=[],
        )
        assert extractor.infer_genre(feat) == "techno"

    def test_ambient_tag(self, extractor: EmbeddingExtractor) -> None:
        feat = EmbeddingFeatures(
            filepath=Path("t.mp3"),
            file_hash="x",
            embedding=np.zeros(128, dtype=np.float32),
            top_tags=[("ambient", 0.85)],
            moods=[],
        )
        assert extractor.infer_genre(feat) == "ambient"

    def test_dnb_aliases(self, extractor: EmbeddingExtractor) -> None:
        for tag in ("drum and bass", "dnb", "jungle"):
            feat = EmbeddingFeatures(
                filepath=Path("t.mp3"),
                file_hash="x",
                embedding=np.zeros(128, dtype=np.float32),
                top_tags=[(tag, 0.9)],
                moods=[],
            )
            assert extractor.infer_genre(feat) == "dnb"

    def test_unknown_tags_return_none(self, extractor: EmbeddingExtractor) -> None:
        feat = EmbeddingFeatures(
            filepath=Path("t.mp3"),
            file_hash="x",
            embedding=np.zeros(128, dtype=np.float32),
            top_tags=[("jazz", 0.9), ("classical", 0.7)],
            moods=[],
        )
        assert extractor.infer_genre(feat) is None

    def test_highest_confidence_wins(self, extractor: EmbeddingExtractor) -> None:
        """When multiple genre tags are present the first (highest conf) wins."""
        feat = EmbeddingFeatures(
            filepath=Path("t.mp3"),
            file_hash="x",
            embedding=np.zeros(128, dtype=np.float32),
            top_tags=[("house", 0.95), ("techno", 0.85)],
            moods=[],
        )
        assert extractor.infer_genre(feat) == "house"

    def test_case_insensitive(self, extractor: EmbeddingExtractor) -> None:
        feat = EmbeddingFeatures(
            filepath=Path("t.mp3"),
            file_hash="x",
            embedding=np.zeros(128, dtype=np.float32),
            top_tags=[("Techno", 0.9)],
            moods=[],
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
            mood_model_path=tmp_path / "fake_mood.pb",
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
        assert loaded.moods == feat.moods
        assert loaded.file_hash == feat.file_hash

    def test_npy_file_created(self, extractor: EmbeddingExtractor) -> None:
        feat = make_embedding(Path("track.mp3"))
        extractor._save_to_cache(feat.file_hash, feat)

        npy_files = list(extractor.cache_dir.glob("*.npy"))
        assert len(npy_files) == 1

    def test_metadata_json_created(self, extractor: EmbeddingExtractor) -> None:
        feat = make_embedding(Path("track.mp3"))
        extractor._save_to_cache(feat.file_hash, feat)

        json_files = list(extractor.cache_dir.glob("*_metadata.json"))
        assert len(json_files) == 1

    def test_load_from_legacy_tags_json(
        self, extractor: EmbeddingExtractor, tmp_path: Path
    ) -> None:
        """Verify backward compatibility: load top_tags from old _tags.json."""
        file_hash = "legacy_123"
        extractor.cache_dir.mkdir(parents=True, exist_ok=True)
        # Write valid .npy
        np.save(str(extractor._get_cache_path(file_hash)), np.zeros(128, dtype=np.float32))
        # Write old _tags.json
        tags_path = extractor.cache_dir / f"{file_hash}_tags.json"
        with open(tags_path, "w") as f:
            json.dump([["house", 0.8]], f)

        loaded = extractor._load_from_cache(file_hash)
        assert loaded is not None
        assert loaded.top_tags == [("house", 0.8)]
        assert loaded.moods == []  # empty for legacy cache

    def test_partial_cache_miss_returns_none(self, extractor: EmbeddingExtractor) -> None:
        """If only .npy exists (no metadata JSON), load returns None."""
        feat = make_embedding(Path("track.mp3"))
        # Only write the .npy, not the metadata JSON
        np.save(str(extractor._get_cache_path(feat.file_hash)), feat.embedding)
        assert extractor._load_from_cache(feat.file_hash) is None


# ── TestEmbeddingExtractorAnalysis ────────────────────────────────────────────


class TestEmbeddingExtractorAnalysis:
    """Test analyze() with fully mocked Essentia models."""

    _N_FRAMES = 5
    _N_TAGS = 50
    _N_MOODS = 5

    def _make_mock_model_class(self) -> type:
        """Return a mock _EssentiaModel class whose instances return synthetic arrays."""
        n_frames = self._N_FRAMES
        n_tags = self._N_TAGS
        n_moods = self._N_MOODS

        class MockModel:
            def __init__(
                self,
                graphFilename: str,
                output: str = "",
                input: str = "",
                inputs: list[Any] | None = None,
                outputs: list[Any] | None = None,
                **kwargs: object,
            ):
                self.output = output

            def __call__(self, audio: np.ndarray) -> Any:
                from playchitect.core.embedding_extractor import (
                    _EMB_OUTPUT_LAYER as EMB_OUT,
                )
                from playchitect.core.embedding_extractor import (
                    _MIREX_FEATS_LAYER as MIREX_OUT,
                )
                from playchitect.core.embedding_extractor import (
                    _MOOD_OUTPUT_LAYER as MOOD_OUT,
                )

                rng = np.random.default_rng(0)
                if self.output == MOOD_OUT:
                    return np.abs(rng.standard_normal((n_frames, n_moods))).astype(np.float32)

                if self.output == EMB_OUT:
                    return rng.standard_normal((n_frames, 128)).astype(np.float32)

                if self.output == MIREX_OUT:
                    return np.abs(rng.standard_normal((n_frames, 200))).astype(np.float32)

                # TAG_OUTPUT_LAYER — sigmoid activations in [0, 1]
                return np.abs(rng.standard_normal((n_frames, n_tags))).astype(np.float32)

        return MockModel

    @pytest.fixture()
    def extractor_with_mock_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> EmbeddingExtractor:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)
        mock_cls = self._make_mock_model_class()
        monkeypatch.setattr(emb_mod, "_EssentiaModel", mock_cls)
        monkeypatch.setattr(emb_mod, "TensorflowPredict2D", mock_cls)

        # Create fake .pb files so _ensure_model skips download
        model_file = tmp_path / "fake.pb"
        model_file.write_bytes(b"fake_model")
        mood_model_file = tmp_path / "fake_mood.pb"
        mood_model_file.write_bytes(b"fake_mood_model")

        # Create fake metadata JSONs
        meta_file = tmp_path / "fake.json"
        labels = [f"tag_{i}" for i in range(self._N_TAGS)]
        meta_file.write_text(json.dumps({"classes": labels}))

        mood_meta_file = tmp_path / "fake_mood.json"
        mood_labels = [f"mood_{i}" for i in range(self._N_MOODS)]
        mood_meta_file.write_text(json.dumps({"classes": mood_labels}))

        return EmbeddingExtractor(
            model_path=model_file,
            mood_model_path=mood_model_file,
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

    def test_analyze_produces_moods(
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

        assert len(feat.moods) == self._N_MOODS
        # Moods should be sorted descending by probability
        probs = [p for _, p in feat.moods]
        assert probs == sorted(probs, reverse=True)
        assert feat.primary_mood is not None
        assert feat.primary_mood.startswith("mood_")

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
        meta_files = list(extractor_with_mock_model.cache_dir.glob("*_metadata.json"))
        assert len(meta_files) == 1

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
            def __init__(
                self,
                graphFilename: str,
                output: str = "",
                inputs: list | None = None,
                outputs: list | None = None,
                **kwargs: object,
            ):
                self.output = output
                self.outputs = outputs

            def __call__(self, audio: np.ndarray) -> Any:
                from playchitect.core.embedding_extractor import _MOOD_OUTPUT_LAYER as MOOD_OUT

                if self.outputs and MOOD_OUT in self.outputs:
                    return [np.zeros((n_frames, 5), dtype=np.float32)]
                if self.output == _EMB_OUTPUT_LAYER:
                    return fixed_frames
                return np.zeros((n_frames, self._N_TAGS), dtype=np.float32)

            _N_TAGS = 50

        monkeypatch.setattr(emb_mod, "_EssentiaModel", FixedModel)
        monkeypatch.setattr(emb_mod, "TensorflowPredict2D", FixedModel)

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
        meta = target.with_suffix(".json")
        calls: list[tuple[str, Path]] = []

        def fake_urlretrieve(url: str, dest: object) -> None:
            calls.append((url, Path(str(dest))))

        monkeypatch.setattr(
            "playchitect.core.embedding_extractor.urllib.request.urlretrieve",
            fake_urlretrieve,
        )

        extractor = EmbeddingExtractor(model_path=target, cache_enabled=False)
        extractor._download_model(target, _MSD_MUSICNN_URL, "https://fake.json")

        assert len(calls) == 2
        assert calls[0] == (_MSD_MUSICNN_URL, target)
        assert calls[1] == ("https://fake.json", meta)

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
        extractor._download_model(custom_path, _MSD_MUSICNN_URL, "https://fake.json")

        assert retrieved[0] == custom_path


# ── TestEmbeddingExtractorFileHash ────────────────────────────────────────────


class TestEmbeddingExtractorFileHash:
    """Test _compute_file_hash."""

    @pytest.fixture()
    def extractor(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EmbeddingExtractor:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)
        return EmbeddingExtractor(cache_enabled=False, model_path=tmp_path / "fake.pb")

    def test_hash_is_md5_hex(self, extractor: EmbeddingExtractor, tmp_path: Path) -> None:
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"hello world")
        h = extractor._compute_file_hash(f)
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)

    def test_identical_content_same_hash(
        self, extractor: EmbeddingExtractor, tmp_path: Path
    ) -> None:
        data = b"same content" * 100
        f1 = tmp_path / "a.mp3"
        f2 = tmp_path / "b.mp3"
        f1.write_bytes(data)
        f2.write_bytes(data)
        assert extractor._compute_file_hash(f1) == extractor._compute_file_hash(f2)

    def test_different_content_different_hash(
        self, extractor: EmbeddingExtractor, tmp_path: Path
    ) -> None:
        f1 = tmp_path / "a.mp3"
        f2 = tmp_path / "b.mp3"
        f1.write_bytes(b"content_a")
        f2.write_bytes(b"content_b")
        assert extractor._compute_file_hash(f1) != extractor._compute_file_hash(f2)


# ── TestBuildTopTags ──────────────────────────────────────────────────────────


class TestBuildTopTags:
    """Test _build_top_tags directly."""

    @pytest.fixture()
    def extractor(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EmbeddingExtractor:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)
        return EmbeddingExtractor(cache_enabled=False, model_path=tmp_path / "fake.pb")

    def test_fallback_to_numeric_when_labels_none(self, extractor: EmbeddingExtractor) -> None:
        """No metadata file → _load_tag_labels returns None → numeric labels."""
        activations = np.array([0.1, 0.9, 0.5], dtype=np.float32)
        tags = extractor._build_top_tags(activations)
        # Numeric labels, sorted descending by confidence
        assert tags[0] == ("1", pytest.approx(0.9))
        assert tags[1] == ("2", pytest.approx(0.5))
        assert tags[2] == ("0", pytest.approx(0.1))

    def test_fallback_to_numeric_when_label_count_mismatch(
        self, extractor: EmbeddingExtractor, tmp_path: Path
    ) -> None:
        """3 labels but 5 activations → numeric fallback."""
        meta = tmp_path / "fake.json"
        meta.write_text('{"classes": ["a", "b", "c"]}')
        extractor.model_path = tmp_path / "fake.pb"
        activations = np.ones(5, dtype=np.float32)
        tags = extractor._build_top_tags(activations)
        # Labels should be numeric 0-4
        assert all(t[0].isdigit() for t in tags)

    def test_sorted_descending_by_confidence(
        self, extractor: EmbeddingExtractor, tmp_path: Path
    ) -> None:
        """Output must be sorted highest confidence first."""
        meta = tmp_path / "fake.json"
        meta.write_text('{"classes": ["low", "high", "mid"]}')
        extractor.model_path = tmp_path / "fake.pb"
        activations = np.array([0.1, 0.9, 0.5], dtype=np.float32)
        tags = extractor._build_top_tags(activations)
        confs = [c for _, c in tags]
        assert confs == sorted(confs, reverse=True)
        assert tags[0][0] == "high"


# ── TestLoadTagLabels ─────────────────────────────────────────────────────────


class TestLoadTagLabels:
    """Test _load_tag_labels."""

    @pytest.fixture()
    def extractor(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EmbeddingExtractor:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)
        return EmbeddingExtractor(cache_enabled=False, model_path=tmp_path / "fake.pb")

    def test_no_meta_file_returns_none(self, extractor: EmbeddingExtractor) -> None:
        assert extractor._load_tag_labels() is None

    def test_corrupt_meta_file_returns_none(
        self, extractor: EmbeddingExtractor, tmp_path: Path
    ) -> None:
        meta = tmp_path / "fake.json"
        meta.write_text("NOT VALID JSON {{{")
        assert extractor._load_tag_labels() is None

    def test_valid_meta_returns_labels(self, extractor: EmbeddingExtractor, tmp_path: Path) -> None:
        meta = tmp_path / "fake.json"
        meta.write_text('{"classes": ["tag_0", "tag_1", "tag_2"]}')
        labels = extractor._load_tag_labels()
        assert labels == ["tag_0", "tag_1", "tag_2"]

    def test_cached_in_memory_on_second_call(
        self, extractor: EmbeddingExtractor, tmp_path: Path
    ) -> None:
        """Second call returns cached result without re-reading the file."""
        meta = tmp_path / "fake.json"
        meta.write_text('{"classes": ["a", "b"]}')
        first = extractor._load_tag_labels()
        # Delete the file; second call should still return cached value
        meta.unlink()
        second = extractor._load_tag_labels()
        assert first == second == ["a", "b"]


# ── TestAnalyzeAdditional ─────────────────────────────────────────────────────


class TestAnalyzeAdditional:
    """Additional analyze() coverage tests."""

    @pytest.fixture()
    def extractor(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EmbeddingExtractor:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)
        return EmbeddingExtractor(
            cache_dir=tmp_path / "cache",
            cache_enabled=True,
            model_path=tmp_path / "fake.pb",
            mood_model_path=tmp_path / "fake_mood.pb",
        )

    def test_analyze_nonexistent_file_raises(
        self, extractor: EmbeddingExtractor, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError):
            extractor.analyze(tmp_path / "does_not_exist.mp3")

    def test_analyze_returns_cached_on_second_call(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Second analyze() call hits the cache without touching the model."""
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)

        n_frames = 3
        n_tags = 4
        n_moods = 5

        class MockModel:
            def __init__(
                self,
                graphFilename: str,
                output: str = "",
                inputs: list[Any] | None = None,
                outputs: list[Any] | None = None,
                input: str = "",
                **kwargs: object,
            ):
                self.output = output
                MockModel.call_count += 1

            def __call__(self, audio: np.ndarray) -> Any:
                from playchitect.core.embedding_extractor import (
                    _EMB_OUTPUT_LAYER as EMB_OUT,
                )
                from playchitect.core.embedding_extractor import (
                    _MIREX_FEATS_LAYER as MIREX_OUT,
                )
                from playchitect.core.embedding_extractor import (
                    _MOOD_OUTPUT_LAYER as MOOD_OUT,
                )

                if self.output == MOOD_OUT:
                    return np.ones((n_frames, n_moods), dtype=np.float32)
                if self.output == EMB_OUT:
                    return np.ones((n_frames, 128), dtype=np.float32)
                if self.output == MIREX_OUT:
                    return np.ones((n_frames, 200), dtype=np.float32)
                return np.ones((n_frames, n_tags), dtype=np.float32)

            call_count = 0

        monkeypatch.setattr(emb_mod, "_EssentiaModel", MockModel)
        monkeypatch.setattr(emb_mod, "TensorflowPredict2D", MockModel)

        model_file = tmp_path / "fake.pb"
        model_file.write_bytes(b"x")
        mood_file = tmp_path / "fake_mood.pb"
        mood_file.write_bytes(b"y")

        meta_file = tmp_path / "fake.json"
        meta_file.write_text(
            '{"classes": ' + str([f"t{i}" for i in range(n_tags)]).replace("'", '"') + "}"
        )
        mood_meta = tmp_path / "fake_mood.json"
        mood_meta.write_text(
            '{"classes": ' + str([f"m{i}" for i in range(n_moods)]).replace("'", '"') + "}"
        )

        extractor = EmbeddingExtractor(
            model_path=model_file,
            mood_model_path=mood_file,
            cache_dir=tmp_path / "cache",
            cache_enabled=True,
        )

        audio_file = tmp_path / "track.wav"
        audio_file.write_bytes(b"\x00" * 200)

        import librosa as _librosa  # noqa: PLC0415

        monkeypatch.setattr(
            _librosa, "load", lambda *a, **kw: (np.zeros(16000, dtype=np.float32), 16000)
        )

        MockModel.call_count = 0
        feat1 = extractor.analyze(audio_file)
        feat2 = extractor.analyze(audio_file)  # Should hit cache

        # Embedding content matches
        np.testing.assert_array_equal(feat1.embedding, feat2.embedding)
        # Model constructor called exactly 4 times (emb + tags + mirex_feats + moods)
        # on first analyze only
        assert MockModel.call_count == 4


# ── TestAnalyzeBatch ──────────────────────────────────────────────────────────


class TestAnalyzeBatch:
    """Test analyze_batch()."""

    @pytest.fixture()
    def extractor(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EmbeddingExtractor:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)
        return EmbeddingExtractor(
            cache_enabled=False,
            model_path=tmp_path / "fake.pb",
            mood_model_path=tmp_path / "fake_mood.pb",
        )

    def test_batch_skips_nonexistent_files(
        self, extractor: EmbeddingExtractor, tmp_path: Path
    ) -> None:
        missing = tmp_path / "ghost.mp3"
        result = extractor.analyze_batch([missing])
        assert missing not in result
        assert len(result) == 0

    def test_batch_returns_successful_subset(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """analyze_batch returns only successfully-analyzed paths."""
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)

        class MockModel:
            def __init__(
                self,
                graphFilename: str,
                output: str = "",
                input: str = "",
                inputs: list[Any] | None = None,
                outputs: list[Any] | None = None,
                **kwargs: object,
            ):
                self.output = output

            def __call__(self, audio: np.ndarray) -> Any:
                from playchitect.core.embedding_extractor import (
                    _EMB_OUTPUT_LAYER as EMB_OUT,
                )
                from playchitect.core.embedding_extractor import (
                    _MIREX_FEATS_LAYER as MIREX_OUT,
                )
                from playchitect.core.embedding_extractor import (
                    _MOOD_OUTPUT_LAYER as MOOD_OUT,
                )

                if self.output == MOOD_OUT:
                    return np.ones((2, 5), dtype=np.float32)
                if self.output == EMB_OUT:
                    return np.ones((2, 128), dtype=np.float32)
                if self.output == MIREX_OUT:
                    return np.ones((2, 200), dtype=np.float32)
                return np.ones((2, 4), dtype=np.float32)

        monkeypatch.setattr(emb_mod, "_EssentiaModel", MockModel)
        monkeypatch.setattr(emb_mod, "TensorflowPredict2D", MockModel)

        model_file = tmp_path / "fake.pb"
        model_file.write_bytes(b"x")
        mood_file = tmp_path / "fake_mood.pb"
        mood_file.write_bytes(b"y")

        meta = tmp_path / "fake.json"
        meta.write_text('{"classes": ["a", "b", "c", "d"]}')
        mood_meta = tmp_path / "fake_mood.json"
        mood_meta.write_text('{"classes": ["m1", "m2", "m3", "m4", "m5"]}')

        extractor = EmbeddingExtractor(
            model_path=model_file,
            mood_model_path=mood_file,
            cache_enabled=False,
        )

        good = tmp_path / "good.wav"
        good.write_bytes(b"\x00" * 200)
        bad = tmp_path / "bad.mp3"  # doesn't exist

        import librosa as _librosa  # noqa: PLC0415

        monkeypatch.setattr(
            _librosa, "load", lambda *a, **kw: (np.zeros(16000, dtype=np.float32), 16000)
        )

        results = extractor.analyze_batch([good, bad])
        assert good in results
        assert bad not in results


# ── TestEnsureModel ───────────────────────────────────────────────────────────


class TestEnsureModel:
    """Test _ensure_model lazy initialization."""

    def test_models_initialized_only_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Calling _ensure_model twice creates each model instance exactly once."""
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)

        init_calls: list[str] = []

        class MockModel:
            def __init__(
                self,
                graphFilename: str,
                output: str = "",
                input: str = "",
                inputs: list | None = None,
                outputs: list | None = None,
                **kwargs: object,
            ):
                init_calls.append(output)

        monkeypatch.setattr(emb_mod, "_EssentiaModel", MockModel)
        monkeypatch.setattr(emb_mod, "TensorflowPredict2D", MockModel)

        model_file = tmp_path / "fake.pb"
        model_file.write_bytes(b"x")
        mood_file = tmp_path / "fake_mood.pb"
        mood_file.write_bytes(b"y")

        extractor = EmbeddingExtractor(
            model_path=model_file,
            mood_model_path=mood_file,
            cache_enabled=False,
        )
        extractor._ensure_model()
        extractor._ensure_model()  # Second call — should not re-create

        # Each model output layer created exactly once
        assert init_calls.count(_EMB_OUTPUT_LAYER) == 1
        assert init_calls.count(_TAG_OUTPUT_LAYER) == 1
        assert init_calls.count(_MIREX_FEATS_LAYER) == 1
        assert init_calls.count(_MOOD_OUTPUT_LAYER) == 1

    def test_ensure_model_triggers_download_when_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_ensure_model calls _download_model when .pb does not exist."""
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)

        downloaded: list[Path] = []

        class MockModel:
            def __init__(
                self,
                graphFilename: str,
                output: str = "",
                inputs: list | None = None,
                outputs: list | None = None,
                **kwargs: object,
            ):
                pass

        monkeypatch.setattr(emb_mod, "_EssentiaModel", MockModel)
        monkeypatch.setattr(emb_mod, "TensorflowPredict2D", MockModel)

        model_file = tmp_path / "missing.pb"
        mood_file = tmp_path / "missing_mood.pb"

        def fake_download(target: Path, *args: Any) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fake_model")  # create it
            downloaded.append(target)

        extractor = EmbeddingExtractor(
            model_path=model_file,
            mood_model_path=mood_file,
            cache_enabled=False,
        )
        extractor._download_model = fake_download  # type: ignore[method-assign]
        extractor._ensure_model()

        assert len(downloaded) == 2
        assert model_file in downloaded
        assert mood_file in downloaded

    def test_ensure_model_propagates_init_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the model constructor raises, the exception propagates from _ensure_model."""
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)

        class BrokenModel:
            def __init__(
                self,
                graphFilename: str,
                output: str = "",
                inputs: list | None = None,
                outputs: list | None = None,
                **kwargs: object,
            ):
                raise RuntimeError("model load failed")

        monkeypatch.setattr(emb_mod, "_EssentiaModel", BrokenModel)
        monkeypatch.setattr(emb_mod, "TensorflowPredict2D", BrokenModel)

        model_file = tmp_path / "fake.pb"
        model_file.write_bytes(b"x")
        mood_file = tmp_path / "fake_mood.pb"
        mood_file.write_bytes(b"y")

        extractor = EmbeddingExtractor(
            model_path=model_file,
            mood_model_path=mood_file,
            cache_enabled=False,
        )
        with pytest.raises(RuntimeError, match="model load failed"):
            extractor._ensure_model()


# ── TestDownloadFailure ───────────────────────────────────────────────────────


class TestDownloadFailure:
    """Test that download failures propagate correctly."""

    def test_download_network_error_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A network error from urlretrieve propagates out of _download_model."""
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)

        def failing_urlretrieve(url: str, dest: object) -> None:
            raise OSError("network error")

        monkeypatch.setattr(
            "playchitect.core.embedding_extractor.urllib.request.urlretrieve",
            failing_urlretrieve,
        )

        target = tmp_path / "msd-musicnn-1.pb"
        extractor = EmbeddingExtractor(model_path=target, cache_enabled=False)
        with pytest.raises(OSError, match="network error"):
            extractor._download_model(target, _MSD_MUSICNN_URL, "https://fake.json")


# ── TestCorruptCache ──────────────────────────────────────────────────────────


class TestCorruptCache:
    """Test that corrupted cache files are handled gracefully."""

    @pytest.fixture()
    def extractor(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EmbeddingExtractor:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)
        return EmbeddingExtractor(
            cache_dir=tmp_path / "cache",
            cache_enabled=True,
            model_path=tmp_path / "fake.pb",
        )

    def test_corrupt_npy_returns_none(self, extractor: EmbeddingExtractor) -> None:
        """A broken .npy file causes _load_from_cache to return None."""
        file_hash = "deadbeefcafe1234"
        extractor.cache_dir.mkdir(parents=True, exist_ok=True)
        # Write garbage bytes that np.load cannot parse
        (extractor.cache_dir / f"{file_hash}.npy").write_bytes(b"NOT VALID NPY DATA")
        (extractor.cache_dir / f"{file_hash}_tags.json").write_text('[["techno", 0.9]]')

        result = extractor._load_from_cache(file_hash)
        assert result is None

    def test_corrupt_tags_json_returns_none(self, extractor: EmbeddingExtractor) -> None:
        """A broken _tags.json file causes _load_from_cache to return None."""
        file_hash = "cafe1234deadbeef"
        extractor.cache_dir.mkdir(parents=True, exist_ok=True)
        # Write a valid .npy
        valid_emb = np.zeros(128, dtype=np.float32)
        np.save(str(extractor._get_cache_path(file_hash)), valid_emb)
        # Write corrupt tags JSON
        (extractor.cache_dir / f"{file_hash}_tags.json").write_text("NOT JSON {{{{")

        result = extractor._load_from_cache(file_hash)
        assert result is None
