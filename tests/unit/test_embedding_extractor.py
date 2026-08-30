"""
Unit tests for embedding_extractor module.
"""

import hashlib
import json
import logging
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

        mood_model_file = tmp_path / "fake_mood.pb"
        mood_model_file.write_bytes(b"x")

        extractor = EmbeddingExtractor(
            model_path=model_file,
            mood_model_path=mood_model_file,
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
        # TASK-31 rail 3: _download_model now verifies the downloaded .pb's
        # sha256, so the fake must actually write bytes to `dest` (as the
        # real urlretrieve would) for that check to have something to hash.
        fake_content = b"fake msd-musicnn model bytes"

        def fake_urlretrieve(url: str, dest: object) -> None:
            calls.append((url, Path(str(dest))))
            Path(str(dest)).write_bytes(fake_content)

        monkeypatch.setattr(
            "playchitect.core.embedding_extractor.urllib.request.urlretrieve",
            fake_urlretrieve,
        )

        extractor = EmbeddingExtractor(model_path=target, cache_enabled=False)
        extractor._download_model(
            target,
            _MSD_MUSICNN_URL,
            "https://fake.json",
            expected_sha256=hashlib.sha256(fake_content).hexdigest(),
        )

        assert len(calls) == 2
        assert calls[0] == (_MSD_MUSICNN_URL, target)
        assert calls[1] == ("https://fake.json", meta)

    def test_download_target_path_matches_model_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)

        custom_path = tmp_path / "my_models" / "msd-musicnn-1.pb"
        retrieved: list[Path] = []
        fake_content = b"fake msd-musicnn model bytes"

        def fake_urlretrieve(url: str, dest: object) -> None:
            retrieved.append(Path(str(dest)))
            Path(str(dest)).write_bytes(fake_content)

        monkeypatch.setattr(
            "playchitect.core.embedding_extractor.urllib.request.urlretrieve",
            fake_urlretrieve,
        )

        extractor = EmbeddingExtractor(model_path=custom_path, cache_enabled=False)
        extractor._download_model(
            custom_path,
            _MSD_MUSICNN_URL,
            "https://fake.json",
            expected_sha256=hashlib.sha256(fake_content).hexdigest(),
        )

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

        # TASK-31 rail 3: _ensure_model() now passes a required keyword-only
        # expected_sha256 through to _download_model, so the fake must
        # accept (and ignore) it too.
        def fake_download(target: Path, *args: Any, **kwargs: Any) -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fake_model")  # create it
            downloaded.append(target)

        extractor = EmbeddingExtractor(
            model_path=model_file,
            mood_model_path=mood_file,
            cache_enabled=False,
        )
        extractor._download_model = fake_download  # ty: ignore[invalid-assignment]
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
            extractor._download_model(
                target,
                _MSD_MUSICNN_URL,
                "https://fake.json",
                expected_sha256=emb_mod._MSD_MUSICNN_SHA256,  # ty: ignore[unresolved-attribute]
            )


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


# ── TestDiscogsEffnetOption ───────────────────────────────────────────────────
#
# TASK-19 extends EmbeddingExtractor with a discogs-effnet model option for the
# embedding-cache ETL pipeline (playchitect/core/embedding_cache.py and
# scripts/embed_library.py), auto-downloading into ~/.cache/playchitect/models/
# following the exact pattern already used for the MusiCNN model below
# (module-level URL constants + a lazily-initialised, auto-downloading model
# reached through the existing generic `_download_model` helper).
#
# This part of the contract is NOT covered by prd.json's acceptance criteria
# (those only specify tests/unit/test_embedding_cache.py) and the task brief
# gives no fixed method/attribute names for it, so these tests encode one
# reasonable proposal rather than a locked-in requirement:
#   * module constant `_DISCOGS_EFFNET_URL` (mirrors `_MSD_MUSICNN_URL`)
#   * constructor kwarg `discogs_effnet_model_path: Path | None = None`,
#     defaulting to `_DEFAULT_MODEL_DIR / "discogs-effnet-bs64-1.pb"`
#   * private `_ensure_discogs_effnet_model()` that downloads via the
#     existing `_download_model(target, pb_url, meta_url)` helper, exactly
#     like `_ensure_model()` does for the MusiCNN pair of models.
# See the Test Writer report for this task for the explicit flag that this
# sub-contract is negotiable and may need renaming to match the eventual
# implementation.


class TestDiscogsEffnetOption:
    """Contract for the discogs-effnet auto-download model option."""

    def test_discogs_effnet_url_constant_defined(self) -> None:
        url = emb_mod._DISCOGS_EFFNET_URL  # ty: ignore[unresolved-attribute]
        assert isinstance(url, str)
        assert url.startswith("https://")
        assert "discogs-effnet" in url

    def test_default_discogs_effnet_path_under_default_model_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)
        extractor = EmbeddingExtractor(
            model_path=tmp_path / "fake.pb",
            mood_model_path=tmp_path / "fake_mood.pb",
            cache_enabled=False,
        )

        discogs_path = extractor.discogs_effnet_model_path  # ty: ignore[unresolved-attribute]
        assert discogs_path.parent == emb_mod._DEFAULT_MODEL_DIR
        assert "discogs-effnet" in discogs_path.name

    def test_custom_discogs_effnet_model_path_respected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)
        custom = tmp_path / "custom_models" / "discogs-effnet-bs64-1.pb"

        extractor = EmbeddingExtractor(
            model_path=tmp_path / "fake.pb",
            mood_model_path=tmp_path / "fake_mood.pb",
            discogs_effnet_model_path=custom,  # ty: ignore[unknown-argument]
            cache_enabled=False,
        )

        assert extractor.discogs_effnet_model_path == custom  # ty: ignore[unresolved-attribute]

    def test_ensure_discogs_effnet_model_triggers_download_when_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Mirrors TestEnsureModel.test_ensure_model_triggers_download_when_missing
        for the discogs-effnet model: when the .pb is absent, the extractor
        must call _download_model with the discogs-effnet target path and URL.
        """
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)
        discogs_target = tmp_path / "models" / "discogs-effnet-bs64-1.pb"

        extractor = EmbeddingExtractor(
            model_path=tmp_path / "fake.pb",
            mood_model_path=tmp_path / "fake_mood.pb",
            discogs_effnet_model_path=discogs_target,  # ty: ignore[unknown-argument]
            cache_enabled=False,
        )

        downloaded: list[tuple[Path, str]] = []

        # TASK-31 rail 3: _ensure_discogs_effnet_model() now passes a
        # required keyword-only expected_sha256 through to _download_model,
        # so the fake must accept (and ignore) it too.
        def fake_download(target: Path, pb_url: str, meta_url: str, **kwargs: Any) -> None:
            downloaded.append((target, pb_url))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fake discogs-effnet model")

        extractor._download_model = fake_download  # ty: ignore[invalid-assignment]

        extractor._ensure_discogs_effnet_model()  # ty: ignore[unresolved-attribute]

        assert len(downloaded) == 1
        assert downloaded[0][0] == discogs_target
        assert downloaded[0][1] == emb_mod._DISCOGS_EFFNET_URL  # ty: ignore[unresolved-attribute]

    def test_ensure_discogs_effnet_model_skips_download_when_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)
        discogs_target = tmp_path / "models" / "discogs-effnet-bs64-1.pb"
        discogs_target.parent.mkdir(parents=True, exist_ok=True)
        discogs_target.write_bytes(b"already downloaded")

        extractor = EmbeddingExtractor(
            model_path=tmp_path / "fake.pb",
            mood_model_path=tmp_path / "fake_mood.pb",
            discogs_effnet_model_path=discogs_target,  # ty: ignore[unknown-argument]
            cache_enabled=False,
        )

        downloaded: list[Path] = []
        extractor._download_model = lambda target, *a: downloaded.append(  # ty: ignore[invalid-assignment]
            target
        )

        extractor._ensure_discogs_effnet_model()  # ty: ignore[unresolved-attribute]

        assert downloaded == []


# ── TestMusiCNNZeroFrameGuard ─────────────────────────────────────────────────
#
# TASK-32. MusiCNN patches audio into 187 mel frames (frameSize=512,
# hopSize=256 at 16 kHz, essentia's centred framing with startFromZero=False,
# giving frames = floor((N-1)/256) + 2). Solving frames >= 187 gives
# N >= 185 * 256 + 1 = 47361 samples = 2.9600625 s. Measured by binary search
# against the installed msd-musicnn-1.pb on 2026-08-29: 47360 samples yields
# 0 frames, 47361 samples yields 187 frames.
#
# Below that threshold TensorflowPredictMusiCNN returns an EMPTY LIST. The
# current analyze() then does np.mean([], axis=0), which does not raise --
# it emits a RuntimeWarning and collapses to a 0-d np.float64(nan). The NaN
# scalar is then passed to _build_top_tags(), which calls len() on it and
# raises an opaque "object of type 'numpy.float64' has no len()" TypeError.
# analyze() must instead detect the zero-frame case and raise
# AudioTooShortForEmbeddingError, exactly as analyze_discogs_effnet() already
# does for its own (different, shorter) 2.048 s threshold.

# The measured MusiCNN minimum, in samples at 16 kHz and in seconds.
_MEASURED_MUSICNN_MIN_SAMPLES = 47361
_MEASURED_MUSICNN_MIN_SECONDS = 2.9600625

# Name the production constant that must carry the measurement above.
_MUSICNN_MIN_CONST_NAME = "_MUSICNN_MIN_AUDIO_SECONDS"


def _musicnn_min_seconds() -> float:
    """
    Return the production MusiCNN minimum-duration constant.

    Fails the calling test (rather than erroring at import time) when the
    constant does not exist yet, so the missing-constant case reports as a
    readable assertion rather than a collection error.
    """
    value = getattr(emb_mod, _MUSICNN_MIN_CONST_NAME, None)
    assert value is not None, (
        f"embedding_extractor must define a module-level "
        f"{_MUSICNN_MIN_CONST_NAME} constant recording the measured MusiCNN "
        f"minimum audio duration ({_MEASURED_MUSICNN_MIN_SECONDS}s = "
        f"{_MEASURED_MUSICNN_MIN_SAMPLES} samples at 16 kHz)"
    )
    return float(value)


def _make_length_aware_musicnn_model_class(empty_value: Any) -> type:
    """
    Build a mock Essentia model class that mimics the real MusiCNN framing.

    Instances return ``empty_value`` (the caller chooses an empty list or an
    empty ndarray, both of which the real bindings can produce) whenever the
    supplied audio is shorter than the measured 47361-sample minimum, and
    plausible frame-level arrays otherwise. Every output layer goes empty
    together, because the zero-frame condition is a property of the framing,
    not of the graph output node.
    """
    n_frames = 187

    class LengthAwareMockModel:
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

        def __call__(self, audio: Any) -> Any:
            rng = np.random.default_rng(0)

            # The MIREX head is fed the 200-D MusiCNN feature frames, not raw
            # audio, so the raw-audio framing rule below does not apply to it.
            if self.output == _MOOD_OUTPUT_LAYER:
                return np.abs(rng.standard_normal((len(audio), 5))).astype(np.float32)

            if len(audio) < _MEASURED_MUSICNN_MIN_SAMPLES:
                return empty_value

            if self.output == _EMB_OUTPUT_LAYER:
                return rng.standard_normal((n_frames, 128)).astype(np.float32)
            if self.output == _MIREX_FEATS_LAYER:
                return np.abs(rng.standard_normal((n_frames, 200))).astype(np.float32)
            # _TAG_OUTPUT_LAYER — sigmoid activations in [0, 1]
            return rng.random((n_frames, 50)).astype(np.float32)

    return LengthAwareMockModel


class TestMusiCNNZeroFrameGuard:
    """analyze() must refuse to mean-pool zero MusiCNN frames."""

    def _build_extractor(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        empty_value: Any,
    ) -> EmbeddingExtractor:
        monkeypatch.setattr(emb_mod, "_ESSENTIA_AVAILABLE", True)
        mock_cls = _make_length_aware_musicnn_model_class(empty_value)
        monkeypatch.setattr(emb_mod, "_EssentiaModel", mock_cls)
        monkeypatch.setattr(emb_mod, "TensorflowPredict2D", mock_cls)

        model_file = tmp_path / "fake.pb"
        model_file.write_bytes(b"fake_model")
        mood_model_file = tmp_path / "fake_mood.pb"
        mood_model_file.write_bytes(b"fake_mood_model")
        (tmp_path / "fake.json").write_text(
            json.dumps({"classes": [f"tag_{i}" for i in range(50)]})
        )
        (tmp_path / "fake_mood.json").write_text(
            json.dumps({"classes": [f"mood_{i}" for i in range(5)]})
        )

        return EmbeddingExtractor(
            model_path=model_file,
            mood_model_path=mood_model_file,
            cache_dir=tmp_path / "cache",
            cache_enabled=True,
        )

    @staticmethod
    def _patch_audio_length(monkeypatch: pytest.MonkeyPatch, n_samples: int) -> None:
        import librosa as _librosa  # noqa: PLC0415

        monkeypatch.setattr(
            _librosa,
            "load",
            lambda *a, **kw: (np.zeros(n_samples, dtype=np.float32), 16000),
        )

    @staticmethod
    def _make_audio_file(tmp_path: Path, name: str = "way_too_short.flac") -> Path:
        audio_file = tmp_path / name
        audio_file.write_bytes(b"\x00" * 100)
        return audio_file

    # ── the constant itself ───────────────────────────────────────────────

    def test_musicnn_minimum_constant_matches_the_measurement(self) -> None:
        """
        The MusiCNN minimum must be recorded as a named constant carrying the
        measured 2.9600625 s value (47361 samples at 16 kHz). A tolerance of
        1e-3 s is allowed so the constant may be written either exactly
        (2.9600625) or rounded up (2.961); it is tight enough to reject the
        nominal-but-wrong 187 * 256 / 16000 = 2.992 s figure, which overstates
        the requirement because essentia's centred framing shaves ~32 ms.
        """
        const = _musicnn_min_seconds()

        assert const == pytest.approx(_MEASURED_MUSICNN_MIN_SECONDS, abs=1e-3)
        # Must not sit BELOW the measured floor, or it would advertise a
        # duration that still yields zero frames. The -1 sample of slack
        # absorbs float rounding in seconds -> samples.
        assert const * emb_mod._EMBEDDING_SAMPLE_RATE >= _MEASURED_MUSICNN_MIN_SAMPLES - 1

    def test_musicnn_minimum_is_not_the_discogs_effnet_minimum(self) -> None:
        """
        The two models frame audio differently -- discogs-effnet needs
        2.0320625 s (pinned at 2.048), MusiCNN needs 2.9600625 s. Reusing the
        discogs constant for MusiCNN would silently under-report the
        requirement by nearly a second.
        """
        const = _musicnn_min_seconds()

        assert const != emb_mod._DISCOGS_EFFNET_MIN_AUDIO_SECONDS
        assert const > emb_mod._DISCOGS_EFFNET_MIN_AUDIO_SECONDS

    # ── the guard ─────────────────────────────────────────────────────────

    def test_zero_frames_as_empty_list_raises_audio_too_short(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real bindings return a plain empty list, not an ndarray."""
        extractor = self._build_extractor(tmp_path, monkeypatch, empty_value=[])
        audio_file = self._make_audio_file(tmp_path)
        self._patch_audio_length(monkeypatch, 8000)  # 0.5 s at 16 kHz

        with pytest.raises(emb_mod.AudioTooShortForEmbeddingError):
            extractor.analyze(audio_file)

    def test_zero_frames_as_empty_ndarray_raises_audio_too_short(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard must also cope with a (0, 128)-shaped ndarray."""
        empty = np.empty((0, 128), dtype=np.float32)
        extractor = self._build_extractor(tmp_path, monkeypatch, empty_value=empty)
        audio_file = self._make_audio_file(tmp_path)
        self._patch_audio_length(monkeypatch, 8000)

        with pytest.raises(emb_mod.AudioTooShortForEmbeddingError):
            extractor.analyze(audio_file)

    def test_zero_frames_does_not_raise_opaque_type_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Regression guard for the observed failure: np.mean of zero frames
        collapsing to a NaN scalar and surfacing as
        "object of type 'numpy.float64' has no len()".
        """
        extractor = self._build_extractor(tmp_path, monkeypatch, empty_value=[])
        audio_file = self._make_audio_file(tmp_path)
        self._patch_audio_length(monkeypatch, 8000)

        with pytest.raises(Exception) as exc_info:  # noqa: B017 - type is the assertion
            extractor.analyze(audio_file)

        assert not isinstance(exc_info.value, TypeError)
        assert "has no len()" not in str(exc_info.value)

    def test_error_message_names_duration_minimum_and_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The message must be actionable on its own: what, how long, how short."""
        const = _musicnn_min_seconds()
        extractor = self._build_extractor(tmp_path, monkeypatch, empty_value=[])
        audio_file = self._make_audio_file(tmp_path, name="tiny_clip.flac")
        self._patch_audio_length(monkeypatch, 8000)  # exactly 0.5 s

        with pytest.raises(emb_mod.AudioTooShortForEmbeddingError) as exc_info:
            extractor.analyze(audio_file)

        message = str(exc_info.value)
        assert "tiny_clip.flac" in message
        # Actual duration of the clip (0.5 s), at any reasonable precision.
        assert "0.5" in message
        # Required minimum, at 2 dp, 3 dp or full precision.
        assert any(
            candidate in message for candidate in (f"{const:.2f}", f"{const:.3f}", repr(const))
        ), f"message must cite the {const}s minimum: {message!r}"

    def test_error_message_reads_the_module_constant_not_a_literal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Repointing the module constant must change the reported minimum. A
        hard-coded literal in the message would keep reporting 2.96 and fail
        here. 12.5 is chosen because it renders identically at 1, 2 and 3
        decimal places ("12.5", "12.50", "12.500" all contain "12.5"), so
        this does not constrain the implementer's format string.
        """
        _musicnn_min_seconds()  # assert the constant exists before repointing it
        monkeypatch.setattr(emb_mod, _MUSICNN_MIN_CONST_NAME, 12.5, raising=False)

        extractor = self._build_extractor(tmp_path, monkeypatch, empty_value=[])
        audio_file = self._make_audio_file(tmp_path)
        self._patch_audio_length(monkeypatch, 8000)

        with pytest.raises(emb_mod.AudioTooShortForEmbeddingError) as exc_info:
            extractor.analyze(audio_file)

        assert "12.5" in str(exc_info.value)
        assert "2.96" not in str(exc_info.value)

    def test_guard_fires_before_build_top_tags_is_reached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        _build_top_tags() is where the NaN scalar currently detonates. The
        guard must short-circuit before it is ever called.
        """
        extractor = self._build_extractor(tmp_path, monkeypatch, empty_value=[])
        audio_file = self._make_audio_file(tmp_path)
        self._patch_audio_length(monkeypatch, 8000)

        calls: list[Any] = []
        original = extractor._build_top_tags

        def spy(activations: Any) -> Any:
            calls.append(activations)
            return original(activations)

        monkeypatch.setattr(extractor, "_build_top_tags", spy)

        with pytest.raises(emb_mod.AudioTooShortForEmbeddingError):
            extractor.analyze(audio_file)

        assert calls == [], (
            "_build_top_tags must not be reached for zero-frame audio; "
            f"it was called with {calls!r}"
        )

    def test_too_short_audio_is_not_written_to_the_embedding_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A NaN 'embedding' must never be persisted for a later cache hit."""
        extractor = self._build_extractor(tmp_path, monkeypatch, empty_value=[])
        audio_file = self._make_audio_file(tmp_path)
        self._patch_audio_length(monkeypatch, 8000)

        with pytest.raises(emb_mod.AudioTooShortForEmbeddingError):
            extractor.analyze(audio_file)

        assert list(extractor.cache_dir.glob("*.npy")) == []
        assert list(extractor.cache_dir.glob("*_metadata.json")) == []

    # ── the measured boundary ─────────────────────────────────────────────

    def test_one_sample_below_the_measured_minimum_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """47360 samples at 16 kHz yields 0 frames (measured)."""
        extractor = self._build_extractor(tmp_path, monkeypatch, empty_value=[])
        audio_file = self._make_audio_file(tmp_path)
        self._patch_audio_length(monkeypatch, _MEASURED_MUSICNN_MIN_SAMPLES - 1)

        with pytest.raises(emb_mod.AudioTooShortForEmbeddingError):
            extractor.analyze(audio_file)

    def test_audio_at_the_measured_minimum_still_analyses_normally(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        47361 samples yields 187 frames (measured), so the guard must not
        over-trigger: a clip exactly at the boundary produces a finite,
        mean-pooled embedding plus populated tags and moods.
        """
        extractor = self._build_extractor(tmp_path, monkeypatch, empty_value=[])
        audio_file = self._make_audio_file(tmp_path, name="just_long_enough.flac")
        self._patch_audio_length(monkeypatch, _MEASURED_MUSICNN_MIN_SAMPLES)

        feat = extractor.analyze(audio_file)

        assert feat.embedding.ndim == 1
        assert feat.embedding.shape == (128,)
        assert np.all(np.isfinite(feat.embedding))
        assert len(feat.top_tags) == 50
        assert len(feat.moods) == 5
        assert feat.primary_mood is not None

    def test_analyze_batch_keeps_long_tracks_and_drops_short_ones(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """
        The batch path must degrade per-track, not abort: one too-short clip
        must not cost us the embedding of a long one, and the warning it logs
        must name the real cause rather than the opaque NaN-len() TypeError.
        """
        extractor = self._build_extractor(tmp_path, monkeypatch, empty_value=[])
        short_file = self._make_audio_file(tmp_path, name="short.flac")
        long_file = self._make_audio_file(tmp_path, name="long.flac")
        # Distinct bytes so the two files do not share a cache hash.
        long_file.write_bytes(b"\x01" * 100)

        lengths = {
            short_file: 8000,
            long_file: _MEASURED_MUSICNN_MIN_SAMPLES + 16000,
        }

        import librosa as _librosa  # noqa: PLC0415

        def fake_load(path: Any, *a: Any, **kw: Any) -> Any:
            return np.zeros(lengths[Path(path)], dtype=np.float32), 16000

        monkeypatch.setattr(_librosa, "load", fake_load)

        with caplog.at_level(logging.WARNING, logger=emb_mod.__name__):
            results = extractor.analyze_batch([short_file, long_file])

        assert list(results) == [long_file]
        assert np.all(np.isfinite(results[long_file].embedding))

        short_warnings = [r.getMessage() for r in caplog.records if "short.flac" in r.getMessage()]
        assert short_warnings, f"no warning logged for the skipped clip: {caplog.text!r}"
        assert "has no len()" not in " ".join(short_warnings)
        assert "too short" in " ".join(short_warnings).lower()
