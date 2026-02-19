"""Unit tests for genre_resolver module."""

import tempfile
from pathlib import Path

import numpy as np

from playchitect.core.embedding_extractor import EmbeddingFeatures
from playchitect.core.genre_resolver import (
    InferGenreProtocol,
    load_genre_map,
    resolve_genres,
)
from playchitect.core.metadata_extractor import TrackMetadata


class TestLoadGenreMap:
    """Tests for load_genre_map."""

    def test_valid_yaml_returns_assignments(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(b'manual_assignments:\n  "track1.mp3": "techno"\n  "track2.mp3": "house"\n')
            path = Path(f.name)
        try:
            result = load_genre_map(path)
            assert result == {"track1.mp3": "techno", "track2.mp3": "house"}
        finally:
            path.unlink(missing_ok=True)

    def test_nonexistent_returns_empty(self) -> None:
        result = load_genre_map(Path("/nonexistent/genre_map.yaml"))
        assert result == {}

    def test_invalid_genre_skipped(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(
                b'manual_assignments:\n  "track1.mp3": "techno"\n  "track2.mp3": "invalid_genre"\n'
            )
            path = Path(f.name)
        try:
            result = load_genre_map(path)
            assert result == {"track1.mp3": "techno"}
        finally:
            path.unlink(missing_ok=True)

    def test_empty_file_returns_empty(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(b"{}")
            path = Path(f.name)
        try:
            result = load_genre_map(path)
            assert result == {}
        finally:
            path.unlink(missing_ok=True)

    def test_missing_manual_assignments_key_returns_empty(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(b"other_key: value\n")
            path = Path(f.name)
        try:
            result = load_genre_map(path)
            assert result == {}
        finally:
            path.unlink(missing_ok=True)


class TestResolveGenres:
    """Tests for resolve_genres."""

    def test_manual_override_takes_priority(self) -> None:
        p = Path("/music/track.mp3")
        meta = TrackMetadata(filepath=p, bpm=120, genre="house")
        metadata_dict = {p: meta}
        genre_map = {str(p): "techno"}

        result = resolve_genres(
            metadata_dict,
            None,
            genre_map,
            music_root=Path("/music"),
            infer_genre_fn=None,
        )
        assert result[p] == "techno"

    def test_metadata_genre_used_when_no_override(self) -> None:
        p = Path("/music/track.mp3")
        meta = TrackMetadata(filepath=p, bpm=120, genre="house")
        metadata_dict = {p: meta}

        result = resolve_genres(
            metadata_dict, None, {}, music_root=Path("/music"), infer_genre_fn=None
        )
        assert result[p] == "house"

    def test_filename_match_in_genre_map(self) -> None:
        p = Path("/music/subdir/track.mp3")
        meta = TrackMetadata(filepath=p, bpm=120)
        metadata_dict = {p: meta}
        genre_map = {"track.mp3": "ambient"}

        result = resolve_genres(
            metadata_dict,
            None,
            genre_map,
            music_root=Path("/music"),
            infer_genre_fn=None,
        )
        assert result[p] == "ambient"

    def test_unknown_when_no_sources(self) -> None:
        p = Path("/music/track.mp3")
        meta = TrackMetadata(filepath=p, bpm=120)
        metadata_dict = {p: meta}

        result = resolve_genres(metadata_dict, None, {}, music_root=None, infer_genre_fn=None)
        assert result[p] == "unknown"

    def test_infer_genre_fn_used_when_embeddings(self) -> None:
        p = Path("/music/track.mp3")
        meta = TrackMetadata(filepath=p, bpm=120)
        metadata_dict = {p: meta}

        class MockInferGenreDnb(InferGenreProtocol):
            def __call__(self, features: EmbeddingFeatures) -> str | None:
                return "dnb"

        fake_infer = MockInferGenreDnb()

        embedding_dict = {
            p: EmbeddingFeatures(
                filepath=Path(""),
                file_hash="",
                embedding=np.array([], dtype=np.float32),
                top_tags=[],
            )
        }

        result = resolve_genres(
            metadata_dict,
            embedding_dict,
            {},
            music_root=None,
            infer_genre_fn=fake_infer,
        )
        assert result[p] == "dnb"

    def test_relative_path_match_in_genre_map(self) -> None:
        """Genre map with relative path (e.g. subdir/track.mp3) resolves when music_root given."""
        p = Path("/music/electro/track.mp3")
        meta = TrackMetadata(filepath=p, bpm=120)
        metadata_dict = {p: meta}
        genre_map = {"electro/track.mp3": "techno"}  # relative to /music
        result = resolve_genres(
            metadata_dict,
            None,
            genre_map,
            music_root=Path("/music"),
            infer_genre_fn=None,
        )
        assert result[p] == "techno"

    def test_infer_genre_fn_returns_none_fallback_to_unknown(self) -> None:
        """When infer_genre_fn returns None, track gets 'unknown'."""
        p = Path("/music/track.mp3")
        meta = TrackMetadata(filepath=p, bpm=120)
        metadata_dict = {p: meta}

        class MockInferGenreNone(InferGenreProtocol):
            def __call__(self, features: EmbeddingFeatures) -> str | None:
                return None

        fake_infer = MockInferGenreNone()

        embedding_dict = {
            p: EmbeddingFeatures(
                filepath=Path(""),
                file_hash="",
                embedding=np.array([], dtype=np.float32),
                top_tags=[],
            )
        }
        result = resolve_genres(
            metadata_dict,
            embedding_dict,
            {},
            music_root=None,
            infer_genre_fn=fake_infer,
        )
        assert result[p] == "unknown"

    def test_infer_genre_fn_returns_empty_string_fallback_to_unknown(self) -> None:
        """When infer_genre_fn returns empty string, track gets 'unknown'."""
        p = Path("/music/track.mp3")
        meta = TrackMetadata(filepath=p, bpm=120)
        metadata_dict = {p: meta}

        class MockInferGenreEmpty(InferGenreProtocol):
            def __call__(self, features: EmbeddingFeatures) -> str | None:
                return ""

        fake_infer = MockInferGenreEmpty()

        embedding_dict = {
            p: EmbeddingFeatures(
                filepath=Path(""),
                file_hash="",
                embedding=np.array([], dtype=np.float32),
                top_tags=[],
            )
        }
        result = resolve_genres(
            metadata_dict,
            embedding_dict,
            {},
            music_root=None,
            infer_genre_fn=fake_infer,
        )
        assert result[p] == "unknown"

    def test_infer_genre_fn_returns_unsupported_genre_fallback_to_unknown(self) -> None:
        """When infer_genre_fn returns genre not in SUPPORTED_GENRES, track gets 'unknown'."""
        p = Path("/music/track.mp3")
        meta = TrackMetadata(filepath=p, bpm=120)
        metadata_dict = {p: meta}

        class MockInferGenreUnsupported(InferGenreProtocol):
            def __call__(self, features: EmbeddingFeatures) -> str | None:
                return "rock"  # not in techno, house, ambient, dnb

        fake_infer = MockInferGenreUnsupported()

        embedding_dict = {
            p: EmbeddingFeatures(
                filepath=Path(""),
                file_hash="",
                embedding=np.array([], dtype=np.float32),
                top_tags=[],
            )
        }
        result = resolve_genres(
            metadata_dict,
            embedding_dict,
            {},
            music_root=None,
            infer_genre_fn=fake_infer,
        )
        assert result[p] == "unknown"

    def test_infer_genre_fn_raises_fallback_to_unknown(self) -> None:
        """When infer_genre_fn raises, track gets 'unknown' (exception logged)."""
        p = Path("/music/track.mp3")
        meta = TrackMetadata(filepath=p, bpm=120)
        metadata_dict = {p: meta}

        class MockInferGenreRaises(InferGenreProtocol):
            def __call__(self, features: EmbeddingFeatures) -> str | None:
                raise ValueError("mock inference error")

        fake_infer = MockInferGenreRaises()

        embedding_dict = {
            p: EmbeddingFeatures(
                filepath=Path(""),
                file_hash="",
                embedding=np.array([], dtype=np.float32),
                top_tags=[],
            )
        }
        result = resolve_genres(
            metadata_dict,
            embedding_dict,
            {},
            music_root=None,
            infer_genre_fn=fake_infer,
        )
        assert result[p] == "unknown"
