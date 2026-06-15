"""Tests for playchitect.core.seed_playlist — core seed playlist engine.

These tests must fail at import (ModuleNotFoundError) until TASK-06 creates
playchitect/core/seed_playlist.py.
"""

from pathlib import Path

import numpy as np
import pytest
from playchitect.core.seed_playlist import (
    fill_to_duration,
    generate_playlist_from_seed,
    rank_by_similarity,
)

from playchitect.core.clustering import ClusterResult
from playchitect.core.export import M3UExporter
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata


def make_metadata(
    name: str, bpm: float | None = 128.0, duration: float = 360.0
) -> TrackMetadata:
    """Create TrackMetadata for testing."""
    return TrackMetadata(filepath=Path(name), bpm=bpm, duration=duration)


def make_intensity(
    name: str,
    rms: float = 0.5,
    brightness: float = 0.5,
    sub_bass: float = 0.3,
    kick: float = 0.6,
    harmonics: float = 0.4,
    perc: float = 0.5,
    onset: float = 0.5,
) -> IntensityFeatures:
    """Create IntensityFeatures for testing."""
    return IntensityFeatures(
        file_path=Path(name),
        file_hash="deadbeef",  # pragma: allowlist secret
        rms_energy=rms,
        brightness=brightness,
        sub_bass_energy=sub_bass,
        kick_energy=kick,
        bass_harmonics=harmonics,
        percussiveness=perc,
        onset_strength=onset,
        camelot_key="8B",
        key_index=0.0,
    )


class TestRankBySimilarity:
    """Tests for rank_by_similarity(seed_vec, candidates, *, genre, weight_overrides).

    Returns list[tuple[Path, float]] sorted by distance, nearest first.
    """

    def test_identical_seed_ranks_zero_distance(self) -> None:
        """A candidate identical to the seed should rank first with distance ~0."""
        seed_vec = np.array([128.0, 0.5, 0.5, 0.3, 0.6, 0.4, 0.5, 0.5])
        p1 = Path("track1.flac")
        p2 = Path("track2.flac")
        candidates: dict[Path, np.ndarray] = {
            p1: np.array([128.0, 0.5, 0.5, 0.3, 0.6, 0.4, 0.5, 0.5]),
            p2: np.array([130.0, 0.7, 0.6, 0.4, 0.5, 0.3, 0.6, 0.6]),
        }
        result = rank_by_similarity(seed_vec, candidates)
        assert len(result) > 0
        assert result[0][0] == p1
        # Distance to identical candidate should be near zero
        assert result[0][1] < 1e-6

    def test_ordering_is_nearest_first(self) -> None:
        """Candidates should be sorted by ascending distance."""
        seed_vec = np.array([128.0, 0.5, 0.5, 0.3, 0.6, 0.4, 0.5, 0.5])
        p_far = Path("far.flac")
        p_near = Path("near.flac")
        p_mid = Path("mid.flac")
        # near: identical (dist 0), mid: modest diff, far: large diff
        candidates: dict[Path, np.ndarray] = {
            p_far: np.array([200.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]),
            p_near: np.array([128.0, 0.5, 0.5, 0.3, 0.6, 0.4, 0.5, 0.5]),
            p_mid: np.array([130.0, 0.6, 0.6, 0.4, 0.5, 0.4, 0.55, 0.55]),
        }
        result = rank_by_similarity(seed_vec, candidates)
        assert len(result) == 3
        assert result[0][0] == p_near  # distance ~0
        assert result[1][0] == p_mid  # moderate distance
        assert result[2][0] == p_far  # largest distance

    def test_empty_candidates_returns_empty_list(self) -> None:
        """Empty dict should produce empty list."""
        seed_vec = np.array([128.0, 0.5, 0.5, 0.3, 0.6, 0.4, 0.5, 0.5])
        result = rank_by_similarity(seed_vec, {})
        assert result == []

    def test_single_candidate(self) -> None:
        """Single candidate returns a list of length 1."""
        seed_vec = np.array([128.0, 0.5, 0.5, 0.3, 0.6, 0.4, 0.5, 0.5])
        p = Path("only.flac")
        candidates = {p: np.array([130.0, 0.6, 0.6, 0.4, 0.5, 0.3, 0.5, 0.5])}
        result = rank_by_similarity(seed_vec, candidates)
        assert len(result) == 1
        assert result[0][0] == p

    def test_changing_genre_changes_order(self) -> None:
        """Genre should affect which candidates rank closer."""
        # Use a seed vector and three candidates at different positions
        # in feature space so that genre-specific weighting changes the order
        seed_vec = np.array([128.0, 0.5, 0.3, 0.8, 0.2, 0.8, 0.3, 0.5])
        p_a = Path("a.flac")
        p_b = Path("b.flac")
        p_c = Path("c.flac")
        # A: high BPM, high percussiveness (techno-like)
        # B: low BPM, low brightness (ambient-like)
        # C: balanced
        candidates = {
            p_a: np.array([150.0, 0.7, 0.3, 0.9, 0.2, 0.9, 0.3, 0.7]),
            p_b: np.array([100.0, 0.3, 0.1, 0.7, 0.2, 0.7, 0.3, 0.3]),
            p_c: np.array([125.0, 0.5, 0.2, 0.8, 0.2, 0.8, 0.3, 0.5]),
        }
        result_techno = rank_by_similarity(
            seed_vec, candidates, genre="techno"
        )
        result_ambient = rank_by_similarity(
            seed_vec, candidates, genre="ambient"
        )
        # The orderings should differ because genre changes the weight vector
        order_techno = [p for p, _ in result_techno]
        order_ambient = [p for p, _ in result_ambient]
        assert order_techno != order_ambient, (
            f"Genre should change ranking. techno={order_techno}, "
            f"ambient={order_ambient}"
        )


class TestFillToDuration:
    """Tests for fill_to_duration(ranked, metadata_dict, target_secs, tolerance, seed_path).

    Returns list[Path] greedily filled to target duration.
    """

    def test_seed_always_included(self) -> None:
        """The seed path should always appear in the result, even without ranking."""
        seed = Path("seed.flac")
        track = Path("other.flac")
        ranked = [(track, 1.0)]
        meta = {
            seed: make_metadata("seed.flac", duration=300.0),
            track: make_metadata("other.flac", duration=400.0),
        }
        result = fill_to_duration(ranked, meta, target_secs=100.0, tolerance=0.1, seed_path=seed)
        assert seed in result

    def test_total_duration_within_tolerance(self) -> None:
        """With tolerance=0.1 and target=3600s, total should be within ±10%."""
        # 20 tracks × 360s = 7200s total. Target 3600 → should fill about 10 tracks.
        seed = Path("seed.flac")
        tracks = [Path(f"t{i}.flac") for i in range(20)]
        ranked = [(p, float(i)) for i, p in enumerate(tracks)]
        meta = {p: make_metadata(str(p), duration=360.0) for p in tracks + [seed]}
        result = fill_to_duration(ranked, meta, target_secs=3600.0, tolerance=0.1, seed_path=seed)
        total = sum(meta[p].duration or 0.0 for p in result)
        lower = 3600.0 * (1 - 0.1)
        upper = 3600.0 * (1 + 0.1)
        assert lower <= total <= upper, (
            f"Total {total}s outside [{lower}, {upper}]"
        )

    def test_short_library_returns_all(self) -> None:
        """When target exceeds all tracks combined, return everything."""
        seed = Path("seed.flac")
        tracks = [Path(f"t{i}.flac") for i in range(3)]
        ranked = [(p, float(i)) for i, p in enumerate(tracks)]
        meta = {p: make_metadata(str(p), duration=100.0) for p in tracks + [seed]}
        result = fill_to_duration(ranked, meta, target_secs=99999.0, tolerance=0.1, seed_path=seed)
        # All 3 tracks + seed should be returned
        assert len(result) == 4
        assert all(p in result for p in tracks)

    def test_zero_duration_tracks_skipped(self) -> None:
        """Tracks with duration=0 should never appear in the result."""
        seed = Path("seed.flac")
        zero_track = Path("zero.flac")
        good_track = Path("good.flac")
        ranked = [(zero_track, 0.0), (good_track, 1.0)]
        meta = {
            seed: make_metadata("seed.flac", duration=300.0),
            zero_track: make_metadata("zero.flac", duration=0.0),
            good_track: make_metadata("good.flac", duration=300.0),
        }
        result = fill_to_duration(ranked, meta, target_secs=600.0, tolerance=0.1, seed_path=seed)
        assert zero_track not in result
        assert good_track in result

    def test_tolerance_boundary_respected(self) -> None:
        """With tolerance=0.0, only tracks that fit exactly at or under target are included."""
        seed = Path("seed.flac")
        # seed: 300s, t0: 600s (fits easily), t1: 200s (pushes over)
        t0 = Path("t0.flac")
        t1 = Path("t1.flac")
        ranked = [(t0, 0.0), (t1, 1.0)]
        meta = {
            seed: make_metadata("seed.flac", duration=300.0),
            t0: make_metadata("t0.flac", duration=600.0),
            t1: make_metadata("t1.flac", duration=200.0),
        }
        # target_secs=900: seed(300) + t0(600) = 900 exactly → fits
        # target_secs=899: seed(300) + t0(600) = 900 > 899 → t0 excluded
        result_loose = fill_to_duration(
            ranked, meta, target_secs=900.0, tolerance=0.0, seed_path=seed
        )
        result_tight = fill_to_duration(
            ranked, meta, target_secs=899.0, tolerance=0.0, seed_path=seed
        )
        assert t0 in result_loose, "t0 should fit at target 900"
        assert t0 not in result_tight, "t0 should be excluded at target 899 with tolerance=0"


class TestGeneratePlaylistFromSeed:
    """Tests for generate_playlist_from_seed() — the public API.

    Signature:
        generate_playlist_from_seed(
            seed_path: Path,
            candidate_features: dict[Path, IntensityFeatures],
            metadata_dict: dict[Path, TrackMetadata],
            target_duration_mins: float,
            *,
            genre: str | None = None,
            weight_overrides=None,
            tolerance: float = 0.1,
            sequence_mode: str = 'ramp',
        ) -> ClusterResult
    """

    def _make_synthetic_library(
        self, n_tracks: int = 10, seed_idx: int = 0, duration_secs: float = 360.0
    ) -> tuple[Path, dict[Path, IntensityFeatures], dict[Path, TrackMetadata]]:
        """Build a small synthetic library for testing."""
        paths = [Path(f"track_{i}.flac") for i in range(n_tracks)]
        seed_path = paths[seed_idx]

        features_dict: dict[Path, IntensityFeatures] = {}
        metadata_dict: dict[Path, TrackMetadata] = {}
        for i, p in enumerate(paths):
            features_dict[p] = make_intensity(
                str(p),
                rms=0.3 + 0.02 * i,
                brightness=0.4 + 0.02 * i,
                sub_bass=0.2 + 0.03 * i,
                kick=0.5 + 0.02 * i,
                harmonics=0.3 + 0.02 * i,
                perc=0.4 + 0.03 * i,
                onset=0.5 + 0.02 * i,
            )
            metadata_dict[p] = make_metadata(
                str(p), bpm=120.0 + float(i) * 5.0, duration=duration_secs
            )
        return seed_path, features_dict, metadata_dict

    def test_returns_cluster_result(self) -> None:
        """Result should be an instance of ClusterResult."""
        seed_path, feats, meta = self._make_synthetic_library(n_tracks=10)
        result = generate_playlist_from_seed(
            seed_path=seed_path,
            candidate_features=feats,
            metadata_dict=meta,
            target_duration_mins=30.0,
        )
        assert isinstance(result, ClusterResult)

    def test_seed_included_in_result(self) -> None:
        """The seed track must appear in the result tracks list."""
        seed_path, feats, meta = self._make_synthetic_library(n_tracks=10)
        result = generate_playlist_from_seed(
            seed_path=seed_path,
            candidate_features=feats,
            metadata_dict=meta,
            target_duration_mins=30.0,
        )
        assert seed_path in result.tracks

    def test_duration_within_tolerance(self) -> None:
        """Total duration should be within tolerance band around target."""
        seed_path, feats, meta = self._make_synthetic_library(
            n_tracks=20, duration_secs=300.0
        )
        target_mins = 25.0
        result = generate_playlist_from_seed(
            seed_path=seed_path,
            candidate_features=feats,
            metadata_dict=meta,
            target_duration_mins=target_mins,
            tolerance=0.1,
        )
        target_secs = target_mins * 60.0
        lower = target_secs * (1 - 0.1)
        upper = target_secs * (1 + 0.1)
        assert lower <= result.total_duration <= upper, (
            f"Duration {result.total_duration}s outside [{lower}, {upper}]"
        )

    def test_cluster_result_name_contains_seed_title(self) -> None:
        """Result should be named 'Like: <seed title>'."""
        seed_path = Path("my_seed_track.flac")
        feats: dict[Path, IntensityFeatures] = {
            seed_path: make_intensity("my_seed_track.flac"),
            Path("other.flac"): make_intensity("other.flac", rms=0.8),
        }
        meta: dict[Path, TrackMetadata] = {
            seed_path: make_metadata(
                "my_seed_track.flac", bpm=128.0, duration=360.0
            ),
            Path("other.flac"): make_metadata(
                "other.flac", bpm=130.0, duration=360.0
            ),
        }
        # Set title on the seed metadata
        meta[seed_path].title = "My Seed Track"

        result = generate_playlist_from_seed(
            seed_path=seed_path,
            candidate_features=feats,
            metadata_dict=meta,
            target_duration_mins=5.0,
        )
        # Name stored in genre field: 'Like: <seed_title>'
        assert result.genre is not None
        assert "Like:" in result.genre

    def test_ramp_and_build_give_different_order(self) -> None:
        """'ramp' and 'build' strategies should produce different track orderings."""
        seed_path, feats, meta = self._make_synthetic_library(
            n_tracks=20, duration_secs=180.0
        )
        result_ramp = generate_playlist_from_seed(
            seed_path=seed_path,
            candidate_features=feats,
            metadata_dict=meta,
            target_duration_mins=15.0,
            sequence_mode="ramp",
        )
        result_build = generate_playlist_from_seed(
            seed_path=seed_path,
            candidate_features=feats,
            metadata_dict=meta,
            target_duration_mins=15.0,
            sequence_mode="build",
        )
        # ramp and build should produce different orderings
        order_ramp = [str(p) for p in result_ramp.tracks]
        order_build = [str(p) for p in result_build.tracks]
        assert order_ramp != order_build, (
            f"ramp and build should differ. ramp={order_ramp}, build={order_build}"
        )

    def test_invalid_target_raises_value_error(self) -> None:
        """target_duration_mins=0 should raise ValueError."""
        seed_path, feats, meta = self._make_synthetic_library(n_tracks=5)
        with pytest.raises(ValueError):
            generate_playlist_from_seed(
                seed_path=seed_path,
                candidate_features=feats,
                metadata_dict=meta,
                target_duration_mins=0.0,
            )

    def test_empty_candidates_raises_value_error(self) -> None:
        """Empty candidate_features dict should raise ValueError."""
        seed_path = Path("seed.flac")
        with pytest.raises(ValueError):
            generate_playlist_from_seed(
                seed_path=seed_path,
                candidate_features={},
                metadata_dict={seed_path: make_metadata("seed.flac")},
                target_duration_mins=10.0,
            )

    def test_seed_absent_from_candidates_raises_value_error(self) -> None:
        """Seed path not in candidate_features should raise ValueError."""
        seed_path = Path("seed.flac")
        other_path = Path("other.flac")
        feats = {other_path: make_intensity("other.flac")}
        meta = {
            seed_path: make_metadata("seed.flac"),
            other_path: make_metadata("other.flac"),
        }
        with pytest.raises(ValueError):
            generate_playlist_from_seed(
                seed_path=seed_path,
                candidate_features=feats,
                metadata_dict=meta,
                target_duration_mins=10.0,
            )

    def test_result_is_exportable(self, tmp_path: Path) -> None:
        """A ClusterResult from generate_playlist_from_seed should be exportable to M3U."""
        seed_path, feats, meta = self._make_synthetic_library(
            n_tracks=5, duration_secs=300.0
        )
        result = generate_playlist_from_seed(
            seed_path=seed_path,
            candidate_features=feats,
            metadata_dict=meta,
            target_duration_mins=10.0,
        )
        exporter = M3UExporter(output_dir=tmp_path, playlist_prefix="test")
        output_path = exporter.export_cluster(result, metadata_dict=meta)
        assert output_path.exists()
        content = output_path.read_text()
        assert len(content) > 0
        assert str(seed_path) in content
