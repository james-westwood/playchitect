"""
Unit tests for weighting module.
"""

from pathlib import Path

import numpy as np
import pytest

from playchitect.core.weighting import (
    FEATURE_NAMES,
    SUPPORTED_GENRES,
    WeightProfile,
    ewkm_refine,
    get_heuristic_weights,
    get_uniform_weights,
    learn_weights_pca,
    select_weights,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_structured_data(n: int = 100, seed: int = 0) -> np.ndarray:
    """Synthetic 8-feature data with clear structure for PCA stability."""
    rng = np.random.default_rng(seed)
    # Two latent factors drive most variance
    f1 = rng.standard_normal(n)
    f2 = rng.standard_normal(n)
    X = np.column_stack(
        [
            f1 * 2.0 + rng.standard_normal(n) * 0.2,  # bpm
            f1 * 1.5 + rng.standard_normal(n) * 0.2,  # rms_energy
            f2 * 1.8 + rng.standard_normal(n) * 0.2,  # brightness
            f2 * 0.8 + rng.standard_normal(n) * 0.3,  # sub_bass
            f1 * 1.2 + rng.standard_normal(n) * 0.4,  # kick_energy
            f2 * 0.6 + rng.standard_normal(n) * 0.3,  # bass_harmonics
            f1 * 0.9 + rng.standard_normal(n) * 0.5,  # percussiveness
            f1 * 0.7 + rng.standard_normal(n) * 0.4,  # onset_strength
        ]
    )
    # Standardise so test doesn't depend on raw scale
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X


# ── WeightProfile ─────────────────────────────────────────────────────────────


class TestWeightProfile:
    def test_as_dict_has_all_feature_names(self) -> None:
        w = np.ones(8) / 8
        profile = WeightProfile(weights=w, source="uniform")
        d = profile.as_dict()
        assert set(d.keys()) == set(FEATURE_NAMES)

    def test_as_dict_values_match_weights(self) -> None:
        w = np.array([0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1])
        profile = WeightProfile(weights=w, source="heuristic", genre="techno")
        d = profile.as_dict()
        for i, name in enumerate(FEATURE_NAMES):
            assert abs(d[name] - float(w[i])) < 1e-9

    def test_optional_fields_default_to_none(self) -> None:
        profile = WeightProfile(weights=np.ones(8) / 8, source="uniform")
        assert profile.genre is None
        assert profile.ci_width is None
        assert profile.n_tracks == 0


# ── get_uniform_weights ───────────────────────────────────────────────────────


class TestGetUniformWeights:
    def test_shape(self) -> None:
        profile = get_uniform_weights()
        assert profile.weights.shape == (len(FEATURE_NAMES),)

    def test_sums_to_one(self) -> None:
        profile = get_uniform_weights()
        assert abs(profile.weights.sum() - 1.0) < 1e-9

    def test_all_equal(self) -> None:
        profile = get_uniform_weights()
        expected = 1.0 / len(FEATURE_NAMES)
        assert np.allclose(profile.weights, expected)

    def test_source_label(self) -> None:
        assert get_uniform_weights().source == "uniform"

    def test_n_tracks_stored(self) -> None:
        assert get_uniform_weights(n_tracks=42).n_tracks == 42


# ── get_heuristic_weights ─────────────────────────────────────────────────────


class TestGetHeuristicWeights:
    def test_all_supported_genres_return_profile(self) -> None:
        for genre in SUPPORTED_GENRES:
            profile = get_heuristic_weights(genre)
            assert isinstance(profile, WeightProfile)

    def test_weights_sum_to_one(self) -> None:
        for genre in SUPPORTED_GENRES:
            profile = get_heuristic_weights(genre)
            assert abs(profile.weights.sum() - 1.0) < 1e-9, f"Failed for {genre}"

    def test_shape(self) -> None:
        for genre in SUPPORTED_GENRES:
            assert get_heuristic_weights(genre).weights.shape == (len(FEATURE_NAMES),)

    def test_source_label(self) -> None:
        assert get_heuristic_weights("techno").source == "heuristic"

    def test_genre_stored(self) -> None:
        assert get_heuristic_weights("techno").genre == "techno"

    def test_case_insensitive(self) -> None:
        p1 = get_heuristic_weights("Techno")
        p2 = get_heuristic_weights("techno")
        assert np.allclose(p1.weights, p2.weights)

    def test_unknown_genre_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown genre"):
            get_heuristic_weights("polka")

    def test_techno_emphasises_kick_and_bpm(self) -> None:
        profile = get_heuristic_weights("techno")
        d = profile.as_dict()
        # kick_energy and bpm should both be among the top-2 weights
        top2 = sorted(d, key=lambda k: d[k], reverse=True)[:2]
        assert "kick_energy" in top2 or "bpm" in top2

    def test_ambient_deemphasises_kick(self) -> None:
        techno = get_heuristic_weights("techno").as_dict()
        ambient = get_heuristic_weights("ambient").as_dict()
        assert ambient["kick_energy"] < techno["kick_energy"]

    def test_ambient_emphasises_onset_strength(self) -> None:
        ambient = get_heuristic_weights("ambient").as_dict()
        # onset_strength should be high (near-zero onset defines ambient)
        top3 = sorted(ambient, key=lambda k: ambient[k], reverse=True)[:3]
        assert "onset_strength" in top3

    def test_dnb_has_highest_bpm_weight(self) -> None:
        dnb = get_heuristic_weights("dnb").as_dict()
        assert dnb["bpm"] == max(dnb.values())


# ── learn_weights_pca ─────────────────────────────────────────────────────────


class TestLearnWeightsPca:
    def test_returns_profile_for_sufficient_structured_data(self) -> None:
        X = _make_structured_data(n=120)
        profile = learn_weights_pca(X, n_bootstrap=20, random_state=0)
        assert profile is not None
        assert profile.source == "pca"
        assert profile.weights.shape == (len(FEATURE_NAMES),)

    def test_weights_sum_to_one(self) -> None:
        X = _make_structured_data(n=120)
        profile = learn_weights_pca(X, n_bootstrap=20, random_state=0)
        assert profile is not None
        assert abs(profile.weights.sum() - 1.0) < 1e-9

    def test_all_weights_positive(self) -> None:
        X = _make_structured_data(n=120)
        profile = learn_weights_pca(X, n_bootstrap=20, random_state=0)
        assert profile is not None
        assert np.all(profile.weights > 0)

    def test_ci_width_stored(self) -> None:
        X = _make_structured_data(n=120)
        profile = learn_weights_pca(X, n_bootstrap=20, random_state=0)
        assert profile is not None
        assert profile.ci_width is not None
        assert profile.ci_width >= 0.0

    def test_returns_none_when_ci_threshold_zero(self) -> None:
        # Setting ci_threshold=0 forces instability detection
        X = _make_structured_data(n=80)
        profile = learn_weights_pca(X, n_bootstrap=20, ci_threshold=0.0, random_state=0)
        assert profile is None

    def test_reproducible_with_same_seed(self) -> None:
        X = _make_structured_data(n=120)
        p1 = learn_weights_pca(X, n_bootstrap=20, random_state=7)
        p2 = learn_weights_pca(X, n_bootstrap=20, random_state=7)
        assert p1 is not None and p2 is not None
        assert np.allclose(p1.weights, p2.weights)


# ── select_weights ────────────────────────────────────────────────────────────


class TestSelectWeights:
    def test_small_library_no_genre_returns_uniform(self) -> None:
        X = _make_structured_data(n=10)
        profile = select_weights(X)
        assert profile.source == "uniform"

    def test_small_library_with_genre_returns_heuristic(self) -> None:
        X = _make_structured_data(n=10)
        profile = select_weights(X, genre="techno")
        assert profile.source == "heuristic"
        assert profile.genre == "techno"

    def test_large_library_returns_pca_or_heuristic(self) -> None:
        X = _make_structured_data(n=120)
        profile = select_weights(X)
        # With structured data, PCA should be stable enough
        assert profile.source in ("pca", "heuristic", "uniform")

    def test_genre_overrides_when_pca_unstable(self) -> None:
        # Very small dataset forces PCA to be skipped; genre should be used
        X = _make_structured_data(n=15)
        profile = select_weights(X, genre="house")
        assert profile.source == "heuristic"
        assert profile.genre == "house"

    def test_returns_weight_profile_instance(self) -> None:
        X = _make_structured_data(n=50)
        assert isinstance(select_weights(X), WeightProfile)

    def test_weights_always_sum_to_one(self) -> None:
        for n, genre in [(10, None), (10, "techno"), (60, None), (120, None)]:
            X = _make_structured_data(n=n)
            profile = select_weights(X, genre=genre)
            assert abs(profile.weights.sum() - 1.0) < 1e-9, f"Failed for n={n}, genre={genre}"


# ── ewkm_refine ───────────────────────────────────────────────────────────────


class TestEwkmRefine:
    def _two_cluster_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return X, labels, centroids for a simple 2-cluster dataset."""
        rng = np.random.default_rng(42)
        n = 40
        X = np.zeros((n, 8))
        X[:20, 0] = rng.standard_normal(20) * 0.1 + 2.0  # cluster 0: high feature 0
        X[20:, 0] = rng.standard_normal(20) * 0.1 - 2.0  # cluster 1: low feature 0
        X[:, 1:] = rng.standard_normal((n, 7)) * 0.5
        labels = np.array([0] * 20 + [1] * 20)
        centroids = np.array([X[:20].mean(axis=0), X[20:].mean(axis=0)])
        return X, labels, centroids

    def test_output_shapes(self) -> None:
        X, labels, centroids = self._two_cluster_data()
        new_labels, weights = ewkm_refine(X, labels, centroids)
        assert new_labels.shape == (len(X),)
        assert weights.shape == (2, 8)

    def test_per_cluster_weights_sum_to_one(self) -> None:
        X, labels, centroids = self._two_cluster_data()
        _, weights = ewkm_refine(X, labels, centroids)
        for k in range(weights.shape[0]):
            assert abs(weights[k].sum() - 1.0) < 1e-6, f"Cluster {k} weights don't sum to 1"

    def test_all_weights_positive(self) -> None:
        X, labels, centroids = self._two_cluster_data()
        _, weights = ewkm_refine(X, labels, centroids)
        assert np.all(weights > 0)

    def test_low_dispersion_feature_gets_high_weight(self) -> None:
        """Feature with near-zero within-cluster variance should get the highest weight."""
        rng = np.random.default_rng(0)
        n = 30
        X = rng.standard_normal((n, 8)) * 2.0  # noisy features
        X[:, 3] = 0.5 + rng.standard_normal(n) * 0.01  # feature 3: extremely tight

        labels = np.zeros(n, dtype=int)  # single cluster
        centroids = X.mean(axis=0, keepdims=True)

        _, weights = ewkm_refine(X, labels, centroids, gamma=0.1)
        assert np.argmax(weights[0]) == 3

    def test_does_not_modify_input_arrays(self) -> None:
        X, labels, centroids = self._two_cluster_data()
        labels_orig = labels.copy()
        centroids_orig = centroids.copy()
        ewkm_refine(X, labels, centroids)
        assert np.array_equal(labels, labels_orig)
        assert np.allclose(centroids, centroids_orig)

    def test_convergence_with_well_separated_clusters(self) -> None:
        """EWKM should converge quickly and maintain cluster structure."""
        rng = np.random.default_rng(1)
        n_per = 25
        X = np.vstack(
            [
                rng.standard_normal((n_per, 8)) + np.array([3, 0, 0, 0, 0, 0, 0, 0]),
                rng.standard_normal((n_per, 8)) + np.array([-3, 0, 0, 0, 0, 0, 0, 0]),
            ]
        )
        labels = np.array([0] * n_per + [1] * n_per)
        centroids = np.array([X[:n_per].mean(axis=0), X[n_per:].mean(axis=0)])

        new_labels, _ = ewkm_refine(X, labels, centroids)
        # All originally-cluster-0 tracks should stay in the same cluster
        assert len(set(new_labels[:n_per])) == 1
        assert len(set(new_labels[n_per:])) == 1


# ── Integration: cluster_by_features with genre ───────────────────────────────


class TestClusterByFeaturesWeighted:
    """Smoke tests: cluster_by_features respects genre and weight_source."""

    def _make_library(self, n: int, bpm: float = 138.0) -> tuple[dict, dict]:
        from playchitect.core.intensity_analyzer import IntensityFeatures
        from playchitect.core.metadata_extractor import TrackMetadata

        meta = {}
        intensity = {}
        rng = np.random.default_rng(42)
        for i in range(n):
            p = Path(f"t{i}.mp3")
            meta[p] = TrackMetadata(filepath=p, bpm=bpm + rng.uniform(-2, 2), duration=360.0)
            intensity[p] = IntensityFeatures(
                filepath=p,
                file_hash="x",
                rms_energy=float(rng.uniform(0.3, 0.9)),
                brightness=float(rng.uniform(0.2, 0.8)),
                sub_bass_energy=float(rng.uniform(0.1, 0.6)),
                kick_energy=float(rng.uniform(0.4, 0.9)),
                bass_harmonics=float(rng.uniform(0.2, 0.7)),
                percussiveness=float(rng.uniform(0.5, 0.95)),
                onset_strength=float(rng.uniform(0.4, 0.8)),
            )
        return meta, intensity

    def test_weight_source_uniform_for_tiny_library(self) -> None:
        from playchitect.core.clustering import PlaylistClusterer

        meta, intensity = self._make_library(8)
        clusterer = PlaylistClusterer(target_tracks_per_playlist=4, min_clusters=2)
        results = clusterer.cluster_by_features(meta, intensity)
        assert all(r.weight_source == "uniform" for r in results)

    def test_weight_source_heuristic_when_genre_given(self) -> None:
        from playchitect.core.clustering import PlaylistClusterer

        meta, intensity = self._make_library(8)
        clusterer = PlaylistClusterer(target_tracks_per_playlist=4, min_clusters=2)
        results = clusterer.cluster_by_features(meta, intensity, genre="techno")
        assert all(r.weight_source == "heuristic" for r in results)

    def test_feature_importance_present_in_weighted_results(self) -> None:
        from playchitect.core.clustering import PlaylistClusterer

        meta, intensity = self._make_library(12)
        clusterer = PlaylistClusterer(target_tracks_per_playlist=6, min_clusters=2)
        results = clusterer.cluster_by_features(meta, intensity, genre="techno")
        for r in results:
            assert r.feature_importance is not None
            assert set(r.feature_importance.keys()) == set(FEATURE_NAMES)
            assert abs(sum(r.feature_importance.values()) - 1.0) < 1e-6

    def test_results_still_cover_all_tracks(self) -> None:
        from playchitect.core.clustering import PlaylistClusterer

        n = 20
        meta, intensity = self._make_library(n)
        clusterer = PlaylistClusterer(target_tracks_per_playlist=10, min_clusters=2)
        results = clusterer.cluster_by_features(meta, intensity, genre="house")
        assert sum(r.track_count for r in results) == n
