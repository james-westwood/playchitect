"""
Regression tests for TASK-28: silent track loss in ``_deduplicate_clusters``.

Bug summary
-----------
``cluster_by_features`` runs K-means, then (for >= 80 tracks, no embeddings)
refines the labels with EWKM per-cluster weight refinement. EWKM can move a
track from its initial K-means cluster to a different one. ``_build_cluster_results``
builds each ``ClusterResult.tracks`` from the EWKM-refined labels, but stores
``centroid=cluster_centers[cid]`` — the RAW (pre-EWKM) K-means centroid.

``_deduplicate_clusters`` then recomputes each track's "home" cluster as the
nearest RAW K-means centroid and keeps only tracks whose current cluster
membership matches that recomputed home. Because the loop is remove-only, any
track EWKM moved into cluster A (whose nearest raw centroid is actually
cluster B) gets removed from A and is never added to B — it vanishes from the
clustering output entirely, with no log line and no error.

A second defect in the same function: the rebuilt ``ClusterResult`` recomputes
``track_count`` from the surviving tracks but copies ``bpm_mean``, ``bpm_std``,
``total_duration``, and ``feature_means`` verbatim from the pre-dedup
(pre-shrink) result, so a cluster that lost members reports stale statistics.

Fixture geometry
-----------------
``_build_boundary_dataset`` builds 90 synthetic tracks (>= ``_MIN_TRACKS_EWKM``)
in two overlapping blobs. Cluster A is tight (low std) on rms/brightness/
sub_bass/harmonics/percussiveness/onset and diffuse on kick_energy; cluster B
has the opposite dispersion pattern (tight on kick_energy, diffuse elsewhere).
This asymmetric per-feature dispersion is exactly the signal EWKM's per-cluster
entropy weighting responds to, so EWKM reassigns several tracks relative to the
initial (uniformly/PCA-weighted) K-means labels. Verified empirically (see
scratch probes during test authoring): with ``random_state=42`` and dataset
``seed=99``, EWKM changes 9 of 90 labels relative to initial K-means, and the
current buggy ``_deduplicate_clusters`` drops exactly those 9 tracks from the
returned clustering (90 in -> 81 out).
"""

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from playchitect.core.clustering import FEATURE_NAMES, ClusterResult, PlaylistClusterer
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata

# ── Fixture geometry ─────────────────────────────────────────────────────────

# feature order for the 8-value Gaussian draw: bpm, rms, brightness, sub_bass,
# kick, harmonics, perc, onset — matches FEATURE_NAMES with bpm prepended.
_CENTER_A = np.array([115.0, 0.30, 0.30, 0.30, 0.50, 0.30, 0.30, 0.30])
_CENTER_B = np.array([125.0, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50])
# Cluster A: tight on everything except kick_energy (diffuse).
_SIGMA_A = np.array([4.0, 0.02, 0.02, 0.02, 0.25, 0.02, 0.02, 0.02])
# Cluster B: tight on kick_energy, diffuse on everything else.
_SIGMA_B = np.array([4.0, 0.25, 0.25, 0.25, 0.02, 0.25, 0.25, 0.25])
_N_PER_CLUSTER = 45
_DATASET_SEED = 99
_CLUSTERER_RANDOM_STATE = 42


def _build_boundary_dataset(
    seed: int = _DATASET_SEED,
) -> tuple[dict[Path, TrackMetadata], dict[Path, IntensityFeatures]]:
    """Build a synthetic library sized/shaped to trigger EWKM cross-boundary moves.

    Returns 90 tracks (well above ``_MIN_TRACKS_EWKM`` = 80) in two overlapping
    blobs with deliberately asymmetric per-feature dispersion, so that EWKM's
    per-cluster entropy weighting reassigns several tracks relative to the
    initial K-means labels.
    """
    rng = np.random.default_rng(seed)
    meta: dict[Path, TrackMetadata] = {}
    intensity: dict[Path, IntensityFeatures] = {}
    i = 0

    for center, sigma, prefix in ((_CENTER_A, _SIGMA_A, "a"), (_CENTER_B, _SIGMA_B, "b")):
        for _ in range(_N_PER_CLUSTER):
            p = Path(f"{prefix}_{i}.mp3")
            vals = rng.normal(center, sigma)
            meta[p] = TrackMetadata(filepath=p, bpm=float(vals[0]), duration=300.0)
            intensity[p] = IntensityFeatures(
                file_path=p,
                file_hash=f"hash{prefix}{i}",
                rms_energy=float(vals[1]),
                brightness=float(vals[2]),
                sub_bass_energy=float(vals[3]),
                kick_energy=float(vals[4]),
                bass_harmonics=float(vals[5]),
                percussiveness=float(vals[6]),
                onset_strength=float(vals[7]),
                camelot_key="8B",
                key_index=0.0,
            )
            i += 1

    return meta, intensity


def _bpm(meta: dict[Path, TrackMetadata], track: Path) -> float:
    """Return a track's BPM, asserting it is populated (fixture always sets it)."""
    bpm = meta[track].bpm
    assert bpm is not None, f"{track} has no BPM in fixture metadata"
    return bpm


def _expected_feature_means(
    tracks: list[Path],
    meta: dict[Path, TrackMetadata],
    intensity: dict[Path, IntensityFeatures],
) -> dict[str, float]:
    """Independently recompute per-feature means (+hardness) for a set of tracks.

    Mirrors the public formula used by ``_build_cluster_results`` (bpm column +
    7-value intensity vector, plus mean ``hardness``) without depending on any
    private clustering internals — used as an oracle to detect stale stats.
    """
    rows = [np.concatenate(([_bpm(meta, t)], intensity[t].to_feature_vector())) for t in tracks]
    matrix = np.vstack(rows)
    means = {name: float(matrix[:, i].mean()) for i, name in enumerate(FEATURE_NAMES)}
    means["hardness"] = float(np.mean([intensity[t].hardness for t in tracks]))
    return means


@dataclass
class DedupFixtureData:
    """Typed container for the module-scoped dedup fixture's outputs."""

    meta: dict[Path, TrackMetadata]
    intensity: dict[Path, IntensityFeatures]
    pre_dedup: list[ClusterResult]
    results: list[ClusterResult]


@pytest.fixture(scope="module")
def dedup_fixture() -> DedupFixtureData:
    """Run cluster_by_features once on the boundary dataset, capturing pre-dedup state.

    Spies on ``_deduplicate_clusters`` (via ``wraps``-style side_effect calling
    the real implementation) purely to capture the pre-dedup ``ClusterResult``
    list for stale-stats comparison. The real dedup logic still executes
    unmodified and unmocked.
    """
    meta, intensity = _build_boundary_dataset()
    clusterer = PlaylistClusterer(
        target_tracks_per_playlist=100,
        min_clusters=2,
        random_state=_CLUSTERER_RANDOM_STATE,
    )

    captured: dict[str, list[ClusterResult]] = {}
    original = clusterer._deduplicate_clusters

    def _spy(
        results: list[ClusterResult], features: np.ndarray, track_order: list[Path]
    ) -> list[ClusterResult]:
        captured["pre_dedup"] = results
        return original(results, features, track_order)

    with patch.object(clusterer, "_deduplicate_clusters", side_effect=_spy):
        final_results = clusterer.cluster_by_features(meta, intensity)

    assert "pre_dedup" in captured, "Fixture bug: _deduplicate_clusters was never called"

    return DedupFixtureData(
        meta=meta,
        intensity=intensity,
        pre_dedup=captured["pre_dedup"],
        results=final_results,
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestDedupConservation:
    """Every input track must end up in exactly one output cluster, or none lost."""

    def test_union_of_cluster_tracks_equals_input_set(
        self, dedup_fixture: DedupFixtureData
    ) -> None:
        """The union of tracks across all returned clusters equals the input set exactly."""
        results = dedup_fixture.results
        meta = dedup_fixture.meta

        returned_tracks: list[Path] = [t for r in results for t in r.tracks]
        expected_paths = sorted(meta.keys())
        actual_union_paths = sorted(set(returned_tracks))

        # Assert on sorted paths (not just counts) per the acceptance criteria.
        assert actual_union_paths == expected_paths, (
            f"Lost {len(expected_paths) - len(actual_union_paths)} tracks in dedup: "
            f"missing={sorted(set(expected_paths) - set(actual_union_paths))}"
        )

    def test_no_track_appears_in_two_clusters(self, dedup_fixture: DedupFixtureData) -> None:
        """No track appears in more than one returned cluster (BUG-02 guarantee)."""
        results = dedup_fixture.results

        returned_tracks: list[Path] = [t for r in results for t in r.tracks]
        assert len(returned_tracks) == len(set(returned_tracks)), (
            f"Found duplicate tracks: {len(returned_tracks)} total vs "
            f"{len(set(returned_tracks))} unique"
        )

    def test_track_counts_sum_to_input_size(self, dedup_fixture: DedupFixtureData) -> None:
        """Sum of track_count across clusters equals the number of input valid tracks."""
        results = dedup_fixture.results
        meta = dedup_fixture.meta

        total_track_count = sum(r.track_count for r in results)
        assert total_track_count == len(meta), (
            f"track_count sum ({total_track_count}) != input size ({len(meta)})"
        )

    def test_each_result_track_count_matches_tracks_length(
        self, dedup_fixture: DedupFixtureData
    ) -> None:
        """Each ClusterResult.track_count matches len(ClusterResult.tracks)."""
        results = dedup_fixture.results

        for r in results:
            assert r.track_count == len(r.tracks), (
                f"Cluster {r.cluster_id}: track_count={r.track_count} "
                f"!= len(tracks)={len(r.tracks)}"
            )

    def test_fixture_actually_triggers_ewkm_reassignment(
        self, dedup_fixture: DedupFixtureData
    ) -> None:
        """Sanity check: at least one cluster's membership actually changed during dedup.

        This guards against the fixture silently stopping to trigger the bug
        (e.g. after an unrelated dependency upgrade changes KMeans/EWKM
        numerics) and thereby making the rest of this module vacuous.
        """
        pre_dedup = dedup_fixture.pre_dedup
        results = dedup_fixture.results

        pre_by_id = {r.cluster_id: set(r.tracks) for r in pre_dedup}
        post_by_id = {r.cluster_id: set(r.tracks) for r in results}

        changed_clusters = [
            cid for cid, pre_tracks in pre_by_id.items() if pre_tracks != post_by_id.get(cid)
        ]
        assert changed_clusters, (
            "Fixture did not trigger any membership change during dedup — "
            "the boundary geometry no longer causes EWKM to cross a K-means "
            "boundary. This test file must be re-tuned; it is not exercising "
            "the TASK-28 bug."
        )


class TestDedupFreshStats:
    """A cluster whose membership shrank must report stats from its survivors."""

    def test_shrunken_cluster_reports_fresh_bpm_and_duration_stats(
        self, dedup_fixture: DedupFixtureData
    ) -> None:
        """bpm_mean/bpm_std/total_duration are recomputed from survivors, not copied stale."""
        pre_dedup = dedup_fixture.pre_dedup
        results = dedup_fixture.results
        meta = dedup_fixture.meta

        pre_by_id = {r.cluster_id: r for r in pre_dedup}
        post_by_id = {r.cluster_id: r for r in results}

        shrunken = [
            cid for cid, pre in pre_by_id.items() if post_by_id[cid].track_count < pre.track_count
        ]
        assert shrunken, "Fixture did not shrink any cluster; cannot test stale-stats fix"

        for cid in shrunken:
            pre = pre_by_id[cid]
            post = post_by_id[cid]
            survivor_bpms = [_bpm(meta, t) for t in post.tracks]
            expected_bpm_mean = float(np.mean(survivor_bpms))
            expected_bpm_std = float(np.std(survivor_bpms))
            expected_duration = float(sum(meta[t].duration or 0.0 for t in post.tracks))

            # Must match a fresh recomputation from the surviving members.
            assert post.bpm_mean == pytest.approx(expected_bpm_mean, abs=1e-9), (
                f"Cluster {cid}: bpm_mean={post.bpm_mean} does not match fresh "
                f"recomputation {expected_bpm_mean} from {post.track_count} survivors"
            )
            assert post.bpm_std == pytest.approx(expected_bpm_std, abs=1e-9), (
                f"Cluster {cid}: bpm_std={post.bpm_std} does not match fresh "
                f"recomputation {expected_bpm_std} from {post.track_count} survivors"
            )
            assert post.total_duration == pytest.approx(expected_duration, abs=1e-9), (
                f"Cluster {cid}: total_duration={post.total_duration} does not match "
                f"fresh recomputation {expected_duration} from {post.track_count} survivors"
            )

            # And must NOT equal the stale pre-dedup parent values (proves the
            # stats were actually recomputed, not merely coincidentally equal).
            assert post.bpm_mean != pytest.approx(pre.bpm_mean, abs=1e-9), (
                f"Cluster {cid}: bpm_mean still equals stale pre-dedup value {pre.bpm_mean}"
            )
            assert post.total_duration != pytest.approx(pre.total_duration, abs=1e-9), (
                f"Cluster {cid}: total_duration still equals stale pre-dedup "
                f"value {pre.total_duration}"
            )

    def test_shrunken_cluster_reports_fresh_feature_means(
        self, dedup_fixture: DedupFixtureData
    ) -> None:
        """feature_means are recomputed from survivors, not copied from the stale parent."""
        pre_dedup = dedup_fixture.pre_dedup
        results = dedup_fixture.results
        meta = dedup_fixture.meta
        intensity = dedup_fixture.intensity

        pre_by_id = {r.cluster_id: r for r in pre_dedup}
        post_by_id = {r.cluster_id: r for r in results}

        shrunken = [
            cid for cid, pre in pre_by_id.items() if post_by_id[cid].track_count < pre.track_count
        ]
        assert shrunken, "Fixture did not shrink any cluster; cannot test stale-stats fix"

        for cid in shrunken:
            pre = pre_by_id[cid]
            post = post_by_id[cid]
            assert post.feature_means is not None
            assert pre.feature_means is not None

            expected_means = _expected_feature_means(post.tracks, meta, intensity)

            for name, expected_value in expected_means.items():
                assert post.feature_means[name] == pytest.approx(expected_value, abs=1e-9), (
                    f"Cluster {cid}: feature_means[{name!r}]={post.feature_means[name]} "
                    f"does not match fresh recomputation {expected_value}"
                )

            # At least one feature mean must differ from the stale parent value —
            # otherwise the recomputation would be indistinguishable from a copy.
            differing = [
                name
                for name in expected_means
                if post.feature_means[name] != pytest.approx(pre.feature_means[name], abs=1e-9)
            ]
            assert differing, (
                f"Cluster {cid}: feature_means are identical to the stale pre-dedup "
                f"parent values; stats were not recomputed"
            )


class TestDeduplicateClustersUnit:
    """Focused unit test of _deduplicate_clusters in isolation."""

    def test_track_reassigned_by_caller_is_kept_not_dropped(self) -> None:
        """A track whose current cluster differs from its nearest centroid is reassigned,
        not silently dropped.

        Hand-builds two ClusterResult objects plus a feature matrix where
        ``p_swing`` is listed under cluster 0's tracks (mimicking what EWKM
        would have produced) but its feature vector is far closer to cluster
        1's centroid. The remove-only bug drops it entirely (0 appearances in
        the output); the fix must place it in exactly one output cluster.
        """
        clusterer = PlaylistClusterer(target_tracks_per_playlist=10, min_clusters=2)

        p0 = Path("p0.mp3")
        p1 = Path("p1.mp3")
        p2 = Path("p2.mp3")
        p_swing = Path("p_swing.mp3")

        track_order = [p0, p1, p2, p_swing]
        # 2D toy feature space: cluster 0 lives near (0, 0), cluster 1 near (10, 10).
        features = np.array(
            [
                [0.0, 0.0],  # p0 — near centroid 0
                [0.5, 0.5],  # p1 — near centroid 0
                [10.0, 10.0],  # p2 — near centroid 1
                [9.0, 9.0],  # p_swing — geometrically near centroid 1
            ]
        )

        results = [
            ClusterResult(
                cluster_id=0,
                # p_swing is listed here (its "current" EWKM-assigned home),
                # even though it is geometrically closest to cluster 1.
                tracks=[p0, p1, p_swing],
                bpm_mean=120.0,
                bpm_std=1.0,
                track_count=3,
                total_duration=900.0,
                centroid=np.array([0.0, 0.0]),
            ),
            ClusterResult(
                cluster_id=1,
                tracks=[p2],
                bpm_mean=140.0,
                bpm_std=0.0,
                track_count=1,
                total_duration=300.0,
                centroid=np.array([10.0, 10.0]),
            ),
        ]

        deduped = clusterer._deduplicate_clusters(results, features, track_order)

        appearances = [t for r in deduped for t in r.tracks if t == p_swing]
        assert len(appearances) == 1, (
            f"p_swing appeared {len(appearances)} times in deduped output "
            f"(expected exactly 1) — it must be reassigned to its nearest "
            f"centroid's cluster, not dropped"
        )

        # It should specifically land in cluster 1 (its geometrically nearest centroid).
        owning_clusters = [r.cluster_id for r in deduped if p_swing in r.tracks]
        assert owning_clusters == [1], (
            f"p_swing ended up in cluster(s) {owning_clusters}, expected [1] (nearest centroid)"
        )

        # No other track should have been disturbed.
        all_tracks = [t for r in deduped for t in r.tracks]
        assert sorted(all_tracks, key=str) == sorted([p0, p1, p2, p_swing], key=str)
