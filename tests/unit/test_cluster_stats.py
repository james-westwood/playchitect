"""Unit tests for playchitect.gui.widgets.cluster_stats.

All tests are GTK-free — ClusterStats is a pure-Python dataclass.
"""

from __future__ import annotations

import pytest

from playchitect.core.clustering import ClusterResult
from playchitect.gui.widgets.cluster_stats import (
    _BAR_WIDTH,
    _HIGH_THRESHOLD,
    _LOW_THRESHOLD,
    _VERY_HIGH_THRESHOLD,
    ClusterStats,
)


def _make_result(
    cluster_id: int | str = 1,
    bpm_mean: float = 125.0,
    bpm_std: float = 2.5,
    track_count: int = 20,
    total_duration: float = 3600.0,
    feature_means: dict[str, float] | None = None,
    feature_importance: dict[str, float] | None = None,
) -> ClusterResult:
    return ClusterResult(
        cluster_id=cluster_id,
        tracks=[],
        bpm_mean=bpm_mean,
        bpm_std=bpm_std,
        track_count=track_count,
        total_duration=total_duration,
        feature_means=feature_means,
        feature_importance=feature_importance,
    )


# ── TestClusterStatsFromResult ────────────────────────────────────────────────


class TestClusterStatsFromResult:
    def test_basic_fields_populated(self):
        result = _make_result(cluster_id=3, bpm_mean=128.0, bpm_std=3.0, track_count=15)
        stats = ClusterStats.from_result(result)

        assert stats.cluster_id == 3
        assert stats.track_count == 15
        assert stats.bpm_mean == 128.0

    def test_bpm_range_from_std(self):
        result = _make_result(bpm_mean=120.0, bpm_std=5.0)
        stats = ClusterStats.from_result(result)

        assert stats.bpm_min == pytest.approx(115.0)
        assert stats.bpm_max == pytest.approx(125.0)

    def test_bpm_min_clamped_to_one(self):
        # Even if mean - std < 1, bpm_min must be ≥ 1.
        result = _make_result(bpm_mean=3.0, bpm_std=5.0)
        stats = ClusterStats.from_result(result)

        assert stats.bpm_min >= 1.0

    def test_bpm_max_always_greater_than_min(self):
        result = _make_result(bpm_mean=100.0, bpm_std=0.0)
        stats = ClusterStats.from_result(result)

        assert stats.bpm_max > stats.bpm_min

    def test_intensity_from_rms_energy(self):
        result = _make_result(feature_means={"rms_energy": 0.75, "spectral_centroid": 0.5})
        stats = ClusterStats.from_result(result)

        assert stats.intensity_mean == pytest.approx(0.75)

    def test_intensity_falls_back_to_zero_when_no_feature_means(self):
        result = _make_result(feature_means=None)
        stats = ClusterStats.from_result(result)

        assert stats.intensity_mean == 0.0

    def test_intensity_falls_back_to_zero_when_rms_missing(self):
        result = _make_result(feature_means={"spectral_centroid": 0.5})
        stats = ClusterStats.from_result(result)

        assert stats.intensity_mean == 0.0

    def test_feature_importance_sorted_descending(self):
        result = _make_result(
            feature_importance={"bpm": 0.3, "rms_energy": 0.5, "spectral_centroid": 0.2}
        )
        stats = ClusterStats.from_result(result)

        names = [name for name, _ in stats.feature_importance]
        assert names == ["rms_energy", "bpm", "spectral_centroid"]

    def test_zero_weight_features_excluded(self):
        result = _make_result(
            feature_importance={"bpm": 0.5, "hf_energy": 0.0, "percussiveness": 0.3}
        )
        stats = ClusterStats.from_result(result)

        names = [name for name, _ in stats.feature_importance]
        assert "hf_energy" not in names

    def test_no_feature_importance_gives_empty_list(self):
        result = _make_result(feature_importance=None)
        stats = ClusterStats.from_result(result)

        assert stats.feature_importance == []

    def test_string_cluster_id_preserved(self):
        result = _make_result(cluster_id="1a")
        stats = ClusterStats.from_result(result)

        assert stats.cluster_id == "1a"

    def test_opener_and_closer_populated(self):
        from pathlib import Path

        opener = Path("/music/opener.flac")
        closer = Path("/music/closer.flac")
        result = _make_result()
        result.opener = opener
        result.closer = closer
        stats = ClusterStats.from_result(result)

        assert stats.opener_name == "opener.flac"
        assert stats.closer_name == "closer.flac"

    def test_opener_and_closer_none_by_default(self):
        result = _make_result()
        result.opener = None
        result.closer = None
        stats = ClusterStats.from_result(result)

        assert stats.opener_name is None
        assert stats.closer_name is None


# ── TestBpmRangeStr ───────────────────────────────────────────────────────────


class TestBpmRangeStr:
    def test_typical_range(self):
        stats = ClusterStats(
            cluster_id=1,
            track_count=10,
            bpm_min=120.0,
            bpm_max=125.0,
            bpm_mean=122.5,
            intensity_mean=0.5,
            hardness_mean=0.5,
            total_duration=3600.0,
        )
        assert stats.bpm_range_str == "120–125 BPM"

    def test_fractional_values_rounded(self):
        stats = ClusterStats(
            cluster_id=1,
            track_count=5,
            bpm_min=119.6,
            bpm_max=125.4,
            bpm_mean=122.0,
            intensity_mean=0.4,
            hardness_mean=0.4,
            total_duration=1800.0,
        )
        assert stats.bpm_range_str == "120–125 BPM"

    def test_bpm_mean_str_integer(self):
        stats = ClusterStats(
            cluster_id=1,
            track_count=5,
            bpm_min=120.0,
            bpm_max=125.0,
            bpm_mean=122.7,
            intensity_mean=0.4,
            hardness_mean=0.4,
            total_duration=1800.0,
        )
        assert stats.bpm_mean_str == "123"


# ── TestIntensityLabel ────────────────────────────────────────────────────────


class TestIntensityLabel:
    def _make(self, intensity: float) -> ClusterStats:
        return ClusterStats(
            cluster_id=1,
            track_count=5,
            bpm_min=120.0,
            bpm_max=125.0,
            bpm_mean=122.5,
            intensity_mean=intensity,
            hardness_mean=intensity,
            total_duration=1800.0,
        )

    def test_low(self):
        assert self._make(0.0).intensity_label == "Low"
        assert self._make(_LOW_THRESHOLD - 0.01).intensity_label == "Low"

    def test_medium(self):
        assert self._make(_LOW_THRESHOLD).intensity_label == "Medium"
        assert self._make(0.5).intensity_label == "Medium"

    def test_high(self):
        assert self._make(_HIGH_THRESHOLD).intensity_label == "High"
        assert self._make(0.75).intensity_label == "High"

    def test_very_high(self):
        assert self._make(_VERY_HIGH_THRESHOLD).intensity_label == "Very High"
        assert self._make(1.0).intensity_label == "Very High"


# ── TestIntensityBars ─────────────────────────────────────────────────────────


class TestIntensityBars:
    def _make(self, intensity: float) -> ClusterStats:
        return ClusterStats(
            cluster_id=1,
            track_count=5,
            bpm_min=120.0,
            bpm_max=125.0,
            bpm_mean=122.5,
            intensity_mean=intensity,
            hardness_mean=intensity,
            total_duration=1800.0,
        )

    def test_length_always_bar_width(self):
        for intensity in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert len(self._make(intensity).intensity_bars) == _BAR_WIDTH

    def test_zero_intensity_all_empty(self):
        assert self._make(0.0).intensity_bars == "░" * _BAR_WIDTH

    def test_full_intensity_all_full(self):
        assert self._make(1.0).intensity_bars == "█" * _BAR_WIDTH

    def test_half_intensity(self):
        bars = self._make(0.5).intensity_bars
        assert bars.count("█") == 5
        assert bars.count("░") == 5

    def test_clamped_above_one(self):
        assert self._make(2.0).intensity_bars == "█" * _BAR_WIDTH

    def test_clamped_below_zero(self):
        assert self._make(-1.0).intensity_bars == "░" * _BAR_WIDTH


# ── TestDurationStr ───────────────────────────────────────────────────────────


class TestDurationStr:
    def _make(self, duration: float) -> ClusterStats:
        return ClusterStats(
            cluster_id=1,
            track_count=5,
            bpm_min=120.0,
            bpm_max=125.0,
            bpm_mean=122.5,
            intensity_mean=0.5,
            hardness_mean=0.5,
            total_duration=duration,
        )

    def test_minutes_only(self):
        assert self._make(2700.0).duration_str == "45m"

    def test_hours_and_minutes(self):
        assert self._make(5700.0).duration_str == "1h 35m"

    def test_exactly_one_hour(self):
        assert self._make(3600.0).duration_str == "1h 0m"

    def test_zero_duration(self):
        assert self._make(0.0).duration_str == "0m"

    def test_fractional_seconds_truncated(self):
        # 90.9 seconds → 1m (integer truncation, not rounding)
        assert self._make(90.9).duration_str == "1m"


# ── TestTrackCountStr ─────────────────────────────────────────────────────────


class TestTrackCountStr:
    def _make(self, count: int) -> ClusterStats:
        return ClusterStats(
            cluster_id=1,
            track_count=count,
            bpm_min=120.0,
            bpm_max=125.0,
            bpm_mean=122.5,
            intensity_mean=0.5,
            hardness_mean=0.5,
            total_duration=1800.0,
        )

    def test_singular(self):
        assert self._make(1).track_count_str == "1 track"

    def test_plural(self):
        assert self._make(23).track_count_str == "23 tracks"

    def test_zero(self):
        assert self._make(0).track_count_str == "0 tracks"


# ── TestClusterLabel ──────────────────────────────────────────────────────────


class TestClusterLabel:
    def test_integer_id(self):
        stats = ClusterStats(
            cluster_id=2,
            track_count=10,
            bpm_min=120.0,
            bpm_max=130.0,
            bpm_mean=125.0,
            intensity_mean=0.5,
            hardness_mean=0.5,
            total_duration=3600.0,
        )
        assert stats.cluster_label == "Cluster 2"

    def test_string_id(self):
        stats = ClusterStats(
            cluster_id="1a",
            track_count=5,
            bpm_min=120.0,
            bpm_max=125.0,
            bpm_mean=122.5,
            intensity_mean=0.3,
            hardness_mean=0.3,
            total_duration=1800.0,
        )
        assert stats.cluster_label == "Cluster 1a"


# ── TestTopFeatures ───────────────────────────────────────────────────────────


class TestTopFeatures:
    def test_top_three_returned(self):
        stats = ClusterStats(
            cluster_id=1,
            track_count=10,
            bpm_min=120.0,
            bpm_max=130.0,
            bpm_mean=125.0,
            intensity_mean=0.5,
            hardness_mean=0.5,
            total_duration=3600.0,
            feature_importance=[
                ("rms_energy", 0.5),
                ("bpm", 0.3),
                ("spectral_centroid", 0.2),
                ("percussiveness", 0.1),
            ],
        )
        assert len(stats.top_features) == 3
        assert stats.top_features[0] == ("rms_energy", 0.5)

    def test_fewer_than_three_returns_all(self):
        stats = ClusterStats(
            cluster_id=1,
            track_count=10,
            bpm_min=120.0,
            bpm_max=130.0,
            bpm_mean=125.0,
            intensity_mean=0.5,
            hardness_mean=0.5,
            total_duration=3600.0,
            feature_importance=[("rms_energy", 0.5)],
        )
        assert len(stats.top_features) == 1

    def test_empty_importance_returns_empty(self):
        stats = ClusterStats(
            cluster_id=1,
            track_count=10,
            bpm_min=120.0,
            bpm_max=130.0,
            bpm_mean=125.0,
            intensity_mean=0.5,
            hardness_mean=0.5,
            total_duration=3600.0,
        )
        assert stats.top_features == []


# ── TestBpmRangeFraction ──────────────────────────────────────────────────────


class TestBpmRangeFraction:
    def _make(self, bpm_mean: float) -> ClusterStats:
        return ClusterStats(
            cluster_id=1,
            track_count=5,
            bpm_min=bpm_mean - 2,
            bpm_max=bpm_mean + 2,
            bpm_mean=bpm_mean,
            intensity_mean=0.5,
            hardness_mean=0.5,
            total_duration=1800.0,
        )

    def test_at_global_min(self):
        assert self._make(100.0).bpm_range_fraction(100.0, 200.0) == pytest.approx(0.0)

    def test_at_global_max(self):
        assert self._make(200.0).bpm_range_fraction(100.0, 200.0) == pytest.approx(1.0)

    def test_midpoint(self):
        assert self._make(150.0).bpm_range_fraction(100.0, 200.0) == pytest.approx(0.5)

    def test_zero_span_returns_zero(self):
        assert self._make(120.0).bpm_range_fraction(120.0, 120.0) == 0.0

    def test_clamped_below_zero(self):
        assert self._make(80.0).bpm_range_fraction(100.0, 200.0) == pytest.approx(0.0)

    def test_clamped_above_one(self):
        assert self._make(250.0).bpm_range_fraction(100.0, 200.0) == pytest.approx(1.0)


# ── TestGlobalBpmRange ────────────────────────────────────────────────────────


class TestGlobalBpmRange:
    def _make_stats(self, bpm_min: float, bpm_max: float) -> ClusterStats:
        return ClusterStats(
            cluster_id=1,
            track_count=5,
            bpm_min=bpm_min,
            bpm_max=bpm_max,
            bpm_mean=(bpm_min + bpm_max) / 2,
            intensity_mean=0.5,
            hardness_mean=0.5,
            total_duration=1800.0,
        )

    def test_single_cluster(self):
        stats = [self._make_stats(120.0, 130.0)]
        lo, hi = ClusterStats.global_bpm_range(stats)
        assert lo == pytest.approx(120.0)
        assert hi == pytest.approx(130.0)

    def test_multiple_clusters(self):
        stats = [
            self._make_stats(110.0, 118.0),
            self._make_stats(120.0, 130.0),
            self._make_stats(128.0, 140.0),
        ]
        lo, hi = ClusterStats.global_bpm_range(stats)
        assert lo == pytest.approx(110.0)
        assert hi == pytest.approx(140.0)

    def test_empty_list_returns_default(self):
        lo, hi = ClusterStats.global_bpm_range([])
        assert lo == pytest.approx(0.0)
        assert hi == pytest.approx(200.0)


# ── TestFromResults ───────────────────────────────────────────────────────────


class TestFromResults:
    def test_converts_all_results(self):
        results = [_make_result(cluster_id=i) for i in range(4)]
        stats_list = ClusterStats.from_results(results)
        assert len(stats_list) == 4
        assert [s.cluster_id for s in stats_list] == [0, 1, 2, 3]

    def test_empty_list(self):
        assert ClusterStats.from_results([]) == []


# ── TestBpmFillBars ───────────────────────────────────────────────────────────


class TestBpmFillBars:
    def _make(self, bpm_mean: float) -> ClusterStats:
        return ClusterStats(
            cluster_id=1,
            track_count=5,
            bpm_min=bpm_mean - 2,
            bpm_max=bpm_mean + 2,
            bpm_mean=bpm_mean,
            intensity_mean=0.5,
            hardness_mean=0.5,
            total_duration=1800.0,
        )

    def test_length_always_bar_width(self):
        for bpm in [60.0, 120.0, 170.0, 200.0]:
            assert len(self._make(bpm).bpm_fill_bars) == _BAR_WIDTH

    def test_200_bpm_is_full(self):
        assert self._make(200.0).bpm_fill_bars == "█" * _BAR_WIDTH

    def test_100_bpm_is_half(self):
        bars = self._make(100.0).bpm_fill_bars
        assert bars.count("█") == 5


# ── TestStr ───────────────────────────────────────────────────────────────────


class TestIntensityFraction:
    def _make(self, intensity: float) -> ClusterStats:
        return ClusterStats(
            cluster_id=1,
            track_count=5,
            bpm_min=120.0,
            bpm_max=125.0,
            bpm_mean=122.5,
            intensity_mean=intensity,
            hardness_mean=intensity,
            total_duration=1800.0,
        )

    def test_normal_value_unchanged(self):
        assert self._make(0.5).intensity_fraction() == pytest.approx(0.5)

    def test_clamped_below_zero(self):
        assert self._make(-0.5).intensity_fraction() == pytest.approx(0.0)

    def test_clamped_above_one(self):
        assert self._make(1.5).intensity_fraction() == pytest.approx(1.0)


class TestStr:
    def test_str_contains_cluster_label(self):
        stats = ClusterStats(
            cluster_id=3,
            track_count=15,
            bpm_min=120.0,
            bpm_max=128.0,
            bpm_mean=124.0,
            intensity_mean=0.6,
            hardness_mean=0.6,
            total_duration=5400.0,
        )
        result = str(stats)
        assert "Cluster 3" in result
        assert "120–128 BPM" in result
        assert "15 tracks" in result
        assert "1h 30m" in result
