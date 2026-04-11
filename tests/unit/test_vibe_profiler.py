"""Unit tests for vibe_profiler module."""

from pathlib import Path

import pytest

from playchitect.core.clustering import ClusterResult
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.naming.vibe_profiler import (
    VibeProfile,
    bucket_bpm,
    bucket_energy,
    compute_vibe_profile,
    score_salience,
)


class TestComputeVibeProfile:
    """Test compute_vibe_profile function."""

    def _make_cluster(
        self,
        tracks: list[Path],
        bpm_mean: float = 125.0,
        bpm_std: float = 2.0,
    ) -> ClusterResult:
        """Create a ClusterResult for testing."""
        return ClusterResult(
            cluster_id=0,
            tracks=tracks,
            bpm_mean=bpm_mean,
            bpm_std=bpm_std,
            track_count=len(tracks),
            total_duration=300.0,
        )

    def _make_features(
        self,
        filepath: Path,
        rms: float = 0.5,
        brightness: float = 0.5,
        percussiveness: float = 0.5,
        vocal_presence: float = 0.3,
        mood: str = "Ethereal",
    ) -> IntensityFeatures:
        """Create an IntensityFeatures with specified values."""
        features = IntensityFeatures(
            file_path=filepath,
            file_hash="hash123",
            rms_energy=rms,
            brightness=brightness,
            sub_bass_energy=0.3,
            kick_energy=0.4,
            bass_harmonics=0.3,
            percussiveness=percussiveness,
            onset_strength=0.5,
            camelot_key="8B",
            key_index=0.0,
            vocal_presence=vocal_presence,
        )
        features.mood_label = mood
        return features

    def test_compute_vibe_profile_returns_correct_mean_bpm(self) -> None:
        """compute_vibe_profile on a 3-track cluster returns correct mean_bpm."""
        # Create 3 test tracks
        tracks = [Path(f"/music/track{i}.mp3") for i in range(3)]

        # Create cluster with known BPM mean
        cluster = self._make_cluster(tracks, bpm_mean=125.0, bpm_std=1.5)

        # Create features for each track with different values
        features = {
            tracks[0]: self._make_features(
                tracks[0],
                rms=0.4,
                brightness=0.6,
                percussiveness=0.3,
                vocal_presence=0.2,
                mood="Dark",
            ),
            tracks[1]: self._make_features(
                tracks[1],
                rms=0.6,
                brightness=0.4,
                percussiveness=0.7,
                vocal_presence=0.4,
                mood="Aggressive",
            ),
            tracks[2]: self._make_features(
                tracks[2],
                rms=0.5,
                brightness=0.5,
                percussiveness=0.5,
                vocal_presence=0.3,
                mood="Ethereal",
            ),
        }

        profile = compute_vibe_profile(cluster, features)

        # Assert mean_bpm matches cluster's pre-computed mean
        assert profile.mean_bpm == 125.0

    def test_compute_vibe_profile_correct_feature_means(self) -> None:
        """Test that feature means are correctly averaged."""
        tracks = [Path(f"/music/track{i}.mp3") for i in range(3)]
        cluster = self._make_cluster(tracks, bpm_mean=120.0)

        features = {
            tracks[0]: self._make_features(
                tracks[0], rms=0.3, brightness=0.3, percussiveness=0.3, vocal_presence=0.1
            ),
            tracks[1]: self._make_features(
                tracks[1], rms=0.5, brightness=0.5, percussiveness=0.5, vocal_presence=0.3
            ),
            tracks[2]: self._make_features(
                tracks[2], rms=0.7, brightness=0.7, percussiveness=0.7, vocal_presence=0.5
            ),
        }

        profile = compute_vibe_profile(cluster, features)

        # Expected means: (0.3 + 0.5 + 0.7) / 3 = 0.5
        assert profile.mean_rms == pytest.approx(0.5, abs=1e-6)
        assert profile.mean_brightness == pytest.approx(0.5, abs=1e-6)
        assert profile.mean_percussiveness == pytest.approx(0.5, abs=1e-6)
        assert profile.mean_vocal_presence == pytest.approx(0.3, abs=1e-6)

    def test_compute_vibe_profile_dominant_mood(self) -> None:
        """Test that dominant_mood is the most frequent mood."""
        tracks = [Path(f"/music/track{i}.mp3") for i in range(4)]
        cluster = self._make_cluster(tracks)

        features = {
            tracks[0]: self._make_features(tracks[0], mood="Dark"),
            tracks[1]: self._make_features(tracks[1], mood="Dark"),
            tracks[2]: self._make_features(tracks[2], mood="Aggressive"),
            tracks[3]: self._make_features(tracks[3], mood="Ethereal"),
        }

        profile = compute_vibe_profile(cluster, features)

        # Dark appears twice, others once → Dark should be dominant
        assert profile.dominant_mood == "Dark"
        assert profile.mood_distribution["Dark"] == 0.5
        assert profile.mood_distribution["Aggressive"] == 0.25
        assert profile.mood_distribution["Ethereal"] == 0.25

    def test_compute_vibe_profile_empty_cluster_raises(self) -> None:
        """Empty cluster should raise ValueError."""
        cluster = self._make_cluster([])
        features: dict[Path, IntensityFeatures] = {}

        with pytest.raises(ValueError, match="empty cluster"):
            compute_vibe_profile(cluster, features)

    def test_compute_vibe_profile_missing_features_raises(self) -> None:
        """Cluster with no matching features should raise ValueError."""
        tracks = [Path("/music/track1.mp3")]
        cluster = self._make_cluster(tracks)

        features: dict[Path, IntensityFeatures] = {}

        with pytest.raises(ValueError, match="No intensity features"):
            compute_vibe_profile(cluster, features)


class TestScoreSalience:
    """Test score_salience function."""

    def _make_profile(
        self,
        cluster_id: int = 0,
        mean_bpm: float = 125.0,
        mean_rms: float = 0.5,
        mean_brightness: float = 0.5,
        mean_percussiveness: float = 0.5,
        mean_vocal_presence: float = 0.3,
    ) -> VibeProfile:
        """Create a VibeProfile with specified values."""
        return VibeProfile(
            cluster_id=cluster_id,
            mean_bpm=mean_bpm,
            mean_rms=mean_rms,
            mean_brightness=mean_brightness,
            mean_percussiveness=mean_percussiveness,
            mean_vocal_presence=mean_vocal_presence,
            dominant_mood="Ethereal",
            mood_distribution={"Ethereal": 1.0},
        )

    def test_score_salience_identical_profiles_empty(self) -> None:
        """score_salience returns empty dict when all profiles are identical."""
        # Create 3 identical profiles
        profiles = [
            self._make_profile(cluster_id=0, mean_bpm=125.0, mean_rms=0.5),
            self._make_profile(cluster_id=1, mean_bpm=125.0, mean_rms=0.5),
            self._make_profile(cluster_id=2, mean_bpm=125.0, mean_rms=0.5),
        ]

        # Score salience of first profile against all (identical)
        result = score_salience(profiles[0], profiles)

        # With identical profiles, std=0, so all features should be skipped
        assert result == {}

    def test_score_salience_high_zscore_detected(self) -> None:
        """Features with abs(z) > 1.5 should be included."""
        # Library of profiles with mean_bpm around 125
        library = [
            self._make_profile(cluster_id=0, mean_bpm=125.0, mean_rms=0.5),
            self._make_profile(cluster_id=1, mean_bpm=126.0, mean_rms=0.5),
            self._make_profile(cluster_id=2, mean_bpm=124.0, mean_rms=0.5),
        ]

        # Profile with very high BPM (z-score > 1.5)
        profile = self._make_profile(cluster_id=3, mean_bpm=135.0, mean_rms=0.5)

        result = score_salience(profile, library)

        # mean_bpm should be salient (z-score > 1.5)
        assert "mean_bpm" in result
        assert abs(result["mean_bpm"]) > 1.5

    def test_score_salience_low_zscore_excluded(self) -> None:
        """Features with abs(z) <= 1.5 should be excluded."""
        library = [
            self._make_profile(cluster_id=0, mean_bpm=120.0, mean_rms=0.5),
            self._make_profile(cluster_id=1, mean_bpm=125.0, mean_rms=0.5),
            self._make_profile(cluster_id=2, mean_bpm=130.0, mean_rms=0.5),
        ]

        # Profile with BPM near mean (z-score < 1.5)
        profile = self._make_profile(cluster_id=3, mean_bpm=125.0, mean_rms=0.5)

        result = score_salience(profile, library)

        # mean_bpm should NOT be salient (z-score ~ 0)
        assert "mean_bpm" not in result

    def test_score_salience_empty_library_raises(self) -> None:
        """Empty library should raise ValueError."""
        profile = self._make_profile()

        with pytest.raises(ValueError, match="empty library"):
            score_salience(profile, [])

    def test_score_salience_positive_and_negative_zscores(self) -> None:
        """Test both positive and negative z-scores are detected."""
        library = [
            self._make_profile(cluster_id=0, mean_rms=0.5),
            self._make_profile(cluster_id=1, mean_rms=0.6),
            self._make_profile(cluster_id=2, mean_rms=0.4),
        ]

        # Very low RMS (negative z-score < -1.5)
        low_profile = self._make_profile(cluster_id=3, mean_rms=0.1)
        result_low = score_salience(low_profile, library)
        assert "mean_rms" in result_low
        assert result_low["mean_rms"] < -1.5

        # Very high RMS (positive z-score > 1.5)
        high_profile = self._make_profile(cluster_id=4, mean_rms=0.9)
        result_high = score_salience(high_profile, library)
        assert "mean_rms" in result_high
        assert result_high["mean_rms"] > 1.5


class TestBucketBpm:
    """Test bucket_bpm function."""

    def test_bucket_bpm_slow(self) -> None:
        """BPM < 100 should return 'Slow'."""
        assert bucket_bpm(99) == "Slow"
        assert bucket_bpm(95) == "Slow"
        assert bucket_bpm(50) == "Slow"

    def test_bucket_bpm_mid_tempo(self) -> None:
        """BPM 100-119 should return 'Mid-Tempo'."""
        assert bucket_bpm(100) == "Mid-Tempo"
        assert bucket_bpm(110) == "Mid-Tempo"
        assert bucket_bpm(119) == "Mid-Tempo"

    def test_bucket_bpm_peak_hour(self) -> None:
        """BPM 120-135 should return 'Peak Hour'."""
        assert bucket_bpm(120) == "Peak Hour"
        assert bucket_bpm(125) == "Peak Hour"
        assert bucket_bpm(135) == "Peak Hour"

    def test_bucket_bpm_high_energy(self) -> None:
        """BPM > 135 should return 'High Energy'."""
        assert bucket_bpm(136) == "High Energy"
        assert bucket_bpm(140) == "High Energy"
        assert bucket_bpm(170) == "High Energy"


class TestBucketEnergy:
    """Test bucket_energy function."""

    def test_bucket_energy_subtle(self) -> None:
        """RMS < 0.3 should return 'Subtle'."""
        assert bucket_energy(0.29) == "Subtle"
        assert bucket_energy(0.1) == "Subtle"
        assert bucket_energy(0.0) == "Subtle"

    def test_bucket_energy_groovy(self) -> None:
        """RMS 0.3-0.49 should return 'Groovy'."""
        assert bucket_energy(0.3) == "Groovy"
        assert bucket_energy(0.4) == "Groovy"
        assert bucket_energy(0.49) == "Groovy"

    def test_bucket_energy_energetic(self) -> None:
        """RMS 0.5-0.7 should return 'Energetic'."""
        assert bucket_energy(0.5) == "Energetic"
        assert bucket_energy(0.6) == "Energetic"
        assert bucket_energy(0.7) == "Energetic"

    def test_bucket_energy_intense(self) -> None:
        """RMS > 0.7 should return 'Intense'."""
        assert bucket_energy(0.71) == "Intense"
        assert bucket_energy(0.8) == "Intense"
        assert bucket_energy(1.0) == "Intense"
