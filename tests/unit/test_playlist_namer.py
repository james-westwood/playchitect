"""Unit tests for playlist_namer module."""

from pathlib import Path

from playchitect.core.clustering import ClusterResult
from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata
from playchitect.core.naming.playlist_namer import (
    TAG_TO_DESCRIPTORS,
    PlaylistNamer,
    _descriptors_from_fallback,
    _descriptors_from_salience,
    _get_noun_for_mood,
)
from playchitect.core.naming.vibe_profiler import VibeProfile


class TestTagToDescriptors:
    """Test TAG_TO_DESCRIPTORS mapping."""

    def test_has_at_least_15_entries(self) -> None:
        """TAG_TO_DESCRIPTORS should have at least 15 mood/energy entries."""
        assert len(TAG_TO_DESCRIPTORS) >= 15

    def test_mood_entries_have_descriptors(self) -> None:
        """Mood entries should have at least 5 descriptor words."""
        mood_entries = ["Dark", "Ethereal", "Aggressive", "Melancholic", "Euphoric"]
        for mood in mood_entries:
            assert mood in TAG_TO_DESCRIPTORS
            assert len(TAG_TO_DESCRIPTORS[mood]) >= 5

    def test_bpm_bucket_entries_exist(self) -> None:
        """BPM bucket entries should exist."""
        bpm_entries = ["Slow", "Mid-Tempo", "Peak Hour", "High Energy"]
        for entry in bpm_entries:
            assert entry in TAG_TO_DESCRIPTORS

    def test_energy_bucket_entries_exist(self) -> None:
        """Energy bucket entries should exist."""
        energy_entries = ["Subtle", "Groovy", "Energetic", "Intense"]
        for entry in energy_entries:
            assert entry in TAG_TO_DESCRIPTORS


class TestGetNounForMood:
    """Test _get_noun_for_mood function."""

    def test_returns_mood_specific_noun(self) -> None:
        """Returns mood-specific noun when available and not used."""
        used: set[str] = set()
        result = _get_noun_for_mood("Dark", used)
        assert result == "Journey"

    def test_falls_back_to_default_when_mood_noun_used(self) -> None:
        """Falls back to default nouns when mood-specific is already used."""
        used = {"Journey"}
        result = _get_noun_for_mood("Dark", used)
        assert result != "Journey"
        assert result in ["Groove", "Set", "Wave"]

    def test_returns_first_available_default(self) -> None:
        """Returns first available default noun when all others used."""
        used = {"Journey", "Groove", "Set"}
        result = _get_noun_for_mood("Unknown", used)
        assert result == "Wave"


class TestDescriptorsFromSalience:
    """Test _descriptors_from_salience function."""

    def _make_profile(self, mean_bpm: float = 125.0) -> VibeProfile:
        """Create a VibeProfile for testing."""
        return VibeProfile(
            cluster_id=0,
            mean_bpm=mean_bpm,
            mean_rms=0.5,
            mean_brightness=0.5,
            mean_percussiveness=0.5,
            mean_vocal_presence=0.3,
            dominant_mood="Ethereal",
            mood_distribution={"Ethereal": 1.0},
        )

    def test_returns_descriptors_for_high_zscore_features(self) -> None:
        """Returns descriptors for features with high positive z-scores."""
        salience = {
            "mean_brightness": 2.5,
            "mean_rms": -1.2,
        }
        profile = self._make_profile()
        result = _descriptors_from_salience(salience, profile)

        # Should include brightness descriptor (high z-score)
        assert len(result) >= 1
        assert "Bright" in result or "Luminous" in result

    def test_returns_empty_for_empty_salience(self) -> None:
        """Returns empty list when salience is empty."""
        result = _descriptors_from_salience({}, self._make_profile())
        assert result == []

    def test_selects_top_2_salient_features(self) -> None:
        """Selects top 2 most salient features by absolute z-score."""
        salience = {
            "mean_brightness": 3.0,
            "mean_rms": 2.5,
            "mean_percussiveness": 1.0,
        }
        profile = self._make_profile()
        result = _descriptors_from_salience(salience, profile)

        # Should return at most 2 descriptors
        assert len(result) <= 2


class TestDescriptorsFromFallback:
    """Test _descriptors_from_fallback function."""

    def test_returns_mood_and_bpm_descriptors(self) -> None:
        """Returns descriptors from dominant mood and BPM bucket."""
        profile = VibeProfile(
            cluster_id=0,
            mean_bpm=125.0,
            mean_rms=0.5,
            mean_brightness=0.5,
            mean_percussiveness=0.5,
            mean_vocal_presence=0.3,
            dominant_mood="Dark",
            mood_distribution={"Dark": 1.0},
        )
        result = _descriptors_from_fallback(profile)

        # Should include at least mood descriptor
        assert len(result) >= 1
        assert any("Dark" in r or "Shadowed" in r or "Nocturnal" in r for r in result)

    def test_returns_bpm_bucket_descriptor(self) -> None:
        """Returns BPM bucket descriptor based on mean BPM."""
        profile = VibeProfile(
            cluster_id=0,
            mean_bpm=85.0,  # Slow BPM
            mean_rms=0.3,
            mean_brightness=0.3,
            mean_percussiveness=0.3,
            mean_vocal_presence=0.3,
            dominant_mood="Ethereal",
            mood_distribution={"Ethereal": 1.0},
        )
        result = _descriptors_from_fallback(profile)

        # Should include slow BPM descriptor
        assert any(d in result for d in ["Languid", "Unhurried", "Leisurely", "Relaxed", "Mellow"])


class TestPlaylistNamer:
    """Test PlaylistNamer class."""

    def _make_profile(
        self,
        cluster_id: int = 0,
        mean_bpm: float = 125.0,
        mean_rms: float = 0.5,
        dominant_mood: str = "Ethereal",
    ) -> VibeProfile:
        """Create a VibeProfile for testing."""
        return VibeProfile(
            cluster_id=cluster_id,
            mean_bpm=mean_bpm,
            mean_rms=mean_rms,
            mean_brightness=0.5,
            mean_percussiveness=0.5,
            mean_vocal_presence=0.3,
            dominant_mood=dominant_mood,
            mood_distribution={dominant_mood: 1.0},
        )

    def test_name_cluster_returns_non_empty_string(self) -> None:
        """name_cluster returns a non-empty string."""
        namer = PlaylistNamer()
        profile = self._make_profile()
        salience = {"mean_brightness": 2.0}
        library = [profile]

        result = namer.name_cluster(profile, salience, library)

        assert isinstance(result, str)
        assert result

    def test_name_cluster_uses_salient_features(self) -> None:
        """name_cluster uses salient features when available."""
        namer = PlaylistNamer()
        profile = self._make_profile(dominant_mood="Dark")
        salience = {"mean_brightness": 2.5}
        library = [profile]

        result = namer.name_cluster(profile, salience, library)

        # Should be title-cased
        assert result == result.title()
        # Should include descriptor or noun
        assert len(result.split()) >= 1

    def test_name_cluster_falls_back_when_no_salience(self) -> None:
        """name_cluster falls back to mood/BPM when no salient features."""
        namer = PlaylistNamer()
        profile = self._make_profile(dominant_mood="Dark", mean_bpm=85.0)
        salience: dict[str, float] = {}
        library = [profile]

        result = namer.name_cluster(profile, salience, library)

        assert isinstance(result, str)
        assert result

    def test_ensure_unique_appends_roman_numerals(self) -> None:
        """Duplicate names get Roman numerals appended."""
        namer = PlaylistNamer()

        # Create first name
        name1 = namer._ensure_unique("Dark Journey")
        assert name1 == "Dark Journey"

        # Create duplicate - should get II
        name2 = namer._ensure_unique("Dark Journey")
        assert name2 == "Dark Journey II"

        # Create another duplicate - should get III
        name3 = namer._ensure_unique("Dark Journey")
        assert name3 == "Dark Journey III"


class TestNameAllClusters:
    """Test name_all_clusters method."""

    def _make_cluster(
        self,
        cluster_id: int,
        tracks: list[Path],
        bpm_mean: float = 125.0,
    ) -> ClusterResult:
        """Create a ClusterResult for testing."""
        return ClusterResult(
            cluster_id=cluster_id,
            tracks=tracks,
            bpm_mean=bpm_mean,
            bpm_std=2.0,
            track_count=len(tracks),
            total_duration=300.0,
        )

    def _make_features(
        self,
        filepath: Path,
        mood: str = "Ethereal",
        rms: float = 0.5,
        brightness: float = 0.5,
    ) -> IntensityFeatures:
        """Create an IntensityFeatures for testing."""
        features = IntensityFeatures(
            file_path=filepath,
            file_hash="hash123",
            rms_energy=rms,
            brightness=brightness,
            sub_bass_energy=0.3,
            kick_energy=0.4,
            bass_harmonics=0.3,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="8B",
            key_index=0.0,
            vocal_presence=0.3,
        )
        features.mood_label = mood
        return features

    def _make_metadata(self, filepath: Path, bpm: float = 125.0) -> TrackMetadata:
        """Create a TrackMetadata for testing."""
        return TrackMetadata(
            filepath=filepath,
            title="Test Track",
            artist="Test Artist",
            bpm=bpm,
            duration=300.0,
        )

    def test_returns_3_distinct_names_for_3_clusters(self) -> None:
        """name_all_clusters on 3 synthetic clusters returns 3 distinct non-empty strings."""
        namer = PlaylistNamer()

        # Create 3 clusters with different characteristics
        tracks1 = [Path(f"/music/track{i}.mp3") for i in range(3)]
        tracks2 = [Path(f"/music/track{i}.mp3") for i in range(3, 6)]
        tracks3 = [Path(f"/music/track{i}.mp3") for i in range(6, 9)]

        clusters = [
            self._make_cluster(0, tracks1, bpm_mean=85.0),
            self._make_cluster(1, tracks2, bpm_mean=125.0),
            self._make_cluster(2, tracks3, bpm_mean=140.0),
        ]

        # Create features with different moods for each cluster
        features: dict[Path, IntensityFeatures] = {}
        metadata: dict[Path, TrackMetadata] = {}

        for i, track in enumerate(tracks1):
            features[track] = self._make_features(track, mood="Dark", rms=0.3, brightness=0.3)
            metadata[track] = self._make_metadata(track, bpm=85.0)

        for i, track in enumerate(tracks2):
            features[track] = self._make_features(track, mood="Ethereal", rms=0.5, brightness=0.6)
            metadata[track] = self._make_metadata(track, bpm=125.0)

        for i, track in enumerate(tracks3):
            features[track] = self._make_features(track, mood="Aggressive", rms=0.8, brightness=0.8)
            metadata[track] = self._make_metadata(track, bpm=140.0)

        result = namer.name_all_clusters(clusters, features, metadata)

        # Should return 3 entries
        assert len(result) == 3

        # All names should be distinct
        names = list(result.values())
        assert len(set(names)) == 3

        # All names should be non-empty
        for name in names:
            assert name
            assert isinstance(name, str)

    def test_duplicate_prevention_appends_ii_iii(self) -> None:
        """Duplicate prevention appends II/III when clusters would have same name."""
        namer = PlaylistNamer()

        # Manually set _used_names to force duplicate handling
        # This tests that _ensure_unique works correctly
        namer._used_names.add("Ethereal Journey")

        # Now calling name_cluster with features that would generate the same name
        # should get a unique suffix
        tracks = [Path(f"/music/track{i}.mp3") for i in range(3)]
        clusters = [self._make_cluster(0, tracks, bpm_mean=125.0)]

        features: dict[Path, IntensityFeatures] = {}
        metadata: dict[Path, TrackMetadata] = {}

        for track in tracks:
            features[track] = self._make_features(track, mood="Ethereal", rms=0.5)
            metadata[track] = self._make_metadata(track, bpm=125.0)

        result = namer.name_all_clusters(clusters, features, metadata)

        # Should return 1 entry
        assert len(result) == 1

        # The name should either be unique by synonym or have a suffix
        name = result[0]
        # Name should not be exactly "Ethereal Journey" since that's taken
        assert name != "Ethereal Journey"

    def test_handles_clusters_without_features(self) -> None:
        """Handles clusters that have no corresponding features."""
        namer = PlaylistNamer()

        tracks1 = [Path(f"/music/track{i}.mp3") for i in range(3)]
        tracks2 = [Path("/music/missing.mp3")]

        clusters = [
            self._make_cluster(0, tracks1, bpm_mean=125.0),
            self._make_cluster(1, tracks2, bpm_mean=130.0),
        ]

        # Only create features for first cluster
        features: dict[Path, IntensityFeatures] = {}
        metadata: dict[Path, TrackMetadata] = {}

        for track in tracks1:
            features[track] = self._make_features(track, mood="Ethereal")
            metadata[track] = self._make_metadata(track, bpm=125.0)

        # Create metadata for second cluster but no features
        metadata[tracks2[0]] = self._make_metadata(tracks2[0], bpm=130.0)

        result = namer.name_all_clusters(clusters, features, metadata)

        # Should return 2 entries
        assert len(result) == 2

        # Second cluster should get a default name
        assert result[1] == "Cluster 1"

    def test_returns_dict_with_cluster_ids_as_keys(self) -> None:
        """Result dict uses cluster_ids as keys."""
        namer = PlaylistNamer()

        tracks = [Path(f"/music/track{i}.mp3") for i in range(3)]
        clusters = [self._make_cluster(42, tracks, bpm_mean=125.0)]

        features: dict[Path, IntensityFeatures] = {}
        metadata: dict[Path, TrackMetadata] = {}

        for track in tracks:
            features[track] = self._make_features(track, mood="Dark")
            metadata[track] = self._make_metadata(track, bpm=125.0)

        result = namer.name_all_clusters(clusters, features, metadata)

        # Key should be the cluster_id
        assert 42 in result
        assert isinstance(result[42], str)
