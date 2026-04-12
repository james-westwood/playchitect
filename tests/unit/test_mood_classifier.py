"""Unit tests for mood_classifier module."""

from pathlib import Path

import pytest

from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.mood_classifier import classify_mood


class TestMoodClassifier:
    """Test mood classification based on intensity features."""

    def _make_features(
        self,
        rms_energy: float = 0.5,
        brightness: float = 0.5,
        percussiveness: float = 0.5,
        onset_strength: float = 0.5,
        vocal_presence: float = 0.0,
        dynamic_range: float = 0.0,
    ) -> IntensityFeatures:
        """Create an IntensityFeatures with specified values for mood testing."""
        return IntensityFeatures(
            file_path=Path("test.mp3"),
            file_hash="abc123",
            rms_energy=rms_energy,
            brightness=brightness,
            sub_bass_energy=0.3,
            kick_energy=0.4,
            bass_harmonics=0.3,
            percussiveness=percussiveness,
            onset_strength=onset_strength,
            camelot_key="8B",
            key_index=0.0,
            dynamic_range=dynamic_range,
            vocal_presence=vocal_presence,
        )

    def test_aggressive_high_percussive_high_onset(self) -> None:
        """High percussiveness (>0.7) + high onset strength (>0.6) should return 'Aggressive'."""
        features = self._make_features(
            percussiveness=0.75,
            onset_strength=0.65,
        )
        mood = classify_mood(features)
        assert mood == "Aggressive"

    def test_dark_low_brightness_low_vocal(self) -> None:
        """Low brightness (<0.3) + low vocal presence (<0.3) should return 'Dark'."""
        features = self._make_features(
            brightness=0.2,
            vocal_presence=0.2,
        )
        mood = classify_mood(features)
        assert mood == "Dark"

    def test_euphoric_high_energy_high_brightness(self) -> None:
        """High energy (>0.7) + high brightness (>0.6) should return 'Euphoric'."""
        features = self._make_features(
            rms_energy=0.75,
            brightness=0.65,
        )
        mood = classify_mood(features)
        assert mood == "Euphoric"

    def test_melancholic_low_energy_high_vocal(self) -> None:
        """Low energy (<0.3) + high vocal presence (>0.5) should return 'Melancholic'."""
        features = self._make_features(
            rms_energy=0.2,
            vocal_presence=0.6,
        )
        mood = classify_mood(features)
        assert mood == "Melancholic"

    def test_dreamy_low_percussive_high_brightness(self) -> None:
        """Low percussiveness (<0.3) + high brightness (>0.5) should return 'Dreamy'."""
        features = self._make_features(
            percussiveness=0.2,
            brightness=0.6,
        )
        mood = classify_mood(features)
        assert mood == "Dreamy"

    def test_hypnotic_mid_energy_low_dynamic_range(self) -> None:
        """Mid energy (0.3-0.6) + low dynamic range (<0.3) should return 'Hypnotic'."""
        features = self._make_features(
            rms_energy=0.45,
            dynamic_range=0.2,
        )
        mood = classify_mood(features)
        assert mood == "Hypnotic"

    def test_groovy_mid_high_percussive_mid_energy(self) -> None:
        """Mid-high percussiveness (>0.5) + mid energy (0.4-0.7) should return 'Groovy'."""
        # Use energy outside Hypnotic range (>0.6) to avoid Hypnotic taking precedence
        features = self._make_features(
            percussiveness=0.6,
            rms_energy=0.65,  # Above Hypnotic range (0.3-0.6), in Groovy range (0.4-0.7)
            dynamic_range=0.5,  # High enough to avoid Hypnotic
        )
        mood = classify_mood(features)
        assert mood == "Groovy"

    def test_ethereal_default_low_everything(self) -> None:
        """Default low-everything values should return 'Ethereal' (catch-all)."""
        # Need to avoid Dark condition (brightness < 0.3 AND vocal_presence < 0.3)
        features = self._make_features(
            rms_energy=0.1,
            brightness=0.4,  # Above Dark threshold to avoid Dark classification
            percussiveness=0.1,
            onset_strength=0.1,
            vocal_presence=0.1,
            dynamic_range=0.5,
        )
        mood = classify_mood(features)
        assert mood == "Ethereal"

    def test_ethereal_borderline_values(self) -> None:
        """Borderline values that don't match any specific mood should return 'Ethereal'."""
        features = self._make_features(
            rms_energy=0.35,  # Too high for Melancholic, too low for Euphoric/Groovy
            brightness=0.4,  # Too high for Dark, too low for Dreamy/Euphoric
            percussiveness=0.4,  # Too high for Dreamy, too low for Aggressive/Groovy
            onset_strength=0.4,  # Too low for Aggressive
            vocal_presence=0.4,  # Too low for Melancholic, too high for Dark
            dynamic_range=0.5,  # Too high for Hypnotic
        )
        mood = classify_mood(features)
        assert mood == "Ethereal"

    def test_backwards_compat_missing_vocal_presence(self) -> None:
        """Missing vocal_presence field should be treated as 0.0."""
        features = IntensityFeatures(
            file_path=Path("test.mp3"),
            file_hash="abc123",
            rms_energy=0.2,
            brightness=0.2,
            sub_bass_energy=0.3,
            kick_energy=0.4,
            bass_harmonics=0.3,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="8B",
            key_index=0.0,
        )
        # vocal_presence not set, should default to 0.0 via getattr
        mood = classify_mood(features)
        # With low brightness (0.2) and default vocal_presence (0.0), should be Dark
        assert mood == "Dark"

    def test_backwards_compat_missing_dynamic_range(self) -> None:
        """Missing dynamic_range field should be treated as 0.0."""
        features = IntensityFeatures(
            file_path=Path("test.mp3"),
            file_hash="abc123",
            rms_energy=0.45,
            brightness=0.5,
            sub_bass_energy=0.3,
            kick_energy=0.4,
            bass_harmonics=0.3,
            percussiveness=0.5,
            onset_strength=0.5,
            camelot_key="8B",
            key_index=0.0,
        )
        # dynamic_range not set, should default to 0.0 via getattr
        mood = classify_mood(features)
        # With mid energy (0.45) and default dynamic_range (0.0), should be Hypnotic
        assert mood == "Hypnotic"

    def test_dark_priority_over_hypnotic(self) -> None:
        """Dark should take priority over Hypnotic when both conditions are met."""
        features = self._make_features(
            brightness=0.2,  # Dark threshold
            vocal_presence=0.2,  # Dark threshold
            rms_energy=0.45,  # Hypnotic energy range
            dynamic_range=0.2,  # Hypnotic threshold
        )
        mood = classify_mood(features)
        # Dark has higher priority than Hypnotic
        assert mood == "Dark"

    def test_euphoric_priority_over_groovy(self) -> None:
        """Euphoric should take priority over Groovy when both conditions are met."""
        features = self._make_features(
            rms_energy=0.75,  # Euphoric threshold, also in Groovy range
            brightness=0.65,  # Euphoric threshold
            percussiveness=0.6,  # Groovy threshold
        )
        mood = classify_mood(features)
        # Euphoric has higher priority than Groovy
        assert mood == "Euphoric"

    def test_melancholic_priority_over_ethereal(self) -> None:
        """Melancholic should take priority over Ethereal when conditions are met."""
        features = self._make_features(
            rms_energy=0.2,  # Low energy
            vocal_presence=0.6,  # High vocal
            brightness=0.1,
            percussiveness=0.1,
        )
        mood = classify_mood(features)
        assert mood == "Melancholic"

    def test_exact_threshold_values(self) -> None:
        """Test exact threshold values for boundary conditions."""
        # Exactly at Dark threshold
        features = self._make_features(brightness=0.3, vocal_presence=0.3)
        mood = classify_mood(features)
        # At exact threshold, conditions are not < 0.3, so not Dark
        assert mood != "Dark"

        # Exactly at Euphoric threshold
        features = self._make_features(rms_energy=0.7, brightness=0.6)
        mood = classify_mood(features)
        # At exact threshold, conditions are not > 0.7/0.6, so not Euphoric
        assert mood != "Euphoric"

    def test_all_mood_labels_defined(self) -> None:
        """Verify all expected mood labels can be returned."""
        test_cases = [
            ("Dark", self._make_features(brightness=0.2, vocal_presence=0.2)),
            ("Euphoric", self._make_features(rms_energy=0.75, brightness=0.65)),
            ("Melancholic", self._make_features(rms_energy=0.2, vocal_presence=0.6)),
            ("Aggressive", self._make_features(percussiveness=0.75, onset_strength=0.65)),
            ("Dreamy", self._make_features(percussiveness=0.2, brightness=0.6)),
            ("Hypnotic", self._make_features(rms_energy=0.45, dynamic_range=0.2)),
            # Groovy: needs percussive > 0.5 AND energy in 0.4-0.7
            # BUT energy outside Hypnotic range (0.3-0.6)
            ("Groovy", self._make_features(percussiveness=0.6, rms_energy=0.65, dynamic_range=0.5)),
            # Ethereal: avoid Dark by having brightness >= 0.3 OR vocal >= 0.3
            ("Ethereal", self._make_features(rms_energy=0.1, brightness=0.4, percussiveness=0.1)),
        ]

        for expected_mood, features in test_cases:
            actual_mood = classify_mood(features)
            assert actual_mood == expected_mood, f"Expected {expected_mood}, got {actual_mood}"


class TestElectronicMusicMoodClassification:
    """Tests for electronic music mood classification (techno/house/dnb).

    These tests verify the fix for BUG-06 where electronic/techno tracks
    were incorrectly classified as 'Ethereal' due to miscalibrated thresholds.
    """

    @pytest.mark.parametrize(
        "rms_energy,brightness,percussiveness,onset_strength,expected_mood",
        [
            pytest.param(
                0.15,
                0.6,
                0.6,
                0.4,
                "Energetic",
                id="techno-high-energy",
            ),
            pytest.param(
                0.12,
                0.55,
                0.51,
                0.3,
                "Energetic",
                id="house-typical",
            ),
            pytest.param(
                0.4,
                0.7,
                0.8,
                0.7,
                "Aggressive",
                id="dnb-aggressive",
            ),
            pytest.param(
                0.2,
                0.65,
                0.55,
                0.35,
                "Energetic",
                id="techno-bright",
            ),
            pytest.param(
                0.18,
                0.7,
                0.52,
                0.3,
                "Energetic",
                id="house-bright",
            ),
        ],
    )
    def test_electronic_music_not_ethereal(
        self,
        rms_energy: float,
        brightness: float,
        percussiveness: float,
        onset_strength: float,
        expected_mood: str,
    ) -> None:
        """High RMS + high percussiveness + high brightness should NOT be Ethereal."""
        features = IntensityFeatures(
            file_path=Path("test.mp3"),
            file_hash="abc123",
            rms_energy=rms_energy,
            brightness=brightness,
            sub_bass_energy=0.3,
            kick_energy=0.4,
            bass_harmonics=0.3,
            percussiveness=percussiveness,
            onset_strength=onset_strength,
            camelot_key="8B",
            key_index=0.0,
            dynamic_range=0.4,
            vocal_presence=0.1,
        )
        mood = classify_mood(features)
        assert mood != "Ethereal", (
            f"Electronic music with rms={rms_energy}, brightness={brightness}, "
            f"percussiveness={percussiveness} should not be Ethereal"
        )
        assert mood == expected_mood, f"Expected {expected_mood}, got {mood}"

    def test_acceptance_criteria_high_rms_percussive_bright_not_ethereal(
        self,
    ) -> None:
        """Acceptance criteria: high RMS (>0.1) + high percussiveness (>0.5) + high brightness."""
        features = IntensityFeatures(
            file_path=Path("test.mp3"),
            file_hash="abc123",
            rms_energy=0.15,
            brightness=0.6,
            sub_bass_energy=0.3,
            kick_energy=0.4,
            bass_harmonics=0.3,
            percussiveness=0.6,
            onset_strength=0.4,
            camelot_key="8B",
            key_index=0.0,
        )
        mood = classify_mood(features)
        assert mood in ("Energetic", "Aggressive"), f"Expected Energetic or Aggressive, got {mood}"
