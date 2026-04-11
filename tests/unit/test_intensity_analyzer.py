"""
Unit tests for intensity_analyzer module.
"""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from playchitect.core.intensity_analyzer import IntensityAnalyzer, IntensityFeatures


class TestIntensityFeatures:
    """Test IntensityFeatures dataclass."""

    def test_feature_creation(self) -> None:
        """Test creating IntensityFeatures."""
        features = IntensityFeatures(
            file_path=Path("test.mp3"),
            file_hash="abc123",
            rms_energy=0.5,
            brightness=0.6,
            sub_bass_energy=0.3,
            kick_energy=0.7,
            bass_harmonics=0.4,
            percussiveness=0.8,
            onset_strength=0.65,
            camelot_key="8B",
            key_index=0.0,
        )

        assert features.rms_energy == 0.5
        assert features.brightness == 0.6
        assert features.kick_energy == 0.7
        assert features.camelot_key == "8B"
        assert features.key_index == 0.0

    def test_to_feature_vector(self) -> None:
        """Test conversion to feature vector."""
        features = IntensityFeatures(
            file_path=Path("test.mp3"),
            file_hash="abc123",
            rms_energy=0.5,
            brightness=0.6,
            sub_bass_energy=0.3,
            kick_energy=0.7,
            bass_harmonics=0.4,
            percussiveness=0.8,
            onset_strength=0.65,
            camelot_key="8B",
            key_index=0.0,
        )

        vector = features.to_feature_vector()

        assert isinstance(vector, np.ndarray)
        assert len(vector) == 7
        assert vector[0] == 0.5  # rms
        assert vector[1] == 0.6  # brightness
        assert vector[2] == 0.3  # sub_bass
        assert vector[3] == 0.7  # kick
        assert vector[4] == 0.4  # harmonics
        assert vector[5] == 0.8  # percussiveness
        assert vector[6] == 0.65  # onset

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        features = IntensityFeatures(
            file_path=Path("test.mp3"),
            file_hash="abc123",
            rms_energy=0.5,
            brightness=0.6,
            sub_bass_energy=0.3,
            kick_energy=0.7,
            bass_harmonics=0.4,
            percussiveness=0.8,
            onset_strength=0.65,
            camelot_key="8B",
            key_index=0.0,
        )

        data = features.to_dict()

        assert isinstance(data, dict)
        assert data["filepath"] == "test.mp3"
        assert data["rms_energy"] == 0.5
        assert data["camelot_key"] == "8B"
        assert data["key_index"] == 0.0

    def test_from_dict(self) -> None:
        """Test creating from dictionary."""
        data = {
            "filepath": "test.mp3",
            "file_hash": "abc123",
            "rms_energy": 0.5,
            "brightness": 0.6,
            "sub_bass_energy": 0.3,
            "kick_energy": 0.7,
            "bass_harmonics": 0.4,
            "percussiveness": 0.8,
            "onset_strength": 0.65,
            "camelot_key": "3B",
            "key_index": 1.0,
        }

        features = IntensityFeatures.from_dict(data)

        assert features.rms_energy == 0.5
        assert features.filepath == Path("test.mp3")
        assert features.camelot_key == "3B"
        assert features.key_index == 1.0

    def test_from_dict_backward_compat(self) -> None:
        """Test creating from old dictionary without harmonic fields."""
        data = {
            "filepath": "test.mp3",
            "file_hash": "abc123",
            "rms_energy": 0.5,
            "brightness": 0.6,
            "sub_bass_energy": 0.3,
            "kick_energy": 0.7,
            "bass_harmonics": 0.4,
            "percussiveness": 0.8,
            "onset_strength": 0.65,
        }

        features = IntensityFeatures.from_dict(data)

        # Should default to 8B (C major)
        assert features.camelot_key == "8B"
        assert features.key_index == 0.0

    def test_from_dict_energy_flow_backward_compat(self) -> None:
        """Test creating from old dictionary without energy flow fields."""
        data = {
            "filepath": "test.mp3",
            "file_hash": "abc123",
            "rms_energy": 0.5,
            "brightness": 0.6,
            "sub_bass_energy": 0.3,
            "kick_energy": 0.7,
            "bass_harmonics": 0.4,
            "percussiveness": 0.8,
            "onset_strength": 0.65,
            "camelot_key": "8B",
            "key_index": 0.0,
        }

        features = IntensityFeatures.from_dict(data)

        # Should default to 0.0
        assert features.dynamic_range == 0.0
        assert features.energy_gradient == 0.0

    def test_from_dict_timbre_texture_backward_compat(self) -> None:
        """Test creating from old dictionary without timbre/texture fields."""
        data = {
            "filepath": "test.mp3",
            "file_hash": "abc123",
            "rms_energy": 0.5,
            "brightness": 0.6,
            "sub_bass_energy": 0.3,
            "kick_energy": 0.7,
            "bass_harmonics": 0.4,
            "percussiveness": 0.8,
            "onset_strength": 0.65,
            "camelot_key": "8B",
            "key_index": 0.0,
            "dynamic_range": 0.5,
            "energy_gradient": 0.1,
            "drop_density": 0.2,
        }

        features = IntensityFeatures.from_dict(data)

        # Should default to 0.0 for missing timbre/texture fields
        assert features.spectral_flatness == 0.0
        assert features.zero_crossing_rate == 0.0
        assert features.mfcc_variance == 0.0
        assert features.spectral_rolloff_85 == 0.0

    def test_from_dict_structural_vocal_backward_compat(self) -> None:
        """Test creating from old dictionary without structural/vocal fields."""
        data = {
            "filepath": "test.mp3",
            "file_hash": "abc123",
            "rms_energy": 0.5,
            "brightness": 0.6,
            "sub_bass_energy": 0.3,
            "kick_energy": 0.7,
            "bass_harmonics": 0.4,
            "percussiveness": 0.8,
            "onset_strength": 0.65,
            "camelot_key": "8B",
            "key_index": 0.0,
            "dynamic_range": 0.5,
            "energy_gradient": 0.1,
            "drop_density": 0.2,
            "spectral_flatness": 0.3,
            "zero_crossing_rate": 0.2,
            "mfcc_variance": 0.4,
            "spectral_rolloff_85": 0.5,
        }

        features = IntensityFeatures.from_dict(data)

        # Should default to 0.0 for missing structural/vocal fields
        assert features.vocal_presence == 0.0
        assert features.intro_length_secs == 0.0


class TestHarmonicCompatibility:
    """Test harmonic compatibility function."""

    def test_same_number_compatible(self) -> None:
        """Same Camelot number with different letter is compatible."""
        from playchitect.core.intensity_analyzer import harmonic_compatibility

        assert harmonic_compatibility("8A", "8B") is True
        assert harmonic_compatibility("3A", "3B") is True

    def test_adjacent_number_same_letter_compatible(self) -> None:
        """Adjacent numbers with same letter are compatible."""
        from playchitect.core.intensity_analyzer import harmonic_compatibility

        assert harmonic_compatibility("8B", "9B") is True
        assert harmonic_compatibility("8B", "7B") is True
        assert harmonic_compatibility("1B", "12B") is True  # Wrap around
        assert harmonic_compatibility("12B", "1B") is True  # Wrap around

    def test_incompatible_keys(self) -> None:
        """Non-adjacent keys are incompatible."""
        from playchitect.core.intensity_analyzer import harmonic_compatibility

        assert harmonic_compatibility("8B", "1B") is False
        assert harmonic_compatibility("8B", "3B") is False
        assert harmonic_compatibility("8A", "10B") is False

    def test_invalid_key_format(self) -> None:
        """Invalid key format raises ValueError."""
        from playchitect.core.intensity_analyzer import harmonic_compatibility

        with pytest.raises(ValueError):
            harmonic_compatibility("8", "9B")

        with pytest.raises(ValueError):
            harmonic_compatibility("8X", "9B")

        with pytest.raises(ValueError):
            harmonic_compatibility("13B", "9B")


class TestKeyDetection:
    """Test key detection in IntensityAnalyzer."""

    def test_synthetic_chroma_bin_0_returns_8b(self, tmp_path: Path) -> None:
        """Synthetic chroma with max at bin 0 should return 8B (C major)."""
        from unittest.mock import patch

        import numpy as np

        # Create synthetic chroma where bin 0 (C) is dominant
        synthetic_chroma = np.zeros((12, 100))
        synthetic_chroma[0, :] = 1.0  # Bin 0 is max
        synthetic_chroma[1:, :] = 0.1  # Others are low

        with patch(
            "playchitect.core.intensity_analyzer.librosa.feature.chroma_cqt",
            return_value=synthetic_chroma,
        ):
            analyzer = IntensityAnalyzer(sample_rate=22050, cache_enabled=False)
            camelot_key, key_index = analyzer._calculate_key(np.zeros(1000), 22050)

        assert camelot_key == "8B"
        assert key_index == 0.0

    def test_synthetic_chroma_bin_1_returns_3b(self, tmp_path: Path) -> None:
        """Synthetic chroma with max at bin 1 should return 3B (C# major)."""
        from unittest.mock import patch

        import numpy as np

        # Create synthetic chroma where bin 1 (C#) is dominant
        synthetic_chroma = np.zeros((12, 100))
        synthetic_chroma[1, :] = 1.0  # Bin 1 is max
        synthetic_chroma[0, :] = 0.1
        synthetic_chroma[2:, :] = 0.1

        with patch(
            "playchitect.core.intensity_analyzer.librosa.feature.chroma_cqt",
            return_value=synthetic_chroma,
        ):
            analyzer = IntensityAnalyzer(sample_rate=22050, cache_enabled=False)
            camelot_key, key_index = analyzer._calculate_key(np.zeros(1000), 22050)

        assert camelot_key == "3B"
        assert key_index == 1.0

    def test_cache_roundtrip_includes_camelot_key(self, tmp_path: Path) -> None:
        """JSON cache round-trip preserves camelot_key."""
        import soundfile as sf

        cache_dir = tmp_path / "cache"
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

        test_file = tmp_path / "test_key.wav"
        sf.write(test_file, audio, sample_rate)

        analyzer = IntensityAnalyzer(
            sample_rate=sample_rate, cache_dir=cache_dir, cache_enabled=True
        )

        # First analysis - should create cache with key info
        features1 = analyzer.analyze(test_file)
        assert hasattr(features1, "camelot_key")
        assert features1.camelot_key in [
            "1B",
            "2B",
            "3B",
            "4B",
            "5B",
            "6B",
            "7B",
            "8B",
            "9B",
            "10B",
            "11B",
            "12B",
        ]

        # Second analysis - should load from cache with key info intact
        features2 = analyzer.analyze(test_file)
        assert features2.camelot_key == features1.camelot_key
        assert features2.key_index == features1.key_index

    def test_all_camelot_keys_mapped(self) -> None:
        """Verify all 12 chroma bins map to valid Camelot keys."""
        from playchitect.core.intensity_analyzer import _CHROMA_TO_CAMELOT

        assert len(_CHROMA_TO_CAMELOT) == 12

        for i in range(12):
            assert i in _CHROMA_TO_CAMELOT
            camelot_key, key_index = _CHROMA_TO_CAMELOT[i]
            # Verify format: number (1-12) + letter (A or B)
            assert len(camelot_key) >= 2
            assert camelot_key[-1] in ("A", "B")
            number = int(camelot_key[:-1])
            assert 1 <= number <= 12
            assert key_index == float(i)


class TestIntensityAnalyzer:
    """Test IntensityAnalyzer class."""

    def test_analyzer_initialization(self) -> None:
        """Test analyzer initialization."""
        analyzer = IntensityAnalyzer(sample_rate=22050, cache_enabled=False)

        assert analyzer.sample_rate == 22050
        assert analyzer.cache_enabled is False

    def test_cache_dir_priority_arg(self, tmp_path: Path) -> None:
        """Test that explicit cache_dir argument takes highest priority."""
        arg_path = tmp_path / "arg_cache"

        with patch.dict(os.environ, {"PLAYCHITECT_CACHE_DIR": "/env/path"}):
            with patch("playchitect.core.intensity_analyzer.get_config") as mock_get_config:
                mock_get_config.return_value.get_cache_dir.return_value = Path("/config/path")

                analyzer = IntensityAnalyzer(cache_dir=arg_path, cache_enabled=False)
                assert analyzer.cache_dir == arg_path

    def test_cache_dir_priority_env(self, tmp_path: Path) -> None:
        """Test that PLAYCHITECT_CACHE_DIR env var takes priority over config."""
        env_path = tmp_path / "env_cache"

        with patch.dict(os.environ, {"PLAYCHITECT_CACHE_DIR": str(env_path)}):
            with patch("playchitect.core.intensity_analyzer.get_config") as mock_get_config:
                mock_get_config.return_value.get_cache_dir.return_value = Path("/config/path")

                analyzer = IntensityAnalyzer(cache_dir=None, cache_enabled=False)
                assert analyzer.cache_dir == env_path / "intensity"

    def test_cache_dir_priority_config(self) -> None:
        """Test that config is used when no arg or env var is set."""
        config_path = Path("/config/path")

        with patch.dict(os.environ, {}, clear=True):
            with patch("playchitect.core.intensity_analyzer.get_config") as mock_get_config:
                mock_get_config.return_value.get_cache_dir.return_value = config_path

                analyzer = IntensityAnalyzer(cache_dir=None, cache_enabled=False)
                assert analyzer.cache_dir == config_path / "intensity"

    def test_analyze_nonexistent_file(self) -> None:
        """Test analyzing nonexistent file raises error."""
        analyzer = IntensityAnalyzer(cache_enabled=False)

        with pytest.raises(FileNotFoundError):
            analyzer.analyze(Path("/nonexistent/file.mp3"))

    def test_analyze_with_synthetic_audio(self, tmp_path: Path) -> None:
        """Test analysis with synthetic audio."""
        # Create a simple synthetic audio file
        sample_rate = 22050
        duration = 2.0  # seconds
        frequency = 440.0  # A4 note

        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Save as WAV
        import soundfile as sf

        test_file = tmp_path / "test_tone.wav"
        sf.write(test_file, audio, sample_rate)

        # Analyze
        analyzer = IntensityAnalyzer(sample_rate=sample_rate, cache_enabled=False)
        features = analyzer.analyze(test_file)

        # Verify features are in valid range
        assert 0.0 <= features.rms_energy <= 1.0
        assert 0.0 <= features.brightness <= 1.0
        assert 0.0 <= features.sub_bass_energy <= 1.0
        assert 0.0 <= features.kick_energy <= 1.0
        assert 0.0 <= features.bass_harmonics <= 1.0
        assert 0.0 <= features.percussiveness <= 1.0
        assert 0.0 <= features.onset_strength <= 1.0

        # Sine wave should have low percussiveness (harmonic)
        assert features.percussiveness < 0.5

    def test_feature_vector_dimensions(self, tmp_path: Path) -> None:
        """Test feature vector has correct dimensions."""
        # Create minimal audio
        sample_rate = 22050
        audio = np.random.randn(sample_rate).astype(np.float32) * 0.1

        import soundfile as sf

        test_file = tmp_path / "test_noise.wav"
        sf.write(test_file, audio, sample_rate)

        analyzer = IntensityAnalyzer(sample_rate=sample_rate, cache_enabled=False)
        features = analyzer.analyze(test_file)

        vector = features.to_feature_vector()
        assert vector.shape == (7,)

    def test_caching_functionality(self, tmp_path: Path) -> None:
        """Test that caching works correctly."""
        cache_dir = tmp_path / "cache"

        # Create test audio
        sample_rate = 22050
        audio = np.random.randn(sample_rate).astype(np.float32) * 0.1

        import soundfile as sf

        test_file = tmp_path / "test_cache.wav"
        sf.write(test_file, audio, sample_rate)

        # Analyze with caching enabled
        analyzer = IntensityAnalyzer(
            sample_rate=sample_rate, cache_dir=cache_dir, cache_enabled=True
        )

        # First analysis - should create cache
        features1 = analyzer.analyze(test_file)

        # Check cache file was created
        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) == 1

        # Second analysis - should use cache
        features2 = analyzer.analyze(test_file)

        # Should be identical
        assert features1.rms_energy == features2.rms_energy
        assert features1.brightness == features2.brightness

    def test_clear_cache(self, tmp_path: Path) -> None:
        """Test clearing cache."""
        cache_dir = tmp_path / "cache"

        # Create test audio
        sample_rate = 22050
        audio = np.random.randn(sample_rate).astype(np.float32) * 0.1

        import soundfile as sf

        test_file = tmp_path / "test_clear.wav"
        sf.write(test_file, audio, sample_rate)

        analyzer = IntensityAnalyzer(
            sample_rate=sample_rate, cache_dir=cache_dir, cache_enabled=True
        )

        # Analyze to create cache
        analyzer.analyze(test_file)
        assert len(list(cache_dir.glob("*.json"))) == 1

        # Clear cache
        analyzer.clear_cache()
        assert len(list(cache_dir.glob("*.json"))) == 0

    def test_bass_energy_sum_reasonable(self, tmp_path: Path) -> None:
        """Test that bass energy components are all non-zero for bass-heavy audio."""
        # Create test audio with bass content
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Mix of bass frequencies
        audio = (
            np.sin(2 * np.pi * 40 * t)  # sub-bass
            + np.sin(2 * np.pi * 80 * t)  # kick
            + np.sin(2 * np.pi * 150 * t)  # harmonics
        )
        audio = audio.astype(np.float32) * 0.3

        import soundfile as sf

        test_file = tmp_path / "test_bass.wav"
        sf.write(test_file, audio, sample_rate)

        analyzer = IntensityAnalyzer(sample_rate=sample_rate, cache_enabled=False)
        features = analyzer.analyze(test_file)

        # All bass components should be non-zero for bass-heavy audio
        assert features.sub_bass_energy > 0.0
        assert features.kick_energy > 0.0
        assert features.bass_harmonics > 0.0

        # All should be in valid range
        assert 0.0 <= features.sub_bass_energy <= 1.0
        assert 0.0 <= features.kick_energy <= 1.0
        assert 0.0 <= features.bass_harmonics <= 1.0

    def test_rms_weighting_concept(self, tmp_path: Path) -> None:
        """Test that louder audio produces higher RMS energy."""
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        base_audio = np.sin(2 * np.pi * 440 * t)

        import soundfile as sf

        # Quiet audio
        quiet_audio = base_audio.astype(np.float32) * 0.1
        quiet_file = tmp_path / "quiet.wav"
        sf.write(quiet_file, quiet_audio, sample_rate)

        # Loud audio
        loud_audio = base_audio.astype(np.float32) * 0.8
        loud_file = tmp_path / "loud.wav"
        sf.write(loud_file, loud_audio, sample_rate)

        analyzer = IntensityAnalyzer(sample_rate=sample_rate, cache_enabled=False)

        quiet_features = analyzer.analyze(quiet_file)
        loud_features = analyzer.analyze(loud_file)

        # Loud audio should have higher RMS
        assert loud_features.rms_energy > quiet_features.rms_energy

    def test_percussive_vs_harmonic(self, tmp_path: Path) -> None:
        """Test that percussive sounds score higher than harmonic."""
        sample_rate = 22050
        duration = 1.0

        import soundfile as sf

        # Harmonic (sine wave)
        t = np.linspace(0, duration, int(sample_rate * duration))
        harmonic = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
        harmonic_file = tmp_path / "harmonic.wav"
        sf.write(harmonic_file, harmonic, sample_rate)

        # Percussive (white noise bursts)
        percussive = np.zeros(int(sample_rate * duration), dtype=np.float32)
        # Add sharp transients
        for i in range(0, len(percussive), sample_rate // 4):
            if i + 1000 < len(percussive):
                percussive[i : i + 1000] = np.random.randn(1000) * 0.5
        percussive_file = tmp_path / "percussive.wav"
        sf.write(percussive_file, percussive.astype(np.float32), sample_rate)

        analyzer = IntensityAnalyzer(sample_rate=sample_rate, cache_enabled=False)

        harmonic_features = analyzer.analyze(harmonic_file)
        percussive_features = analyzer.analyze(percussive_file)

        # Percussive should score higher
        assert percussive_features.percussiveness > harmonic_features.percussiveness

        # Harmonic should have low percussiveness
        assert harmonic_features.percussiveness < 0.5

    def test_energy_flow_features_synthetic_audio(self, tmp_path: Path) -> None:
        """Test energy flow features with synthetic sine-wave audio."""
        sample_rate = 22050
        duration = 2.0  # seconds
        frequency = 440.0  # A4 note

        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Save as WAV
        import soundfile as sf

        test_file = tmp_path / "test_energy_flow.wav"
        sf.write(test_file, audio, sample_rate)

        # Analyze
        analyzer = IntensityAnalyzer(sample_rate=sample_rate, cache_enabled=False)
        features = analyzer.analyze(test_file)

        # Verify energy flow features are valid
        assert 0.0 <= features.dynamic_range <= 1.0
        assert -1.0 <= features.energy_gradient <= 1.0
        assert isinstance(features.energy_gradient, float)
        assert 0.0 <= features.drop_density <= 1.0

        # Sine wave should have zero drop density (no drops in steady tone)
        # Threshold is mean + 2*std; for steady signal, std ≈ 0, so no frames exceed threshold
        assert features.drop_density < 0.1

    def test_timbre_texture_features_synthetic_audio(self, tmp_path: Path) -> None:
        """Test timbre/texture features with synthetic sine-wave audio."""
        sample_rate = 22050
        duration = 2.0  # seconds
        frequency = 440.0  # A4 note

        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Save as WAV
        import soundfile as sf

        test_file = tmp_path / "test_timbre.wav"
        sf.write(test_file, audio, sample_rate)

        # Analyze
        analyzer = IntensityAnalyzer(sample_rate=sample_rate, cache_enabled=False)
        features = analyzer.analyze(test_file)

        # Verify timbre/texture features are in valid range [0, 1]
        assert 0.0 <= features.spectral_flatness <= 1.0
        assert 0.0 <= features.zero_crossing_rate <= 1.0
        assert 0.0 <= features.mfcc_variance <= 1.0
        assert 0.0 <= features.spectral_rolloff_85 <= 1.0

        # Sine wave should have low spectral flatness (tonal, not noisy)
        assert features.spectral_flatness < 0.5

        # Sine wave should have low ZCR (smooth, few crossings)
        assert features.zero_crossing_rate < 0.1

    def test_timbre_texture_features_noisy_audio(self, tmp_path: Path) -> None:
        """Test timbre/texture features with noisy/percussive audio."""
        sample_rate = 22050
        duration = 1.0

        import soundfile as sf

        # Percussive (white noise with sharp transients)
        percussive = np.zeros(int(sample_rate * duration), dtype=np.float32)
        # Add sharp transients
        for i in range(0, len(percussive), sample_rate // 4):
            if i + 1000 < len(percussive):
                percussive[i : i + 1000] = np.random.randn(1000) * 0.5
        percussive_file = tmp_path / "percussive_timbre.wav"
        sf.write(percussive_file, percussive.astype(np.float32), sample_rate)

        analyzer = IntensityAnalyzer(sample_rate=sample_rate, cache_enabled=False)
        percussive_features = analyzer.analyze(percussive_file)

        # Noisy/percussive audio should have high spectral flatness
        assert percussive_features.spectral_flatness > 0.3

        # Should still be in valid range
        assert 0.0 <= percussive_features.zero_crossing_rate <= 1.0
        assert 0.0 <= percussive_features.mfcc_variance <= 1.0
        assert 0.0 <= percussive_features.spectral_rolloff_85 <= 1.0

    def test_timbre_texture_cache_roundtrip(self, tmp_path: Path) -> None:
        """Test that timbre/texture features are cached and loaded correctly."""
        import soundfile as sf

        cache_dir = tmp_path / "cache"
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

        test_file = tmp_path / "test_timbre_cache.wav"
        sf.write(test_file, audio, sample_rate)

        analyzer = IntensityAnalyzer(
            sample_rate=sample_rate, cache_dir=cache_dir, cache_enabled=True
        )

        # First analysis - should create cache with timbre/texture info
        features1 = analyzer.analyze(test_file)
        assert hasattr(features1, "spectral_flatness")
        assert hasattr(features1, "zero_crossing_rate")
        assert hasattr(features1, "mfcc_variance")
        assert hasattr(features1, "spectral_rolloff_85")

        # Second analysis - should load from cache with timbre/texture info intact
        features2 = analyzer.analyze(test_file)
        assert features2.spectral_flatness == features1.spectral_flatness
        assert features2.zero_crossing_rate == features1.zero_crossing_rate
        assert features2.mfcc_variance == features1.mfcc_variance
        assert features2.spectral_rolloff_85 == features1.spectral_rolloff_85

    def test_analyze_emits_no_audioread_or_soundfile_warnings(self, tmp_path: Path) -> None:
        """analyze() on a valid audio fixture must not emit WARNING-level log
        records from the audioread or soundfile loggers.

        This guards against backend-negotiation chatter (e.g. 'PySoundFile
        failed', 'Trying audioread…') leaking into user-visible output.
        """
        import soundfile as sf

        # Build a minimal valid WAV fixture — 1 second of 440 Hz sine wave.
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
        fixture = tmp_path / "valid_tone.wav"
        sf.write(fixture, audio, sample_rate)

        # Capture WARNING+ records from the target loggers.
        noisy_logger_names = ("audioread", "soundfile", "librosa")
        captured_records: list[logging.LogRecord] = []

        class _CapturingHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured_records.append(record)

        capturing_handler = _CapturingHandler(level=logging.WARNING)
        loggers_under_test = [logging.getLogger(name) for name in noisy_logger_names]
        original_levels = [lg.level for lg in loggers_under_test]

        for lg in loggers_under_test:
            lg.addHandler(capturing_handler)
            # Ensure the logger itself is not filtering out WARNING records
            # before they reach our handler.
            if lg.level == logging.NOTSET or lg.level > logging.WARNING:
                lg.setLevel(logging.WARNING)

        try:
            analyzer = IntensityAnalyzer(sample_rate=sample_rate, cache_enabled=False)
            analyzer.analyze(fixture)
        finally:
            for lg, level in zip(loggers_under_test, original_levels):
                lg.removeHandler(capturing_handler)
                lg.setLevel(level)

        warning_records = [r for r in captured_records if r.levelno >= logging.WARNING]
        assert warning_records == [], (
            f"Expected no WARNING-level log records from audioread/soundfile/librosa "
            f"loggers, but got: {[r.getMessage() for r in warning_records]}"
        )


class TestStructuralVocalFeatures:
    """Test structural and vocal features."""

    def test_vocal_presence_with_harmonic_mid_range(self, tmp_path: Path) -> None:
        """Synthetic audio with loud harmonic mid-range (200-800Hz) yields vocal_presence > 0."""
        import soundfile as sf

        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create harmonic content in vocal range (440Hz and 880Hz, both in 200-800Hz)
        # Use sine waves for pure harmonic content
        audio = (
            np.sin(2 * np.pi * 440 * t) * 0.4  # A4 - in vocal range
            + np.sin(2 * np.pi * 220 * t) * 0.3  # A3 - in vocal range
        )
        audio = audio.astype(np.float32)

        test_file = tmp_path / "test_vocal.wav"
        sf.write(test_file, audio, sample_rate)

        analyzer = IntensityAnalyzer(sample_rate=sample_rate, cache_enabled=False)
        features = analyzer.analyze(test_file)

        # Should have vocal presence > 0
        assert features.vocal_presence > 0.0
        assert 0.0 <= features.vocal_presence <= 1.0

    def test_vocal_presence_no_vocal_range_content(self, tmp_path: Path) -> None:
        """Audio without 200-800Hz content should have low vocal presence."""
        import soundfile as sf

        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Create low bass content only (below vocal range)
        audio = (
            np.sin(2 * np.pi * 50 * t) * 0.5  # Sub-bass, below vocal range
            + np.sin(2 * np.pi * 100 * t) * 0.5  # Kick, below vocal range
        )
        audio = audio.astype(np.float32)

        test_file = tmp_path / "test_no_vocal.wav"
        sf.write(test_file, audio, sample_rate)

        analyzer = IntensityAnalyzer(sample_rate=sample_rate, cache_enabled=False)
        features = analyzer.analyze(test_file)

        # Should have very low vocal presence
        assert features.vocal_presence < 0.3

    def test_intro_length_with_silent_prefix(self, tmp_path: Path) -> None:
        """Silent prefix followed by loud portion yields intro_length_secs > 0."""
        import soundfile as sf

        sample_rate = 22050
        # Create audio with silent intro then loud content
        silent_duration = 1.0  # 1 second of silence
        loud_duration = 2.0  # 2 seconds of loud content

        # Silent prefix (zeros)
        silent_samples = int(sample_rate * silent_duration)
        silent = np.zeros(silent_samples, dtype=np.float32)

        # Loud portion (sine wave at high amplitude)
        t = np.linspace(0, loud_duration, int(sample_rate * loud_duration))
        loud = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.8

        # Concatenate
        audio = np.concatenate([silent, loud])

        test_file = tmp_path / "test_intro.wav"
        sf.write(test_file, audio, sample_rate)

        analyzer = IntensityAnalyzer(sample_rate=sample_rate, cache_enabled=False)
        features = analyzer.analyze(test_file)

        # Should detect intro length > 0 (approximately 1 second, with some tolerance)
        assert features.intro_length_secs > 0.5
        assert features.intro_length_secs < 1.5

    def test_intro_length_no_intro(self, tmp_path: Path) -> None:
        """Audio that starts loud immediately should have intro_length_secs ≈ 0."""
        import soundfile as sf

        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))

        # Loud from the start
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.8

        test_file = tmp_path / "test_no_intro.wav"
        sf.write(test_file, audio, sample_rate)

        analyzer = IntensityAnalyzer(sample_rate=sample_rate, cache_enabled=False)
        features = analyzer.analyze(test_file)

        # Should have intro length ≈ 0
        assert features.intro_length_secs < 0.2

    def test_structural_vocal_cache_roundtrip(self, tmp_path: Path) -> None:
        """Test that structural/vocal features are cached and loaded correctly."""
        import soundfile as sf

        cache_dir = tmp_path / "cache"
        sample_rate = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5

        test_file = tmp_path / "test_structural_vocal_cache.wav"
        sf.write(test_file, audio, sample_rate)

        analyzer = IntensityAnalyzer(
            sample_rate=sample_rate, cache_dir=cache_dir, cache_enabled=True
        )

        # First analysis - should create cache with structural/vocal info
        features1 = analyzer.analyze(test_file)
        assert hasattr(features1, "vocal_presence")
        assert hasattr(features1, "intro_length_secs")

        # Second analysis - should load from cache with structural/vocal info intact
        features2 = analyzer.analyze(test_file)
        assert features2.vocal_presence == features1.vocal_presence
        assert features2.intro_length_secs == features1.intro_length_secs


class TestIntegration:
    """Integration tests for intensity analysis."""

    def test_full_pipeline_with_multiple_files(self, tmp_path: Path) -> None:
        """Test analyzing multiple files."""
        sample_rate = 22050
        duration = 0.5  # Short for speed

        import soundfile as sf

        files = []
        for i in range(3):
            t = np.linspace(0, duration, int(sample_rate * duration))
            freq = 440 * (i + 1)  # Different frequencies
            audio = np.sin(2 * np.pi * freq * t).astype(np.float32) * 0.5

            test_file = tmp_path / f"test_{i}.wav"
            sf.write(test_file, audio, sample_rate)
            files.append(test_file)

        analyzer = IntensityAnalyzer(sample_rate=sample_rate, cache_enabled=False)

        features_list = []
        for file in files:
            features = analyzer.analyze(file)
            features_list.append(features)

        # All analyses should succeed
        assert len(features_list) == 3

        # All features should be in valid range
        for features in features_list:
            vector = features.to_feature_vector()
            assert np.all((vector >= 0.0) & (vector <= 1.0))


def _make_fixture_features(filepath: Path) -> IntensityFeatures:
    return IntensityFeatures(
        file_path=filepath,
        file_hash="deadbeef",
        rms_energy=0.5,
        brightness=0.6,
        sub_bass_energy=0.3,
        kick_energy=0.7,
        bass_harmonics=0.4,
        percussiveness=0.8,
        onset_strength=0.65,
        camelot_key="8B",
        key_index=0.0,
    )


class TestAnalyzeBatch:
    """Tests for IntensityAnalyzer.analyze_batch() and estimate_batch_seconds()."""

    def test_cached_tracks_skip_pool(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Files already cached should not cause ProcessPoolExecutor to be constructed."""
        filepaths = [tmp_path / f"track_{i}.mp3" for i in range(3)]
        for p in filepaths:
            p.touch()

        # Patch _load_from_cache to always return a fixture result
        monkeypatch.setattr(
            IntensityAnalyzer,
            "_load_from_cache",
            lambda self, file_hash: _make_fixture_features(tmp_path / "fixture.mp3"),
        )
        # Patch _compute_file_hash so we don't need real audio
        monkeypatch.setattr(IntensityAnalyzer, "_compute_file_hash", lambda self, p: "deadbeef")

        with patch("playchitect.core.intensity_analyzer.ProcessPoolExecutor") as mock_ppe:
            analyzer = IntensityAnalyzer(cache_dir=tmp_path, cache_enabled=False)
            result = analyzer.analyze_batch(filepaths)

        mock_ppe.assert_not_called()
        assert len(result) == 3

    def test_progress_callback_called(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """progress_callback must be called once per file with (n_done, n_total)."""
        filepaths = [tmp_path / f"track_{i}.mp3" for i in range(3)]
        for p in filepaths:
            p.touch()

        monkeypatch.setattr(
            IntensityAnalyzer,
            "_load_from_cache",
            lambda self, file_hash: _make_fixture_features(tmp_path / "fixture.mp3"),
        )
        monkeypatch.setattr(IntensityAnalyzer, "_compute_file_hash", lambda self, p: "deadbeef")

        calls: list[tuple[int, int]] = []

        def cb(n_done: int, n_total: int) -> None:
            calls.append((n_done, n_total))

        analyzer = IntensityAnalyzer(cache_dir=tmp_path, cache_enabled=False)
        analyzer.analyze_batch(filepaths, progress_callback=cb)

        assert len(calls) == 3
        assert calls[0] == (1, 3)
        assert calls[1] == (2, 3)
        assert calls[2] == (3, 3)

    def test_error_is_omitted_not_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A worker error should omit the track from results; no exception should propagate."""
        filepath = tmp_path / "bad_track.mp3"
        filepath.touch()

        # Cache miss — force the file into the pool
        monkeypatch.setattr(IntensityAnalyzer, "_load_from_cache", lambda self, h: None)
        monkeypatch.setattr(IntensityAnalyzer, "_compute_file_hash", lambda self, p: "deadbeef")

        # Make the future raise ValueError
        failing_future: MagicMock = MagicMock()
        failing_future.result.side_effect = ValueError("corrupt audio")

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.return_value = failing_future

        with patch(
            "playchitect.core.intensity_analyzer.ProcessPoolExecutor", return_value=mock_executor
        ):
            with patch(
                "playchitect.core.intensity_analyzer.as_completed",
                return_value=[failing_future],
            ):
                analyzer = IntensityAnalyzer(cache_dir=tmp_path, cache_enabled=False)
                result = analyzer.analyze_batch([filepath])

        assert result == {}

    def test_estimate_batch_seconds(self) -> None:
        """estimate_batch_seconds should return n_uncached * 2.0."""
        assert IntensityAnalyzer.estimate_batch_seconds(100) == 200.0
        assert IntensityAnalyzer.estimate_batch_seconds(0) == 0.0
        assert IntensityAnalyzer.estimate_batch_seconds(1) == 2.0

    def test_default_max_workers_is_capped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With cpu_count() == 32, max_workers should be capped at 6."""
        filepath = tmp_path / "track.mp3"
        filepath.touch()

        monkeypatch.setattr(IntensityAnalyzer, "_load_from_cache", lambda self, h: None)
        monkeypatch.setattr(IntensityAnalyzer, "_compute_file_hash", lambda self, p: "aabbccdd")

        captured: list[int] = []

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.submit.return_value = MagicMock()

        def fake_ppe(max_workers: int | None = None, **kwargs: object) -> MagicMock:
            captured.append(max_workers or 0)
            return mock_executor

        with patch("playchitect.core.intensity_analyzer.ProcessPoolExecutor", side_effect=fake_ppe):
            with patch("playchitect.core.intensity_analyzer.as_completed", return_value=[]):
                with patch("os.cpu_count", return_value=32):
                    analyzer = IntensityAnalyzer(cache_dir=tmp_path, cache_enabled=False)
                    analyzer.analyze_batch([filepath])

        assert captured == [6]
