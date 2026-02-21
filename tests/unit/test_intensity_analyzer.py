"""
Unit tests for intensity_analyzer module.
"""

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
            filepath=Path("test.mp3"),
            file_hash="abc123",
            rms_energy=0.5,
            brightness=0.6,
            sub_bass_energy=0.3,
            kick_energy=0.7,
            bass_harmonics=0.4,
            percussiveness=0.8,
            onset_strength=0.65,
        )

        assert features.rms_energy == 0.5
        assert features.brightness == 0.6
        assert features.kick_energy == 0.7

    def test_to_feature_vector(self) -> None:
        """Test conversion to feature vector."""
        features = IntensityFeatures(
            filepath=Path("test.mp3"),
            file_hash="abc123",
            rms_energy=0.5,
            brightness=0.6,
            sub_bass_energy=0.3,
            kick_energy=0.7,
            bass_harmonics=0.4,
            percussiveness=0.8,
            onset_strength=0.65,
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
            filepath=Path("test.mp3"),
            file_hash="abc123",
            rms_energy=0.5,
            brightness=0.6,
            sub_bass_energy=0.3,
            kick_energy=0.7,
            bass_harmonics=0.4,
            percussiveness=0.8,
            onset_strength=0.65,
        )

        data = features.to_dict()

        assert isinstance(data, dict)
        assert data["filepath"] == "test.mp3"
        assert data["rms_energy"] == 0.5

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
        }

        features = IntensityFeatures.from_dict(data)

        assert features.rms_energy == 0.5
        assert features.filepath == Path("test.mp3")


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
        filepath=filepath,
        file_hash="deadbeef",
        rms_energy=0.5,
        brightness=0.6,
        sub_bass_energy=0.3,
        kick_energy=0.7,
        bass_harmonics=0.4,
        percussiveness=0.8,
        onset_strength=0.65,
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
