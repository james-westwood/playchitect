"""
Robustness tests for handling corrupt or problematic audio files.
"""

import pytest

from playchitect.core.intensity_analyzer import IntensityAnalyzer
from playchitect.core.metadata_extractor import MetadataExtractor


def test_intensity_analyzer_corrupt_file(tmp_path):
    """Test that IntensityAnalyzer handles a non-audio file gracefully."""
    bad_file = tmp_path / "not_audio.txt"
    bad_file.write_text("This is not an audio file.")

    analyzer = IntensityAnalyzer(cache_enabled=False)

    # analyze() should raise ValueError but with a descriptive message
    with pytest.raises(ValueError, match="Failed to analyze"):
        analyzer.analyze(bad_file)


def test_intensity_analyzer_empty_file(tmp_path):
    """Test that IntensityAnalyzer handles an empty file gracefully."""
    empty_file = tmp_path / "empty.wav"
    empty_file.touch()

    analyzer = IntensityAnalyzer(cache_enabled=False)

    with pytest.raises(ValueError, match="Failed to analyze"):
        analyzer.analyze(empty_file)


def test_intensity_analyzer_batch_robustness(tmp_path, monkeypatch):
    """Test that analyze_batch continues even if some files are corrupt."""
    # This is a placeholder for a more complex test if needed
    pass


def test_intensity_analyzer_batch_exception_handling(tmp_path):
    """Test that analyze_batch handles exceptions from workers."""
    from concurrent.futures import Future
    from unittest.mock import patch

    analyzer = IntensityAnalyzer(cache_enabled=False)

    # Create actual dummy files for hashing
    f1 = tmp_path / "fail.mp3"
    f1.write_text("fail")
    f2 = tmp_path / "success.mp3"
    f2.write_text("success")

    with patch("playchitect.core.intensity_analyzer.ProcessPoolExecutor") as mock_executor:
        executor_instance = mock_executor.return_value.__enter__.return_value

        # Create a future that raises an exception
        fail_future = Future()
        fail_future.set_exception(ValueError("Worker crashed"))

        # Create a future that succeeds
        success_future = Future()
        success_future.set_result(
            (
                "success.mp3",
                {
                    "filepath": str(f2),
                    "file_hash": "h1",
                    "rms_energy": 0.5,
                    "brightness": 0.5,
                    "sub_bass_energy": 0.5,
                    "kick_energy": 0.5,
                    "bass_harmonics": 0.5,
                    "percussiveness": 0.5,
                    "onset_strength": 0.5,
                },
            )
        )

        # Mock executor behavior
        executor_instance.submit.side_effect = [fail_future, success_future]

        # Mock as_completed to return our futures
        with patch("playchitect.core.intensity_analyzer.as_completed") as mock_as_completed:
            mock_as_completed.return_value = [fail_future, success_future]
            results = analyzer.analyze_batch([f1, f2])

            # Should have one result (the success one)
            assert len(results) == 1
            assert f2 in results


def test_metadata_extractor_calculate_bpm_robustness(tmp_path):
    """Test that MetadataExtractor.calculate_bpm handles corrupt files."""
    bad_file = tmp_path / "not_audio.txt"
    bad_file.write_text("This is not an audio file.")

    extractor = MetadataExtractor(cache_enabled=False)
    bpm = extractor.calculate_bpm(bad_file)

    # Should return None, not raise
    assert bpm is None
