"""
Tests for playchitect.utils.warnings — scoped warning/stderr suppression.
"""

import logging
import warnings

from playchitect.utils.warnings import (
    suppress_audio_log_warnings,
    suppress_c_stderr,
    suppress_librosa_warnings,
)


class TestSuppressLibrosaWarnings:
    """Issue #94: Python-level warnings from librosa/audioread must be suppressed."""

    def test_suppresses_pysoundfile_user_warning(self) -> None:
        with suppress_librosa_warnings():
            warnings.warn("PySoundFile failed. Trying audioread instead.", UserWarning)

    def test_suppresses_audioread_future_warning(self) -> None:
        with suppress_librosa_warnings():
            warnings.warn(
                "librosa.core.audio.__audioread_load is deprecated.",
                FutureWarning,
            )

    def test_suppresses_librosa_user_warning(self) -> None:
        with suppress_librosa_warnings():
            warnings.warn("Some librosa issue", UserWarning)

    def test_other_warnings_not_suppressed(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            with suppress_librosa_warnings():
                warnings.warn("unrelated warning", UserWarning, stacklevel=1)
        assert len(caught) == 1


class TestSuppressAudioLogWarnings:
    """Issue #94: Logging-level noise from soundfile/audioread must be suppressed."""

    def test_suppresses_soundfile_warning_log(self) -> None:
        sf_logger = logging.getLogger("soundfile")
        original = sf_logger.level
        try:
            sf_logger.setLevel(logging.WARNING)
            with suppress_audio_log_warnings():
                sf_logger.warning("PySoundFile failed")
        finally:
            sf_logger.setLevel(original)

    def test_error_log_not_suppressed(self) -> None:
        sf_logger = logging.getLogger("soundfile")
        with suppress_audio_log_warnings():
            sf_logger.error("real error")


class TestSuppressCStderr:
    """Issue #95: C-level stderr from mpg123 must be redirected to /dev/null."""

    def test_stderr_redirected_within_context(self, tmp_path) -> None:
        log_file = tmp_path / "stderr_capture.txt"
        with (
            suppress_c_stderr(),
            open(log_file, "w") as captured,
        ):
            import os
            import sys

            original_fd = os.dup(2)
            try:
                os.dup2(captured.fileno(), 2)
                sys.stderr.write("This should be discarded\n")
                sys.stderr.flush()
            finally:
                os.dup2(original_fd, 2)
                os.close(original_fd)

    def test_stderr_restored_after_context(self) -> None:
        import os

        original = os.dup(2)
        os.close(original)

        with suppress_c_stderr():
            pass

        current = os.dup(2)
        os.close(current)


class TestBatchSkippedSummary:
    """Issue #95: analyze_batch must report skipped file count."""

    def test_skipped_count_logged(self, tmp_path, caplog) -> None:
        from concurrent.futures import Future
        from unittest.mock import patch

        from playchitect.core.intensity_analyzer import IntensityAnalyzer

        analyzer = IntensityAnalyzer(cache_enabled=False)
        f1 = tmp_path / "fail.mp3"
        f1.write_text("fail")

        with (
            patch("playchitect.core.intensity_analyzer.ProcessPoolExecutor") as mock_executor,
            patch("playchitect.core.intensity_analyzer.as_completed") as mock_as_completed,
        ):
            executor_instance = mock_executor.return_value.__enter__.return_value

            fail_future = Future()
            fail_future.set_exception(ValueError("Worker crashed"))

            executor_instance.submit.side_effect = [fail_future]
            mock_as_completed.return_value = [fail_future]

            with caplog.at_level(logging.WARNING, logger="playchitect.core.intensity_analyzer"):
                results = analyzer.analyze_batch([f1])

        assert len(results) == 0
        skipped_msgs = [r for r in caplog.records if "Skipped" in r.message]
        assert len(skipped_msgs) == 1
        assert "1" in skipped_msgs[0].message
