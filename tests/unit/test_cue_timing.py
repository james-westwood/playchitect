"""Unit tests for CUE sheet timing utilities."""

import pytest

from playchitect.core.cue_timing import (
    cumulative_offsets,
    seconds_to_cue_time,
    validate_cue_time,
)


class TestSecondsToTime:
    def test_zero(self):
        assert seconds_to_cue_time(0.0) == "00:00:00"

    def test_one_second(self):
        assert seconds_to_cue_time(1.0) == "00:01:00"

    def test_one_minute(self):
        assert seconds_to_cue_time(60.0) == "01:00:00"

    def test_large_duration(self):
        # 3661.0 s = 61 m 1 s 0 f
        assert seconds_to_cue_time(3661.0) == "61:01:00"

    def test_half_second(self):
        # round(0.5 * 75) = 38 frames -> "00:00:38"
        assert seconds_to_cue_time(0.5) == "00:00:38"

    def test_negative_clamped(self):
        assert seconds_to_cue_time(-10.0) == "00:00:00"

    def test_manual_calc(self):
        # 386.4 * 75 = 28980.0
        # 28980 // 75 = 386 s
        # 386 % 60 = 26 s
        # 386 // 60 = 6 m
        # 28980 % 75 = 30 f
        assert seconds_to_cue_time(386.4) == "06:26:30"

    def test_one_frame(self):
        # 1/75 s = 1 frame
        assert seconds_to_cue_time(1 / 75) == "00:00:01"

    def test_rounding_up(self):
        # round(0.9999 * 75) = 75 frames -> 1 second, 0 frames
        assert seconds_to_cue_time(0.9999) == "00:01:00"


class TestCumulativeOffsets:
    def test_empty(self):
        assert cumulative_offsets([]) == []

    def test_single(self):
        assert cumulative_offsets([300.0]) == [0.0]

    def test_multiple(self):
        durations = [300.0, 400.0, 250.0]
        expected = [0.0, 300.0, 700.0]
        assert cumulative_offsets(durations) == pytest.approx(expected)

    def test_first_is_zero(self):
        assert cumulative_offsets([123.4, 567.8])[0] == 0.0

    def test_result_length(self):
        durations = [1.0, 2.0, 3.0, 4.0]
        assert len(cumulative_offsets(durations)) == len(durations)


class TestValidateCueTime:
    def test_valid_zero(self):
        assert validate_cue_time("00:00:00") is True

    def test_valid_arbitrary(self):
        assert validate_cue_time("01:23:45") is True

    def test_valid_max_frames(self):
        assert validate_cue_time("99:59:74") is True

    def test_invalid_frames_out_of_range(self):
        assert validate_cue_time("00:00:75") is False

    def test_invalid_seconds_out_of_range(self):
        assert validate_cue_time("00:60:00") is False

    def test_invalid_alphabetic(self):
        assert validate_cue_time("abc") is False

    def test_invalid_empty(self):
        assert validate_cue_time("") is False

    def test_invalid_too_few_fields(self):
        assert validate_cue_time("00:00") is False

    def test_invalid_too_many_fields(self):
        assert validate_cue_time("00:00:00:00") is False

    def test_invalid_negative_not_allowed(self):
        assert validate_cue_time("-1:00:00") is False
