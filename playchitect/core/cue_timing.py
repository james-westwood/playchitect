"""CUE sheet timing utilities.

CUE sheet time format: MM:SS:FF (minutes, seconds, frames).
75 frames per second — the CD standard used by all CUE players.
"""

from __future__ import annotations

_FRAMES_PER_SECOND: int = 75


def seconds_to_cue_time(seconds: float) -> str:
    """Convert a duration in seconds to CUE sheet MM:SS:FF format.

    Negative values are clamped to 0. Rounds to the nearest frame.

    Args:
        seconds: Duration in seconds (float).

    Returns:
        String in "MM:SS:FF" format, e.g. "05:23:00", "00:00:38".
    """
    total_frames = round(max(0.0, seconds) * _FRAMES_PER_SECOND)
    frames = total_frames % _FRAMES_PER_SECOND
    total_secs = total_frames // _FRAMES_PER_SECOND
    secs = total_secs % 60
    mins = total_secs // 60
    return f"{mins:02d}:{secs:02d}:{frames:02d}"


def cumulative_offsets(durations: list[float]) -> list[float]:
    """Return cumulative start-time offsets for a sequence of track durations.

    The first track always starts at 0.0.  Each subsequent offset is the sum
    of all preceding durations.

    Args:
        durations: List of track durations in seconds.

    Returns:
        List of float offsets with the same length as *durations*.
        Returns an empty list when *durations* is empty.

    Example:
        >>> cumulative_offsets([300.0, 400.0, 250.0])
        [0.0, 300.0, 700.0]
    """
    if not durations:
        return []
    offsets: list[float] = []
    total = 0.0
    for d in durations:
        offsets.append(total)
        total += d
    return offsets


def validate_cue_time(time_str: str) -> bool:
    """Return True if *time_str* is a valid CUE sheet time (MM:SS:FF).

    Rules:
        - Exactly 3 colon-separated fields.
        - All fields are non-negative integers (no leading sign, no spaces).
        - Seconds field: 0–59.
        - Frames field: 0–74.
        - Minutes field: any non-negative integer.

    Args:
        time_str: String to validate.

    Returns:
        True if valid, False otherwise.
    """
    parts = time_str.split(":")
    if len(parts) != 3:
        return False
    try:
        mins, secs, frames = (int(p) for p in parts)
    except ValueError:
        return False
    # Reject any field that had a leading sign (e.g. "-1")
    for raw in parts:
        if raw.startswith(("+", "-")):
            return False
    return mins >= 0 and 0 <= secs <= 59 and 0 <= frames <= 74
