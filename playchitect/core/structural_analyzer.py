"""
Audio structural analysis to predict mix points and inject cues.

Uses librosa to detect energy-based boundaries (intro, outro, drops)
for DJ mix point prediction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import librosa
import numpy as np
from scipy.signal import find_peaks

from playchitect.utils.warnings import (
    suppress_audio_log_warnings,
    suppress_c_stderr,
    suppress_librosa_warnings,
)

if TYPE_CHECKING:
    from playchitect.core.metadata_extractor import TrackMetadata

logger = logging.getLogger(__name__)

# Constants for boundary detection
_INTRO_ENERGY_THRESHOLD_PERCENTILE: float = 20.0  # % of peak RMS for intro end
_OUTRO_ENERGY_THRESHOLD_PERCENTILE: float = 20.0  # % of peak RMS for outro start
_DROP_DETECTION_PERCENTILE: float = 75.0  # % of RMS for drop detection
_MIN_PEAK_DISTANCE_SECS: float = 10.0  # Minimum seconds between drops
_SAMPLE_RATE: int = 22050  # Sample rate for librosa analysis
_RMS_FRAME_LENGTH: int = 2048  # Frame length for RMS computation
_RMS_HOP_LENGTH: int = 512  # Hop length for RMS computation
_TARGET_SAMPLE_RATE_CUES: int = 44100  # Sample rate for cue positions (Mixxx standard)


@dataclass
class StructuralAnalysis:
    """Container for structural analysis results.

    All time values are in milliseconds (ms).
    """

    intro_end_ms: float  # Time when energy first exceeds threshold (intro end)
    outro_start_ms: float  # Time when energy drops below threshold (outro start)
    breakdowns: list[float]  # List of breakdown timestamps (low energy sections)
    drops: list[float]  # List of drop timestamps (energy peaks)


def predict_cue_points(analysis: StructuralAnalysis) -> dict[str, float]:
    """
    Predict cue points from structural analysis.

    Returns cue points suitable for hot cue injection into DJ software.
    cue_1_ms is the intro end (first major energy rise).
    cue_2_ms is the outro start (last major energy drop).

    Args:
        analysis: StructuralAnalysis from StructuralAnalyzer.analyze()

    Returns:
        Dict with 'cue_1_ms' and 'cue_2_ms' keys mapping to millisecond positions
    """
    return {
        "cue_1_ms": analysis.intro_end_ms,
        "cue_2_ms": analysis.outro_start_ms,
    }


class StructuralAnalyzer:
    """Analyzes audio structure to detect mix points.

    Uses energy-based analysis to identify:
    - Intro end: first significant energy rise
    - Outro start: last significant energy drop
    - Breakdowns: sections of sustained low energy
    - Drops: local energy maxima above threshold
    """

    def __init__(self, sample_rate: int = _SAMPLE_RATE):
        """
        Initialize structural analyzer.

        Args:
            sample_rate: Target sample rate for librosa analysis
        """
        self.sample_rate = sample_rate

    def analyze(self, filepath: Path, metadata: TrackMetadata | None = None) -> StructuralAnalysis:
        """
        Analyze audio file for structural boundaries.

        Detects intro/outro boundaries and energy events (drops/breakdowns)
        using RMS energy analysis.

        Args:
            filepath: Path to audio file
            metadata: Optional track metadata (not currently used, for future extensions)

        Returns:
            StructuralAnalysis with detected boundaries

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be analyzed
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        logger.debug(f"Analyzing structure: {filepath.name}")

        # Load audio with warnings suppressed
        with suppress_librosa_warnings(), suppress_audio_log_warnings(), suppress_c_stderr():
            try:
                y, sr = librosa.load(
                    filepath,
                    sr=self.sample_rate,
                    mono=True,
                    duration=600,  # Limit to 10 min
                )

                if y is None or len(y) < 100:
                    raise ValueError("Audio file is empty or too short")

            except Exception as e:
                raise ValueError(f"Failed to load audio: {e}") from e

        # Compute RMS energy envelope
        rms = librosa.feature.rms(y=y, frame_length=_RMS_FRAME_LENGTH, hop_length=_RMS_HOP_LENGTH)[
            0
        ]

        if len(rms) == 0 or np.max(rms) < 1e-7:
            raise ValueError("No signal detected in audio")

        # Convert frame indices to milliseconds
        hop_duration_ms = (_RMS_HOP_LENGTH / sr) * 1000.0

        # Detect boundaries
        intro_end_ms = self._detect_intro_end(rms, hop_duration_ms)
        outro_start_ms = self._detect_outro_start(rms, hop_duration_ms)
        drops = self._detect_drops(rms, hop_duration_ms, int(sr))
        breakdowns = self._detect_breakdowns(rms, hop_duration_ms)

        analysis = StructuralAnalysis(
            intro_end_ms=intro_end_ms,
            outro_start_ms=outro_start_ms,
            drops=drops,
            breakdowns=breakdowns,
        )

        logger.debug(
            f"Structure analysis complete: intro={intro_end_ms:.0f}ms, "
            f"outro={outro_start_ms:.0f}ms, {len(drops)} drops, {len(breakdowns)} breakdowns"
        )

        return analysis

    def _detect_intro_end(self, rms: np.ndarray, hop_duration_ms: float) -> float:
        """
        Detect the end of the intro section.

        Finds the first frame where energy exceeds the threshold (20% of peak).

        Args:
            rms: RMS energy envelope
            hop_duration_ms: Duration of each hop in milliseconds

        Returns:
            Time in milliseconds where intro ends
        """
        peak_rms = np.max(rms)
        threshold = peak_rms * (_INTRO_ENERGY_THRESHOLD_PERCENTILE / 100.0)

        # Find first frame exceeding threshold
        intro_frames = int(np.argmax(rms > threshold))

        # If no frame exceeds threshold (shouldn't happen), use first frame
        if intro_frames == 0 and rms[0] <= threshold:
            # Find first non-zero frame as fallback
            non_zero = np.where(rms > 0)[0]
            if len(non_zero) > 0:
                intro_frames = non_zero[0]

        return float(intro_frames * hop_duration_ms)

    def _detect_outro_start(self, rms: np.ndarray, hop_duration_ms: float) -> float:
        """
        Detect the start of the outro section.

        Finds the last frame where energy exceeds the threshold (20% of peak).

        Args:
            rms: RMS energy envelope
            hop_duration_ms: Duration of each hop in milliseconds

        Returns:
            Time in milliseconds where outro starts
        """
        peak_rms = np.max(rms)
        threshold = peak_rms * (_OUTRO_ENERGY_THRESHOLD_PERCENTILE / 100.0)

        # Find last frame exceeding threshold
        # Reverse the array and find first match (which is the last in original)
        reversed_rms = rms[::-1]
        reversed_idx = int(np.argmax(reversed_rms > threshold))

        # Convert back to original index
        outro_frames = len(rms) - 1 - reversed_idx

        # If no frame exceeds threshold, use last frame
        if reversed_idx == 0 and reversed_rms[0] <= threshold:
            # Find last non-zero frame as fallback
            non_zero = np.where(rms > 0)[0]
            if len(non_zero) > 0:
                outro_frames = non_zero[-1]

        return float(outro_frames * hop_duration_ms)

    def _detect_drops(self, rms: np.ndarray, hop_duration_ms: float, sr: int) -> list[float]:
        """
        Detect drop points (energy peaks above 75th percentile).

        Uses scipy.signal.find_peaks to identify local maxima.

        Args:
            rms: RMS energy envelope
            hop_duration_ms: Duration of each hop in milliseconds
            sr: Sample rate for distance calculation

        Returns:
            List of drop timestamps in milliseconds
        """
        threshold = np.percentile(rms, _DROP_DETECTION_PERCENTILE)

        # Minimum distance between peaks (in samples)
        min_distance = int(_MIN_PEAK_DISTANCE_SECS * sr / _RMS_HOP_LENGTH)

        # Find peaks above threshold
        peaks, _ = find_peaks(rms, height=threshold, distance=min_distance)

        # Convert to milliseconds
        drops = [float(p * hop_duration_ms) for p in peaks]

        return drops

    def _detect_breakdowns(self, rms: np.ndarray, hop_duration_ms: float) -> list[float]:
        """
        Detect breakdown sections (sustained low energy).

        Currently returns empty list as a placeholder for future implementation.
        Breakdown detection requires more complex analysis of sustained low-energy
        sections between drops.

        Args:
            rms: RMS energy envelope
            hop_duration_ms: Duration of each hop in milliseconds

        Returns:
            List of breakdown timestamps (currently empty)
        """
        # TODO: Implement breakdown detection
        # This would detect sections where energy drops below a threshold
        # for a sustained period between drops
        return []
