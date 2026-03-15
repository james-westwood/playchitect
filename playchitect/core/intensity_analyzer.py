"""
Audio intensity analysis using librosa.

Analyzes tracks to extract intensity features including RMS energy,
spectral brightness, bass energy (3-way split), percussiveness, and onset strength.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

if TYPE_CHECKING:
    from playchitect.core.cache_db import CacheDB

import librosa
import numpy as np

from playchitect.utils.config import get_config
from playchitect.utils.warnings import suppress_audio_log_warnings, suppress_librosa_warnings

logger = logging.getLogger(__name__)

# Default timeout for individual file analysis in batch mode (seconds)
_WORKER_TIMEOUT_SECS: int = 30


def harmonic_compatibility(key_a: str, key_b: str) -> bool:
    """
    Check if two Camelot Wheel keys are harmonically compatible.

    Compatible if:
    - Same number (±0 letter change): e.g., 8B ↔ 8A
    - Adjacent number (±1 same letter): e.g., 8B ↔ 9B or 7B
    - Same letter (±1 number): e.g., 8B ↔ 9B (already covered)

    Args:
        key_a: Camelot notation (e.g., '8B', '12A')
        key_b: Camelot notation (e.g., '8B', '3A')

    Returns:
        True if keys are compatible for mixing

    Raises:
        ValueError: If keys are not in valid Camelot format
    """

    # Parse Camelot notation
    def parse_key(key: str) -> tuple[int, str]:
        if len(key) < 2 or not key[-1].isalpha():
            raise ValueError(f"Invalid Camelot key format: {key}")
        try:
            number = int(key[:-1])
            letter = key[-1].upper()
        except ValueError as e:
            raise ValueError(f"Invalid Camelot key format: {key}") from e
        if number < 1 or number > 12:
            raise ValueError(f"Camelot number must be 1-12: {key}")
        if letter not in ("A", "B"):
            raise ValueError(f"Camelot letter must be A or B: {key}")
        return number, letter

    num_a, letter_a = parse_key(key_a)
    num_b, letter_b = parse_key(key_b)

    # Same number: always compatible (adjacent on wheel, mode switch)
    if num_a == num_b:
        return True

    # Same letter with adjacent numbers
    if letter_a == letter_b:
        # Adjacent on the wheel (wrapping: 12 adjacent to 1)
        diff = abs(num_a - num_b)
        if diff == 1 or diff == 11:  # 11 because 1 and 12 are adjacent
            return True

    return False


# Normalization constants
_RMS_NORM_FACTOR: float = 0.3  # Typical peak RMS for normalized audio
_ONSET_NORM_FACTOR: float = 10.0  # Typical peak onset strength envelope mean
_MAX_GRADIENT_RMS_PER_FRAME: float = 0.001  # Max expected RMS change per frame for gradient

# Frequency band limits (Hz) — optimized for techno/electronic music
_FREQ_SUB_BASS_LOW: int = 20
_FREQ_SUB_BASS_HIGH: int = 60  # sub-bass upper / kick lower boundary
_FREQ_KICK_HIGH: int = 120  # kick upper / harmonics lower boundary
_FREQ_BASS_HARMONICS_HIGH: int = 250

# Energy gating threshold — frames below this RMS percentile are excluded from
# brightness calculation to avoid silence skewing the spectral centroid.
_ENERGY_GATE_PERCENTILE: int = 25

# Default sample rate for librosa audio loading (Hz)
_DEFAULT_SAMPLE_RATE: int = 22050

# Camelot Wheel mapping for major keys (chromatic scale C=0)
# Maps chroma index 0-11 to (camelot_notation, key_index)
# Major keys only - minor detection out of scope
_CHROMA_TO_CAMELOT: dict[int, tuple[str, int]] = {
    0: ("8B", 0),  # C Major
    1: ("3B", 1),  # C# Major
    2: ("10B", 2),  # D Major
    3: ("5B", 3),  # D# Major
    4: ("12B", 4),  # E Major
    5: ("7B", 5),  # F Major
    6: ("2B", 6),  # F# Major
    7: ("9B", 7),  # G Major
    8: ("4B", 8),  # G# Major
    9: ("11B", 9),  # A Major
    10: ("6B", 10),  # A# Major
    11: ("1B", 11),  # B Major
}


def _analyze_worker(args: tuple[str, str]) -> tuple[str, dict[str, Any]]:
    """
    Module-level worker for ProcessPoolExecutor.

    Must be at module level (not a bound method) so it can be pickled
    across process boundaries.

    Args:
        args: (filepath_str, cache_dir_str)

    Returns:
        (filepath_str, features_dict) — plain dict avoids cross-process pickling issues

    Raises:
        Exception propagated to the future on error
    """
    filepath_str, cache_dir_str = args
    analyzer = IntensityAnalyzer(cache_dir=Path(cache_dir_str))
    with suppress_librosa_warnings(), suppress_audio_log_warnings():
        features = analyzer.analyze(Path(filepath_str))
    return (filepath_str, features.to_dict())


@dataclass
class IntensityFeatures:
    """Container for intensity analysis features."""

    # File identification
    filepath: Path
    file_hash: str  # MD5 hash for cache validation

    # Energy features
    rms_energy: float  # Overall loudness (0-1)

    # Spectral features
    brightness: float  # Spectral centroid, RMS-weighted (0-1)

    # Bass energy (3-way split for techno)
    sub_bass_energy: float  # 20-60Hz (sub-kick, rumble)
    kick_energy: float  # 60-120Hz (main kick fundamental)
    bass_harmonics: float  # 120-250Hz (bass notes)

    # Rhythmic features
    percussiveness: float  # HPSS ratio (0-1)
    onset_strength: float  # Beat intensity (0-1)

    # Harmonic features
    camelot_key: str  # Camelot Wheel notation (e.g., '8B')
    key_index: float  # Chroma bin index 0-11 as float

    # Energy flow features (metadata, not clustering dimensions)
    dynamic_range: float = 0.0  # Range of RMS energy (0-1)
    energy_gradient: float = 0.0  # Trend of energy over time (-1 to 1)
    drop_density: float = 0.0  # Normalised drops per minute (0-1)

    # Timbre/texture features (metadata, not clustering dimensions)
    spectral_flatness: float = 0.0  # Noisiness vs tonal (0-1)
    zero_crossing_rate: float = 0.0  # Noisiness / high-freq content (0-1)
    mfcc_variance: float = 0.0  # Spectral shape complexity (0-1)
    spectral_rolloff_85: float = 0.0  # Frequency where 85% energy is below (0-1)

    # Structural/vocal features (metadata, not clustering dimensions)
    vocal_presence: float = 0.0  # Vocal content in 200-800Hz harmonic band (0-1)
    intro_length_secs: float = 0.0  # Time until energy exceeds threshold (seconds)

    @property
    def hardness(self) -> float:
        """
        Combined hardness/intensity score (0.0–1.0).
        Weights emphasize treble brightness and percussive drive (techno-optimized).
        """
        score = (
            0.4 * self.brightness
            + 0.2 * self.rms_energy
            + 0.2 * self.percussiveness
            + 0.2 * self.onset_strength
        )
        return float(np.clip(score, 0.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["filepath"] = str(self.filepath)
        return result

    @overload
    def to_feature_vector(self, include_filepath: Literal[False] = ...) -> np.ndarray: ...

    @overload
    def to_feature_vector(self, include_filepath: Literal[True]) -> dict[str, Any]: ...

    def to_feature_vector(self, include_filepath: bool = False) -> np.ndarray | dict[str, Any]:
        """
        Convert to numpy feature vector (7 dimensions, excludes file_hash).

        Args:
            include_filepath: If True, return as dict with filepath and features keys.

        Returns:
            7-dimensional numpy array, or dict with 'filepath' and 'features' keys
            if include_filepath is True.
        """
        vector = np.array(
            [
                self.rms_energy,
                self.brightness,
                self.sub_bass_energy,
                self.kick_energy,
                self.bass_harmonics,
                self.percussiveness,
                self.onset_strength,
            ]
        )

        if include_filepath:
            return {"filepath": self.filepath, "features": vector}
        return vector

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IntensityFeatures:
        """Create from dictionary.

        Handles backward compatibility with old cache files that may not
        have the camelot_key, key_index, dynamic_range, energy_gradient,
        or drop_density fields.
        """
        data["filepath"] = Path(data["filepath"])
        # Handle old cache files missing harmonic fields
        if "camelot_key" not in data:
            data["camelot_key"] = "8B"  # Default to C major
        if "key_index" not in data:
            data["key_index"] = 0.0
        # Handle old cache files missing energy flow fields
        if "dynamic_range" not in data:
            data["dynamic_range"] = 0.0
        if "energy_gradient" not in data:
            data["energy_gradient"] = 0.0
        if "drop_density" not in data:
            data["drop_density"] = 0.0
        # Handle old cache files missing timbre/texture fields
        if "spectral_flatness" not in data:
            data["spectral_flatness"] = 0.0
        if "zero_crossing_rate" not in data:
            data["zero_crossing_rate"] = 0.0
        if "mfcc_variance" not in data:
            data["mfcc_variance"] = 0.0
        if "spectral_rolloff_85" not in data:
            data["spectral_rolloff_85"] = 0.0
        # Handle old cache files missing structural/vocal fields
        if "vocal_presence" not in data:
            data["vocal_presence"] = 0.0
        if "intro_length_secs" not in data:
            data["intro_length_secs"] = 0.0
        return cls(**data)


class IntensityAnalyzer:
    """Analyzes audio intensity using librosa."""

    def __init__(
        self,
        sample_rate: int = _DEFAULT_SAMPLE_RATE,
        cache_dir: Path | None = None,
        cache_enabled: bool = True,
        cache_db: CacheDB | None = None,
    ):
        """
        Initialize intensity analyzer.

        Args:
            sample_rate:  Target sample rate for analysis.
            cache_dir:    Directory for the legacy JSON cache. Still used by
                          subprocess workers regardless of ``cache_db``.
            cache_enabled: Whether to use any caching at all.
            cache_db:     Optional SQLite-backed cache.  When provided,
                          ``analyze_batch`` loads the full cache in a single
                          query instead of N individual JSON reads, and new
                          results are stored in the DB after analysis.
                          Falls back to JSON cache when ``None``.
        """
        self.sample_rate = sample_rate
        self.cache_enabled = cache_enabled
        self.cache_db = cache_db
        self._db_migrated: bool = False

        if cache_dir is None:
            env_cache_dir = os.environ.get("PLAYCHITECT_CACHE_DIR")
            if env_cache_dir:
                cache_dir = Path(env_cache_dir) / "intensity"
            else:
                cache_dir = get_config().get_cache_dir() / "intensity"
        self.cache_dir = Path(cache_dir)

        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def analyze(self, filepath: Path) -> IntensityFeatures:
        """
        Analyze audio file for intensity features.

        Args:
            filepath: Path to audio file

        Returns:
            IntensityFeatures object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be read
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        # Check cache first
        file_hash = self._compute_file_hash(filepath)

        if self.cache_db is not None:
            cached = self.cache_db.get_intensity(file_hash)
            if cached is not None:
                logger.debug(f"Using cached analysis for: {filepath.name}")
                cached.filepath = filepath
                return cached
        elif self.cache_enabled:
            cached = self._load_from_cache(file_hash)
            if cached is not None:
                logger.debug(f"Using cached analysis for: {filepath.name}")
                # Update filepath in case file was moved
                cached.filepath = filepath
                return cached

        logger.debug(f"Analyzing: {filepath.name}")

        # Load audio — suppress backend-negotiation warnings from librosa /
        # audioread / soundfile so they never appear in user-facing output.
        with suppress_librosa_warnings(), suppress_audio_log_warnings():
            try:
                # Use duration=300 limit to avoid OOM on huge files, enough for intensity
                y, _ = librosa.load(filepath, sr=self.sample_rate, mono=True, duration=300)

                # Defensive check for empty or near-empty audio
                if y is None or len(y) < 100:
                    raise ValueError("Audio file is empty or too short")

                # Compute STFT once
                S = np.abs(librosa.stft(y))

                # Ensure S has content
                if S.size == 0 or np.max(S) < 1e-7:
                    raise ValueError("No signal detected in audio")

                rms = self._calculate_rms_energy(S)
                brightness = self._calculate_brightness(S, self.sample_rate)
                sub_bass, kick, harmonics = self._calculate_bass_energy(S, self.sample_rate)
                percussiveness = self._calculate_percussiveness(S)
                onset = self._calculate_onset_strength(S, self.sample_rate)

                # Compute chroma features for key detection
                camelot_key, key_index = self._calculate_key(y, self.sample_rate)

                # Compute energy flow features from RMS frames
                dynamic_range, energy_gradient, drop_density = self._calculate_energy_flow_features(
                    y
                )

                # Compute timbre/texture features
                spectral_flatness = self._calculate_spectral_flatness(y)
                zcr = self._calculate_zero_crossing_rate(y)
                mfcc_var = self._calculate_mfcc_variance(y, self.sample_rate)
                sr_85 = self._calculate_spectral_rolloff_85(y, self.sample_rate)

                # Compute structural/vocal features
                vocal_presence = self._calculate_vocal_presence(y, self.sample_rate)
                intro_length_secs = self._calculate_intro_length(y, self.sample_rate)

            except Exception as e:
                # Re-raise as ValueError with context for analyze_batch to catch
                raise ValueError(f"Failed to analyze {filepath.name}: {e}") from e

        features = IntensityFeatures(
            filepath=filepath,
            file_hash=file_hash,
            rms_energy=rms,
            brightness=brightness,
            sub_bass_energy=sub_bass,
            kick_energy=kick,
            bass_harmonics=harmonics,
            percussiveness=percussiveness,
            onset_strength=onset,
            camelot_key=camelot_key,
            key_index=key_index,
            dynamic_range=dynamic_range,
            energy_gradient=energy_gradient,
            drop_density=drop_density,
            spectral_flatness=spectral_flatness,
            zero_crossing_rate=zcr,
            mfcc_variance=mfcc_var,
            spectral_rolloff_85=sr_85,
            vocal_presence=vocal_presence,
            intro_length_secs=intro_length_secs,
        )

        # Cache results
        if self.cache_db is not None:
            self.cache_db.put_intensity(file_hash, features)
        elif self.cache_enabled:
            self._save_to_cache(file_hash, features)

        return features

    def _calculate_rms_energy(self, S: np.ndarray) -> float:
        """
        Calculate RMS energy (overall loudness) from magnitude spectrogram.

        Args:
            S: Magnitude spectrogram

        Returns:
            Normalized RMS energy (0-1)
        """
        rms = librosa.feature.rms(S=S)[0]
        rms_mean = float(np.mean(rms))
        return float(np.clip(rms_mean / _RMS_NORM_FACTOR, 0.0, 1.0))

    def _calculate_brightness(self, S: np.ndarray, sr: int) -> float:
        """
        Calculate brightness (spectral centroid) with RMS weighting.

        Louder frames count more. Energy gating avoids silence skewing the result.

        Args:
            S: Magnitude spectrogram
            sr: Sample rate

        Returns:
            Normalized brightness (0-1)
        """
        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        frame_rms = librosa.feature.rms(S=S)[0]

        # Energy gating: only use frames above the configured percentile
        energy_threshold = np.percentile(frame_rms, _ENERGY_GATE_PERCENTILE)
        valid_frames = frame_rms > energy_threshold

        if np.sum(valid_frames) == 0:
            # All frames too quiet, fall back to unweighted mean
            brightness_hz = float(np.mean(centroid))
        else:
            centroid_gated = centroid[valid_frames]
            rms_gated = frame_rms[valid_frames]
            weights = rms_gated / np.sum(rms_gated)
            brightness_hz = float(np.sum(centroid_gated * weights))

        brightness_normalized = brightness_hz / (sr / 2.0)
        return float(np.clip(brightness_normalized, 0.0, 1.0))

    def _calculate_bass_energy(self, S: np.ndarray, sr: int) -> tuple[float, float, float]:
        """
        Calculate bass energy in 3 frequency bands.

        For techno: sub-bass (20-60Hz), kick (60-120Hz), harmonics (120-250Hz)

        Args:
            S: Magnitude spectrogram
            sr: Sample rate

        Returns:
            Tuple of (sub_bass, kick, harmonics) energies, normalized 0-1
        """
        freqs = librosa.fft_frequencies(sr=sr)

        sub_bass_mask = (freqs >= _FREQ_SUB_BASS_LOW) & (freqs < _FREQ_SUB_BASS_HIGH)
        kick_mask = (freqs >= _FREQ_SUB_BASS_HIGH) & (freqs < _FREQ_KICK_HIGH)
        harmonics_mask = (freqs >= _FREQ_KICK_HIGH) & (freqs < _FREQ_BASS_HARMONICS_HIGH)

        sub_bass_energy = float(np.mean(S[sub_bass_mask, :]))
        kick_energy = float(np.mean(S[kick_mask, :]))
        harmonics_energy = float(np.mean(S[harmonics_mask, :]))

        # Normalize relative to total low-freq energy (20-250Hz)
        low_freq_mask = (freqs >= _FREQ_SUB_BASS_LOW) & (freqs < _FREQ_BASS_HARMONICS_HIGH)
        total_low_energy = float(np.mean(S[low_freq_mask, :]))

        if total_low_energy > 0:
            sub_bass_norm = sub_bass_energy / total_low_energy
            kick_norm = kick_energy / total_low_energy
            harmonics_norm = harmonics_energy / total_low_energy
        else:
            return 0.0, 0.0, 0.0

        return (
            float(np.clip(sub_bass_norm, 0.0, 1.0)),
            float(np.clip(kick_norm, 0.0, 1.0)),
            float(np.clip(harmonics_norm, 0.0, 1.0)),
        )

    def _calculate_percussiveness(self, S: np.ndarray) -> float:
        """
        Calculate percussiveness using HPSS on the pre-computed spectrogram.

        Uses librosa.decompose.hpss directly on the magnitude spectrogram,
        avoiding the iSTFT round-trip that librosa.effects.hpss performs for
        a ~2x speedup on this step.

        Args:
            S: Magnitude spectrogram

        Returns:
            Percussive ratio (0-1)
        """
        _, P = librosa.decompose.hpss(S)

        percussive_rms = float(np.sqrt(np.mean(P**2)))
        total_rms = float(np.sqrt(np.mean(S**2)))

        if total_rms > 0:
            perc_ratio = percussive_rms / total_rms
        else:
            perc_ratio = 0.0

        return float(np.clip(perc_ratio, 0.0, 1.0))

    def _calculate_onset_strength(self, S: np.ndarray, sr: int) -> float:
        """
        Calculate onset strength (beat intensity) from pre-computed spectrogram.

        Args:
            S: Magnitude spectrogram
            sr: Sample rate

        Returns:
            Normalized onset strength (0-1)
        """
        onset_env = librosa.onset.onset_strength(S=S, sr=sr)
        onset_mean = float(np.mean(onset_env))
        return float(np.clip(onset_mean / _ONSET_NORM_FACTOR, 0.0, 1.0))

    def _calculate_key(self, y: np.ndarray, sr: int) -> tuple[str, float]:
        """
        Calculate musical key using chroma features.

        Uses Constant-Q Transform (CQT) chroma analysis to determine
        the dominant pitch class, then maps to Camelot Wheel notation.
        Major keys only - minor detection is out of scope.

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Tuple of (camelot_key_str, key_index_float)
        """
        # Compute chroma features using CQT
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

        # Average across time frames to get 12-bin chroma vector
        chroma_mean = chroma.mean(axis=1)

        # Find dominant chroma bin (0-11, where 0=C, 1=C#, ..., 11=B)
        dominant_bin = int(np.argmax(chroma_mean))

        # Map to Camelot notation
        camelot_key, key_index = _CHROMA_TO_CAMELOT[dominant_bin]

        return camelot_key, float(key_index)

    def _calculate_energy_flow_features(self, y: np.ndarray) -> tuple[float, float, float]:
        """
        Calculate energy flow features: dynamic range, gradient, and drop density.

        Args:
            y: Audio time series

        Returns:
            Tuple of (dynamic_range, energy_gradient, drop_density)
        """
        # Compute RMS frames
        rms_frames = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

        # Dynamic range: normalized difference between max and min RMS
        dynamic_range = float(
            np.clip((rms_frames.max() - rms_frames.min()) / (_RMS_NORM_FACTOR + 1e-9), 0, 1)
        )

        # Energy gradient: trend of RMS over time, normalized to [-1, 1]
        if len(rms_frames) > 1:
            gradient = np.polyfit(np.arange(len(rms_frames)), rms_frames, 1)[0]
            # Clip at ±_MAX_GRADIENT_RMS_PER_FRAME then divide to normalize
            gradient_clipped = np.clip(
                gradient, -_MAX_GRADIENT_RMS_PER_FRAME, _MAX_GRADIENT_RMS_PER_FRAME
            )
            energy_gradient = float(gradient_clipped / _MAX_GRADIENT_RMS_PER_FRAME)
        else:
            energy_gradient = 0.0

        # Drop density: count high-energy frames divided by duration in minutes
        # Normalised to [0, 1] by clipping at 10 drops/min
        mean_rms = rms_frames.mean()
        std_rms = rms_frames.std()
        threshold = mean_rms + 2 * std_rms
        drops = np.sum(rms_frames > threshold)

        # Track duration in minutes
        hop_length = 512
        sr = self.sample_rate
        duration_minutes = (len(rms_frames) * hop_length) / (sr * 60)

        if duration_minutes > 0:
            drops_per_minute = drops / duration_minutes
            drop_density = float(np.clip(drops_per_minute / 10, 0, 1))
        else:
            drop_density = 0.0

        return dynamic_range, energy_gradient, drop_density

    def _calculate_spectral_flatness(self, y: np.ndarray) -> float:
        """
        Calculate spectral flatness (noisiness vs tonal).

        Args:
            y: Audio time series

        Returns:
            Normalized spectral flatness (0-1), clipped at 0.5
        """
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        # Normalize to [0, 1] by clipping at 0.5 and dividing
        return float(np.clip(flatness / 0.5, 0.0, 1.0))

    def _calculate_zero_crossing_rate(self, y: np.ndarray) -> float:
        """
        Calculate zero crossing rate (noisiness indicator).

        Args:
            y: Audio time series

        Returns:
            Normalized ZCR (0-1), clipped at 0.5
        """
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)[0]))
        # Normalize to [0, 1] by clipping at 0.5 and dividing
        return float(np.clip(zcr / 0.5, 0.0, 1.0))

    def _calculate_mfcc_variance(self, y: np.ndarray, sr: int) -> float:
        """
        Calculate MFCC variance (spectral shape complexity).

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Normalized MFCC variance (0-1), divided by 100 and clipped
        """
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Calculate variance along time axis for each MFCC coefficient
        variances = np.var(mfccs, axis=1)
        # Mean variance across all coefficients
        mfcc_var = float(np.mean(variances))
        # Normalize to [0, 1] by dividing by 100 and clipping
        return float(np.clip(mfcc_var / 100.0, 0.0, 1.0))

    def _calculate_spectral_rolloff_85(self, y: np.ndarray, sr: int) -> float:
        """
        Calculate spectral rolloff at 85% (frequency below which 85% of energy resides).

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Normalized rolloff (0-1), normalized to Nyquist frequency (sr/2)
        """
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        rolloff_mean = float(np.mean(rolloff))
        # Normalize to [0, 1] by dividing by Nyquist frequency (sr/2)
        nyquist = sr / 2.0
        return float(np.clip(rolloff_mean / nyquist, 0.0, 1.0))

    def _calculate_vocal_presence(self, y: np.ndarray, sr: int) -> float:
        """
        Calculate vocal presence as ratio of harmonic energy in 200-800Hz band.

        Uses HPSS to separate harmonic component, then measures what proportion
        of the harmonic energy falls in the typical vocal frequency range.

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Vocal presence score (0-1), scaled to account for vocal band being a subset
        """
        # Separate harmonic component using HPSS
        y_harmonic, _ = librosa.effects.hpss(y)

        # Compute harmonic spectrogram
        harmonic_spec = np.abs(librosa.stft(y_harmonic))

        # Find frequency bins for 200-800Hz (typical vocal range)
        freqs = librosa.fft_frequencies(sr=sr)
        vocal_bins = (freqs >= 200) & (freqs < 800)

        # Calculate vocal band energy and total harmonic energy
        vocal_energy = harmonic_spec[vocal_bins].sum()
        total_harmonic_energy = harmonic_spec.sum() + 1e-9  # Small epsilon to avoid div by zero

        # Scale by 10 because vocal band is a subset of total spectrum
        vocal_presence = float(np.clip(vocal_energy / total_harmonic_energy * 10, 0, 1))

        return vocal_presence

    def _calculate_intro_length(self, y: np.ndarray, sr: int) -> float:
        """
        Calculate intro length as time until energy exceeds threshold.

        Uses RMS frames to find when the track's energy first exceeds
        50% of the mean RMS energy, indicating the end of the intro.

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Intro length in seconds (0 if no intro detected)
        """
        # Compute RMS frames (same parameters as energy_gradient step)
        rms_frames = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]

        if len(rms_frames) == 0:
            return 0.0

        # Threshold is 50% of mean RMS energy
        threshold = 0.5 * np.mean(rms_frames)

        # Find first frame that exceeds threshold
        intro_frames = int(np.argmax(rms_frames > threshold))

        # If no frame exceeds threshold, argmax returns 0; verify it actually exceeds
        if intro_frames == 0 and rms_frames[0] <= threshold:
            return 0.0

        # Convert frames to seconds: frame_length=2048, hop_length=512
        intro_length_secs = float(intro_frames * 512 / sr)

        return intro_length_secs

    def _compute_file_hash(self, filepath: Path) -> str:
        """
        Compute MD5 hash of file for cache validation.

        Only hashes first 1MB for speed.

        Args:
            filepath: Path to file

        Returns:
            MD5 hash string
        """
        md5 = hashlib.md5()

        with open(filepath, "rb") as f:
            chunk = f.read(1024 * 1024)
            md5.update(chunk)

        return md5.hexdigest()

    def _get_cache_path(self, file_hash: str) -> Path:
        """Get cache file path for given file hash."""
        return self.cache_dir / f"{file_hash}.json"

    def _save_to_cache(self, file_hash: str, features: IntensityFeatures) -> None:
        """Save features to cache."""
        cache_path = self._get_cache_path(file_hash)

        try:
            with open(cache_path, "w") as f:
                json.dump(features.to_dict(), f, indent=2)
            logger.debug(f"Cached analysis: {cache_path.name}")
        except Exception as e:
            logger.warning(f"Failed to cache analysis: {e}")

    def _load_from_cache(self, file_hash: str) -> IntensityFeatures | None:
        """Load features from cache."""
        cache_path = self._get_cache_path(file_hash)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)
            return IntensityFeatures.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load cached analysis: {e}")
            return None

    @staticmethod
    def estimate_batch_seconds(n_uncached: int) -> float:
        """
        Rough wall-clock estimate for user-facing warnings.

        Based on ~2 s/track with default parallel workers on a modern 8-core machine.
        """
        return n_uncached * 2.0

    def analyze_batch(
        self,
        filepaths: list[Path],
        max_workers: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[Path, IntensityFeatures]:
        """
        Analyse multiple audio files in parallel using ProcessPoolExecutor.

        Cache pre-flight: files already cached are resolved immediately without
        spawning workers. Only uncached files are sent to the pool.

        Args:
            filepaths:          Files to analyse.
            max_workers:        Worker processes. Defaults to min(cpu_count()//2, 6).
            progress_callback:  Called as callback(n_done, n_total) after each
                                file completes (cached or computed). Optional.

        Returns:
            Mapping of Path -> IntensityFeatures. Files that fail are omitted
            and logged at ERROR level — no exception is raised.
        """
        if max_workers is None:
            max_workers = max(1, min((os.cpu_count() or 1) // 2, 6))

        # One-time migration from JSON on first DB-backed batch
        if self.cache_db is not None and not self._db_migrated:
            from playchitect.core.cache_db import migrate_json_cache

            migrate_json_cache(self.cache_dir, self.cache_db)
            self._db_migrated = True

        n_total = len(filepaths)
        n_done = 0
        results: dict[Path, IntensityFeatures] = {}
        uncached: list[Path] = []

        # Bulk-load DB cache in one query (replaces N individual JSON reads)
        db_cache: dict[str, IntensityFeatures] = {}
        if self.cache_db is not None:
            db_cache = self.cache_db.load_all_intensity()

        # Cache pre-flight — resolve hits without touching the pool
        for filepath in filepaths:
            file_hash = self._compute_file_hash(filepath)
            cached: IntensityFeatures | None
            if self.cache_db is not None:
                cached = db_cache.get(file_hash)
            else:
                cached = self._load_from_cache(file_hash)  # always check, same as original
            if cached is not None:
                cached.filepath = filepath
                results[filepath] = cached
                n_done += 1
                if progress_callback is not None:
                    progress_callback(n_done, n_total)
            else:
                uncached.append(filepath)

        if not uncached:
            return results

        # Submit uncached files to the process pool
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(_analyze_worker, (str(p), str(self.cache_dir))): p for p in uncached
            }
            for future in as_completed(future_to_path):
                original_path = future_to_path[future]
                try:
                    # Timeout protects against individual file analysis hanging (Issue #95)
                    _filepath_str, features_dict = future.result(timeout=_WORKER_TIMEOUT_SECS)
                    features = IntensityFeatures.from_dict(features_dict)
                    results[original_path] = features
                    if self.cache_db is not None:
                        self.cache_db.put_intensity(features.file_hash, features)
                except Exception as exc:
                    logger.error("Failed to analyse %s: %s", original_path, exc)
                finally:
                    n_done += 1
                    if progress_callback is not None:
                        progress_callback(n_done, n_total)

        return results

    def clear_cache(self) -> None:
        """Clear all cached analysis results."""
        if not self.cache_dir.exists():
            return

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        logger.info(f"Cleared {count} cached analyses")
