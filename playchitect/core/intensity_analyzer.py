"""
Audio intensity analysis using librosa.

Analyzes tracks to extract intensity features including RMS energy,
spectral brightness, bass energy (3-way split), percussiveness, and onset strength.
"""

import hashlib
import json
import logging
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, overload

import librosa
import numpy as np

from playchitect.utils.config import get_config

logger = logging.getLogger(__name__)

# Normalization constants
_RMS_NORM_FACTOR: float = 0.3  # Typical peak RMS for normalized audio
_ONSET_NORM_FACTOR: float = 10.0  # Typical peak onset strength envelope mean

# Frequency band limits (Hz) — optimized for techno/electronic music
_FREQ_SUB_BASS_LOW: int = 20
_FREQ_SUB_BASS_HIGH: int = 60  # sub-bass upper / kick lower boundary
_FREQ_KICK_HIGH: int = 120  # kick upper / harmonics lower boundary
_FREQ_BASS_HARMONICS_HIGH: int = 250

# Energy gating threshold — frames below this RMS percentile are excluded from
# brightness calculation to avoid silence skewing the spectral centroid.
_ENERGY_GATE_PERCENTILE: int = 25


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
    def from_dict(cls, data: dict[str, Any]) -> "IntensityFeatures":
        """Create from dictionary."""
        data["filepath"] = Path(data["filepath"])
        return cls(**data)


class IntensityAnalyzer:
    """Analyzes audio intensity using librosa."""

    def __init__(
        self,
        sample_rate: int = 22050,
        cache_dir: Path | None = None,
        cache_enabled: bool = True,
    ):
        """
        Initialize intensity analyzer.

        Args:
            sample_rate: Target sample rate for analysis
            cache_dir: Directory for caching analysis results
            cache_enabled: Whether to use caching
        """
        self.sample_rate = sample_rate
        self.cache_enabled = cache_enabled

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

        if self.cache_enabled:
            cached = self._load_from_cache(file_hash)
            if cached is not None:
                logger.debug(f"Using cached analysis for: {filepath.name}")
                # Update filepath in case file was moved
                cached.filepath = filepath
                return cached

        logger.debug(f"Analyzing: {filepath.name}")

        # Load audio
        try:
            y, _ = librosa.load(filepath, sr=self.sample_rate, mono=True)
        except Exception as e:
            raise ValueError(f"Error loading audio file {filepath}: {e}")

        # Compute STFT once — all feature methods reuse this to avoid
        # redundant transform computation across the 5 feature extractors.
        S = np.abs(librosa.stft(y))

        rms = self._calculate_rms_energy(S)
        brightness = self._calculate_brightness(S, self.sample_rate)
        sub_bass, kick, harmonics = self._calculate_bass_energy(S, self.sample_rate)
        percussiveness = self._calculate_percussiveness(S)
        onset = self._calculate_onset_strength(S, self.sample_rate)

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
        )

        # Cache results
        if self.cache_enabled:
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
    ) -> dict[Path, "IntensityFeatures"]:
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

        n_total = len(filepaths)
        n_done = 0
        results: dict[Path, IntensityFeatures] = {}
        uncached: list[Path] = []

        # Cache pre-flight — resolve hits without touching the pool
        for filepath in filepaths:
            file_hash = self._compute_file_hash(filepath)
            cached = self._load_from_cache(file_hash)
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
                    _filepath_str, features_dict = future.result()
                    results[original_path] = IntensityFeatures.from_dict(features_dict)
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
