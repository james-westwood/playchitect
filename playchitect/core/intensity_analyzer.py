"""
Audio intensity analysis using librosa.

Analyzes tracks to extract intensity features including RMS energy,
spectral brightness, bass energy (3-way split), percussiveness, and onset strength.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

import numpy as np
import librosa

logger = logging.getLogger(__name__)


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["filepath"] = str(self.filepath)
        return result

    def to_feature_vector(self, include_filepath: bool = False) -> np.ndarray:
        """
        Convert to numpy feature vector (7 dimensions, excludes file_hash).

        Args:
            include_filepath: If True, return as dict with filepath key

        Returns:
            7-dimensional numpy array of features
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
    def from_dict(cls, data: Dict[str, Any]) -> "IntensityFeatures":
        """Create from dictionary."""
        data["filepath"] = Path(data["filepath"])
        return cls(**data)


class IntensityAnalyzer:
    """Analyzes audio intensity using librosa."""

    def __init__(
        self,
        sample_rate: int = 22050,
        cache_dir: Optional[Path] = None,
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
            cache_dir = Path.home() / ".cache" / "playchitect" / "intensity"
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
            y, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)
        except Exception as e:
            raise ValueError(f"Error loading audio file {filepath}: {e}")

        # Calculate features
        rms = self._calculate_rms_energy(y)
        brightness = self._calculate_brightness(y, sr, rms)
        sub_bass, kick, harmonics = self._calculate_bass_energy(y, sr)
        percussiveness = self._calculate_percussiveness(y)
        onset = self._calculate_onset_strength(y, sr)

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

    def _calculate_rms_energy(self, y: np.ndarray) -> float:
        """
        Calculate RMS energy (overall loudness).

        Args:
            y: Audio time series

        Returns:
            Normalized RMS energy (0-1)
        """
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = float(np.mean(rms))

        # Normalize to 0-1 (typical RMS range is 0-0.3 for normalized audio)
        rms_normalized = float(np.clip(rms_mean / 0.3, 0.0, 1.0))

        return rms_normalized

    def _calculate_brightness(self, y: np.ndarray, sr: int, rms: float) -> float:
        """
        Calculate brightness (spectral centroid) with RMS weighting.

        Louder frames count more. Energy gating avoids silence issues.

        Args:
            y: Audio time series
            sr: Sample rate
            rms: Pre-calculated RMS energy

        Returns:
            Normalized brightness (0-1)
        """
        # Calculate spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

        # Calculate frame RMS for weighting
        frame_rms = librosa.feature.rms(y=y)[0]

        # Energy gating: only use frames above 25th percentile
        energy_threshold = np.percentile(frame_rms, 25)
        valid_frames = frame_rms > energy_threshold

        if np.sum(valid_frames) == 0:
            # All frames too quiet, use unweighted mean
            brightness_hz = float(np.mean(centroid))
        else:
            # RMS-weighted mean of gated frames
            centroid_gated = centroid[valid_frames]
            rms_gated = frame_rms[valid_frames]

            # Normalize weights
            weights = rms_gated / np.sum(rms_gated)
            brightness_hz = float(np.sum(centroid_gated * weights))

        # Normalize to 0-1 (typical range: 0 - sr/2)
        brightness_normalized = brightness_hz / (sr / 2.0)
        brightness_normalized = np.clip(brightness_normalized, 0.0, 1.0)

        return brightness_normalized

    def _calculate_bass_energy(self, y: np.ndarray, sr: int) -> tuple[float, float, float]:
        """
        Calculate bass energy in 3 frequency bands.

        For techno: sub-bass (20-60Hz), kick (60-120Hz), harmonics (120-250Hz)

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Tuple of (sub_bass, kick, harmonics) energies, normalized 0-1
        """
        # Compute STFT
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)

        # Define frequency bands
        sub_bass_mask = (freqs >= 20) & (freqs < 60)
        kick_mask = (freqs >= 60) & (freqs < 120)
        harmonics_mask = (freqs >= 120) & (freqs < 250)

        # Calculate mean energy in each band
        sub_bass_energy = float(np.mean(S[sub_bass_mask, :]))
        kick_energy = float(np.mean(S[kick_mask, :]))
        harmonics_energy = float(np.mean(S[harmonics_mask, :]))

        # Normalize relative to total low-freq energy (20-250Hz)
        low_freq_mask = (freqs >= 20) & (freqs < 250)
        total_low_energy = float(np.mean(S[low_freq_mask, :]))

        if total_low_energy > 0:
            sub_bass_norm = sub_bass_energy / total_low_energy
            kick_norm = kick_energy / total_low_energy
            harmonics_norm = harmonics_energy / total_low_energy
        else:
            sub_bass_norm = 0.0
            kick_norm = 0.0
            harmonics_norm = 0.0

        # Clip to 0-1 (ratios should already be in range, but ensure it)
        sub_bass_norm = np.clip(sub_bass_norm, 0.0, 1.0)
        kick_norm = np.clip(kick_norm, 0.0, 1.0)
        harmonics_norm = np.clip(harmonics_norm, 0.0, 1.0)

        return sub_bass_norm, kick_norm, harmonics_norm

    def _calculate_percussiveness(self, y: np.ndarray) -> float:
        """
        Calculate percussiveness using HPSS.

        Args:
            y: Audio time series

        Returns:
            Percussive ratio (0-1)
        """
        # Harmonic-percussive source separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # Calculate RMS of percussive and total components
        percussive_rms = np.sqrt(np.mean(y_percussive**2))
        total_rms = np.sqrt(np.mean(y**2))

        if total_rms > 0:
            perc_ratio = float(percussive_rms / total_rms)
        else:
            perc_ratio = 0.0

        # Already in 0-1 range as a ratio
        perc_ratio = np.clip(perc_ratio, 0.0, 1.0)

        return perc_ratio

    def _calculate_onset_strength(self, y: np.ndarray, sr: int) -> float:
        """
        Calculate onset strength (beat intensity).

        Args:
            y: Audio time series
            sr: Sample rate

        Returns:
            Normalized onset strength (0-1)
        """
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_mean = float(np.mean(onset_env))

        # Normalize (typical range is 0-10 for onset strength)
        onset_normalized = float(np.clip(onset_mean / 10.0, 0.0, 1.0))

        return onset_normalized

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
            # Only hash first 1MB for speed
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

    def _load_from_cache(self, file_hash: str) -> Optional[IntensityFeatures]:
        """Load features from cache."""
        cache_path = self._get_cache_path(file_hash)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
            return IntensityFeatures.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load cached analysis: {e}")
            return None

    def clear_cache(self) -> None:
        """Clear all cached analysis results."""
        if not self.cache_dir.exists():
            return

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        logger.info(f"Cleared {count} cached analyses")
