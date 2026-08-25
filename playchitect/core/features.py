"Shared 8-D feature vector construction for clustering and similarity ranking."

import logging

import numpy as np

from playchitect.core.intensity_analyzer import IntensityFeatures
from playchitect.core.metadata_extractor import TrackMetadata
from playchitect.core.weighting import (
    FEATURE_NAMES,  # noqa: F401  # authoritative ordering reference
)

logger = logging.getLogger(__name__)

__all__ = ["build_feature_vector"]


def build_feature_vector(
    metadata: TrackMetadata,
    features: IntensityFeatures | None,
) -> np.ndarray | None:
    """Build an 8-D feature vector from metadata and intensity features.

    The 8 values are, in order matching FEATURE_NAMES:
    bpm, rms_energy, brightness, sub_bass_energy, kick_energy,
    bass_harmonics, percussiveness, onset_strength.

    Args:
        metadata: Track metadata containing BPM.
        features: Intensity analysis features (7 audio features).

    Returns:
        An 8-element numpy array, or None if BPM is missing or features
        is None. The returned array is a copy safe for mutation.
    """
    if metadata.bpm is None or features is None:
        return None

    return np.array(
        [
            metadata.bpm,
            features.rms_energy,
            features.brightness,
            features.sub_bass_energy,
            features.kick_energy,
            features.bass_harmonics,
            features.percussiveness,
            features.onset_strength,
        ],
        dtype=float,
    ).copy()
