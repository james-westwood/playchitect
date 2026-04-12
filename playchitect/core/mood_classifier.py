"""Mood classification for audio tracks based on intensity features.

Uses a rule-based decision tree to classify tracks into one of 9 mood categories:
- Dark: Low brightness, low vocal presence
- Euphoric: High energy, high brightness
- Melancholic: Low energy, high vocal presence
- Energetic: High RMS + high percussiveness (catch-all for techno/electronic)
- Aggressive: High percussiveness, high onset strength
- Dreamy: Low percussiveness, high brightness
- Hypnotic: Mid energy, low dynamic range
- Groovy: Mid-high percussiveness, mid energy
- Ethereal: Catch-all default
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playchitect.core.intensity_analyzer import IntensityFeatures

# Mood classification thresholds
# Dark: low brightness + low vocal
_BRIGHTNESS_DARK_THRESHOLD: float = 0.3
_VOCAL_DARK_THRESHOLD: float = 0.3

# Euphoric: high energy + high brightness
_ENERGY_EUPHORIC_THRESHOLD: float = 0.7
_BRIGHTNESS_EUPHORIC_THRESHOLD: float = 0.6

# Melancholic: low energy + high vocal
_ENERGY_MELANCHOLIC_THRESHOLD: float = 0.3
_VOCAL_MELANCHOLIC_THRESHOLD: float = 0.5

# Energetic: high RMS + high percussiveness (techno/electronic)
# Uses higher percussiveness (>=0.505) than Groovy (>=0.5) but lower than Aggressive (>=0.7)
# Requires higher brightness (>=0.54) to distinguish from Groovy
_RMS_ENERGETIC_THRESHOLD: float = 0.11
_PERCUSSIVE_ENERGETIC_THRESHOLD: float = 0.505
_BRIGHTNESS_ENERGETIC_THRESHOLD: float = 0.54

# Aggressive: very high percussiveness + high onset
_PERCUSSIVE_AGGRESSIVE_THRESHOLD: float = 0.7
_ONSET_AGGRESSIVE_THRESHOLD: float = 0.6

# Dreamy: low percussiveness + high brightness
_PERCUSSIVE_DREAMY_THRESHOLD: float = 0.3
_BRIGHTNESS_DREAMY_THRESHOLD: float = 0.5

# Hypnotic: mid energy + low dynamic range
_ENERGY_HYPNOTIC_LOW: float = 0.3
_ENERGY_HYPNOTIC_HIGH: float = 0.6
_DYNAMIC_RANGE_HYPNOTIC_THRESHOLD: float = 0.3

# Groovy: mid-high percussiveness + mid energy
_PERCUSSIVE_GROOVY_THRESHOLD: float = 0.5
_ENERGY_GROOVY_LOW: float = 0.4
_ENERGY_GROOVY_HIGH: float = 0.7

# Default value for missing features
_MISSING_FEATURE_DEFAULT: float = 0.0


def classify_mood(features: IntensityFeatures) -> str:
    """
    Classify a track's mood using a decision tree of feature thresholds.

    The classification follows a priority order:
    1. Dark (low brightness + low vocal)
    2. Euphoric (high energy + high brightness)
    3. Melancholic (low energy + high vocal)
    4. Aggressive (very high percussive + high onset, for harsh electronic)
    5. Energetic (high RMS + high percussiveness, for techno/electronic)
    6. Dreamy (low percussive + high brightness)
    7. Hypnotic (mid energy + low dynamic range)
    8. Groovy (mid-high percussive + mid energy)
    9. Ethereal (catch-all default)

    Args:
        features: IntensityFeatures containing all audio analysis data

    Returns:
        One of: 'Dark', 'Euphoric', 'Melancholic', 'Aggressive', 'Energetic',
                'Dreamy', 'Hypnotic', 'Groovy', 'Ethereal'
    """
    # Get feature values with defaults for backwards compatibility
    brightness = features.brightness
    energy = features.rms_energy
    percussiveness = features.percussiveness
    onset_strength = features.onset_strength
    vocal_presence = getattr(features, "vocal_presence", _MISSING_FEATURE_DEFAULT)
    dynamic_range = getattr(features, "dynamic_range", _MISSING_FEATURE_DEFAULT)

    # Priority 1: Dark - low brightness, low vocal presence
    if brightness < _BRIGHTNESS_DARK_THRESHOLD and vocal_presence < _VOCAL_DARK_THRESHOLD:
        return "Dark"

    # Priority 2: Euphoric - high energy, high brightness
    if energy > _ENERGY_EUPHORIC_THRESHOLD and brightness > _BRIGHTNESS_EUPHORIC_THRESHOLD:
        return "Euphoric"

    # Priority 3: Melancholic - low energy, high vocal presence
    if energy < _ENERGY_MELANCHOLIC_THRESHOLD and vocal_presence > _VOCAL_MELANCHOLIC_THRESHOLD:
        return "Melancholic"

    # Priority 4: Aggressive - very high percussiveness, high onset strength
    # (before Energetic to ensure Aggressive takes precedence for harsh electronic)
    if (
        percussiveness > _PERCUSSIVE_AGGRESSIVE_THRESHOLD
        and onset_strength > _ONSET_AGGRESSIVE_THRESHOLD
    ):
        return "Aggressive"

    # Priority 5: Energetic - high RMS + high percussiveness (techno/electronic)
    # This catches high-energy electronic music that falls through to Ethereal
    if (
        energy > _RMS_ENERGETIC_THRESHOLD
        and percussiveness > _PERCUSSIVE_ENERGETIC_THRESHOLD
        and brightness > _BRIGHTNESS_ENERGETIC_THRESHOLD
    ):
        return "Energetic"

    # Priority 6: Dreamy - low percussiveness, high brightness
    if percussiveness < _PERCUSSIVE_DREAMY_THRESHOLD and brightness > _BRIGHTNESS_DREAMY_THRESHOLD:
        return "Dreamy"

    # Priority 6: Hypnotic - mid energy, low dynamic range
    if (
        _ENERGY_HYPNOTIC_LOW <= energy <= _ENERGY_HYPNOTIC_HIGH
        and dynamic_range < _DYNAMIC_RANGE_HYPNOTIC_THRESHOLD
    ):
        return "Hypnotic"

    # Priority 7: Groovy - mid-high percussiveness, mid energy
    if (
        percussiveness > _PERCUSSIVE_GROOVY_THRESHOLD
        and _ENERGY_GROOVY_LOW <= energy <= _ENERGY_GROOVY_HIGH
    ):
        return "Groovy"

    # Priority 8: Ethereal - catch-all default
    return "Ethereal"
