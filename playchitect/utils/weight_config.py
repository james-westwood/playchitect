"""User-configurable feature weight overrides for clustering.

Provides a WeightOverrides dataclass and functions to load overrides from YAML
and apply them to weight arrays used in K-means clustering.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from playchitect.core.weighting import FEATURE_NAMES


@dataclass
class WeightOverrides:
    """User-configurable feature weight overrides for K-means clustering.

    All fields default to None, meaning "use the computed/default weight".
    When a field is set to a float value, it overrides the corresponding
    feature weight in the clustering algorithm.

    Feature order matches FEATURE_NAMES from weighting.py:
    bpm, rms_energy, brightness, sub_bass, kick_energy,
    bass_harmonics, percussiveness, onset_strength
    """

    bpm: float | None = None
    rms_energy: float | None = None
    brightness: float | None = None
    sub_bass: float | None = None
    kick_energy: float | None = None
    bass_harmonics: float | None = None
    percussiveness: float | None = None
    onset_strength: float | None = None


def load_weight_overrides(path: Path) -> WeightOverrides:
    """Load weight overrides from a YAML file.

    The YAML file should contain a top-level 'weights:' key mapping
    feature names to float values. Features not specified default to None.

    Example YAML:
        weights:
          bpm: 2.0
          rms_energy: 1.5
          # Other features use default weights (None)

    Args:
        path: Path to the YAML configuration file.

    Returns:
        WeightOverrides dataclass with parsed values.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    weights_data = data.get("weights", {})
    if weights_data is None:
        weights_data = {}

    # Build kwargs only for valid WeightOverrides fields
    kwargs: dict[str, float | None] = {}
    valid_fields = {f for f in WeightOverrides.__dataclass_fields__}

    for field_name in valid_fields:
        if field_name in weights_data:
            value = weights_data[field_name]
            if value is not None:
                kwargs[field_name] = float(value)

    return WeightOverrides(**kwargs)


def apply_weight_overrides(
    weights: np.ndarray,
    overrides: WeightOverrides,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> np.ndarray:
    """Apply user overrides to a weight array.

    Replaces entries in the weights array where the corresponding override
    is not None. The order of features in the weights array must match
    the feature_names tuple.

    Args:
        weights: Original weight array, shape (n_features,) or (n_clusters, n_features).
        overrides: WeightOverrides dataclass with user-specified values.
        feature_names: Tuple of feature name strings defining the order.

    Returns:
        New weight array with overrides applied. Original weights are
        copied and not modified in-place.

    Raises:
        ValueError: If weights shape doesn't match feature_names length.
    """
    result = weights.copy()

    # Build a mapping from feature name to its index
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    # Iterate over all fields in the dataclass
    for field_name in WeightOverrides.__dataclass_fields__:
        override_value = getattr(overrides, field_name)
        if override_value is not None and field_name in name_to_idx:
            idx = name_to_idx[field_name]

            # Handle both 1D (global weights) and 2D (per-cluster weights) arrays
            if result.ndim == 1:
                result[idx] = override_value
            elif result.ndim == 2:
                result[:, idx] = override_value
            else:
                raise ValueError(f"Unsupported weights ndim: {result.ndim}")

    return result
