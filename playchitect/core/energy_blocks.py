"""Energy block management for interactive set building.

Energy blocks represent segments of a DJ set with specific energy characteristics,
allowing users to build sets with intentional energy flow (warm-up, build, peak, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from playchitect.core.clustering import ClusterResult
    from playchitect.core.intensity_analyzer import IntensityFeatures


@dataclass
class EnergyBlock:
    """Represents an energy block in a DJ set.

    Attributes:
        id: Unique identifier for the block
        name: Human-readable name (e.g., "Warm-up", "Peak")
        target_duration_min: Target duration in minutes
        energy_min: Minimum energy level (0-1, based on rms_energy)
        energy_max: Maximum energy level (0-1, based on rms_energy)
        cluster_ids: List of cluster IDs assigned to this block
    """

    id: str
    name: str
    target_duration_min: float
    energy_min: float
    energy_max: float
    cluster_ids: list[int | str]


# Energy block definitions with percentile ranges
# Percentiles determine how clusters are distributed across blocks
_WARM_UP_PERCENTILE = 0.20  # Lowest 20%
_BUILD_PERCENTILE = 0.50  # 20-50%
_PEAK_PERCENTILE = 0.80  # 50-80%
_SUSTAIN_PERCENTILE = 0.95  # 80-95%

# Block names
_BLOCK_WARM_UP = "Warm-up"
_BLOCK_BUILD = "Build"
_BLOCK_PEAK = "Peak"
_BLOCK_SUSTAIN = "Sustain"
_BLOCK_WIND_DOWN = "Wind Down"
_BLOCK_CUSTOM = "Custom"

# Duration per track in minutes (average track length assumption)
_TRACK_DURATION_MIN = 6.0


def suggest_blocks(
    clusters: list[ClusterResult],
    features: dict[Path, IntensityFeatures],
) -> list[EnergyBlock]:
    """Suggest energy blocks based on cluster energy levels.

    Sorts clusters by mean RMS energy, then divides them into 4-5 named
    energy blocks with appropriate duration targets.

    Args:
        clusters: List of ClusterResult objects from clustering
        features: Mapping of file paths to IntensityFeatures

    Returns:
        List of EnergyBlock objects sorted by energy level

    Raises:
        ValueError: If clusters list is empty
    """
    if not clusters:
        raise ValueError("Cannot create energy blocks from empty cluster list")

    # Calculate mean RMS energy for each cluster
    cluster_energies: list[tuple[ClusterResult, float]] = []
    for cluster in clusters:
        rms_values = [features[path].rms_energy for path in cluster.tracks if path in features]
        mean_rms = float(np.mean(rms_values)) if rms_values else 0.0
        cluster_energies.append((cluster, mean_rms))

    # Sort by mean RMS energy (ascending)
    cluster_energies.sort(key=lambda x: x[1])

    n_clusters = len(cluster_energies)

    # Handle edge case: single cluster creates a single Warm-up block
    if n_clusters == 1:
        cluster, mean_rms = cluster_energies[0]
        return [
            EnergyBlock(
                id="warm-up",
                name=_BLOCK_WARM_UP,
                target_duration_min=cluster.track_count * _TRACK_DURATION_MIN,
                energy_min=mean_rms,
                energy_max=mean_rms,
                cluster_ids=[cluster.cluster_id],
            )
        ]

    # Determine block boundaries based on percentiles
    # Use integer division to handle edge cases cleanly
    warm_up_end = max(1, int(n_clusters * _WARM_UP_PERCENTILE))
    build_end = max(warm_up_end + 1, int(n_clusters * _BUILD_PERCENTILE))
    peak_end = max(build_end + 1, int(n_clusters * _PEAK_PERCENTILE))
    sustain_end = max(peak_end + 1, int(n_clusters * _SUSTAIN_PERCENTILE))

    blocks: list[EnergyBlock] = []

    # Warm-up block (lowest energy)
    if warm_up_end > 0:
        warm_up_clusters = cluster_energies[:warm_up_end]
        energy_min = min(c[1] for c in warm_up_clusters)
        energy_max = max(c[1] for c in warm_up_clusters)
        track_count = sum(c[0].track_count for c in warm_up_clusters)
        blocks.append(
            EnergyBlock(
                id="warm-up",
                name=_BLOCK_WARM_UP,
                target_duration_min=track_count * _TRACK_DURATION_MIN,
                energy_min=energy_min,
                energy_max=energy_max,
                cluster_ids=[c[0].cluster_id for c in warm_up_clusters],
            )
        )

    # Build block (20-50%)
    build_clusters = cluster_energies[warm_up_end:build_end]
    if build_clusters:
        energy_min = min(c[1] for c in build_clusters)
        energy_max = max(c[1] for c in build_clusters)
        track_count = sum(c[0].track_count for c in build_clusters)
        blocks.append(
            EnergyBlock(
                id="build",
                name=_BLOCK_BUILD,
                target_duration_min=track_count * _TRACK_DURATION_MIN,
                energy_min=energy_min,
                energy_max=energy_max,
                cluster_ids=[c[0].cluster_id for c in build_clusters],
            )
        )

    # Peak block (50-80%)
    peak_clusters = cluster_energies[build_end:peak_end]
    if peak_clusters:
        energy_min = min(c[1] for c in peak_clusters)
        energy_max = max(c[1] for c in peak_clusters)
        track_count = sum(c[0].track_count for c in peak_clusters)
        blocks.append(
            EnergyBlock(
                id="peak",
                name=_BLOCK_PEAK,
                target_duration_min=track_count * _TRACK_DURATION_MIN,
                energy_min=energy_min,
                energy_max=energy_max,
                cluster_ids=[c[0].cluster_id for c in peak_clusters],
            )
        )

    # Sustain block (80-95%)
    sustain_clusters = cluster_energies[peak_end:sustain_end]
    if sustain_clusters:
        energy_min = min(c[1] for c in sustain_clusters)
        energy_max = max(c[1] for c in sustain_clusters)
        track_count = sum(c[0].track_count for c in sustain_clusters)
        blocks.append(
            EnergyBlock(
                id="sustain",
                name=_BLOCK_SUSTAIN,
                target_duration_min=track_count * _TRACK_DURATION_MIN,
                energy_min=energy_min,
                energy_max=energy_max,
                cluster_ids=[c[0].cluster_id for c in sustain_clusters],
            )
        )

    # Wind Down block (top 5% or remaining)
    wind_down_clusters = cluster_energies[sustain_end:]
    if wind_down_clusters:
        energy_min = min(c[1] for c in wind_down_clusters)
        energy_max = max(c[1] for c in wind_down_clusters)
        track_count = sum(c[0].track_count for c in wind_down_clusters)
        blocks.append(
            EnergyBlock(
                id="wind-down",
                name=_BLOCK_WIND_DOWN,
                target_duration_min=track_count * _TRACK_DURATION_MIN,
                energy_min=energy_min,
                energy_max=energy_max,
                cluster_ids=[c[0].cluster_id for c in wind_down_clusters],
            )
        )

    return blocks


def create_custom_block(
    block_id: str,
    target_duration_min: float = 60.0,
    energy_min: float = 0.0,
    energy_max: float = 1.0,
) -> EnergyBlock:
    """Create a custom energy block.

    Args:
        block_id: Unique identifier for the block
        target_duration_min: Target duration in minutes
        energy_min: Minimum energy level (0-1)
        energy_max: Maximum energy level (0-1)

    Returns:
        New EnergyBlock with no assigned clusters
    """
    return EnergyBlock(
        id=block_id,
        name=_BLOCK_CUSTOM,
        target_duration_min=target_duration_min,
        energy_min=energy_min,
        energy_max=energy_max,
        cluster_ids=[],
    )


def get_block_energy_range(blocks: list[EnergyBlock]) -> tuple[float, float]:
    """Get the overall energy range across all blocks.

    Args:
        blocks: List of EnergyBlock objects

    Returns:
        Tuple of (min_energy, max_energy) across all blocks
    """
    if not blocks:
        return (0.0, 1.0)

    all_mins = [b.energy_min for b in blocks]
    all_maxs = [b.energy_max for b in blocks]

    return (min(all_mins), max(all_maxs))
